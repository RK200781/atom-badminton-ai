# -*- coding: utf-8 -*-
"""
Arduino(stattle.ino:ローラーと装填)の操作。

ノートブック Untitled-2.ipynb のセル4(send_serial)と stattle.ino の仕様を元に書き直したもの。

Arduino との取り決め(stattle.ino で確認済み):
- "0,N\\r\\n" … ローラーの強さを N にする(0=停止、1=弱20%、2=中50%、3=強100%)。ino:16〜20、231〜240
- "1\\r\\n"   … 発射。装填用モーターを約1回転させてシャトルをローラーへ送り込む。ino:21、157〜204
- Arduino からは約10msごとに、シャトルを検知していれば "1"、していなければ "0" の行が届く。ino:108

Arduino は指令を無視しても何も返さないので、発射してよいかは Python 側で判定する
(fire_block_reasons)。強さが0のときに送らないことで、Arduino の
「起動直後の1回だけ強さ0でも発射してしまう」不具合(ino:54)も避けている。
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import config

log = logging.getLogger("launcher.feeder")

VALID_SPEEDS = (0, 1, 2, 3)

# 発射の確認結果
FIRE_PENDING = "pending"          # 確認中
FIRE_CONFIRMED = "confirmed"      # 検知が 1→0 に変わった
FIRE_UNCONFIRMED = "unconfirmed"  # 時間内に変わらなかった


def parse_detection_line(line) -> Optional[bool]:
    """Arduino から届いた1行を読む。"1" なら True、"0" なら False、それ以外は None。"""
    if isinstance(line, bytes):
        line = line.decode("ascii", errors="replace")
    text = line.strip()
    if text == "1":
        return True
    if text == "0":
        return False
    return None


# ============================================================
# 発射してよいかの判定
# ============================================================

@dataclass
class FeederSnapshot:
    connected: bool
    error: Optional[str]
    speed: int                       # Python 側で覚えている強さ
    detected: Optional[bool]         # 最後に届いたシャトル検知の値
    detected_at: Optional[float]     # それを受け取った時刻
    roller_ready_at: float           # この時刻を過ぎればローラーが回っているとみなす
    last_fire_at: Optional[float]    # 最後に発射の指令を送った時刻
    fire_result: Optional[str]       # 最後の発射の確認結果


def fire_block_reasons(snap: FeederSnapshot, now: float, hold_running: bool) -> List[str]:
    """発射できない理由の一覧を返す。空なら発射してよい。"""
    reasons = []
    if not snap.connected or snap.error:
        reasons.append("Arduino と通信できていません")
    if snap.speed not in (1, 2, 3):
        reasons.append("強さが0です(2/3/4キーで強さを設定)")
    if snap.detected_at is None or now - snap.detected_at > config.SHUTTLE_FRESH_SEC:
        reasons.append("Arduino からシャトル検知の情報が届いていません")
    elif not snap.detected:
        reasons.append("シャトルを検知していません")
    if snap.speed in (1, 2, 3) and now < snap.roller_ready_at:
        reasons.append(f"ローラーの回転待ち(あと{snap.roller_ready_at - now:.1f}秒)")
    if snap.last_fire_at is not None and now - snap.last_fire_at < config.FIRE_INTERVAL_SEC:
        reasons.append(f"前の発射から間がありません(あと{config.FIRE_INTERVAL_SEC - (now - snap.last_fire_at):.1f}秒)")
    if not hold_running:
        reasons.append("モーターの保持ループが止まっています(R キーで再開)")
    return reasons


# ============================================================
# シリアルポート(実機用。--mock のときは mock.FakeArduinoSerial を使う)
# ============================================================

def open_serial(port: str = config.SERIAL_PORT, baud: int = config.SERIAL_BAUD):
    import serial  # 実機のときだけ必要なので、ここで読み込む
    try:
        ser = serial.Serial(port, baud, timeout=config.SERIAL_TIMEOUT_SEC)
    except serial.SerialException as e:
        raise ConnectionError(
            f"Arduino のポート {port} を開けませんでした({e})。"
            "USB がつながっているか、デバイスマネージャーの「ポート(COM と LPT)」で番号を確認し、"
            "違っていれば --serial-port COM番号 で指定してください。"
            "ノートブックなど他のプログラムがポートを使っていないかも確認してください。"
        ) from e
    log.info("Arduino のポートを開きました: %s %d bps", port, baud)
    return ser


# ============================================================
# Arduino の操作
# ============================================================

class Feeder:
    def __init__(self, serial_factory: Callable[[], object], clock=time.monotonic, sleep=time.sleep):
        self._serial_factory = serial_factory
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._ser = None
        self._reader: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[str] = None

        self._speed = 0
        self._detected: Optional[bool] = None
        self._detected_at: Optional[float] = None
        self._last_present_at = clock()
        self._roller_ready_at = 0.0
        self._last_fire_at: Optional[float] = None
        self._fire_result: Optional[str] = None

    # ---- 接続 --------------------------------------------------------------

    def open(self, start_reader: bool = True) -> None:
        """ポートを開き、Arduino のリセットを待ってから受信スレッドを始める。"""
        self._ser = self._serial_factory()
        log.info("Arduino のリセットを待っています(%.1f秒)...", config.ARDUINO_RESET_WAIT_SEC)
        self._sleep(config.ARDUINO_RESET_WAIT_SEC)
        if hasattr(self._ser, "reset_input_buffer"):
            self._ser.reset_input_buffer()  # リセット前後の途中の行を捨てる
        with self._lock:
            self._last_present_at = self._clock()  # Arduino の 10 秒タイマーはリセットで 0 から始まる
        if start_reader:
            self._stop_event.clear()
            self._reader = threading.Thread(target=self._read_loop, name="arduino-rx", daemon=True)
            self._reader.start()

    def stop_roller(self) -> None:
        """終了時用:ローラーを止める("0,0" を送る)。エラーでも止まらない。"""
        if self._ser is None:
            return
        try:
            self._write("0,0")
            with self._lock:
                self._speed = 0
            log.info("Arduino にローラー停止(0,0)を送りました。")
        except Exception as e:
            log.error("Arduino に停止(0,0)を送れませんでした: %s", e)

    def close(self, send_stop: bool = True) -> None:
        """("0,0" を送ってから)受信スレッドを止めてポートを閉じる。何度呼んでもよい。"""
        if send_stop:
            self.stop_roller()
        self._stop_event.set()
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
                log.info("Arduino のポートを閉じました。")
            except Exception as e:
                log.warning("Arduino のポートを閉じるときにエラー: %s", e)
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)

    # ---- 指令 --------------------------------------------------------------

    def set_speed(self, speed: int) -> None:
        """ローラーの強さを 0〜3 で指定する。"""
        if isinstance(speed, bool) or speed not in VALID_SPEEDS:
            raise ValueError(f"強さは 0〜3 で指定してください: {speed!r}")
        self._write(f"0,{speed}")
        now = self._clock()
        with self._lock:
            if speed != self._speed:
                self._speed = speed
                self._roller_ready_at = now + config.ROLLER_SPINUP_SEC
        log.info("ローラーの強さを %d(%s)にしました。", speed, config.SPEED_LABELS[speed])

    def fire(self, hold_running: bool) -> Tuple[bool, List[str]]:
        """条件がそろっていれば "1" を送る。(送ったか, 送らなかった理由) を返す。"""
        now = self._clock()
        reasons = fire_block_reasons(self.snapshot(), now, hold_running)
        if reasons:
            log.warning("発射しませんでした: %s", " / ".join(reasons))
            return False, reasons
        self._write("1")
        with self._lock:
            self._last_fire_at = now
            self._fire_result = FIRE_PENDING
        log.info("発射の指令を送りました。")
        return True, []

    def block_reasons(self, hold_running: bool) -> List[str]:
        return fire_block_reasons(self.snapshot(), self._clock(), hold_running)

    def _write(self, command: str) -> None:
        if self._ser is None:
            raise RuntimeError("Arduino に接続していません")
        self._ser.write(f"{command}\r\n".encode("ascii"))

    # ---- 状態 --------------------------------------------------------------

    def snapshot(self) -> FeederSnapshot:
        with self._lock:
            return FeederSnapshot(
                connected=self._ser is not None,
                error=self._error,
                speed=self._speed,
                detected=self._detected,
                detected_at=self._detected_at,
                roller_ready_at=self._roller_ready_at,
                last_fire_at=self._last_fire_at,
                fire_result=self._fire_result,
            )

    def shuttle_detected(self) -> Optional[bool]:
        """最新のシャトル検知。届いていない・古いときは None。"""
        snap = self.snapshot()
        if snap.detected_at is None or self._clock() - snap.detected_at > config.SHUTTLE_FRESH_SEC:
            return None
        return snap.detected

    # ---- 受信 --------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                line = self._ser.readline() if self._ser is not None else b""
                now = self._clock()
                if line:
                    self.handle_line(line, now)
                else:
                    self.check_fire_timeout(now)
        except Exception as e:
            if self._stop_event.is_set():
                return  # 閉じている最中のエラーは無視する
            with self._lock:
                self._error = f"{type(e).__name__}: {e}"
            log.error("Arduino からの受信が止まりました: %s", self._error)

    def handle_line(self, line, now: float) -> None:
        """Arduino から届いた1行を処理する(受信スレッドから呼ばれる)。"""
        value = parse_detection_line(line)
        if value is None:
            return
        result = None
        with self._lock:
            if value:
                # Arduino は検知なしが 10 秒続くとローラーを止め、シャトルが入った瞬間に
                # 元の強さで回し直す(ino:119〜121)。そのときは回転待ちをやり直す。
                was_absent_long = now - self._last_present_at > config.ARDUINO_NO_SHUTTLE_TIMEOUT_SEC
                if self._detected is not True and was_absent_long and self._speed in (1, 2, 3):
                    self._roller_ready_at = now + config.ROLLER_SPINUP_SEC
                    log.info("シャトルを検知しました。Arduino がローラーを回し直すので %.1f 秒待ちます。",
                             config.ROLLER_SPINUP_SEC)
                self._last_present_at = now
            self._detected = value
            self._detected_at = now
            if self._fire_result == FIRE_PENDING and not value:
                self._fire_result = result = FIRE_CONFIRMED
        if result == FIRE_CONFIRMED:
            log.info("発射を確認しました(シャトル検知が 1→0)。")
        self.check_fire_timeout(now)

    def check_fire_timeout(self, now: float) -> None:
        with self._lock:
            timed_out = (self._fire_result == FIRE_PENDING and self._last_fire_at is not None
                         and now - self._last_fire_at > config.FIRE_CONFIRM_SEC)
            if timed_out:
                self._fire_result = FIRE_UNCONFIRMED
        if timed_out:
            log.warning("発射を確認できませんでした(%.1f 秒以内にシャトル検知が 0 になりませんでした)。",
                        config.FIRE_CONFIRM_SEC)
