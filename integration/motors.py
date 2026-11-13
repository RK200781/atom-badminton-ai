# -*- coding: utf-8 -*-
"""
CubeMars AK45-10 モーター(yaw=105、pitch=104)の CAN 操作。

ノートブック Untitled-2.ipynb のセル2・セル3 を元に .py として書き直したもの。

- CAN の送受信は1つのスレッド(CAN スレッド)だけが行う。
  50ms ごとに両方のモーターへ目標角を送り(位置の保持ループ)、その合間に受け取った
  フィードバックのフレームから現在角を更新し続ける。
- 可動範囲のチェックは、CAN に送る直前(_send_targets)の1か所だけで行う。
  範囲外の目標角は端に丸めて警告をログに出し、丸めた値を目標角に書き戻す
  (キーを押し続けて範囲を大きく超えても、逆向きのキーですぐ戻れるようにするため)。
- 通信エラーが起きたら CAN スレッドは止まり、error に理由が残る。
  restart() で接続を開き直して再開できる(「動作中」の印が残って再開できない、
  ノートブックの不具合は、スレッドが生きているかどうかで判定することで直している)。
"""

import logging
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import config

log = logging.getLogger("launcher.motors")


# ============================================================
# 軸の定義
# ============================================================

@dataclass(frozen=True)
class Axis:
    name: str
    motor_id: int
    origin_deg: float
    min_deg: float
    max_deg: float


YAW = Axis("yaw", config.YAW_ID, config.YAW_ORIGIN_DEG, config.YAW_MIN_DEG, config.YAW_MAX_DEG)
PITCH = Axis("pitch", config.PITCH_ID, config.PITCH_ORIGIN_DEG, config.PITCH_MIN_DEG, config.PITCH_MAX_DEG)
AXES = (YAW, PITCH)
AXIS_BY_ID = {axis.motor_id: axis for axis in AXES}


def clamp_angle(axis: Axis, angle_deg: float) -> Tuple[float, bool]:
    """原点からの角度を可動範囲に丸める。(丸めた値, 丸めたかどうか) を返す。"""
    clamped = max(axis.min_deg, min(axis.max_deg, float(angle_deg)))
    return clamped, clamped != float(angle_deg)


# ============================================================
# CAN フレームの組み立て・読み取り(セル3:1〜209 と同じ計算)
# ============================================================

def output_rpm_to_erpm(output_rpm: float) -> float:
    """出力軸の回転速度[rpm] → CubeMars の ERPM。セル3:76〜81"""
    return output_rpm * config.GEAR_RATIO * config.POLE_PAIRS


def encode_position_speed(motor_id: int, motor_angle_deg: float,
                          output_speed_rpm: float, output_accel_rpm_s: float) -> Tuple[int, bytes]:
    """Position-Speed Loop の指令フレーム (拡張ID, 8バイト) を作る。セル3:88〜201

    速度・加速度はノートブックと同じく 10 で割らずに送る(config.py の【要確認】参照)。
    """
    speed_rpm = abs(float(output_speed_rpm))
    if speed_rpm > config.MAX_OUTPUT_RPM:
        log.warning("速度 %.1f rpm は上限を超えています。%.1f rpm に制限します。",
                    speed_rpm, config.MAX_OUTPUT_RPM)
        speed_rpm = config.MAX_OUTPUT_RPM

    position_cmd = int(round(motor_angle_deg * 10000.0))
    speed_cmd = int(round(output_rpm_to_erpm(speed_rpm)))
    accel_cmd = int(round(output_accel_rpm_s * config.GEAR_RATIO * config.POLE_PAIRS))
    speed_cmd = max(-32768, min(32767, speed_cmd))
    accel_cmd = max(-32768, min(32767, accel_cmd))

    arbitration_id = (config.CAN_PACKET_SET_POS_SPD << 8) | motor_id
    data = struct.pack(">ihh", position_cmd, speed_cmd, accel_cmd)
    return arbitration_id, data


def decode_feedback(arbitration_id: int, data: bytes) -> Optional[Tuple[int, float]]:
    """状態フィードバックのフレームなら (モーターID, 絶対角度[deg]) を、違えば None を返す。セル3:35〜68"""
    function_id = (arbitration_id >> 8) & 0xFF
    motor_id = arbitration_id & 0xFF
    if function_id != config.CAN_FEEDBACK_FUNCTION_ID or len(data) != 8:
        return None
    position_raw = int.from_bytes(data[0:2], byteorder="big", signed=True)
    return motor_id, position_raw * 0.1


# ============================================================
# CAN ポート(実機用。--mock のときは mock.FakeCanPort を使う)
# ============================================================

class PythonCanPort:
    """python-can の Bus を、送受信に必要な最小限の形で包む。"""

    def __init__(self, interface=config.CAN_INTERFACE, channel=config.CAN_CHANNEL,
                 bitrate=config.CAN_BITRATE):
        import can  # 実機のときだけ必要なので、ここで読み込む
        self._can = can
        try:
            self._bus = can.Bus(interface=interface, channel=channel, bitrate=bitrate)
        except Exception as e:
            raise ConnectionError(
                f"USB-CAN({channel})を開けませんでした({type(e).__name__}: {e})。"
                "USB がつながっているか、デバイスマネージャーの「ポート(COM と LPT)」で番号を確認し、"
                "違っていれば --can-channel COM番号@115200 で指定してください。"
                "ノートブックなど他のプログラムがポートを使っていないかも確認してください。"
            ) from e
        log.info("CAN を開きました: %s %s %d bps", interface, channel, bitrate)

    def send(self, arbitration_id: int, data: bytes) -> None:
        self._bus.send(self._can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True))

    def recv(self, timeout: float):
        msg = self._bus.recv(timeout=timeout)
        if msg is None:
            return None
        return msg.arbitration_id, bytes(msg.data)

    def shutdown(self) -> None:
        self._bus.shutdown()


# ============================================================
# モーターの操作
# ============================================================

@dataclass
class MotorState:
    running: bool                    # CAN スレッド(保持ループ)が動いているか
    error: Optional[str]             # 止まった理由(動いている間は None)
    target_yaw: float                # 原点からの目標角 [deg]
    target_pitch: float
    current_yaw: Optional[float]     # 原点からの現在角 [deg]。届いていない・古いときは None
    current_pitch: Optional[float]


class MotorController:
    def __init__(self, port_factory: Callable[[], object], clock=time.monotonic):
        self._port_factory = port_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._target = {YAW.name: config.INITIAL_YAW_DEG, PITCH.name: config.INITIAL_PITCH_DEG}
        self._current = {YAW.name: None, PITCH.name: None}  # (原点からの角度, 受け取った時刻)
        self._port = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error: Optional[str] = None

    # ---- 状態 -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> Optional[str]:
        return self._error

    def current_angle(self, axis: Axis) -> Optional[float]:
        with self._lock:
            value = self._current[axis.name]
        if value is None or self._clock() - value[1] > config.ANGLE_STALE_SEC:
            return None
        return value[0]

    def state(self) -> MotorState:
        with self._lock:
            target = dict(self._target)
        return MotorState(
            running=self.running,
            error=self._error,
            target_yaw=target[YAW.name],
            target_pitch=target[PITCH.name],
            current_yaw=self.current_angle(YAW),
            current_pitch=self.current_angle(PITCH),
        )

    # ---- 目標角の指定(CAN には CAN スレッドが送る) ----------------------------

    def set_target(self, yaw_deg: float, pitch_deg: float) -> None:
        with self._lock:
            self._target[YAW.name] = float(yaw_deg)
            self._target[PITCH.name] = float(pitch_deg)

    def nudge(self, d_yaw_deg: float, d_pitch_deg: float) -> None:
        with self._lock:
            self._target[YAW.name] += d_yaw_deg
            self._target[PITCH.name] += d_pitch_deg

    def hold_here(self) -> bool:
        """目標角を現在角に置き換えて、その場で保持する。

        現在角が分からない軸は目標角を変えない。両方とも置き換えられたら True。
        """
        replaced = 0
        for axis in AXES:
            current = self.current_angle(axis)
            if current is None:
                log.warning("%s の現在角が分からないため、目標角はそのままにします。", axis.name)
                continue
            with self._lock:
                self._target[axis.name] = current
            replaced += 1
        return replaced == len(AXES)

    # ---- 開始・再開・終了 ---------------------------------------------------

    def start(self) -> None:
        """CAN を開いて保持ループを始める。動いていれば何もしない。"""
        if self.running:
            log.info("保持ループはすでに動いています。")
            return
        if self._port is None:
            self._port = self._port_factory()
        self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="can-loop", daemon=True)
        self._thread.start()
        log.info("保持ループを開始しました(%.0fms 周期)。", config.HOLD_INTERVAL_SEC * 1000)

    def restart(self) -> None:
        """通信エラーで止まった後に、接続を開き直して再開する。"""
        if self.running:
            log.info("保持ループは動いているので、再開の必要はありません。")
            return
        self._close_port()
        self.start()

    def shutdown(self) -> None:
        """保持ループを止めて CAN を閉じる。何度呼んでもよい。"""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.error("CAN スレッドが2秒以内に止まりませんでした。")
        self._close_port()

    def _close_port(self) -> None:
        port, self._port = self._port, None
        if port is None:
            return
        try:
            port.shutdown()
            log.info("CAN を閉じました。")
        except Exception as e:  # 閉じるときのエラーでは止めない
            log.warning("CAN を閉じるときにエラー: %s", e)

    # ---- CAN スレッド ------------------------------------------------------

    def _run(self) -> None:
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self._send_targets()
                next_tick += config.HOLD_INTERVAL_SEC
                self._receive_until(next_tick)
                if time.monotonic() - next_tick > config.HOLD_INTERVAL_SEC:
                    next_tick = time.monotonic()  # 大きく遅れたら周期を取り直す
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            log.error("モーターの通信エラーで保持ループが止まりました: %s(R キーで再開)", self._error)
            self._close_port()
        else:
            log.info("保持ループを止めました。")

    def _send_targets(self) -> None:
        """両方のモーターへ目標角を送る。可動範囲のチェックはここだけで行う。"""
        for axis in AXES:
            with self._lock:
                raw = self._target[axis.name]
            clamped, was_clamped = clamp_angle(axis, raw)
            if was_clamped:
                log.warning("%s の目標角 %+.1f° は可動範囲外(%+.1f〜%+.1f°)のため %+.1f° に丸めました。",
                            axis.name, raw, axis.min_deg, axis.max_deg, clamped)
                with self._lock:
                    if self._target[axis.name] == raw:  # 読んだ後に変わっていなければ書き戻す
                        self._target[axis.name] = clamped
            arbitration_id, data = encode_position_speed(
                axis.motor_id, axis.origin_deg + clamped,
                config.HOLD_SPEED_RPM, config.HOLD_ACCEL_RPM_S,
            )
            self._port.send(arbitration_id, data)

    def _receive_until(self, deadline: float) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            frame = self._port.recv(timeout=remaining)
            if frame is not None:
                self._handle_frame(*frame)

    def _handle_frame(self, arbitration_id: int, data: bytes) -> None:
        decoded = decode_feedback(arbitration_id, data)
        if decoded is None:
            return
        motor_id, motor_angle_deg = decoded
        axis = AXIS_BY_ID.get(motor_id)
        if axis is None:
            return
        with self._lock:
            self._current[axis.name] = (motor_angle_deg - axis.origin_deg, self._clock())
