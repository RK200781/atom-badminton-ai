#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
発射機の統合版(手動操作):起動、キー入力、画面表示。

    python main.py                     実機(CAN: COM6、Arduino: COM5)+カメラ
    python main.py --mock              偽物のモーターと Arduino +カメラ
    python main.py --no-camera         カメラを使わない
    python main.py --camera-file X.mp4 カメラの代わりに動画ファイルを使う

画面は2つ:発射機の操作画面(Launcher Control)と、カメラ映像(Camera)。
カメラではシャトル(YOLO)と人(MediaPipe)を検出して表示・記録するが、発射は自動では行わない。
カメラが使えなくても、発射機の操作はそのまま続けられる。

キーは OpenCV のウィンドウ(どちらでもよい)が前面にあるときだけ受け付ける(ノートブックの pynput と違い、
他のウィンドウで打った文字には反応しない)。

    1        発射
    2/3/4    ローラーの強さ 弱/中/強
    5        ローラー停止
    W/S      pitch -2° / +2°(ノートブックと同じ向き:W がマイナス)
    A/D      yaw   -2° / +2°
    スペース  停止(ローラー停止+その場で保持+強さを設定し直すまで発射不可)
    R        モーターの通信を再開
    Q        終了
"""

import argparse
import collections
import datetime
import logging
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

import config
from feeder import FIRE_CONFIRMED, FIRE_PENDING, FIRE_UNCONFIRMED, Feeder, open_serial
from launcher import Launcher, LauncherStatus
from motors import MotorController, PythonCanPort
from camera import CameraError

log = logging.getLogger("launcher.main")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

QUIT = "quit"


# ============================================================
# キー操作
# ============================================================

def handle_key(launcher, key: int):
    """押されたキーに対応する操作を行う。終了するときは QUIT を返す。"""
    if key < 0 or key == 255:
        return None
    ch = chr(key & 0xFF).lower()
    step = config.KEY_STEP_DEG
    if ch == "q":
        log.info("Q が押されたので終了します。")
        return QUIT
    if ch == "1":
        launcher.fire()
    elif ch in ("2", "3", "4"):
        launcher.set_speed(int(ch) - 1)
    elif ch == "5":
        launcher.set_speed(0)
    elif ch == "w":
        launcher.nudge(0.0, -step)
    elif ch == "s":
        launcher.nudge(0.0, +step)
    elif ch == "a":
        launcher.nudge(-step, 0.0)
    elif ch == "d":
        launcher.nudge(+step, 0.0)
    elif ch == " ":
        launcher.stop()
    elif ch == "r":
        launcher.restart_motors()
    return None


# ============================================================
# ログ(コンソール・ファイル・画面)
# ============================================================

class RecentMessages(logging.Handler):
    """画面に出すため、最近のログを覚えておく。"""

    def __init__(self, maxlen: int = 7):
        super().__init__(level=logging.INFO)
        self._lock_ = threading.Lock()
        self._items = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        stamp = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        with self._lock_:
            self._items.append((stamp, record.levelno, record.getMessage()))

    def items(self):
        with self._lock_:
            return list(self._items)


def setup_logging() -> RecentMessages:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"launcher_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    recent = RecentMessages()
    for handler in (console, file_handler, recent):
        root.addHandler(handler)
    log.info("ログの保存先: %s", log_path)
    return recent


# ============================================================
# 画面
# ============================================================

FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
GREEN = (90, 220, 90)
RED = (90, 90, 255)
YELLOW = (60, 210, 230)


class Screen:
    WIDTH, HEIGHT = 960, 760
    FONT_SIZE = 20
    LINE = 28

    def __init__(self):
        self._wrap_cache = {}
        self._font = None
        try:
            from PIL import ImageFont
            for path in FONT_CANDIDATES:
                if Path(path).exists():
                    self._font = ImageFont.truetype(path, self.FONT_SIZE)
                    break
        except ImportError:
            pass
        if self._font is None:
            log.warning("日本語フォント(または Pillow)が見つからないため、画面の日本語は表示できません。")

    MARGIN = 16

    def _text_width(self, text: str) -> float:
        if self._font is None:
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        return self._font.getlength(text)

    def _wrap_text(self, text: str):
        key = text
        cached = self._wrap_cache.get(key)
        if cached is not None:
            return cached
        max_width = self.WIDTH - 2 * self.MARGIN
        pieces = []
        while self._text_width(text) > max_width:
            lo, hi = 7, len(text)  # 折り返す位置を二分探索(先頭の字下げより後ろで切る)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._text_width(text[:mid]) <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            pieces.append(text[:lo])
            text = "      " + text[lo:]
        pieces.append(text)
        if len(self._wrap_cache) > 500:
            self._wrap_cache.clear()
        self._wrap_cache[key] = pieces
        return pieces

    def _wrap(self, lines):
        """画面の幅に収まらない行を折り返す(2行目以降は字下げ)。"""
        return [(piece, color) for text, color in lines for piece in self._wrap_text(text)]

    def render(self, status: LauncherStatus, camera, messages, mock: bool) -> np.ndarray:
        """camera: CameraStatus、またはカメラを使っていない理由の文字列。"""
        lines = self._layout(status, camera, messages, mock)
        img = np.full((self.HEIGHT, self.WIDTH, 3), 30, np.uint8)
        if self._font is None:
            for i, (text, color) in enumerate(lines):
                cv2.putText(img, text.encode("ascii", "replace").decode(), (self.MARGIN, 30 + i * self.LINE),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for i, (text, color) in enumerate(lines):
            # 配列は OpenCV の BGR のまま描くので、色も BGR で渡す
            draw.text((self.MARGIN, 10 + i * self.LINE), text, font=self._font, fill=color)
        return np.asarray(pil)

    def _layout(self, status: LauncherStatus, camera, messages, mock: bool):
        """状態 → メッセージ → キー操作の順に並べる。入りきらないときは古いメッセージから省く。"""
        footer = self._wrap(self._footer_lines())
        body = self._wrap(self._build_lines(status, mock) + self._camera_lines(camera))
        max_lines = (self.HEIGHT - 20) // self.LINE
        room = max_lines - len(footer) - len(body) - 2
        message_lines = []
        for stamp, level, text in reversed(messages):
            color = RED if level >= logging.ERROR else YELLOW if level >= logging.WARNING else GRAY
            wrapped = self._wrap([(f"  {stamp} {text}", color)])
            if len(message_lines) + len(wrapped) > room:
                break
            message_lines = wrapped + message_lines
        lines = body + [("", WHITE), ("最近のメッセージ:", GRAY)] + message_lines
        lines = lines[:max_lines - len(footer)]
        lines += [("", WHITE)] * (max_lines - len(footer) - len(lines))
        return lines + footer

    def _footer_lines(self):
        return [
            ("1:発射  2/3/4:強さ 弱/中/強  5:ローラー停止  W/S:pitch -/+  A/D:yaw -/+", GRAY),
            ("スペース:停止  R:モーター再開  Q:終了", GRAY),
        ]

    @staticmethod
    def _angle(value):
        return "---" if value is None else f"{value:+6.1f}°"

    def _build_lines(self, s: LauncherStatus, mock: bool):
        m, f = s.motors, s.feeder
        lines = []
        title = "発射機コントロール" + ("  [MOCK:実機なし]" if mock else "")
        lines.append((title, YELLOW if mock else WHITE))

        if m.running:
            lines.append(("モーター保持ループ: 動作中", GREEN))
        else:
            lines.append((f"モーター保持ループ: 停止  {m.error or ''}  → R で再開", RED))
        lines.append((f"  yaw   目標 {self._angle(m.target_yaw)}  現在 {self._angle(m.current_yaw)}"
                      f"   (範囲 {config.YAW_MIN_DEG:+.0f}〜{config.YAW_MAX_DEG:+.0f}°)", WHITE))
        lines.append((f"  pitch 目標 {self._angle(m.target_pitch)}  現在 {self._angle(m.current_pitch)}"
                      f"   (範囲 {config.PITCH_MIN_DEG:+.0f}〜{config.PITCH_MAX_DEG:+.0f}°)", WHITE))

        if f.error:
            lines.append((f"Arduino: 受信エラー  {f.error}", RED))
        lines.append((f"ローラーの強さ: {f.speed}  {config.SPEED_LABELS.get(f.speed, '?')}",
                       GREEN if f.speed else WHITE))
        shuttle_text = {True: ("装填部のシャトル: 検知あり", GREEN), False: ("装填部のシャトル: 検知なし", WHITE),
                        None: ("装填部のシャトル: 不明(Arduino から届いていない)", RED)}[s.shuttle]
        lines.append(shuttle_text)

        if s.fire_reasons:
            lines.append(("発射: できない", RED))
            for reason in s.fire_reasons:
                lines.append((f"   ・{reason}", RED))
        else:
            lines.append(("発射: できる(1 キー)", GREEN))
        result_text = {FIRE_PENDING: ("直前の発射: 確認中...", YELLOW),
                       FIRE_CONFIRMED: ("直前の発射: 確認できた(検知 1→0)", GREEN),
                       FIRE_UNCONFIRMED: ("直前の発射: 確認できず", RED)}.get(f.fire_result)
        if result_text:
            lines.append(result_text)
        return lines

    @staticmethod
    def _camera_lines(camera):
        from camera_worker import STATE_RUNNING
        lines = [("", WHITE)]
        if isinstance(camera, str):
            lines.append((f"カメラ: {camera}", GRAY if camera.startswith("使わない") else RED))
            return lines
        if camera.state != STATE_RUNNING:
            lines.append((f"カメラ: 停止  {camera.message or ''}", RED))
        elif camera.stage is not None:
            # 最初のフレームを処理し終えるまで。長く続くなら、その段階で止まっている
            color = RED if camera.stage_elapsed > 30 else YELLOW
            lines.append((f"カメラ: 準備中  {camera.stage}({camera.stage_elapsed:.0f} 秒)", color))
        else:
            lines.append((f"カメラ: 動作中  {camera.fps:.1f} fps", GREEN))
        sh = camera.shuttle
        lines.append((f"  映像のシャトル: X={sh.X:+.0f} Y={sh.Y:+.0f} Z={sh.Z:.0f} cm(信頼度 {sh.confidence:.2f})"
                      if sh else "  映像のシャトル: 未検出", GREEN if sh else WHITE))
        p = camera.person
        if p:
            source = "足首から" if p.method == "ankle" else f"肩の高さ {config.SHOULDER_HEIGHT_CM:.0f}cm と仮定"
            lines.append((f"  人の立ち位置: X={p.X:+.0f} Z={p.Z:.0f} cm({source})", GREEN))
        else:
            lines.append((f"  人の立ち位置: {camera.person_note or '未検出'}", WHITE))
        lines.append((f"  記録: シャトル {camera.shuttle_rows} 件 / 立ち位置 {camera.person_rows} 件", GRAY))
        return lines


# ============================================================
# 起動
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="発射機の統合版")
    parser.add_argument("--mock", action="store_true", help="実機の代わりに偽物のモーターと Arduino を使う")
    parser.add_argument("--serial-port", default=config.SERIAL_PORT,
                        help=f"Arduino の COM ポート(既定: {config.SERIAL_PORT})")
    parser.add_argument("--can-channel", default=config.CAN_CHANNEL,
                        help=f"USB-CAN の channel(既定: {config.CAN_CHANNEL})")
    parser.add_argument("--mock-can-fail-after", type=float, default=None, metavar="SEC",
                        help="--mock のとき、この秒数後にモーターの通信エラーを起こす(R キーの再開を試す用)")
    parser.add_argument("--no-camera", action="store_true", help="カメラを使わない(発射機の操作だけ)")
    parser.add_argument("--camera-file", type=Path, default=None, metavar="VIDEO",
                        help="カメラの代わりに動画ファイルを使う(カメラなしで確認する用)")
    return parser.parse_args(argv)


def build_launcher(args) -> Launcher:
    if args.mock:
        import mock
        motors = MotorController(lambda: mock.FakeCanPort(fail_after_sec=args.mock_can_fail_after))
        feeder = Feeder(mock.FakeArduinoSerial)
    else:
        motors = MotorController(lambda: PythonCanPort(channel=args.can_channel))
        feeder = Feeder(lambda: open_serial(args.serial_port))
    return Launcher(motors, feeder)


# 最初の映像が届くまで Camera ウィンドウに出す画像
WAITING_FRAME = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 40, np.uint8)
cv2.putText(WAITING_FRAME, "Waiting for the first camera frame... (see Launcher Control)",
            (40, config.CAMERA_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)


def open_camera_window() -> None:
    """カメラ映像のウィンドウを、操作画面と重ならないよう右隣に開く(大きさはマウスで変えられる)。"""
    cv2.namedWindow(config.CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.CAMERA_WINDOW_NAME, *config.CAMERA_WINDOW_SIZE)
    cv2.moveWindow(config.CAMERA_WINDOW_NAME, *config.CAMERA_WINDOW_POSITION)


def build_camera(args, on_raw_frame=None):
    """カメラの処理を準備する。起動できなければ CameraError。

    on_raw_frame: 何も描いていないフレームを受け取る関数(自動版の録画用。手動版では使わない)。
    """
    import camera as cam
    from camera_worker import CameraWorker, make_recorders
    from vision import PoseDetector, ShuttleDetector

    intrinsics = cam.load_calibration()
    shuttle_detector = ShuttleDetector()
    pose_detector = PoseDetector()
    try:
        if args.camera_file is not None:
            cap, width, height = cam.open_video_file(args.camera_file)
        else:
            cap, width, height = cam.open_camera(cam.resolve_camera_index())
    except Exception:
        pose_detector.close()
        raise
    shuttle_rec, person_rec = make_recorders()
    # スレッドはまだ始めない(発射機の準備が終わってから main() で始める)
    return CameraWorker(cap, width, height, intrinsics, shuttle_detector, pose_detector,
                        shuttle_rec, person_rec, is_file=args.camera_file is not None, on_raw_frame=on_raw_frame)


def main(argv=None) -> int:
    args = parse_args(argv)
    recent = setup_logging()
    log.info("起動します(%s)。", "MOCK" if args.mock else f"実機 CAN={args.can_channel} Arduino={args.serial_port}")

    launcher = build_launcher(args)
    camera_worker = None
    camera_note = "使わない(--no-camera)"
    camera_window_open = False
    exit_code = 0
    try:
        # カメラは先に準備する(カメラが複数あるとき、コンソールで番号を選ぶことがあるため)。
        # カメラが使えなくても、発射機の操作は続ける。
        if not args.no_camera:
            try:
                camera_worker = build_camera(args)
            except Exception as e:  # CameraError 以外(OpenCV の内部エラーなど)でも発射機は止めない
                if not isinstance(e, CameraError):
                    e = f"{type(e).__name__}: {e}"
                camera_note = f"起動できませんでした: {e}"
                log.error("カメラを起動できませんでした: %s(発射機の操作は続けられます)", e)
        launcher.start()
        if camera_worker is not None:
            camera_worker.start()  # 発射機の準備が終わってから始める(失敗したときに止める手間を省く)
            open_camera_window()   # 映像が届く前から開いておく(届くまでは「待っています」と表示)
            camera_window_open = True
        screen = Screen()
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(config.WINDOW_NAME, *config.WINDOW_POSITION)
        while True:
            camera_status = camera_worker.status() if camera_worker is not None else camera_note
            cv2.imshow(config.WINDOW_NAME, screen.render(launcher.status(), camera_status, recent.items(), args.mock))
            if camera_worker is not None:
                frame = camera_worker.latest_frame()
                cv2.imshow(config.CAMERA_WINDOW_NAME, frame if frame is not None else WAITING_FRAME)
            key = cv2.waitKey(config.UI_REFRESH_MS)
            if handle_key(launcher, key) == QUIT:
                break
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                log.info("操作画面が閉じられたので終了します。")
                break
            if camera_window_open and cv2.getWindowProperty(config.CAMERA_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                log.info("カメラの画面が閉じられたので終了します。")
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C が押されたので終了します。")
    except ConnectionError as e:  # COM ポートが見つからないなど。対処法をそのまま出す
        log.error("発射機に接続できなかったので終了します: %s", e)
        exit_code = 1
    except Exception:
        log.exception("エラーで終了します。")
        exit_code = 1
    finally:
        # 発射機の終了処理を先に行う(安全に関わるため)。カメラはその後。
        launcher.shutdown()
        if camera_worker is not None:
            try:
                camera_worker.stop()
            except Exception as e:
                log.error("カメラの終了処理でエラー: %s", e)
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
