#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
発射機の統合版(自動・学習版):カメラで見つけた人に向けて自動で狙い、撃った結果をターミナルに文章で入力して学習する。

    python auto_main.py                     実機+カメラ
    python auto_main.py --mock              偽物のモーターと Arduino +カメラ
    python auto_main.py --camera-file X.mp4 カメラの代わりに動画ファイルを使う
    python auto_main.py --no-record         録画しない

手動版(main.py)のキー操作はすべてそのまま使える。追加のキー:
    M        自動狙い オン/オフ(人に向けて yaw・pitch を合わせる)
    F        自動発射 オン/オフ(自動狙いがオンのときだけ。人が止まっていて狙いが合ったら撃つ)
W/A/S/D を押すと自動狙いは切れる(手で合わせたいとき)。スペース(停止)で自動狙い・自動発射とも切れる。

発射の結果は、このプログラムを起動した**ターミナル**に打って Enter(例:「1m足りない」「右に30cm」)。
「help」で入力の例が出る。記録は shots/shots.csv、録画は shots/session_<日時>.mp4。
"""

import argparse
import datetime
import logging
import queue
import sys
import threading
from pathlib import Path

import cv2

import config
import main as manual
from auto_controller import AutoController, AutoStatus
from camera import CameraError
from feedback import HELP_TEXT
from session_video import SessionVideo
from shots import ShotLog

log = logging.getLogger("launcher.auto_main")

GRAY, WHITE, GREEN, RED, YELLOW = manual.GRAY, manual.WHITE, manual.GREEN, manual.RED, manual.YELLOW


# ============================================================
# ターミナルからの入力(別スレッドで1行ずつ読む)
# ============================================================

class ConsoleInput:
    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._queue: "queue.Queue[str]" = queue.Queue()

    def start(self) -> None:
        threading.Thread(target=self._run, name="console-input", daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                line = self._stream.readline()
            except Exception as e:
                log.warning("ターミナルからの入力を読めなくなりました: %s", e)
                return
            if not line:  # 入力が閉じられた
                return
            self._queue.put(line)

    def lines(self):
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return


# ============================================================
# 画面
# ============================================================

class AutoScreen(manual.Screen):
    HEIGHT = 980

    def __init__(self):
        super().__init__()
        self.auto: AutoStatus = None

    def _footer_lines(self):
        return [
            ("M:自動狙い オン/オフ  F:自動発射 オン/オフ  (W/A/S/D で自動狙いは切れる)", GRAY),
            ("1:発射  2/3/4:強さ 弱/中/強  5:ローラー停止  W/S:pitch -/+  A/D:yaw -/+", GRAY),
            ("スペース:停止  R:モーター再開  Q:終了   結果はターミナルに入力(help で例)", GRAY),
        ]

    def _build_lines(self, s, mock):
        lines = super()._build_lines(s, mock)
        a = self.auto
        if a is None:
            return lines
        lines.append(("", WHITE))
        lines.append((f"自動狙い: {'オン' if a.aim_enabled else 'オフ'}   自動発射: {'オン' if a.fire_enabled else 'オフ'}",
                      GREEN if a.aim_enabled else WHITE))
        sol = a.solution
        if a.person is None:
            age = "" if a.person_age_sec is None else f"({a.person_age_sec:.0f} 秒前から)"
            lines.append((f"  狙う人: 見失っています{age}", YELLOW if a.aim_enabled else WHITE))
        elif sol is not None:
            lines.append((f"  狙う人: 発射口から {sol.person_distance_cm / 100:.2f}m、方向 {sol.person_yaw_deg:+.1f}°",
                          WHITE))
            pitch = "---" if sol.pitch_deg is None else f"{sol.pitch_deg:+.1f}°"
            pred = "" if sol.predicted_distance_cm is None else f"(予想飛距離 {sol.predicted_distance_cm / 100:.2f}m)"
            lines.append((f"  計算した狙い: yaw {sol.yaw_deg:+.1f}°  pitch {pitch}{pred}"
                          + ("  合っている" if a.settled else ""), GREEN if a.settled else WHITE))
            for w in sol.warnings:
                lines.append((f"   ・{w}", YELLOW))
        if a.fire_enabled:
            if a.auto_fire_reasons:
                lines.append(("  自動発射: 待っています", YELLOW))
                for r in a.auto_fire_reasons[:4]:
                    lines.append((f"   ・{r}", YELLOW))
            else:
                lines.append(("  自動発射: 条件がそろいました", GREEN))
        lines.append((f"  学習: {a.model_text}", GRAY))
        lines.append((f"  録画: {a.video_name or 'していません'}", GRAY))
        for shot in reversed(a.recent_shots):
            result = shot.feedback_text or "(結果をまだ入力していません)"
            lines.append((f"  #{shot.shot_id} 強さ{shot.speed} pitch {shot.pitch_deg:+.1f}° → {result}",
                          WHITE if shot.feedback_text else YELLOW))
        return lines


def draw_auto_overlay(frame, a: AutoStatus):
    """カメラ映像に、自動版の状態を書き込む(表示用のコピーに描く。録画には入らない)。"""
    if frame is None or a is None:
        return frame
    frame = frame.copy()
    text = f"AUTO AIM {'ON' if a.aim_enabled else 'OFF'}  AUTO FIRE {'ON' if a.fire_enabled else 'OFF'}"
    if a.video_name:
        text += "  REC"
    cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if a.aim_enabled else (200, 200, 200), 2)
    if a.last_shot is not None:
        cv2.putText(frame, f"last shot #{a.last_shot.shot_id}", (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


# ============================================================
# キー操作
# ============================================================

def handle_auto_key(controller: AutoController, launcher, key: int):
    if key < 0 or key == 255:
        return None
    ch = chr(key & 0xFF).lower()
    if ch == "m":
        controller.toggle_aim()
        return None
    if ch == "f":
        controller.toggle_fire()
        return None
    if ch == "1":
        controller.fire("key")   # 記録を残すため、launcher.fire() を直接呼ばない
        return None
    if ch in ("w", "a", "s", "d"):
        controller.disable("手で向きを変えたため")
    elif ch == " ":
        controller.disable("停止したため")
    return manual.handle_key(launcher, key)


# ============================================================
# 起動
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="発射機の統合版(自動・学習版)")
    parser.add_argument("--mock", action="store_true", help="実機の代わりに偽物のモーターと Arduino を使う")
    parser.add_argument("--serial-port", default=config.SERIAL_PORT)
    parser.add_argument("--can-channel", default=config.CAN_CHANNEL)
    parser.add_argument("--mock-can-fail-after", type=float, default=None, metavar="SEC")
    parser.add_argument("--camera-file", type=Path, default=None, metavar="VIDEO",
                        help="カメラの代わりに動画ファイルを使う")
    parser.add_argument("--no-record", action="store_true", help="録画しない")
    parser.add_argument("--shots-csv", type=Path, default=config.SHOTS_CSV,
                        help=f"発射の記録(学習に使う)。既定: {config.SHOTS_CSV}")
    args = parser.parse_args(argv)
    args.no_camera = False   # 自動版はカメラが前提
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    recent = manual.setup_logging()
    log.info("自動版を起動します(%s)。", "MOCK" if args.mock else f"実機 CAN={args.can_channel} Arduino={args.serial_port}")

    session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_log = ShotLog(args.shots_csv)
    video = None if args.no_record else SessionVideo(args.shots_csv.parent, session)
    launcher = manual.build_launcher(args)
    controller = AutoController(launcher, shot_log, session=session, video=video)
    console = ConsoleInput()

    camera_worker = None
    camera_note = None
    exit_code = 0
    try:
        try:
            camera_worker = manual.build_camera(args, on_raw_frame=video.write if video is not None else None)
        except Exception as e:
            if not isinstance(e, CameraError):
                e = f"{type(e).__name__}: {e}"
            camera_note = f"起動できませんでした: {e}"
            log.error("カメラを起動できませんでした: %s(自動で狙えません。手動の操作はできます)", e)
        launcher.start()
        if camera_worker is not None:
            camera_worker.start()
            manual.open_camera_window()
        console.start()
        print(HELP_TEXT, flush=True)
        screen = AutoScreen()
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(config.WINDOW_NAME, *config.WINDOW_POSITION)
        while True:
            camera_status = camera_worker.status() if camera_worker is not None else camera_note
            person = camera_status.person if camera_worker is not None else None
            screen.auto = controller.tick(person)
            for line in console.lines():
                controller.handle_line(line)
            cv2.imshow(config.WINDOW_NAME, screen.render(launcher.status(), camera_status, recent.items(), args.mock))
            if camera_worker is not None:
                frame = camera_worker.latest_frame()
                cv2.imshow(config.CAMERA_WINDOW_NAME,
                           draw_auto_overlay(frame, screen.auto) if frame is not None else manual.WAITING_FRAME)
            key = cv2.waitKey(config.UI_REFRESH_MS)
            if handle_auto_key(controller, launcher, key) == manual.QUIT:
                break
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                log.info("操作画面が閉じられたので終了します。")
                break
            if camera_worker is not None and cv2.getWindowProperty(config.CAMERA_WINDOW_NAME,
                                                                   cv2.WND_PROP_VISIBLE) < 1:
                log.info("カメラの画面が閉じられたので終了します。")
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C が押されたので終了します。")
    except ConnectionError as e:
        log.error("発射機に接続できなかったので終了します: %s", e)
        exit_code = 1
    except Exception:
        log.exception("エラーで終了します。")
        exit_code = 1
    finally:
        controller.disable("終了するため")
        launcher.shutdown()
        if camera_worker is not None:
            try:
                camera_worker.stop()
            except Exception as e:
                log.error("カメラの終了処理でエラー: %s", e)
        if video is not None:
            try:
                video.close()
            except Exception as e:
                log.error("録画の終了処理でエラー: %s", e)
        cv2.destroyAllWindows()
        n = len([s for s in shot_log.all() if s.session == session])
        log.info("このセッションの発射: %d 回(記録: %s)", n, shot_log.path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
