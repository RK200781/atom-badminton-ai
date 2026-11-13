#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
検出テスト用の録画ツール(カメラだけを使う。発射機はつながなくてよい)。

手動版と同じ設定(1280x720・60fps、DirectShow、MJPG)でカメラを開き、何も描き込んでいない映像を
そのまま録画する。録画した動画は compare.py で、YOLO の設定ごとの検出率を比べるのに使う。

    python detection_test/record.py --label 2m

操作(録画ウィンドウが前面のとき):
    V / スペース  録画の開始・停止(押すたびに別のファイルになる)
    Q            終了

保存先: detection_test/recordings/rec_<日時>_<ラベル>.mp4
        と、各フレームの時刻 rec_<日時>_<ラベル>_timestamps.csv(フレーム番号, time.time())
"""

import argparse
import csv
import datetime
import logging
import re
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from camera import CameraError, open_camera, open_video_file, resolve_camera_index  # noqa: E402

log = logging.getLogger("detection_test.record")


class SegmentRecorder:
    """1回分の録画(動画ファイルと、各フレームの時刻の CSV)。"""

    def __init__(self, directory: Path, width: int, height: int, fps: float, label: str = ""):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r"[^\w\-]+", "_", label).strip("_")
        name = f"rec_{stamp}" + (f"_{safe_label}" if safe_label else "")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{name}.mp4"
        self.timestamps_path = directory / f"{name}_timestamps.csv"
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*config.RECORD_FOURCC),
                                       fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"動画ファイルを作れませんでした: {self.path}")
        self._size = (width, height)
        self._ts_file = open(self.timestamps_path, "w", newline="", encoding="utf-8")
        self._ts = csv.writer(self._ts_file)
        self._ts.writerow(["frame", "timestamp"])
        self.frames = 0
        self.started = time.monotonic()

    def write(self, frame, timestamp: float) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            raise ValueError(f"フレームの大きさが録画の大きさと違います: {w}x{h} != {self._size}")
        self._writer.write(frame)
        self._ts.writerow([self.frames, f"{timestamp:.6f}"])
        self.frames += 1

    def close(self) -> Path:
        self._writer.release()
        self._ts_file.close()
        seconds = time.monotonic() - self.started
        fps = self.frames / seconds if seconds > 0 else 0.0
        log.info("録画を保存しました: %s(%d フレーム、%.1f 秒、実際の fps %.1f)", self.path, self.frames, seconds, fps)
        return self.path


def _text(frame, text, org, color, scale=0.8) -> None:
    """黒い縁取りの上に色付きの文字を描く(明るい背景でも読めるように)。"""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_overlay(frame, recorder, fps: float, label: str) -> None:
    """プレビューに状態を描く(録画は描く前のフレームで行うので、動画には入らない)。"""
    h, w = frame.shape[:2]
    if recorder is not None:
        elapsed = time.monotonic() - recorder.started
        cv2.circle(frame, (30, 35), 12, (0, 0, 255), -1)
        _text(frame, f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}  {recorder.frames} frames"
                     f"  -> {recorder.path.name}", (55, 45), (0, 0, 255))
    else:
        _text(frame, "V / Space: start recording   Q: quit", (55, 45), (255, 255, 255))
    _text(frame, f"{w}x{h}  {fps:.1f} fps  label: {label or '-'}", (20, h - 20), (255, 255, 255), 0.7)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="検出テスト用の録画ツール")
    parser.add_argument("--label", default="", help="ファイル名に付けるラベル(例: 2m、3m、flight)")
    parser.add_argument("--camera-file", type=Path, default=None, metavar="VIDEO",
                        help="カメラの代わりに動画ファイルを使う(動作確認用)")
    parser.add_argument("--start", action="store_true", help="起動してすぐ録画を始める")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.camera_file is not None:
            cap, width, height = open_video_file(args.camera_file)
        else:
            cap, width, height = open_camera(resolve_camera_index())
    except CameraError as e:
        log.error("カメラを開けませんでした: %s", e)
        return 1
    if (width, height) != (config.CAMERA_WIDTH, config.CAMERA_HEIGHT):
        log.warning("本番と違う解像度(%dx%d)で録画することになります。", width, height)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, 960, 540)
    recorder = None
    fps, last = 0.0, None
    exit_code = 0
    try:
        if args.start:
            recorder = SegmentRecorder(config.RECORDINGS_DIR, width, height, config.CAMERA_FPS, args.label)
            log.info("録画を始めました: %s", recorder.path)
        while True:
            ok, frame = cap.read()
            t_wall, t_mono = time.time(), time.monotonic()
            if not ok or frame is None:
                log.warning("フレームを取得できなかったので終了します。")
                break
            if recorder is not None:
                recorder.write(frame, t_wall)  # 描く前のフレームを録画する
            if last is not None and t_mono > last:
                fps = 1.0 / (t_mono - last) if fps == 0 else fps * 0.9 + 0.1 / (t_mono - last)
            last = t_mono

            draw_overlay(frame, recorder, fps, args.label)
            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1)
            ch = chr(key & 0xFF).lower() if key >= 0 else ""
            if ch == "q":
                break
            if ch in ("v", " "):
                if recorder is None:
                    recorder = SegmentRecorder(config.RECORDINGS_DIR, width, height, config.CAMERA_FPS, args.label)
                    log.info("録画を始めました: %s", recorder.path)
                else:
                    recorder.close()
                    recorder = None
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        log.info("Ctrl+C が押されたので終了します。")
    except Exception:
        log.exception("エラーで終了します。")
        exit_code = 1
    finally:
        if recorder is not None:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
