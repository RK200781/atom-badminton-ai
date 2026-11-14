# -*- coding: utf-8 -*-
"""
自動版のセッションの録画。

カメラの映像を、何も描き込まずにそのまま1つの動画ファイルに保存する(shots/session_<日時>.mp4)。
各フレームの時刻は session_<日時>_frames.csv に書く(処理が重いと実際の fps は一定にならないため)。
発射の記録(shots.csv)には、発射した瞬間のフレーム番号を書くので、後で動画のその場面を探せる。

書き込みはカメラのスレッドから呼ばれる(CameraWorker の on_raw_frame)。
"""

import csv
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2

import config

log = logging.getLogger("launcher.video")


class SessionVideo:
    def __init__(self, directory: Path, session: str, fps: float = config.SESSION_VIDEO_FPS):
        self.directory = Path(directory)
        self.path = self.directory / f"session_{session}.mp4"
        self.frames_path = self.directory / f"session_{session}_frames.csv"
        self._fps = fps
        self._lock = threading.Lock()
        self._writer = None
        self._frames_file = None
        self._frames_csv = None
        self._count = 0
        self._failed = False
        self._closed = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def frame_count(self) -> int:
        """これまでに書いたフレーム数(次に書くフレームの番号)。"""
        with self._lock:
            return self._count

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._writer is not None and not self._failed and not self._closed

    def write(self, frame, t_wall: float, t_mono: float) -> None:
        with self._lock:
            if self._failed or self._closed:
                return
            if self._writer is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
                if not writer.isOpened():
                    self._failed = True
                    log.error("録画ファイルを作れませんでした: %s(録画せずに続けます)", self.path)
                    return
                self._writer = writer
                self._frames_file = open(self.frames_path, "w", newline="", encoding="utf-8")
                self._frames_csv = csv.writer(self._frames_file)
                self._frames_csv.writerow(["frame", "timestamp", "monotonic"])
                log.info("録画を始めました: %s", self.path)
            self._writer.write(frame)
            self._frames_csv.writerow([self._count, f"{t_wall:.3f}", f"{t_mono:.3f}"])
            self._count += 1

    def close(self) -> Optional[Path]:
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            if self._writer is None:
                return None
            self._writer.release()
            self._frames_file.close()
            log.info("録画を保存しました: %s(%d フレーム)", self.path, self._count)
            return self.path
