# -*- coding: utf-8 -*-
"""
カメラの処理を別スレッドで動かす。

フレームの取得 → シャトルの検出(毎フレーム)→ 人の検出(POSE_INFERENCE_INTERVAL フレームごと)
→ 描画 → CSV 記録、を繰り返し、描画済みの最新フレームと状態を main.py に渡す。

別スレッドにしているのは、YOLO の推論が重くても、発射機の操作画面とキー入力の反応が
遅れないようにするため。カメラ側でエラーが起きてもスレッドが止まるだけで、発射機の操作は続けられる。
"""

import datetime
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

import config
from camera import Intrinsics
from records import CsvRecorder
from vision import (
    PERSON_CSV_HEADER, PERSON_NONE, SHUTTLE_CSV_HEADER,
    PersonObservation, ShuttleObservation, draw_person, draw_shuttle, person_position, shuttle_position,
)

log = logging.getLogger("launcher.camera")

STATE_RUNNING = "running"
STATE_STOPPED = "stopped"

# 画面に「今の検出結果」として出すのは、この時間以内のものだけ
OBSERVATION_FRESH_SEC = 0.5


@dataclass
class CameraStatus:
    state: str                              # running / stopped
    message: Optional[str]                  # 止まった理由など
    fps: float
    shuttle: Optional[ShuttleObservation]   # 最新のシャトル(古ければ None)
    person: Optional[PersonObservation]     # 最新の立ち位置(古ければ None)
    person_note: Optional[str]              # 立ち位置を出せなかった理由
    shuttle_rows: int                       # CSV に記録した件数
    person_rows: int
    frames: int = 0                         # 処理し終えたフレーム数
    stage: Optional[str] = None             # 最初のフレームを処理し終えるまでの、今の段階
    stage_elapsed: float = 0.0              # その段階に入ってからの秒数


def make_recorders(run_stamp: Optional[str] = None):
    run_stamp = run_stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        CsvRecorder(config.SHUTTLE_CSV_DIR, "shuttle_trajectory", SHUTTLE_CSV_HEADER, run_stamp, "シャトルの位置"),
        CsvRecorder(config.PERSON_CSV_DIR, "person_position", PERSON_CSV_HEADER, run_stamp, "人の立ち位置"),
    )


class CameraWorker:
    def __init__(self, capture, width: int, height: int, intrinsics: Intrinsics,
                 shuttle_detector=None, pose_detector=None,
                 shuttle_recorder: Optional[CsvRecorder] = None, person_recorder: Optional[CsvRecorder] = None,
                 is_file: bool = False, on_raw_frame=None):
        """on_raw_frame: 何も描いていないフレームを受け取る関数 f(frame, t_wall, t_mono)(自動版の録画用)。
        カメラのスレッドから呼ぶ。エラーが起きてもカメラの処理は止めない。"""
        self._cap = capture
        self._width, self._height = width, height
        self._intr = intrinsics.scaled_to(width, height)
        self._shuttle_detector = shuttle_detector
        self._pose_detector = pose_detector
        self._shuttle_rec = shuttle_recorder
        self._person_rec = person_recorder
        self._is_file = is_file
        self._on_raw_frame = on_raw_frame
        self._raw_frame_error_logged = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame = None
        self._state = STATE_RUNNING
        self._message: Optional[str] = None
        self._fps = 0.0
        self._shuttle: Optional[ShuttleObservation] = None
        self._shuttle_at = 0.0
        self._person: Optional[PersonObservation] = None
        self._person_at = 0.0
        self._person_note: Optional[str] = PERSON_NONE if pose_detector is not None else None
        self._closed = False
        # 最初のフレームを処理し終えるまでの段階(どこで時間がかかっているか・止まっているかを表示するため)
        self._frames_done = 0
        self._stage: Optional[str] = None
        self._stage_since = time.monotonic()
        # 以下はカメラのスレッド(process_frame)だけが使う
        self._frame_index = 0
        self._start_mono: Optional[float] = None
        self._last_landmarks = None   # 推論しないフレームでも骨格を描き続けるため、前回の結果を持っておく
        self._pose_calls = 0

    # ---- main.py から使う ---------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def latest_frame(self):
        with self._lock:
            return self._frame

    def status(self) -> CameraStatus:
        now = time.monotonic()
        with self._lock:
            return CameraStatus(
                state=self._state if (self._thread is not None and self._thread.is_alive()) else STATE_STOPPED,
                message=self._message,
                fps=self._fps,
                shuttle=self._shuttle if now - self._shuttle_at <= OBSERVATION_FRESH_SEC else None,
                person=self._person if now - self._person_at <= OBSERVATION_FRESH_SEC else None,
                person_note=self._person_note,
                shuttle_rows=self._shuttle_rec.count if self._shuttle_rec else 0,
                person_rows=self._person_rec.count if self._person_rec else 0,
                frames=self._frames_done,
                stage=self._stage,
                stage_elapsed=now - self._stage_since,
            )

    def stop(self) -> None:
        """スレッドを止めて、カメラ・MediaPipe・CSV を閉じる。何度呼んでもよい。"""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.error("カメラのスレッドが3秒以内に止まりませんでした。")
                return  # 使用中のものを閉じると落ちるおそれがあるので、閉じずに終わる
        if self._closed:
            return
        self._closed = True
        for name, close in (("カメラ", self._cap.release),
                            ("MediaPipe", getattr(self._pose_detector, "close", None)),
                            ("シャトルの CSV", getattr(self._shuttle_rec, "close", None)),
                            ("立ち位置の CSV", getattr(self._person_rec, "close", None))):
            if close is None:
                continue
            try:
                close()
            except Exception as e:
                log.warning("%sを閉じるときにエラー: %s", name, e)
        log.info("カメラを止めました。")

    # ---- カメラのスレッド ------------------------------------------------------

    def _enter_stage(self, stage: Optional[str]) -> float:
        """段階を切り替える(最初のフレームのときだけ)。前の段階にかかった秒数を返す。"""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._stage_since
            self._stage, self._stage_since = stage, now
        return elapsed

    def _first_frame(self) -> bool:
        return self._frames_done == 0

    def _set_stopped(self, message: str, level=logging.ERROR) -> None:
        with self._lock:
            self._state = STATE_STOPPED
            self._message = message
        log.log(level, "カメラの処理が止まりました: %s(発射機の操作は続けられます)", message)

    def _run(self) -> None:
        last_frame_at = None
        self._enter_stage("カメラからの最初の映像を待っています")
        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                t_wall, t_mono = time.time(), time.monotonic()  # 取得の直後に時刻を取る
                if not ok or frame is None:
                    if self._is_file:
                        self._set_stopped("動画ファイルの最後まで処理しました", level=logging.INFO)
                    else:
                        self._set_stopped("カメラからフレームを取得できませんでした")
                    return
                self.process_frame(frame, t_wall, t_mono)

                if last_frame_at is not None:
                    dt = t_mono - last_frame_at
                    if dt > 0:
                        with self._lock:
                            self._fps = 1.0 / dt if self._fps == 0 else self._fps * 0.9 + (1.0 / dt) * 0.1
                last_frame_at = t_mono
        except Exception as e:
            self._set_stopped(f"{type(e).__name__}: {e}")

    def process_frame(self, frame, t_wall: float, t_mono: float) -> None:
        """1フレーム分の検出・描画・記録を行う(カメラのスレッドとテストから呼ぶ)。

        YOLO と MediaPipe はどちらも何も描いていない映像で検出し、描くのは最後にまとめて行う
        (先にシャトルの枠や文字を描くと、MediaPipe がそれの描かれた映像で人を探すことになるため)。
        """
        first = self._first_frame()
        if first:
            log.info("カメラから最初の映像を受け取りました(%.1f 秒)。", self._enter_stage("シャトル検出(YOLO)の初回の処理中"))
        if self._start_mono is None:
            self._start_mono = t_mono
        self._frame_index += 1

        if self._on_raw_frame is not None:
            try:
                self._on_raw_frame(frame, t_wall, t_mono)
            except Exception as e:
                if not self._raw_frame_error_logged:
                    log.error("フレームの受け渡し(録画)でエラー: %s(カメラの処理は続けます)", e)
                    self._raw_frame_error_logged = True

        shuttle_drawing = None

        # ---- シャトル(毎フレーム) ----
        if self._shuttle_detector is not None:
            boxes, confs = self._shuttle_detector.detect(frame)
            if first:
                log.info("YOLO の初回の推論が終わりました(%.1f 秒)。", self._enter_stage("人の検出(MediaPipe)の初回の処理中"))
            best, best_box = None, None
            if len(confs) > 0:
                best_idx = int(confs.argmax())
                best_box = boxes[best_idx]
                pos = shuttle_position(best_box, self._intr)
                if pos is not None:
                    best = ShuttleObservation(t_wall, *pos, float(confs[best_idx]))
            shuttle_drawing = (boxes, confs, best, best_box)
            if best is not None:
                if self._shuttle_rec is not None:
                    self._shuttle_rec.add(best.csv_row())
                with self._lock:
                    self._shuttle, self._shuttle_at = best, t_mono

        # ---- 人(POSE_INFERENCE_INTERVAL フレームごと) ----
        if self._pose_detector is not None:
            if self._frame_index % config.POSE_INFERENCE_INTERVAL == 0:
                pose_first = self._pose_calls == 0
                if pose_first:
                    self._enter_stage("人の検出(MediaPipe)の初回の処理中")
                landmarks = self._pose_detector.detect(frame, int((t_mono - self._start_mono) * 1000))
                self._pose_calls += 1
                if pose_first:
                    log.info("MediaPipe の初回の推論が終わりました(%.1f 秒)。", self._enter_stage(None))
                self._last_landmarks = landmarks
                person, note = person_position(landmarks, self._width, self._height, self._intr, t_wall)
                # 記録は推論したフレームだけ(推論しないフレームは前回の結果の使い回しなので記録しない)
                if person is not None and self._person_rec is not None:
                    self._person_rec.add(person.csv_row())
                with self._lock:
                    self._person_note = note
                    if person is not None:
                        self._person, self._person_at = person, t_mono

        # ---- 描画(検出がすべて終わってから) ----
        if shuttle_drawing is not None:
            draw_shuttle(frame, *shuttle_drawing)
        if self._pose_detector is not None and self._last_landmarks is not None:
            self._pose_detector.draw(frame, self._last_landmarks)
            with self._lock:
                person, note = self._person, self._person_note
            draw_person(frame, self._last_landmarks, person if note is None else None, note)

        cv2.putText(frame, f"{self._fps:.1f} fps", (self._width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        with self._lock:
            self._frame = frame
            self._frames_done += 1
            if self._pose_detector is None or self._pose_calls > 0:
                self._stage = None
