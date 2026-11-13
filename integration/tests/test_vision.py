# -*- coding: utf-8 -*-
"""カメラ側(camera.py / vision.py / records.py / camera_worker.py)のテスト。

YOLO と MediaPipe の実物を使うテストは、ライブラリとモデルが無ければ飛ばす。
"""

import csv
import math
import sys
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock as umock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from camera import CameraError, Intrinsics, load_calibration  # noqa: E402
from camera_worker import STATE_RUNNING, STATE_STOPPED, CameraWorker  # noqa: E402
from records import CsvRecorder  # noqa: E402
from test_motors import wait_until  # noqa: E402
from vision import (  # noqa: E402
    LEFT_ANKLE, LEFT_SHOULDER, METHOD_ANKLE, METHOD_SHOULDER, PERSON_CSV_HEADER, PERSON_NO_INTERSECTION,
    PERSON_NONE, PERSON_NOT_VISIBLE, RIGHT_ANKLE, RIGHT_SHOULDER, SHUTTLE_CSV_HEADER,
    estimate_floor_position, estimate_position_on_plane, person_position, shuttle_position,
)

INTR = Intrinsics(fx=760.0, fy=760.0, cx=640.0, cy=360.0, width=1280, height=720)
Landmark = namedtuple("Landmark", "x y visibility presence")


# 計算の仕組みを確かめるテスト用の、床と足首が映るカメラの設置(下向き 15°、高さ 150cm)
DOWNWARD_MOUNT = dict(CAMERA_MOUNT_HEIGHT_CM=150.0, CAMERA_TILT_DEG=15.0)


def make_landmarks(left=(0.45, 0.95), right=(0.55, 0.95), visibility=0.9, presence=0.9,
                   shoulders=((0.45, 0.4), (0.55, 0.4)), shoulder_visibility=0.9):
    """足首と肩の位置・見えやすさを指定した骨格(33点)を作る。"""
    lms = [Landmark(0.5, 0.5, 0.1, 0.1)] * 33  # 他の点は信頼度を低くしておく
    lms = list(lms)
    lms[LEFT_ANKLE] = Landmark(*left, visibility, presence)
    lms[RIGHT_ANKLE] = Landmark(*right, visibility, presence)
    lms[LEFT_SHOULDER] = Landmark(*shoulders[0], shoulder_visibility, 0.9)
    lms[RIGHT_SHOULDER] = Landmark(*shoulders[1], shoulder_visibility, 0.9)
    return lms


class ShuttlePositionTest(unittest.TestCase):
    def test_same_formula_as_realtime_detect(self):
        X, Y, Z, u, v, w, h = shuttle_position((600, 300, 640, 330), INTR)
        self.assertEqual((u, v, w, h), (620.0, 315.0, 40.0, 30.0))
        self.assertAlmostEqual(Z, 760.0 * config.SHUTTLE_DIAMETER_CM / 40.0)
        self.assertAlmostEqual(X, (620.0 - 640.0) * Z / 760.0)
        self.assertAlmostEqual(Y, (315.0 - 360.0) * Z / 760.0)

    def test_empty_box(self):
        self.assertIsNone(shuttle_position((10, 10, 10, 10), INTR))


class FloorPositionTest(unittest.TestCase):
    """tests/test_floor_projection.py と同じ確認を、統合版の関数で行う。"""
    H = 150.0

    def v_below_axis(self, deg):
        return 360.0 + 760.0 * math.tan(math.radians(deg))

    def test_optical_axis(self):
        x, z, valid = estimate_floor_position(640, 360, 760, 760, 640, 360, self.H, 15.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, self.H / math.tan(math.radians(15.0)), places=3)

    def test_below_axis(self):
        _, z, valid = estimate_floor_position(640, self.v_below_axis(20), 760, 760, 640, 360, self.H, 15.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(z, self.H / math.tan(math.radians(35.0)), places=3)

    def test_above_horizon(self):
        self.assertFalse(estimate_floor_position(640, 360, 760, 760, 640, 360, self.H, 0.0)[2])
        self.assertFalse(estimate_floor_position(640, self.v_below_axis(-20), 760, 760, 640, 360, self.H, 15.0)[2])

    def test_right_is_positive(self):
        x, _, _ = estimate_floor_position(840, self.v_below_axis(20), 760, 760, 640, 360, self.H, 15.0)
        self.assertGreater(x, 0)


class PlaneProjectionTest(unittest.TestCase):
    def test_floor_is_plane_at_zero(self):
        a = estimate_floor_position(640, 500, 760, 760, 640, 360, 150.0, 15.0)
        b = estimate_position_on_plane(640, 500, 760, 760, 640, 360, 0.0, 150.0, 15.0)
        self.assertEqual(a, b)

    def test_plane_above_camera_with_upward_tilt(self):
        # 高さ 20cm・35°上向きのカメラの光軸は、120cm 上(床から140cm)の面と 120/tan(35°) 先で交わる
        x, z, valid = estimate_position_on_plane(640, 360, 760, 760, 640, 360, 140.0, 20.0, -35.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, 120.0 / math.tan(math.radians(35.0)), places=3)

    def test_plane_above_camera_needs_upward_ray(self):
        # 下向きの視線は、カメラより高い面とは交わらない
        self.assertFalse(estimate_position_on_plane(640, 360, 760, 760, 640, 360, 140.0, 20.0, 15.0)[2])


class PersonPositionTest(unittest.TestCase):
    """足首と床が映る設置(下向き)での判定。"""

    def setUp(self):
        patcher = umock.patch.multiple(config, **DOWNWARD_MOUNT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reliable_ankles(self):
        person, note = person_position(make_landmarks(), 1280, 720, INTR, 123.0)
        self.assertIsNone(note)
        self.assertEqual(person.method, METHOD_ANKLE)
        self.assertAlmostEqual(person.u, 640.0)
        self.assertAlmostEqual(person.v, 0.95 * 720)
        x, z, _ = estimate_floor_position(640.0, 0.95 * 720, 760, 760, 640, 360)
        self.assertAlmostEqual(person.X, x)
        self.assertAlmostEqual(person.Z, z)
        self.assertEqual(person.csv_row()[0], 123.0)
        self.assertEqual(person.csv_row()[-1], METHOD_ANKLE)
        self.assertEqual(len(person.csv_row()), len(PERSON_CSV_HEADER))

    def test_hidden_ankles_fall_back_to_shoulders(self):
        # 下向き 15° で高さ 150cm のカメラから、肩(床から140cm)は少し下に見える
        lms = make_landmarks(visibility=0.3, shoulders=((0.45, 0.6), (0.55, 0.6)))
        person, note = person_position(lms, 1280, 720, INTR, 0)
        self.assertIsNone(note)
        self.assertEqual(person.method, METHOD_SHOULDER)
        x, z, _ = estimate_position_on_plane(640.0, 0.6 * 720, 760, 760, 640, 360, config.SHOULDER_HEIGHT_CM)
        self.assertAlmostEqual(person.Z, z)

    def test_nothing_visible(self):
        lms = make_landmarks(presence=0.3, shoulder_visibility=0.3)
        self.assertEqual(person_position(lms, 1280, 720, INTR, 0), (None, PERSON_NOT_VISIBLE))

    def test_no_person(self):
        self.assertEqual(person_position(None, 1280, 720, INTR, 0), (None, PERSON_NONE))

    def test_visible_but_cannot_project(self):
        # 足首も肩も画面の一番上(水平より上)で、どちらの面とも交わらない
        lms = make_landmarks(left=(0.45, 0.0), right=(0.55, 0.0), shoulders=((0.45, 0.0), (0.55, 0.0)))
        self.assertEqual(person_position(lms, 1280, 720, INTR, 0), (None, PERSON_NO_INTERSECTION))


class CurrentMountTest(unittest.TestCase):
    """今の設置(床から約20cm・約19°上向き)。画面の下の方は床が映るので、足首が見えれば足首から出す。"""

    def test_config(self):
        self.assertEqual((config.CAMERA_MOUNT_HEIGHT_CM, config.CAMERA_TILT_DEG), (20.0, -19.0))

    def test_bottom_of_image_reaches_the_floor(self):
        # 画面の一番下(v=720)は光軸から atan(360/760)=25.3° 下 = 水平から 6.3° 下
        x, z, valid = estimate_floor_position(640, 720, 760, 760, 640, 360)
        self.assertTrue(valid)
        self.assertAlmostEqual(z, 20.0 / math.tan(math.radians(math.degrees(math.atan(360 / 760)) - 19.0)), places=3)
        # 画面の中央より上は水平より上なので、床とは交わらない
        self.assertFalse(estimate_floor_position(640, 360, 760, 760, 640, 360)[2])

    def test_uses_ankles_when_visible(self):
        person, note = person_position(make_landmarks(), 1280, 720, INTR, 0)  # 足首は v=684(画面の下の方)
        self.assertIsNone(note)
        self.assertEqual(person.method, METHOD_ANKLE)
        below = math.degrees(math.atan((0.95 * 720 - 360) / 760)) - 19.0   # 水平から下へ約 4.1°
        self.assertAlmostEqual(person.Z, 20.0 / math.tan(math.radians(below)), places=3)

    def test_uses_shoulders_when_ankles_hidden(self):
        lms = make_landmarks(visibility=0.3, shoulders=((0.45, 0.3), (0.55, 0.3)))
        person, note = person_position(lms, 1280, 720, INTR, 0)
        self.assertIsNone(note)
        self.assertEqual(person.method, METHOD_SHOULDER)
        # 肩は光軸から atan(144/760) 上 = 水平から約 29.7° 上。120cm 上(床から140cm)の面までの水平距離
        elevation = 19.0 + math.degrees(math.atan((360 - 0.3 * 720) / 760))
        self.assertAlmostEqual(person.Z, 120.0 / math.tan(math.radians(elevation)), places=3)


class CalibrationTest(unittest.TestCase):
    def test_load_and_scale(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calib.yaml"
            path.write_text("camera_matrix: {fx: 2280.0, fy: 2280.0, cx: 1920.0, cy: 1080.0}\n"
                            "image_resolution: {width: 3840, height: 2160}\n", encoding="utf-8")
            intr = load_calibration(path)
        self.assertEqual((intr.fx, intr.width), (2280.0, 3840))
        scaled = intr.scaled_to(1280, 720)
        self.assertAlmostEqual(scaled.fx, 760.0)
        self.assertAlmostEqual(scaled.cy, 360.0)
        self.assertIs(scaled.scaled_to(1280, 720), scaled)

    def test_missing_or_broken(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(CameraError):
                load_calibration(Path(d) / "none.yaml")
            broken = Path(d) / "broken.yaml"
            broken.write_text("foo: 1\n", encoding="utf-8")
            with self.assertRaises(CameraError):
                load_calibration(broken)


class OpenCameraRetryTest(unittest.TestCase):
    """解像度が一度で反映されないカメラ(偽物)で、設定し直し・開き直しを試すこと。"""

    def run_with(self, sizes_per_open):
        """sizes_per_open[i] = i 回目に開いたカメラが、読むたびに返す大きさの列。"""
        import camera
        opens = []

        class FakeVideoCapture:
            def __init__(self, *args):
                self.sizes = list(sizes_per_open[len(opens)])
                opens.append(self)
                self.released = False

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def get(self, *args):
                return 60.0

            def read(self):
                w, h = self.sizes.pop(0) if len(self.sizes) > 1 else self.sizes[0]
                return True, np.zeros((h, w, 3), np.uint8)

            def release(self):
                self.released = True

        with umock.patch.object(camera.cv2, "VideoCapture", FakeVideoCapture):
            cap, w, h = camera.open_camera(0, 1280, 720, 60)
        return (w, h), opens

    def test_ok_first_time(self):
        size, opens = self.run_with([[(1280, 720)]])
        self.assertEqual((size, len(opens)), ((1280, 720), 1))

    def test_fixed_by_setting_again(self):
        size, opens = self.run_with([[(960, 540), (1280, 720)]])
        self.assertEqual((size, len(opens)), ((1280, 720), 1))

    def test_fixed_by_reopening(self):
        size, opens = self.run_with([[(960, 540)], [(1280, 720)]])
        self.assertEqual((size, len(opens)), ((1280, 720), 2))
        self.assertTrue(opens[0].released)

    def test_falls_back_when_never_fixed(self):
        with self.assertLogs("launcher.camera", level="WARNING") as logs:
            size, opens = self.run_with([[(960, 540)], [(960, 540)]])
        self.assertEqual(size, (960, 540))
        self.assertIn("できなかった", logs.output[-1])


class CsvRecorderTest(unittest.TestCase):
    def test_no_rows_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            rec = CsvRecorder(Path(d), "shuttle_trajectory", SHUTTLE_CSV_HEADER, "20260101_000000", "test")
            self.assertIsNone(rec.close())
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_rows_written_with_header(self):
        with tempfile.TemporaryDirectory() as d:
            rec = CsvRecorder(Path(d) / "sub", "person_position", PERSON_CSV_HEADER, "20260101_000000", "test")
            rec.add((1.0, 2.0, 3.0, 4.0, 5.0))
            rec.add((2.0, 2.0, 3.0, 4.0, 5.0))
            path = rec.close()
            self.assertEqual(path.name, "person_position_20260101_000000.csv")
            with open(path, encoding="utf-8") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], PERSON_CSV_HEADER)
        self.assertEqual(len(rows), 3)


# ---- CameraWorker(偽物の入力で) ---------------------------------------------

class FakeCapture:
    def __init__(self, n_frames):
        self.remaining = n_frames
        self.released = False

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        time.sleep(0.005)
        return True, np.zeros((720, 1280, 3), np.uint8)

    def release(self):
        self.released = True


class FakeShuttleDetector:
    def detect(self, frame):
        return np.array([[600.0, 300.0, 640.0, 330.0], [10.0, 10.0, 20.0, 20.0]]), np.array([0.9, 0.3])


class FakePoseDetector:
    def __init__(self, fail=False):
        self.calls = 0
        self.closed = False
        self.fail = fail
        self.timestamps = []

    def detect(self, frame, timestamp_ms):
        if self.fail:
            raise RuntimeError("テスト用の MediaPipe エラー")
        self.calls += 1
        self.timestamps.append(timestamp_ms)
        return make_landmarks()

    def draw(self, frame, landmarks):
        pass

    def close(self):
        self.closed = True


class CameraWorkerTest(unittest.TestCase):
    def setUp(self):
        patcher = umock.patch.multiple(config, **DOWNWARD_MOUNT)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_worker(self, d, n_frames=10, pose=None, is_file=True):
        cap = FakeCapture(n_frames)
        pose = pose or FakePoseDetector()
        shuttle_rec = CsvRecorder(Path(d), "shuttle_trajectory", SHUTTLE_CSV_HEADER, "t", "シャトル")
        person_rec = CsvRecorder(Path(d), "person_position", PERSON_CSV_HEADER, "t", "立ち位置")
        worker = CameraWorker(cap, 1280, 720, INTR, FakeShuttleDetector(), pose, shuttle_rec, person_rec, is_file=is_file)
        return worker, cap, pose

    def test_processes_records_and_stops_at_end_of_file(self):
        with tempfile.TemporaryDirectory() as d:
            worker, cap, pose = self.make_worker(d, n_frames=10)
            worker.start()
            self.assertTrue(wait_until(lambda: worker.status().state == STATE_STOPPED))
            status = worker.status()
            self.assertIn("最後まで", status.message)
            self.assertEqual(status.shuttle_rows, 10)          # 毎フレーム
            self.assertEqual(pose.calls, 10 // config.POSE_INFERENCE_INTERVAL)
            self.assertEqual(status.person_rows, pose.calls)  # 推論したフレームだけ
            self.assertEqual(pose.timestamps, sorted(pose.timestamps))
            self.assertAlmostEqual(status.shuttle.Z, 760.0 * config.SHUTTLE_DIAMETER_CM / 40.0)  # 信頼度の高い方
            self.assertIsNotNone(status.person)
            self.assertIsNotNone(worker.latest_frame())
            worker.stop()
            self.assertTrue(cap.released)
            self.assertTrue(pose.closed)
            with open(Path(d) / "shuttle_trajectory_t.csv", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.reader(f))), 11)
            worker.stop()  # 2回目でもエラーにならない

    def test_detector_error_stops_camera_only(self):
        with tempfile.TemporaryDirectory() as d:
            worker, cap, pose = self.make_worker(d, n_frames=100, pose=FakePoseDetector(fail=True))
            worker.start()
            self.assertTrue(wait_until(lambda: worker.status().state == STATE_STOPPED))
            self.assertIn("テスト用の MediaPipe エラー", worker.status().message)
            worker.stop()
            self.assertTrue(cap.released)

    def test_running_state_and_stop(self):
        with tempfile.TemporaryDirectory() as d:
            worker, cap, _ = self.make_worker(d, n_frames=10_000, is_file=False)
            worker.start()
            self.assertTrue(wait_until(lambda: worker.latest_frame() is not None))
            self.assertEqual(worker.status().state, STATE_RUNNING)
            worker.stop()
            self.assertTrue(cap.released)
            self.assertEqual(worker.status().state, STATE_STOPPED)

    def test_pose_sees_frame_without_shuttle_drawing(self):
        seen = []

        class RecordingPose(FakePoseDetector):
            def detect(self, frame, timestamp_ms):
                seen.append(int(frame.sum()))
                return super().detect(frame, timestamp_ms)

        worker = CameraWorker(FakeCapture(0), 1280, 720, INTR, FakeShuttleDetector(), RecordingPose())
        for i in range(config.POSE_INFERENCE_INTERVAL):
            frame = np.zeros((720, 1280, 3), np.uint8)
            worker.process_frame(frame, 0.0, 1.0 + i)
        self.assertEqual(seen, [0])                 # MediaPipe には真っ黒(何も描かれていない)映像が渡る
        self.assertGreater(int(worker.latest_frame().sum()), 0)  # 表示用の映像には描かれている

    def test_startup_stage_is_reported(self):
        import threading
        gate = threading.Event()

        class SlowShuttleDetector(FakeShuttleDetector):
            def detect(self, frame):
                gate.wait(2.0)  # 初回の推論が終わらない状態を作る
                return super().detect(frame)

        with tempfile.TemporaryDirectory() as d:
            cap = FakeCapture(10_000)
            worker = CameraWorker(cap, 1280, 720, INTR, SlowShuttleDetector(), FakePoseDetector(), is_file=False)
            worker.start()
            try:
                self.assertTrue(wait_until(lambda: worker.status().stage is not None and "YOLO" in worker.status().stage))
                self.assertEqual(worker.status().frames, 0)
                self.assertEqual(worker.status().state, STATE_RUNNING)
                gate.set()
                self.assertTrue(wait_until(lambda: worker.status().stage is None and worker.status().frames > 2))
            finally:
                gate.set()
                worker.stop()

    def test_camera_read_failure(self):
        with tempfile.TemporaryDirectory() as d:
            worker, _, _ = self.make_worker(d, n_frames=3, is_file=False)
            worker.start()
            self.assertTrue(wait_until(lambda: worker.status().state == STATE_STOPPED))
            self.assertIn("取得できません", worker.status().message)
            worker.stop()


class ScreenCameraLinesTest(unittest.TestCase):
    def test_camera_lines_render(self):
        import main
        from camera_worker import CameraStatus
        from vision import PersonObservation, ShuttleObservation
        lines = main.Screen._camera_lines("使わない(--no-camera)")
        self.assertIn("使わない", lines[-1][0])
        status = CameraStatus(STATE_RUNNING, None, 30.0, ShuttleObservation(0, 1, 2, 300, 0, 0, 1, 1, 0.9),
                              PersonObservation(0, 0, 0, 10, 400, METHOD_SHOULDER), None, 5, 3)
        text = " ".join(t for t, _ in main.Screen._camera_lines(status))
        self.assertIn("Z=300", text)
        self.assertIn("Z=400", text)
        self.assertIn("シャトル 5 件", text)


# ---- 実物の YOLO / MediaPipe(あれば) ---------------------------------------------

def _has(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has("ultralytics") and config.YOLO_MODEL_PATH.exists(), "ultralytics か YOLO のモデルが無い")
class RealShuttleDetectorTest(unittest.TestCase):
    def test_detect_on_dataset_image(self):
        import cv2
        from vision import ShuttleDetector
        images = sorted((config.REPO_DIR / "datasets" / "merged" / "test" / "images").glob("*.jpg"))
        if not images:
            self.skipTest("テスト用の画像が無い")
        detector = ShuttleDetector()
        found = 0
        for path in images[:10]:
            boxes, confs = detector.detect(cv2.imread(str(path)))
            self.assertEqual(boxes.shape[1:], (4,))
            self.assertEqual(len(boxes), len(confs))
            found += len(confs) > 0
        self.assertGreater(found, 0)


@unittest.skipUnless(_has("mediapipe") and config.POSE_MODEL_PATH.exists(), "mediapipe かモデルが無い")
class RealPoseDetectorTest(unittest.TestCase):
    def test_blank_image_has_no_person(self):
        from vision import PoseDetector
        detector = PoseDetector()
        try:
            self.assertIsNone(detector.detect(np.zeros((720, 1280, 3), np.uint8), 0))
            self.assertIsNone(detector.detect(np.zeros((720, 1280, 3), np.uint8), 33))
        finally:
            detector.close()


if __name__ == "__main__":
    unittest.main()
