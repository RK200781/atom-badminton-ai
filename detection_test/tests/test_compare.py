# -*- coding: utf-8 -*-
"""compare.py のテスト(偽物の検出器で、集計が正しいか)。

    python -m unittest discover -s detection_test/tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import compare  # noqa: E402
from compare import Setting, best_detection, evaluate, iter_video  # noqa: E402


def fake_detect(frame, imgsz, conf):
    """フレームの左上の画素値で結果を決める偽物。
    値 1: 大きいシャトル(信頼度 0.9)、値 2: 小さいシャトル(imgsz=1280 のときだけ 0.3、640 では 0.12)、
    値 0: 何もない。しきい値 conf 未満は返さない(YOLO と同じ)。"""
    kind = int(frame[0, 0, 0])
    if kind == 1:
        boxes, confs = np.array([[100, 100, 300, 300.0]]), np.array([0.9])
    elif kind == 2:
        boxes, confs = np.array([[10, 10, 30, 25.0]]), np.array([0.3 if imgsz >= 1280 else 0.12])
    else:
        return np.zeros((0, 4)), np.zeros((0,))
    keep = confs >= conf
    return boxes[keep], confs[keep]


def frames_of(kinds):
    for i, k in enumerate(kinds):
        f = np.zeros((40, 40, 3), np.uint8)
        f[0, 0, 0] = k
        yield i, f


class BestDetectionTest(unittest.TestCase):
    def test_picks_highest_above_threshold(self):
        boxes = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]])
        box, c = best_detection(boxes, np.array([0.2, 0.5, 0.4]), 0.3)
        self.assertEqual((box[2], c), (2, 0.5))
        self.assertIsNone(best_detection(boxes, np.array([0.2, 0.1, 0.2]), 0.3))
        self.assertIsNone(best_detection(np.zeros((0, 4)), np.zeros((0,)), 0.1))


class EvaluateTest(unittest.TestCase):
    SETTINGS = [Setting(640, 0.25), Setting(640, 0.1), Setting(1280, 0.25)]

    def test_rates_sizes_and_disagreements(self):
        diffs = []
        result = evaluate(frames_of([1, 2, 0, 2]), fake_detect, self.SETTINGS,
                          on_disagreement=lambda i, f, found: diffs.append(i))
        s640, s640_low, s1280 = (result.stats[s] for s in self.SETTINGS)
        self.assertEqual((s640.frames, s640.detected), (4, 1))        # 大きいシャトルだけ
        self.assertEqual(s640_low.detected, 3)                        # しきい値を下げると小さいのも取れる
        self.assertEqual(s1280.detected, 3)                           # imgsz を上げても取れる
        self.assertAlmostEqual(s640.rate, 0.25)
        self.assertEqual(s1280.bucket_counts(), [2, 0, 0, 1])         # 20px が2つ、200px が1つ
        self.assertEqual(diffs, [1, 3])
        self.assertEqual(result.disagreements, 2)
        self.assertEqual(set(result.ms_per_frame), {640, 1280})

    def test_one_inference_per_imgsz(self):
        calls = []

        def counting(frame, imgsz, conf):
            calls.append((imgsz, conf))
            return fake_detect(frame, imgsz, conf)

        evaluate(frames_of([1, 2]), counting, self.SETTINGS, warmup=False)
        # imgsz=640 は 2 つの設定があっても、一番低いしきい値で1回だけ推論する
        self.assertEqual(calls, [(640, 0.1), (1280, 0.25)] * 2)

    def test_warmup_runs_each_imgsz_once_more(self):
        calls = []
        evaluate(frames_of([1, 2]), lambda f, sz, c: calls.append(sz) or fake_detect(f, sz, c), self.SETTINGS)
        self.assertEqual(calls, [640, 1280] * 3)  # 準備の1回 + 2フレーム分


class VideoTest(unittest.TestCase):
    def test_iter_video_every_and_max(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v.mp4"
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            for i in range(10):
                vw.write(np.full((48, 64, 3), i * 20, np.uint8))
            vw.release()
            self.assertEqual([i for i, _ in iter_video(path)], list(range(10)))
            self.assertEqual([i for i, _ in iter_video(path, every=3)], [0, 3, 6, 9])
            self.assertEqual([i for i, _ in iter_video(path, every=2, max_frames=2)], [0, 2])

    def test_draw_disagreement(self):
        frame = np.zeros((200, 400, 3), np.uint8)
        found = {Setting(640, 0.25): None, Setting(1280, 0.25): (np.array([10, 10, 30, 30]), 0.4)}
        img = compare.draw_disagreement(frame, found, list(found))
        self.assertEqual(img.shape, frame.shape)
        self.assertGreater(int(img.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)  # 元の画像には描かない


if __name__ == "__main__":
    unittest.main()
