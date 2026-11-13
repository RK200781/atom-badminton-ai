# -*- coding: utf-8 -*-
"""record.py のテスト(録画したファイルと時刻の CSV)。"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from record import SegmentRecorder  # noqa: E402


class SegmentRecorderTest(unittest.TestCase):
    def test_writes_video_and_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            rec = SegmentRecorder(Path(d), 320, 240, 30, label="2m 前方")
            self.assertIn("_2m_", rec.path.name + "_")
            for i in range(5):
                rec.write(np.full((240, 320, 3), 40 * i, np.uint8), 1000.0 + i / 30)
            path = rec.close()
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (240, 320, 3))
            with open(rec.timestamps_path, encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["frame", "timestamp"])
            self.assertEqual(len(rows), 6)

    def test_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as d:
            rec = SegmentRecorder(Path(d), 320, 240, 30)
            with self.assertRaises(ValueError):
                rec.write(np.zeros((480, 640, 3), np.uint8), 0.0)
            rec.close()


if __name__ == "__main__":
    unittest.main()
