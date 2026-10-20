# -*- coding: utf-8 -*-
"""realtime_pose_detect.estimate_floor_position() の単体テスト。

    python -m unittest discover -s tests
"""

import math
import unittest

from realtime_pose_detect import estimate_floor_position

FX = FY = 760.0
CX, CY = 640.0, 360.0
HEIGHT = 150.0


def pixel_below_axis(deg):
    """光軸から deg 度下を見る画素の v 座標。"""
    return CY + FY * math.tan(math.radians(deg))


class TestEstimateFloorPosition(unittest.TestCase):
    def test_optical_axis_hits_floor_when_tilted_down(self):
        # 俯角15°なら、画像中心は 150/tan(15°) ≒ 560cm 先の床
        x, z, valid = estimate_floor_position(CX, CY, FX, FY, CX, CY, HEIGHT, 15.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, HEIGHT / math.tan(math.radians(15.0)), places=3)

    def test_pixel_below_axis_adds_to_tilt(self):
        # 光軸から20°下の画素は、俯角15°と合わせて35°下を見ている
        x, z, valid = estimate_floor_position(CX, pixel_below_axis(20.0), FX, FY, CX, CY, HEIGHT, 15.0)
        self.assertTrue(valid)
        self.assertAlmostEqual(z, HEIGHT / math.tan(math.radians(35.0)), places=3)

    def test_level_camera_center_never_hits_floor(self):
        _, _, valid = estimate_floor_position(CX, CY, FX, FY, CX, CY, HEIGHT, 0.0)
        self.assertFalse(valid)

    def test_above_horizon_is_invalid(self):
        # 俯角15°で光軸から20°上 = 水平より5°上
        _, _, valid = estimate_floor_position(CX, pixel_below_axis(-20.0), FX, FY, CX, CY, HEIGHT, 15.0)
        self.assertFalse(valid)

    def test_right_is_positive_x(self):
        x, _, valid = estimate_floor_position(CX + 200.0, pixel_below_axis(20.0), FX, FY, CX, CY, HEIGHT, 15.0)
        self.assertTrue(valid)
        self.assertGreater(x, 0.0)


if __name__ == "__main__":
    unittest.main()
