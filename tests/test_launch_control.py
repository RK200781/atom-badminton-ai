# -*- coding: utf-8 -*-
"""launch_control の単体テスト。

    python -m unittest discover -s tests
"""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from launch_control import aiming, config
from launch_control import fire_controller as fc
from launch_control.fire_controller import FireController, ShotLogger
from launch_control.launcher import MockLauncher

DT = 1.0 / 30.0
OK_POS = (100.0, 400.0)  # 範囲内で狙える立ち位置(カメラ座標系)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Sim:
    """時計・MockLauncher・FireController をまとめて動かすテスト用の道具。"""

    def __init__(self, shot_logger=None, **launcher_kwargs):
        self.clock = FakeClock()
        self.launcher = MockLauncher(clock=self.clock, log=None, **launcher_kwargs)
        self.controller = FireController(
            self.launcher, shot_logger=shot_logger, wall_clock=self.clock, log=None,
        )
        self.states = []

    def run(self, duration, person):
        """duration 秒ぶん進める。person は (x, z)・None・時刻を受け取る関数のいずれか。"""
        end = self.clock.t + duration
        while self.clock.t < end - 1e-9:
            p = person(self.clock.t) if callable(person) else person
            self.states.append(self.controller.update(self.clock.t, p))
            self.clock.t += DT
        return self.controller.state

    @property
    def shots(self):
        return self.controller.shot_count


class TestCameraToLauncher(unittest.TestCase):
    def test_identity_by_default(self):
        self.assertEqual(aiming.camera_to_launcher(30.0, 200.0, config.CameraMount()), (30.0, 200.0))

    def test_offset(self):
        mount = config.CameraMount(offset_x_cm=50.0, offset_z_cm=-20.0)
        self.assertEqual(aiming.camera_to_launcher(10.0, 100.0, mount), (60.0, 80.0))

    def test_yaw_right_90(self):
        # カメラが右を向いている: カメラの正面は発射機の右、カメラの右は発射機の後ろ
        mount = config.CameraMount(yaw_deg=90.0)
        x, z = aiming.camera_to_launcher(0.0, 100.0, mount)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(z, 0.0)
        x, z = aiming.camera_to_launcher(100.0, 0.0, mount)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, -100.0)

    def test_pan_sign(self):
        self.assertAlmostEqual(aiming.pan_angle_deg(100.0, 100.0), 45.0)
        self.assertAlmostEqual(aiming.pan_angle_deg(-100.0, 100.0), -45.0)
        self.assertAlmostEqual(aiming.pan_angle_deg(0.0, 100.0), 0.0)


class TestTiltTable(unittest.TestCase):
    TABLE = [(10.0, 200.0), (30.0, 500.0), (45.0, 600.0), (60.0, 450.0)]

    def test_interpolation(self):
        status, tilt = aiming.tilt_for_distance(350.0, self.TABLE)
        self.assertEqual(status, aiming.STATUS_OK)
        self.assertAlmostEqual(tilt, 20.0)

    def test_table_points_exact(self):
        for t, d in self.TABLE[:3]:
            self.assertAlmostEqual(aiming.tilt_for_distance(d, self.TABLE)[1], t)

    def test_uses_low_trajectory(self):
        # 距離500cm は 30°(低い弾道)と 50°(高い弾道)の2つがあるが、低い方を選ぶ
        status, tilt = aiming.tilt_for_distance(500.0, self.TABLE)
        self.assertEqual(status, aiming.STATUS_OK)
        self.assertAlmostEqual(tilt, 30.0)
        self.assertEqual(aiming.ascending_branch(self.TABLE), self.TABLE[:3])

    def test_too_far_and_too_close(self):
        self.assertEqual(aiming.tilt_for_distance(601.0, self.TABLE), (aiming.STATUS_TOO_FAR, None))
        self.assertEqual(aiming.tilt_for_distance(199.0, self.TABLE), (aiming.STATUS_TOO_CLOSE, None))

    def test_invalid_tables(self):
        with self.assertRaises(ValueError):
            aiming.ascending_branch([(10.0, 300.0)])
        with self.assertRaises(ValueError):
            aiming.ascending_branch([(30.0, 300.0), (10.0, 200.0)])
        with self.assertRaises(ValueError):
            aiming.ascending_branch([(10.0, 300.0), (20.0, 200.0)])  # 最小仰角で最大
        with self.assertRaises(ValueError):
            aiming.ascending_branch([(10.0, 200.0), (20.0, 200.0), (30.0, 300.0)])

    def test_default_table_is_valid(self):
        aiming.ascending_branch(config.TILT_TO_DISTANCE_TABLE)


class TestSolveAim(unittest.TestCase):
    def test_ok(self):
        sol = aiming.solve_aim(*OK_POS)
        self.assertTrue(sol.ok)
        self.assertAlmostEqual(sol.pan_deg, math.degrees(math.atan2(100.0, 400.0)))
        self.assertAlmostEqual(sol.distance_cm, math.hypot(*OK_POS))
        self.assertTrue(config.TILT_MIN_DEG <= sol.tilt_deg <= config.TILT_MAX_DEG)

    def test_out_of_pan(self):
        sol = aiming.solve_aim(400.0, 100.0)  # 右に約76°
        self.assertEqual(sol.status, aiming.STATUS_OUT_OF_PAN)
        self.assertIsNone(sol.tilt_deg)

    def test_too_far(self):
        self.assertEqual(aiming.solve_aim(0.0, 2000.0).status, aiming.STATUS_TOO_FAR)

    def test_too_close(self):
        self.assertEqual(aiming.solve_aim(0.0, 50.0).status, aiming.STATUS_TOO_CLOSE)


class TestMockLauncher(unittest.TestCase):
    def test_moves_at_constant_speed_and_returns_immediately(self):
        clock = FakeClock()
        launcher = MockLauncher(pan_speed_deg_s=90.0, tilt_speed_deg_s=60.0, clock=clock, log=None)
        launcher.move_to(45.0, 40.0)
        self.assertEqual(launcher.get_angles(), (0.0, config.TILT_MIN_DEG))
        clock.t = 0.25
        pan, tilt = launcher.get_angles()
        self.assertAlmostEqual(pan, 22.5)
        self.assertAlmostEqual(tilt, config.TILT_MIN_DEG + 15.0)
        clock.t = 10.0
        self.assertEqual(launcher.get_angles(), (45.0, 40.0))

    def test_clamps_to_range(self):
        clock = FakeClock()
        launcher = MockLauncher(clock=clock, log=None)
        launcher.move_to(100.0, 80.0)
        clock.t = 10.0
        self.assertEqual(launcher.get_angles(), (config.PAN_MAX_DEG, config.TILT_MAX_DEG))


class TestFireController(unittest.TestCase):
    def test_full_cycle(self):
        sim = Sim()
        sim.run(config.STILL_WINDOW_SEC + 2.0, OK_POS)
        self.assertEqual(sim.shots, 1)
        for state in (fc.TRACKING, fc.AIMING, fc.FIRING, fc.COOLDOWN):
            self.assertIn(state, sim.states)
        # 順番どおりに遷移している
        order = [s for i, s in enumerate(sim.states) if i == 0 or s != sim.states[i - 1]]
        self.assertEqual(order[:4], [fc.TRACKING, fc.AIMING, fc.FIRING, fc.COOLDOWN])

        # 待ち時間が過ぎたら TRACKING に戻り、静止していればまた撃つ
        sim.run(config.COOLDOWN_SEC + config.STILL_WINDOW_SEC + 2.0, OK_POS)
        self.assertEqual(sim.shots, 2)

        self.assertFalse(sim.launcher.gate_open)

    def test_does_not_fire_before_arrival(self):
        sim = Sim(pan_speed_deg_s=1.0, tilt_speed_deg_s=1.0)  # 遅いので到達に時間がかかる
        fired_angles = []
        original_open = sim.launcher.open_gate

        def open_gate():
            fired_angles.append(sim.launcher.get_angles())
            original_open()

        sim.launcher.open_gate = open_gate
        sim.run(config.STILL_WINDOW_SEC + config.ARRIVAL_TIMEOUT_SEC - 0.1, OK_POS)
        self.assertEqual(sim.controller.state, fc.AIMING)
        self.assertEqual(sim.shots, 0)

        # 速くすれば到達して撃つ。撃った時の角度は目標の ±1° 以内
        sim.launcher.pan_speed = sim.launcher.tilt_speed = 1000.0
        sim.run(0.5, OK_POS)
        self.assertEqual(sim.shots, 1)
        sol = sim.controller.solution
        pan, tilt = fired_angles[0]
        self.assertLessEqual(abs(pan - sol.pan_deg), config.ARRIVAL_TOLERANCE_DEG)
        self.assertLessEqual(abs(tilt - sol.tilt_deg), config.ARRIVAL_TOLERANCE_DEG)

    def test_arrival_timeout_goes_back_to_tracking(self):
        sim = Sim(pan_speed_deg_s=0.1, tilt_speed_deg_s=0.1)
        sim.run(config.STILL_WINDOW_SEC + config.ARRIVAL_TIMEOUT_SEC + 0.5, OK_POS)
        self.assertEqual(sim.shots, 0)
        # AIMING から(撃たずに)TRACKING へ戻っている
        i = sim.states.index(fc.AIMING)
        self.assertIn(fc.TRACKING, sim.states[i:])
        self.assertNotIn(fc.FIRING, sim.states)

    def test_no_fire_while_moving(self):
        sim = Sim()
        # 普通の歩く速さ(1秒に100cm)で横に歩き続ける。
        # 注: 秒速60cm 以下だと直近1秒の広がりが STILL_RADIUS_CM(30cm)に収まり静止扱いになる
        sim.run(4.0, lambda t: (-200.0 + 100.0 * t, 400.0))
        self.assertEqual(sim.shots, 0)
        self.assertNotIn(fc.AIMING, sim.states)

    def test_reaim_when_player_moves(self):
        sim = Sim(pan_speed_deg_s=5.0, tilt_speed_deg_s=5.0)  # 狙っている途中に動けるよう遅くする
        sim.run(config.STILL_WINDOW_SEC + 0.1, OK_POS)
        self.assertEqual(sim.controller.state, fc.AIMING)
        first_pan = sim.controller.solution.pan_deg

        moved = (-100.0, 400.0)
        sim.run(0.5, moved)
        self.assertEqual(sim.controller.state, fc.TRACKING)
        self.assertEqual(sim.shots, 0)

        sim.launcher.pan_speed = sim.launcher.tilt_speed = 1000.0
        sim.run(config.STILL_WINDOW_SEC + 1.0, moved)
        self.assertEqual(sim.shots, 1)
        self.assertLess(sim.controller.solution.pan_deg, 0.0)
        self.assertNotAlmostEqual(sim.controller.solution.pan_deg, first_pan)

    def test_no_fire_when_lost(self):
        # 静止判定が成立する直前に人が消える: 履歴は「静止」でも撃ってはいけない
        sim = Sim()
        sim.run(config.STILL_WINDOW_SEC - 0.1, OK_POS)
        sim.run(config.LOST_TIMEOUT_SEC + 0.5, None)
        self.assertEqual(sim.shots, 0)
        self.assertEqual(sim.controller.state, fc.IDLE)

    def test_no_fire_while_not_visible_even_if_arrived(self):
        sim = Sim(pan_speed_deg_s=30.0, tilt_speed_deg_s=30.0)
        sim.run(config.STILL_WINDOW_SEC + 0.1, OK_POS)
        self.assertEqual(sim.controller.state, fc.AIMING)
        # 見えていない間に到達しても撃たない(見失いタイムアウト未満)
        sim.run(config.LOST_TIMEOUT_SEC - 0.1, None)
        self.assertTrue(sim.controller._arrived())
        self.assertEqual(sim.shots, 0)
        # そのまま見失えば IDLE、撃たない
        sim.run(0.5, None)
        self.assertEqual(sim.controller.state, fc.IDLE)
        self.assertEqual(sim.shots, 0)

    def test_recovers_from_short_dropout(self):
        sim = Sim()

        def flaky(t):  # 0.4秒ごとに 0.1秒(3フレーム)見失う
            return None if (t % 0.4) > 0.3 else OK_POS

        sim.run(config.STILL_WINDOW_SEC + 2.0, flaky)
        self.assertEqual(sim.shots, 1)
        self.assertNotIn(fc.IDLE, sim.states)

    def test_fires_on_return_after_short_dropout_during_aiming(self):
        sim = Sim(pan_speed_deg_s=30.0, tilt_speed_deg_s=30.0)
        sim.run(config.STILL_WINDOW_SEC + 0.1, OK_POS)
        sim.run(0.5, None)  # 短い見失いの間に到達
        self.assertEqual(sim.shots, 0)
        sim.run(DT, OK_POS)  # 戻ってきた最初のフレームで撃つ
        self.assertEqual(sim.shots, 1)

    def test_too_far_shows_message_and_does_not_fire(self):
        sim = Sim()
        sim.run(5.0, (0.0, 2000.0))
        self.assertEqual(sim.shots, 0)
        self.assertEqual(sim.controller.message, fc.AIM_STATUS_MESSAGES[aiming.STATUS_TOO_FAR])
        self.assertEqual(sim.controller.state, fc.TRACKING)


class TestShotLogger(unittest.TestCase):
    def test_logs_and_continues_ids(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "shots.csv"
            sim = Sim(shot_logger=ShotLogger(path))
            sim.run(config.STILL_WINDOW_SEC + 2.0, OK_POS)

            # 別の実行でも shot_id は続きから振られる
            sim2 = Sim(shot_logger=ShotLogger(path))
            sim2.run(config.STILL_WINDOW_SEC + 2.0, OK_POS)

            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["shot_id"] for r in rows], ["1", "2"])
            self.assertEqual(list(rows[0].keys()), fc.SHOT_FIELDNAMES)
            self.assertEqual(rows[0]["landing_x_cm"], "")
            self.assertEqual(rows[0]["landing_z_cm"], "")
            self.assertAlmostEqual(float(rows[0]["person_x_cm"]), OK_POS[0], places=1)


if __name__ == "__main__":
    unittest.main()
