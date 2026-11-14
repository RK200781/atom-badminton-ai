# -*- coding: utf-8 -*-
"""自動版(ballistics / feedback / shots / aiming / auto_controller)のテスト。実機・カメラは使わない。"""

import io
import math
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ballistics  # noqa: E402
import config  # noqa: E402
from aiming import AimModel, person_from_muzzle  # noqa: E402
from auto_controller import (REASON_AIM_OFF, REASON_NOT_SETTLED, REASON_PERSON_LOST,  # noqa: E402
                             REASON_PERSON_MOVING, AutoController)
from feedback import parse_command, parse_feedback  # noqa: E402
from motors import MotorState  # noqa: E402
from shots import Shot, ShotLog  # noqa: E402


# ============================================================
# ballistics
# ============================================================

class BallisticsTest(unittest.TestCase):
    def test_vacuum_like_limit(self):
        # 終端速度がとても大きければ、空気抵抗なしの放物線とほぼ同じになる
        v, h0 = 5.0, 135.0
        r = ballistics.ranges_cm([0.0], v, h0_cm=h0, target_height_cm=0.0, terminal_velocity_ms=1e6)[0]
        expected = v * math.sqrt(2 * h0 / 100 / config.GRAVITY_MS2) * 100
        self.assertAlmostEqual(r, expected, delta=1.0)

    def test_drag_shortens_and_angle_increases_range(self):
        no_drag = ballistics.ranges_cm([10.0], 10.0, terminal_velocity_ms=1e6)[0]
        drag = ballistics.range_cm(10.0, 10.0)
        self.assertLess(drag, no_drag)
        self.assertLess(ballistics.range_cm(0.0, 10.0), ballistics.range_cm(15.0, 10.0))
        self.assertLess(ballistics.range_cm(10.0, 8.0), ballistics.range_cm(10.0, 12.0))

    def test_unreachable_target_height_is_nan(self):
        r = ballistics.ranges_cm([0.0], 3.0, h0_cm=100.0, target_height_cm=500.0)[0]
        self.assertTrue(math.isnan(r))

    def test_elevation_for_distance_roundtrip(self):
        for d in (360.0, 400.0, 450.0):
            el, result = ballistics.elevation_for_distance(d, 9.0)
            self.assertEqual(result, ballistics.SOLVE_OK)
            self.assertAlmostEqual(ballistics.range_cm(el, 9.0), d, delta=3.0)

    def test_elevation_out_of_range(self):
        self.assertEqual(ballistics.elevation_for_distance(5000.0, 9.0)[1], ballistics.SOLVE_TOO_FAR)
        self.assertEqual(ballistics.elevation_for_distance(10.0, 9.0)[1], ballistics.SOLVE_TOO_NEAR)

    def test_fit_speed_recovers_true_speed(self):
        true_v = 10.5
        els = [0.0, 10.0, 20.0]
        ds = [ballistics.range_cm(e, true_v) for e in els]
        self.assertAlmostEqual(ballistics.fit_speed(els, ds), true_v, delta=0.05)


# ============================================================
# feedback
# ============================================================

class FeedbackTest(unittest.TestCase):
    def check(self, text, distance=None, landing=None, lateral=None):
        fb = parse_feedback(text)
        self.assertEqual((fb.distance_error_cm, fb.landing_distance_cm, fb.lateral_cm), (distance, landing, lateral),
                         msg=text)

    def test_short_and_long(self):
        self.check("1m足りない", distance=-100.0)
        self.check("50cm手前", distance=-50.0)
        self.check("あと1.5メートル", distance=-150.0)
        self.check("2m超えた", distance=200.0)
        self.check("30センチ奥", distance=30.0)
        self.check("1m飛びすぎ", distance=100.0)
        self.check("１ｍ足りない", distance=-100.0)       # 全角
        self.check("1m50足りない", distance=-150.0)
        self.check("1m 50cm足りない", distance=-150.0)
        self.check("70足りない", distance=-70.0)          # 単位なし・20 以上 → cm
        self.check("2足りない", distance=-200.0)          # 単位なし・20 未満 → m

    def test_lateral(self):
        self.check("右に30cm", lateral=30.0)
        self.check("20cm左", lateral=-20.0)

    def test_absolute(self):
        self.check("飛距離4.5m", landing=450.0)
        self.check("4m50地点", landing=450.0)

    def test_combined(self):
        self.check("1m足りなくて右に30cm", distance=-100.0, lateral=30.0)
        self.check("右に30cm、1m足りない", distance=-100.0, lateral=30.0)

    def test_ok(self):
        self.check("ちょうど", distance=0.0, lateral=0.0)
        self.check("OK 右に10cm", lateral=10.0)  # 数字があればそちらを使う(OK だけの文節ではない)

    def test_unparsed(self):
        fb = parse_feedback("少し足りない")
        self.assertFalse(fb.usable)
        self.assertTrue(fb.unparsed)

    def test_commands(self):
        self.assertEqual(parse_command("").kind, "empty")
        self.assertEqual(parse_command("help").kind, "help")
        self.assertEqual(parse_command("取り消し").kind, "undo")
        c = parse_command("#3 1m足りない")
        self.assertEqual((c.kind, c.shot_id, c.feedback.distance_error_cm), ("feedback", 3, -100.0))
        self.assertEqual(parse_command("＃１２ ちょうど").shot_id, 12)


# ============================================================
# shots
# ============================================================

def make_shot(shot_id=1, speed=2, pitch=-10.0, yaw=0.0, person_distance=400.0, person_yaw=0.0, **kw):
    return Shot(shot_id=shot_id, session="s", time="t", trigger="key", speed=speed, yaw_deg=yaw, pitch_deg=pitch,
                auto_aim=True, person_distance_cm=person_distance, person_yaw_deg=person_yaw, **kw)


class ShotLogTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "shots.csv"

    def tearDown(self):
        self.dir.cleanup()

    def test_roundtrip_and_feedback(self):
        log = ShotLog(self.path)
        self.assertEqual(log.next_id(), 1)
        log.add(make_shot(1, person_X_cm=12.5, person_method="ankle", video="v.mp4", video_frame=10))
        log.add(make_shot(2, person_distance=None, person_yaw=None))
        log.apply_feedback(log.get(1), parse_feedback("1m足りない"))
        log.apply_feedback(log.get(1), parse_feedback("右に20cm"))
        again = ShotLog(self.path)
        s1, s2 = again.get(1), again.get(2)
        self.assertEqual(s1.distance_error_cm, -100.0)
        self.assertEqual(s1.lateral_cm, 20.0)
        self.assertEqual(s1.feedback_text, "1m足りない / 右に20cm")
        self.assertEqual((s1.person_X_cm, s1.person_method, s1.video_frame, s1.auto_aim), (12.5, "ankle", 10, True))
        self.assertIsNone(s2.person_distance_cm)
        self.assertEqual(again.next_id(), 3)
        self.assertIs(again.get(None), s2)

    def test_undo(self):
        log = ShotLog(self.path)
        log.add(make_shot(1))
        log.apply_feedback(log.get(1), parse_feedback("1m足りない"))
        log.apply_feedback(log.get(1), parse_feedback("2m超えた"))
        self.assertEqual(log.undo(), 1)
        self.assertEqual(log.get(1).distance_error_cm, -100.0)
        self.assertEqual(log.undo(), 1)
        self.assertIsNone(log.get(1).distance_error_cm)
        self.assertEqual(log.get(1).feedback_text, "")
        self.assertIsNone(log.undo())

    def test_actual_distance_and_direction(self):
        s = make_shot(person_distance=400.0, person_yaw=10.0, distance_error_cm=-100.0, lateral_cm=0.0)
        self.assertEqual(s.actual_distance_cm(), 300.0)
        self.assertAlmostEqual(s.actual_direction_deg(), 10.0)
        s.landing_distance_cm = 500.0
        self.assertEqual(s.actual_distance_cm(), 500.0)
        right = make_shot(person_distance=400.0, person_yaw=0.0, distance_error_cm=0.0, lateral_cm=40.0)
        self.assertAlmostEqual(right.actual_direction_deg(), math.degrees(math.atan2(40, 400)))
        no_person = make_shot(person_distance=None, person_yaw=None, yaw=5.0, landing_distance_cm=400.0,
                              lateral_cm=0.0)
        self.assertAlmostEqual(no_person.actual_direction_deg(), 5.0)


# ============================================================
# aiming
# ============================================================

class AimingTest(unittest.TestCase):
    def test_geometry(self):
        yaw, dist = person_from_muzzle(0.0, 375.0)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(dist, 375.0 + config.LAUNCHER_MUZZLE_BEHIND_CAMERA_CM)
        yaw, _ = person_from_muzzle(100.0, 75.0)   # 右にいる → yaw はプラス(プラスが右)
        self.assertAlmostEqual(yaw, 45.0)

    def test_solve_hits_distance(self):
        m = AimModel()
        sol = m.solve(0.0, 375.0, 2)
        self.assertEqual(sol.warnings, [])
        self.assertLess(sol.pitch_deg, 0)   # 上向き = マイナス
        self.assertAlmostEqual(sol.predicted_distance_cm, 400.0, delta=3.0)

    def test_speed_zero_gives_no_pitch(self):
        sol = AimModel().solve(0.0, 375.0, 0)
        self.assertIsNone(sol.pitch_deg)
        self.assertTrue(sol.warnings)

    def test_learning_recovers_speed_and_yaw_bias(self):
        true_v, true_bias = 12.0, 2.0   # 実際は yaw の指令より 2° 左に飛ぶ(指令 − 実際 = +2°)
        shots = []
        for i, pitch in enumerate((-5.0, -12.0, -20.0)):
            actual = ballistics.range_cm(-pitch, true_v)
            person = 450.0
            lateral = math.tan(math.radians(-true_bias)) * actual   # 人は yaw 0° の正面
            shots.append(make_shot(i + 1, speed=2, pitch=pitch, yaw=0.0, person_distance=person, person_yaw=0.0,
                                   distance_error_cm=actual - person, lateral_cm=lateral))
        m = AimModel()
        st = m.learn(shots)
        self.assertAlmostEqual(st.speeds_ms[2], true_v, delta=0.1)
        self.assertEqual(st.speed_counts[2], 3)
        self.assertAlmostEqual(st.yaw_bias_deg, true_bias, delta=0.05)
        # 記録が無い強さは、同じ倍率で初期値から動かす
        ratio = true_v / config.INITIAL_LAUNCH_SPEED_MS[2]
        self.assertAlmostEqual(st.speeds_ms[1], config.INITIAL_LAUNCH_SPEED_MS[1] * ratio, delta=0.1)
        # yaw のずれを足して狙う
        self.assertAlmostEqual(m.solve(0.0, 400.0, 2).yaw_deg, true_bias, delta=0.05)

    def test_feedback_without_numbers_is_ignored(self):
        m = AimModel()
        st = m.learn([make_shot(1)])
        self.assertEqual(st.speeds_ms, config.INITIAL_LAUNCH_SPEED_MS)
        self.assertEqual(st.yaw_bias_deg, 0.0)


# ============================================================
# auto_controller
# ============================================================

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeLauncher:
    """Launcher の代わり。aim() すると、すぐにその角度に着いたことにする(settle=False なら着かない)。"""

    def __init__(self, speed=2, settle=True, fire_reasons=None):
        self.speed = speed
        self.settle = settle
        self.fire_reasons = fire_reasons or []
        self.target = [0.0, 0.0]
        self.current = [0.0, 0.0]
        self.aims = []
        self.fires = 0

    def aim(self, yaw, pitch):
        self.aims.append((yaw, pitch))
        self.target = [yaw, pitch]
        if self.settle:
            self.current = [yaw, pitch]

    def fire(self):
        if self.fire_reasons:
            return False, list(self.fire_reasons)
        self.fires += 1
        return True, []

    def status(self):
        motors = MotorState(True, None, self.target[0], self.target[1], self.current[0], self.current[1])
        feeder = SimpleNamespace(speed=self.speed)
        return SimpleNamespace(motors=motors, feeder=feeder, fire_reasons=list(self.fire_reasons))


@dataclass
class Person:
    timestamp: float
    X: float
    Z: float
    method: str = "ankle"


class FakeVideo:
    name = "session_x.mp4"
    recording = True
    frame_count = 123


class AutoControllerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = ShotLog(Path(self.dir.name) / "shots.csv")
        self.clock = FakeClock()
        self.launcher = FakeLauncher()
        self.ctl = AutoController(self.launcher, self.log, session="s1", video=FakeVideo(), clock=self.clock)
        self.ts = 0

    def tearDown(self):
        self.dir.cleanup()

    def step(self, person_xz=(50.0, 375.0), dt=0.05, n=1):
        status = None
        for _ in range(n):
            self.clock.t += dt
            self.ts += 1
            person = Person(self.ts, *person_xz) if person_xz is not None else None
            status = self.ctl.tick(person)
        return status

    def test_aim_off_does_not_move(self):
        st = self.step(n=10)
        self.assertEqual(self.launcher.aims, [])
        self.assertIn(REASON_AIM_OFF, st.auto_fire_reasons)
        self.assertIsNotNone(st.solution)   # 計算はして表示する

    def test_aim_on_moves_toward_person(self):
        self.ctl.toggle_aim()
        st = self.step(n=5)
        yaw, pitch = self.launcher.aims[-1]
        self.assertAlmostEqual(yaw, math.degrees(math.atan2(50.0, 400.0)), delta=0.01)
        self.assertLess(pitch, 0)
        self.assertTrue(st.settled)
        # 同じ位置なら送り直さない(不感帯)
        n = len(self.launcher.aims)
        self.step(n=5)
        self.assertEqual(len(self.launcher.aims), n)

    def test_person_lost_holds_position(self):
        self.ctl.toggle_aim()
        self.step(n=5)
        n = len(self.launcher.aims)
        st = self.step(person_xz=None, dt=0.5, n=4)
        self.assertIsNone(st.person)
        self.assertIn(REASON_PERSON_LOST, st.auto_fire_reasons)
        self.assertEqual(len(self.launcher.aims), n)

    def test_auto_fire_needs_stable_person_and_settled_aim(self):
        self.ctl.toggle_aim()
        self.ctl.toggle_fire()
        st = self.step(n=3)
        self.assertIn(REASON_PERSON_MOVING, st.auto_fire_reasons)   # 見つけた直後は撃たない
        self.assertEqual(self.launcher.fires, 0)
        self.step(n=40)   # 2 秒止まっている
        self.assertEqual(self.launcher.fires, 1)
        shot = self.log.get(None)
        self.assertEqual((shot.trigger, shot.speed, shot.session, shot.video_frame), ("auto", 2, "s1", 123))
        self.assertAlmostEqual(shot.person_distance_cm, math.hypot(50.0, 400.0), delta=0.01)
        # 間隔を空ける
        self.step(n=20)
        self.assertEqual(self.launcher.fires, 1)
        self.step(n=int(config.AUTO_FIRE_INTERVAL_SEC / 0.05) + 2)
        self.assertEqual(self.launcher.fires, 2)

    def test_auto_fire_waits_for_motors(self):
        self.launcher.settle = False
        self.ctl.toggle_aim()
        self.ctl.toggle_fire()
        st = self.step(n=40)
        self.assertIn(REASON_NOT_SETTLED, st.auto_fire_reasons)
        self.assertEqual(self.launcher.fires, 0)

    def test_moving_person_blocks_auto_fire(self):
        self.ctl.toggle_aim()
        self.ctl.toggle_fire()
        for i in range(40):
            self.step(person_xz=(50.0 + i * 5, 375.0))
        self.assertEqual(self.launcher.fires, 0)

    def test_fire_aborts_when_launcher_refuses(self):
        self.launcher.fire_reasons = ["シャトルを検知していません"]
        self.assertIsNone(self.ctl.fire("key"))
        self.assertEqual(self.log.all(), [])

    def test_fire_feedback_and_learning(self):
        self.step(n=20)
        shot = self.ctl.fire("key")
        self.assertIsNotNone(shot)
        self.assertFalse(shot.auto_aim)
        before = self.ctl.model.state.speeds_ms[2]
        self.ctl.handle_line("1m足りない\n")
        self.assertEqual(self.log.get(shot.shot_id).distance_error_cm, -100.0)
        self.assertLess(self.ctl.model.state.speeds_ms[2], before)   # 足りなかった → 遅いと学習
        self.ctl.handle_line("取り消し")
        self.assertAlmostEqual(self.ctl.model.state.speeds_ms[2], before)

    def test_feedback_without_shot(self):
        self.ctl.handle_line("1m足りない")   # 落ちない
        self.assertEqual(self.log.all(), [])

    def test_learning_persists_across_sessions(self):
        self.step(n=20)
        self.ctl.fire("key")
        self.ctl.handle_line("2m超えた")
        learned = self.ctl.model.state.speeds_ms[2]
        other = AutoController(FakeLauncher(), ShotLog(self.log.path), session="s2", clock=self.clock)
        self.assertAlmostEqual(other.model.state.speeds_ms[2], learned, delta=0.01)

    def test_toggle_fire_requires_aim(self):
        self.ctl.toggle_fire()
        self.assertFalse(self.ctl.fire_enabled)
        self.ctl.toggle_aim()
        self.ctl.toggle_fire()
        self.assertTrue(self.ctl.fire_enabled)
        self.ctl.disable("test")
        self.assertFalse(self.ctl.aim_enabled or self.ctl.fire_enabled)


class ConsoleInputTest(unittest.TestCase):
    def test_reads_lines(self):
        import time
        from auto_main import ConsoleInput
        c = ConsoleInput(io.StringIO("1m足りない\n右に30cm\n"))
        c.start()
        deadline = time.time() + 2
        got = []
        while len(got) < 2 and time.time() < deadline:
            got += list(c.lines())
            time.sleep(0.01)
        self.assertEqual(got, ["1m足りない\n", "右に30cm\n"])


if __name__ == "__main__":
    unittest.main()
