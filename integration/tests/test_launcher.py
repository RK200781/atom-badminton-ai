# -*- coding: utf-8 -*-
"""launcher.py と main.py のキー操作のテスト。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from feeder import Feeder  # noqa: E402
from launcher import LOCKED_REASON, Launcher  # noqa: E402
from motors import MotorState  # noqa: E402
from test_feeder import FakeClock, make_feeder  # noqa: E402


class StubMotors:
    def __init__(self, fail_shutdown=False):
        self.running = True
        self.calls = []
        self.fail_shutdown = fail_shutdown

    def hold_here(self):
        self.calls.append("hold_here")
        return True

    def set_target(self, yaw, pitch):
        self.calls.append(("set_target", yaw, pitch))

    def nudge(self, d_yaw, d_pitch):
        self.calls.append(("nudge", d_yaw, d_pitch))

    def restart(self):
        self.calls.append("restart")

    def shutdown(self):
        self.calls.append("shutdown")
        if self.fail_shutdown:
            raise IOError("テスト用:CAN を閉じられない")

    def state(self):
        return MotorState(True, None, 0.0, 0.0, None, None)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.feeder, self.ser = make_feeder(self.clock)
        self.motors = StubMotors()
        self.launcher = Launcher(self.motors, self.feeder)

    def make_ready(self):
        self.launcher.set_speed(2)
        self.clock.t += config.ROLLER_SPINUP_SEC + 0.1
        self.feeder.handle_line(b"1\r\n", self.clock.t)

    def test_fire_when_ready(self):
        self.make_ready()
        self.assertEqual(self.launcher.status().fire_reasons, [])
        sent, _ = self.launcher.fire()
        self.assertTrue(sent)
        self.assertEqual(self.ser.written[-1], b"1\r\n")

    def test_stop_locks_until_speed_is_set_again(self):
        self.make_ready()
        self.launcher.stop()
        self.assertEqual(self.ser.written[-1], b"0,0\r\n")
        self.assertIn("hold_here", self.motors.calls)

        sent, reasons = self.launcher.fire()
        self.assertFalse(sent)
        self.assertEqual(reasons, [LOCKED_REASON])
        self.assertEqual(self.launcher.status().fire_reasons[0], LOCKED_REASON)

        self.launcher.set_speed(0)  # 強さ0では解除しない
        self.assertTrue(self.launcher.status().locked)

        self.make_ready()
        self.assertFalse(self.launcher.status().locked)
        self.assertTrue(self.launcher.fire()[0])

    def test_fire_refused_when_hold_loop_stopped(self):
        self.make_ready()
        self.motors.running = False
        sent, reasons = self.launcher.fire()
        self.assertFalse(sent)
        self.assertTrue(any("保持ループ" in r for r in reasons))

    def test_shutdown_always_stops_roller_and_closes_ports(self):
        motors = StubMotors(fail_shutdown=True)
        launcher = Launcher(motors, self.feeder)
        self.launcher.set_speed(3)
        with self.assertLogs("launcher", level="ERROR"):
            launcher.shutdown()
        self.assertEqual(self.ser.written[-1], b"0,0\r\n")
        self.assertIn("shutdown", motors.calls)
        self.assertTrue(self.ser.closed)  # CAN の終了でエラーが起きてもシリアルは閉じる
        launcher.shutdown()  # 2回目は何もしない
        self.assertEqual(motors.calls.count("shutdown"), 1)

    def test_shutdown_when_arduino_was_never_opened(self):
        feeder = Feeder(lambda: None)
        launcher = Launcher(StubMotors(), feeder)
        launcher.shutdown()  # 例外を出さない

    def test_set_speed_error_is_reported_not_raised(self):
        self.ser.write = None  # 書き込みで例外が出る状態
        with self.assertLogs("launcher", level="ERROR"):
            self.launcher.set_speed(2)


class HandleKeyTest(unittest.TestCase):
    class Recorder:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return lambda *args: self.calls.append((name,) + args)

    def press(self, ch):
        import main
        recorder = self.Recorder()
        result = main.handle_key(recorder, ord(ch))
        return result, recorder.calls

    def test_key_map(self):
        step = config.KEY_STEP_DEG
        expected = {
            "1": [("fire",)],
            "2": [("set_speed", 1)],
            "3": [("set_speed", 2)],
            "4": [("set_speed", 3)],
            "5": [("set_speed", 0)],
            "w": [("nudge", 0.0, -step)],   # ノートブックと同じく W がマイナス
            "s": [("nudge", 0.0, +step)],
            "a": [("nudge", -step, 0.0)],
            "d": [("nudge", +step, 0.0)],
            " ": [("stop",)],
            "r": [("restart_motors",)],
            "x": [],
        }
        for ch, calls in expected.items():
            for key in {ch, ch.upper()}:
                result, recorded = self.press(key)
                self.assertIsNone(result, key)
                self.assertEqual(recorded, calls, key)

    def test_quit(self):
        import main
        for ch in "qQ":
            self.assertEqual(self.press(ch), (main.QUIT, []))

    def test_no_key(self):
        import main
        self.assertIsNone(main.handle_key(self.Recorder(), -1))


if __name__ == "__main__":
    unittest.main()


class MainStartupFailureTest(unittest.TestCase):
    """発射機に接続できないとき:カメラの処理は始めず、分かりやすいメッセージで終わる。"""

    def test_connection_error_exits_cleanly_without_starting_camera(self):
        import main
        from unittest import mock as umock

        class FakeWorker:
            def __init__(self):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        class FailingLauncher:
            def __init__(self):
                self.shut_down = False

            def start(self):
                raise ConnectionError("Arduino のポート COM5 を開けませんでした(テスト)")

            def shutdown(self):
                self.shut_down = True

        worker, launcher = FakeWorker(), FailingLauncher()
        with umock.patch.object(main, "setup_logging", return_value=main.RecentMessages()), \
                umock.patch.object(main, "build_camera", return_value=worker), \
                umock.patch.object(main, "build_launcher", return_value=launcher), \
                self.assertLogs("launcher.main", level="ERROR") as logs:
            self.assertEqual(main.main([]), 1)
        self.assertFalse(worker.started)
        self.assertTrue(worker.stopped)
        self.assertTrue(launcher.shut_down)
        self.assertIn("COM5", logs.output[-1])
        self.assertNotIn("Traceback", "\n".join(logs.output))
