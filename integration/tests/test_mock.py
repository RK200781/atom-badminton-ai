# -*- coding: utf-8 -*-
"""偽物(mock.py)と組み合わせた、実機なしの通しテスト。待ち時間は短く差し替える。"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock as umock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import mock  # noqa: E402
from feeder import FIRE_CONFIRMED, Feeder  # noqa: E402
from launcher import Launcher  # noqa: E402
from motors import MotorController  # noqa: E402
from test_motors import wait_until  # noqa: E402

FAST = dict(ARDUINO_RESET_WAIT_SEC=0.0, ROLLER_SPINUP_SEC=0.2, FIRE_INTERVAL_SEC=0.5)


class FakeArduinoTest(unittest.TestCase):
    def test_fire_makes_detection_drop(self):
        ser = mock.FakeArduinoSerial()
        self.assertEqual(ser.readline(), b"1\r\n")
        ser.write(b"0,2\r\n")
        self.assertEqual(ser.speed, 2)
        ser.write(b"1\r\n")
        time.sleep(ser.SHOT_DELAY_SEC + 0.05)
        self.assertEqual(ser.readline(), b"0\r\n")

    def test_fire_ignored_when_speed_zero(self):
        ser = mock.FakeArduinoSerial()
        ser.write(b"1\r\n")
        time.sleep(ser.SHOT_DELAY_SEC + 0.05)
        self.assertEqual(ser.readline(), b"1\r\n")


class EndToEndMockTest(unittest.TestCase):
    def setUp(self):
        patcher = umock.patch.multiple(config, **FAST)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ser = mock.FakeArduinoSerial()
        self.port = mock.FakeCanPort()
        self.launcher = Launcher(MotorController(lambda: self.port), Feeder(lambda: self.ser))
        self.launcher.start()
        self.addCleanup(self.launcher.shutdown)

    def test_aim_moves_motors_and_fire_is_confirmed(self):
        self.assertTrue(wait_until(lambda: self.launcher.status().motors.current_yaw is not None))
        self.launcher.aim(0.0, 0.0)
        self.assertTrue(wait_until(lambda: self.launcher.status().shuttle is True))

        self.launcher.set_speed(1)
        sent, reasons = self.launcher.fire()
        self.assertFalse(sent)
        self.assertTrue(any("回転待ち" in r for r in reasons))

        self.assertTrue(wait_until(lambda: self.launcher.status().fire_reasons == []), self.launcher.status())
        sent, reasons = self.launcher.fire()
        self.assertTrue(sent, reasons)
        self.assertTrue(wait_until(lambda: self.launcher.status().feeder.fire_result == FIRE_CONFIRMED))
        self.assertIs(self.launcher.status().shuttle, False)

    def test_shutdown_sends_stop_and_closes_everything(self):
        self.launcher.set_speed(3)
        self.launcher.shutdown()
        self.assertEqual(self.ser.written[-1], b"0,0\r\n")
        self.assertFalse(self.ser.is_open)
        self.assertTrue(self.port.closed)
        self.assertFalse(self.launcher.motors.running)

    def test_can_failure_then_restart(self):
        self.launcher.shutdown()
        ports = [mock.FakeCanPort(fail_after_sec=0.1), mock.FakeCanPort()]
        launcher = Launcher(MotorController(lambda: ports.pop(0)), Feeder(mock.FakeArduinoSerial))
        launcher.start()
        try:
            self.assertTrue(wait_until(lambda: not launcher.status().motors.running))
            self.assertIn("通信エラー", launcher.status().motors.error)
            self.assertTrue(any("保持ループ" in r for r in launcher.status().fire_reasons))
            launcher.restart_motors()
            self.assertTrue(launcher.status().motors.running)
        finally:
            launcher.shutdown()


if __name__ == "__main__":
    unittest.main()
