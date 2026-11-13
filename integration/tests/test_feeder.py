# -*- coding: utf-8 -*-
"""feeder.py のテスト(受信行の読み取り、fire() の条件判定)。

    python -m unittest discover -s integration/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from feeder import (  # noqa: E402
    FIRE_CONFIRMED, FIRE_PENDING, FIRE_UNCONFIRMED,
    Feeder, FeederSnapshot, fire_block_reasons, parse_detection_line,
)


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class RecordingSerial:
    def __init__(self):
        self.written = []
        self.closed = False

    def write(self, data):
        self.written.append(data)
        return len(data)

    def readline(self):
        return b""

    def close(self):
        self.closed = True


def make_feeder(clock):
    ser = RecordingSerial()
    feeder = Feeder(lambda: ser, clock=clock, sleep=lambda _sec: None)
    feeder.open(start_reader=False)
    return feeder, ser


def ready_snapshot(now, **overrides):
    """発射できる状態のスナップショット。"""
    values = dict(connected=True, error=None, speed=2, detected=True, detected_at=now - 0.05,
                  roller_ready_at=now - 1.0, last_fire_at=None, fire_result=None)
    values.update(overrides)
    return FeederSnapshot(**values)


class ParseDetectionLineTest(unittest.TestCase):
    def test_one_and_zero(self):
        self.assertIs(parse_detection_line(b"1\r\n"), True)
        self.assertIs(parse_detection_line(b"0\r\n"), False)
        self.assertIs(parse_detection_line("1\n"), True)
        self.assertIs(parse_detection_line(b" 0 \r\n"), False)

    def test_other_lines_are_ignored(self):
        for line in (b"", b"\r\n", b"10\r\n", b"TIMEOUT\r\n", b"\xff\xfe1\r\n", b"1,0\r\n"):
            self.assertIsNone(parse_detection_line(line), line)


class FireBlockReasonsTest(unittest.TestCase):
    NOW = 50.0

    def reasons(self, hold_running=True, **overrides):
        return fire_block_reasons(ready_snapshot(self.NOW, **overrides), self.NOW, hold_running)

    def test_all_conditions_met(self):
        self.assertEqual(self.reasons(), [])

    def test_speed_zero(self):
        reasons = self.reasons(speed=0)
        self.assertEqual(len(reasons), 1)
        self.assertIn("強さが0", reasons[0])

    def test_no_shuttle(self):
        self.assertIn("シャトルを検知していません", self.reasons(detected=False))

    def test_stale_detection(self):
        reasons = self.reasons(detected_at=self.NOW - config.SHUTTLE_FRESH_SEC - 0.01)
        self.assertEqual(len(reasons), 1)
        self.assertIn("届いていません", reasons[0])

    def test_never_received(self):
        reasons = self.reasons(detected=None, detected_at=None)
        self.assertIn("届いていません", reasons[0])

    def test_roller_spinning_up(self):
        reasons = self.reasons(roller_ready_at=self.NOW + 0.5)
        self.assertEqual(len(reasons), 1)
        self.assertIn("回転待ち", reasons[0])

    def test_fire_interval(self):
        reasons = self.reasons(last_fire_at=self.NOW - config.FIRE_INTERVAL_SEC + 0.1)
        self.assertEqual(len(reasons), 1)
        self.assertIn("前の発射", reasons[0])
        self.assertEqual(self.reasons(last_fire_at=self.NOW - config.FIRE_INTERVAL_SEC), [])

    def test_hold_loop_stopped(self):
        reasons = self.reasons(hold_running=False)
        self.assertEqual(len(reasons), 1)
        self.assertIn("保持ループ", reasons[0])

    def test_not_connected(self):
        self.assertIn("Arduino と通信できていません", self.reasons(connected=False))
        self.assertIn("Arduino と通信できていません", self.reasons(error="IOError"))

    def test_multiple_reasons(self):
        reasons = self.reasons(hold_running=False, speed=0, detected=False)
        self.assertEqual(len(reasons), 3)


class FeederTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.feeder, self.ser = make_feeder(self.clock)

    def make_ready(self):
        """強さ2にして、回転待ちが終わり、シャトルを検知している状態にする。"""
        self.feeder.set_speed(2)
        self.clock.t += config.ROLLER_SPINUP_SEC + 0.1
        self.feeder.handle_line(b"1\r\n", self.clock.t)

    def test_set_speed_sends_command(self):
        for speed in (0, 1, 2, 3):
            self.feeder.set_speed(speed)
            self.assertEqual(self.ser.written[-1], f"0,{speed}\r\n".encode())
            self.assertEqual(self.feeder.snapshot().speed, speed)

    def test_set_speed_rejects_invalid(self):
        for bad in (4, -1, 100, "2", 1.5, True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.feeder.set_speed(bad)
        self.assertEqual(self.ser.written, [])

    def test_fire_refused_when_speed_zero(self):
        # Arduino は起動直後の1回だけ強さ0でも発射してしまうので、Python 側で必ず止める
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        sent, reasons = self.feeder.fire(hold_running=True)
        self.assertFalse(sent)
        self.assertTrue(any("強さが0" in r for r in reasons))
        self.assertNotIn(b"1\r\n", self.ser.written)

    def test_fire_refused_during_spinup_then_allowed(self):
        self.feeder.set_speed(2)
        self.clock.t += config.ROLLER_SPINUP_SEC - 0.1
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        sent, reasons = self.feeder.fire(hold_running=True)
        self.assertFalse(sent)
        self.assertTrue(any("回転待ち" in r for r in reasons))

        self.clock.t += 0.2
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        sent, reasons = self.feeder.fire(hold_running=True)
        self.assertTrue(sent, reasons)
        self.assertEqual(self.ser.written[-1], b"1\r\n")

    def test_same_speed_again_does_not_restart_spinup(self):
        self.make_ready()
        self.feeder.set_speed(2)
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        self.assertEqual(self.feeder.block_reasons(hold_running=True), [])

    def test_fire_refused_when_hold_loop_stopped(self):
        self.make_ready()
        sent, reasons = self.feeder.fire(hold_running=False)
        self.assertFalse(sent)
        self.assertNotIn(b"1\r\n", self.ser.written)

    def test_fire_interval(self):
        self.make_ready()
        self.assertTrue(self.feeder.fire(hold_running=True)[0])
        self.clock.t += 0.5
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        sent, reasons = self.feeder.fire(hold_running=True)
        self.assertFalse(sent)
        self.assertTrue(any("前の発射" in r for r in reasons))
        self.assertEqual(self.ser.written.count(b"1\r\n"), 1)

    def test_fire_confirmed_when_detection_drops(self):
        self.make_ready()
        self.feeder.fire(hold_running=True)
        self.assertEqual(self.feeder.snapshot().fire_result, FIRE_PENDING)
        self.clock.t += 0.3
        self.feeder.handle_line(b"0\r\n", self.clock.t)
        self.assertEqual(self.feeder.snapshot().fire_result, FIRE_CONFIRMED)

    def test_fire_unconfirmed_when_detection_stays(self):
        self.make_ready()
        self.feeder.fire(hold_running=True)
        self.clock.t += config.FIRE_CONFIRM_SEC + 0.1
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        self.assertEqual(self.feeder.snapshot().fire_result, FIRE_UNCONFIRMED)
        # その後で 0 になっても「確認できた」には変えない
        self.feeder.handle_line(b"0\r\n", self.clock.t + 0.1)
        self.assertEqual(self.feeder.snapshot().fire_result, FIRE_UNCONFIRMED)

    def test_fire_unconfirmed_without_any_line(self):
        self.make_ready()
        self.feeder.fire(hold_running=True)
        self.feeder.check_fire_timeout(self.clock.t + config.FIRE_CONFIRM_SEC + 0.1)
        self.assertEqual(self.feeder.snapshot().fire_result, FIRE_UNCONFIRMED)

    def test_spinup_restarts_after_long_absence(self):
        # Arduino は検知なしが 10 秒続くとローラーを止め、シャトルが入ると回し直す
        self.make_ready()
        self.feeder.handle_line(b"0\r\n", self.clock.t)
        self.clock.t += config.ARDUINO_NO_SHUTTLE_TIMEOUT_SEC + 1.0
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        reasons = self.feeder.block_reasons(hold_running=True)
        self.assertTrue(any("回転待ち" in r for r in reasons), reasons)

    def test_no_spinup_restart_after_short_absence(self):
        self.make_ready()
        self.feeder.handle_line(b"0\r\n", self.clock.t)
        self.clock.t += 3.0
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        self.assertEqual(self.feeder.block_reasons(hold_running=True), [])

    def test_shuttle_detected_becomes_unknown_when_stale(self):
        self.feeder.handle_line(b"1\r\n", self.clock.t)
        self.assertIs(self.feeder.shuttle_detected(), True)
        self.clock.t += config.SHUTTLE_FRESH_SEC + 0.01
        self.assertIsNone(self.feeder.shuttle_detected())

    def test_close_sends_stop_and_closes(self):
        self.feeder.set_speed(3)
        self.feeder.close()
        self.assertEqual(self.ser.written[-1], b"0,0\r\n")
        self.assertTrue(self.ser.closed)
        self.feeder.close()  # 2回目でもエラーにならない


if __name__ == "__main__":
    unittest.main()
