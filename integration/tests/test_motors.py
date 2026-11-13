# -*- coding: utf-8 -*-
"""motors.py のテスト(範囲の丸め、CAN フレーム、エラーからの再開)。"""

import struct
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from motors import (  # noqa: E402
    PITCH, YAW, MotorController, clamp_angle, decode_feedback, encode_position_speed,
)


class RecordingPort:
    def __init__(self, fail_on_send=False):
        self.sent = []
        self.fail_on_send = fail_on_send
        self.closed = False

    def send(self, arbitration_id, data):
        if self.fail_on_send:
            raise IOError("テスト用の通信エラー")
        self.sent.append((arbitration_id, data))

    def recv(self, timeout):
        time.sleep(timeout)
        return None

    def shutdown(self):
        self.closed = True


def feedback_frame(motor_id, motor_angle_deg):
    raw = int(round(motor_angle_deg * 10))
    return (config.CAN_FEEDBACK_FUNCTION_ID << 8) | motor_id, struct.pack(">hhhbb", raw, 0, 0, 25, 0)


def wait_until(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class ClampAngleTest(unittest.TestCase):
    def test_inside_range(self):
        self.assertEqual(clamp_angle(YAW, 10.0), (10.0, False))
        self.assertEqual(clamp_angle(YAW, 60.0), (60.0, False))
        self.assertEqual(clamp_angle(PITCH, -50.0), (-50.0, False))

    def test_outside_range(self):
        self.assertEqual(clamp_angle(YAW, 70.0), (60.0, True))
        self.assertEqual(clamp_angle(YAW, -61.0), (-60.0, True))
        self.assertEqual(clamp_angle(PITCH, 31.0), (30.0, True))
        self.assertEqual(clamp_angle(PITCH, -52.0), (-50.0, True))

    def test_names_and_ids(self):
        self.assertEqual((YAW.name, YAW.motor_id, YAW.origin_deg), ("yaw", 105, 145.0))
        self.assertEqual((PITCH.name, PITCH.motor_id, PITCH.origin_deg), ("pitch", 104, 165.0))


class FrameTest(unittest.TestCase):
    def test_matches_notebook_log(self):
        # ノートブックの実行結果に残っている送信ログと同じバイト列になること
        cases = [
            (105, 145.0, 0x669, "00 16 20 10 00 46 0a f0"),
            (104, 165.0, 0x668, "00 19 2d 50 00 46 0a f0"),
            (105, 161.0, 0x669, "00 18 91 10 00 46 0a f0"),
        ]
        for motor_id, angle, can_id, data_hex in cases:
            arbitration_id, data = encode_position_speed(motor_id, angle, 0.5, 20.0)
            self.assertEqual(arbitration_id, can_id)
            self.assertEqual(data.hex(" "), data_hex)

    def test_speed_is_limited(self):
        _, fast = encode_position_speed(105, 145.0, 999.0, 20.0)
        _, limit = encode_position_speed(105, 145.0, config.MAX_OUTPUT_RPM, 20.0)
        self.assertEqual(fast, limit)

    def test_decode_feedback(self):
        self.assertEqual(decode_feedback(*feedback_frame(104, 160.5)), (104, 160.5))
        motor_id, angle = decode_feedback(*feedback_frame(105, -3.2))
        self.assertEqual(motor_id, 105)
        self.assertAlmostEqual(angle, -3.2)

    def test_decode_ignores_other_frames(self):
        arbitration_id, data = encode_position_speed(105, 145.0, 0.5, 20.0)
        self.assertIsNone(decode_feedback(arbitration_id, data))
        self.assertIsNone(decode_feedback((0x29 << 8) | 105, b"\x00\x01"))


class MotorControllerTest(unittest.TestCase):
    def test_out_of_range_target_is_clamped_before_sending(self):
        port = RecordingPort()
        motors = MotorController(lambda: port)
        motors._port = port
        motors.set_target(70.0, -60.0)
        with self.assertLogs("launcher.motors", level="WARNING") as logs:
            motors._send_targets()
        self.assertEqual(len(logs.records), 2)

        sent = {arbitration_id & 0xFF: struct.unpack(">ihh", data)[0] / 10000.0 for arbitration_id, data in port.sent}
        self.assertEqual(sent[105], 145.0 + 60.0)
        self.assertEqual(sent[104], 165.0 - 50.0)
        # 丸めた値が目標角に書き戻され、逆向きのキーですぐ戻れる
        state = motors.state()
        self.assertEqual((state.target_yaw, state.target_pitch), (60.0, -50.0))

    def test_nudge(self):
        motors = MotorController(RecordingPort)
        motors.nudge(2.0, -2.0)
        motors.nudge(2.0, -2.0)
        state = motors.state()
        self.assertEqual((state.target_yaw, state.target_pitch), (4.0, -4.0))

    def test_current_angle_from_feedback_and_stale(self):
        now = [10.0]
        motors = MotorController(RecordingPort, clock=lambda: now[0])
        motors._handle_frame(*feedback_frame(105, 155.0))
        motors._handle_frame(*feedback_frame(104, 160.0))
        state = motors.state()
        self.assertAlmostEqual(state.current_yaw, 10.0)
        self.assertAlmostEqual(state.current_pitch, -5.0)
        now[0] += config.ANGLE_STALE_SEC + 0.1
        self.assertIsNone(motors.state().current_yaw)

    def test_hold_here(self):
        motors = MotorController(RecordingPort, clock=lambda: 1.0)
        motors.set_target(30.0, 20.0)
        motors._handle_frame(*feedback_frame(105, 150.0))
        motors._handle_frame(*feedback_frame(104, 160.0))
        self.assertTrue(motors.hold_here())
        state = motors.state()
        self.assertAlmostEqual(state.target_yaw, 5.0)
        self.assertAlmostEqual(state.target_pitch, -5.0)

    def test_hold_here_without_feedback_keeps_target(self):
        motors = MotorController(RecordingPort)
        motors.set_target(30.0, 20.0)
        self.assertFalse(motors.hold_here())
        state = motors.state()
        self.assertEqual((state.target_yaw, state.target_pitch), (30.0, 20.0))

    def test_error_stops_loop_and_restart_works(self):
        ports = [RecordingPort(fail_on_send=True), RecordingPort()]
        motors = MotorController(lambda: ports.pop(0))
        motors.start()
        try:
            self.assertTrue(wait_until(lambda: not motors.running))
            self.assertIn("テスト用の通信エラー", motors.error)
            self.assertFalse(motors.state().running)

            # ノートブックでは「動作中」の印が残って再開できなかった。統合版では再開できる。
            motors.restart()
            self.assertTrue(motors.running)
            self.assertIsNone(motors.error)
            good_port = motors._port
            self.assertTrue(wait_until(lambda: len(good_port.sent) >= 4))
        finally:
            motors.shutdown()
        self.assertFalse(motors.running)
        self.assertTrue(good_port.closed)

    def test_loop_sends_every_interval(self):
        port = RecordingPort()
        motors = MotorController(lambda: port)
        motors.start()
        time.sleep(0.32)
        motors.shutdown()
        # 50ms 周期で yaw・pitch の2フレームずつ → 約 0.32 秒で 6〜7 周
        self.assertGreaterEqual(len(port.sent), 10)
        self.assertLessEqual(len(port.sent), 16)
        self.assertTrue(port.closed)
        self.assertFalse(motors.running)


if __name__ == "__main__":
    unittest.main()
