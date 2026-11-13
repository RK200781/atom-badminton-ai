# -*- coding: utf-8 -*-
"""
実機なしで動かすための偽物(main.py --mock で使う)。

- FakeCanPort      : CubeMars モーター2台の代わり。指令された角度へ指令された速さで近づき、
                     約10msごとに状態フィードバック(0x29)のフレームを返す。
- FakeArduinoSerial: stattle.ino の代わり。約10msごとにシャトル検知 "1"/"0" の行を返す。
                     "1" を受けると少しして検知が 0 になり(発射)、しばらくすると 1 に戻る(補充)。
"""

import re
import struct
import threading
import time
from typing import Optional

import config

_NUMBER = re.compile(r"^\s*[-+]?\d+(\.\d*)?")


class FakeCanPort:
    def __init__(self, fail_after_sec: Optional[float] = None,
                 initial_yaw_deg: float = 10.0, initial_pitch_deg: float = -10.0):
        """fail_after_sec を指定すると、その秒数がたった後の送信で通信エラーを起こす(再開の確認用)。"""
        self._angles = {
            config.YAW_ID: config.YAW_ORIGIN_DEG + initial_yaw_deg,
            config.PITCH_ID: config.PITCH_ORIGIN_DEG + initial_pitch_deg,
        }
        self._targets = dict(self._angles)
        self._speeds_deg_s = {mid: 0.0 for mid in self._angles}
        self._created = time.monotonic()
        self._last_advance = self._created
        self._next_feedback = self._created
        self._feedback_order = [config.YAW_ID, config.PITCH_ID]
        self._lock = threading.Lock()
        self.fail_after_sec = fail_after_sec
        self.sent = []
        self.closed = False

    def _advance(self) -> None:
        now = time.monotonic()
        dt = now - self._last_advance
        self._last_advance = now
        for mid, angle in self._angles.items():
            diff = self._targets[mid] - angle
            step = self._speeds_deg_s[mid] * dt
            self._angles[mid] = self._targets[mid] if abs(diff) <= step else angle + (step if diff > 0 else -step)

    def send(self, arbitration_id: int, data: bytes) -> None:
        if self.closed:
            raise IOError("偽物の CAN: 閉じたポートに送信しました")
        if self.fail_after_sec is not None and time.monotonic() - self._created > self.fail_after_sec:
            raise IOError("偽物の CAN: 通信エラー(--mock-can-fail-after による)")
        motor_id = arbitration_id & 0xFF
        position_cmd, speed_cmd, _accel = struct.unpack(">ihh", data)
        with self._lock:
            self._advance()
            if motor_id in self._targets:
                self._targets[motor_id] = position_cmd / 10000.0
                # speed_cmd は ERPM(10 で割らずに送っている)。出力軸 rpm → deg/s
                rpm = speed_cmd / (config.GEAR_RATIO * config.POLE_PAIRS)
                self._speeds_deg_s[motor_id] = rpm * 360.0 / 60.0
            self.sent.append((arbitration_id, data))

    def recv(self, timeout: float):
        now = time.monotonic()
        if now < self._next_feedback:
            time.sleep(min(timeout, self._next_feedback - now))
            if time.monotonic() < self._next_feedback:
                return None
        with self._lock:
            self._advance()
            self._next_feedback = time.monotonic() + 0.01
            motor_id = self._feedback_order[0]
            self._feedback_order.reverse()
            raw = int(round(self._angles[motor_id] * 10))
        data = struct.pack(">hhhbb", raw, 0, 0, 25, 0)
        return (config.CAN_FEEDBACK_FUNCTION_ID << 8) | motor_id, data

    def shutdown(self) -> None:
        self.closed = True


class FakeArduinoSerial:
    SHOT_DELAY_SEC = 0.15   # "1" を受けてからシャトルが出ていくまで
    RELOAD_SEC = 3.0        # シャトルが出てから、次のシャトルが入るまで

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = ""
        self.written = []
        self.speed = 0
        self.present = True
        self._shot_at: Optional[float] = None
        self._empty_since: Optional[float] = None
        self.is_open = True

    @staticmethod
    def _to_number(text: str) -> float:
        """Arduino の String.toFloat() と同じく、先頭の数字だけを読む(読めなければ 0)。"""
        m = _NUMBER.match(text)
        return float(m.group(0)) if m else 0.0

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise IOError("偽物の Arduino: ポートが閉じています")
        with self._lock:
            self.written.append(data)
            self._buffer += data.decode("ascii")
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                fields = line.split(",")
                shot = self._to_number(fields[0])
                if len(fields) > 1:
                    self.speed = int(self._to_number(fields[1]))
                if shot == 1 and self.present and self.speed != 0 and self._shot_at is None:
                    self._shot_at = time.monotonic()
        return len(data)

    def readline(self) -> bytes:
        if not self.is_open:
            raise IOError("偽物の Arduino: ポートが閉じています")
        time.sleep(0.01)
        now = time.monotonic()
        with self._lock:
            if self._shot_at is not None and now - self._shot_at > self.SHOT_DELAY_SEC:
                self.present = False
                self._shot_at = None
                self._empty_since = now
            if not self.present and self._empty_since is not None and now - self._empty_since > self.RELOAD_SEC:
                self.present = True
                self._empty_since = None
            return b"1\r\n" if self.present else b"0\r\n"

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False
