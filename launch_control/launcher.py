# -*- coding: utf-8 -*-
"""
発射機ハードウェアの抽象化。

映像ループを止めないため、どのメソッドも「指令を出してすぐ返る」設計にする。
目標角に到達したかどうかは、呼び出し側が毎フレーム get_angles() で確認する。
"""

import time
from abc import ABC, abstractmethod

from . import config


class Launcher(ABC):
    @abstractmethod
    def move_to(self, pan_deg: float, tilt_deg: float) -> None:
        """目標角を指令する(到達を待たずに返る)。"""

    @abstractmethod
    def get_angles(self):
        """現在の (pan_deg, tilt_deg) を返す。"""

    @abstractmethod
    def open_gate(self) -> None:
        """シャトルがたまっている所の開閉部を開ける(=発射)。"""

    @abstractmethod
    def close_gate(self) -> None:
        """開閉部を閉じる。"""

    @abstractmethod
    def stop(self) -> None:
        """モーターをその場で止める。"""


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class MockLauncher(Launcher):
    """実機の代わりに、一定の角速度で目標角へ近づく様子を時間で模擬する。

    clock を差し替えれば、実時間を待たずにシミュレーションできる。
    """

    def __init__(
        self,
        pan_speed_deg_s: float = 90.0,
        tilt_speed_deg_s: float = 60.0,
        initial_pan_deg: float = 0.0,
        initial_tilt_deg: float = config.TILT_MIN_DEG,
        clock=time.monotonic,
        log=print,
    ):
        self.pan_speed = pan_speed_deg_s
        self.tilt_speed = tilt_speed_deg_s
        self._clock = clock
        self._log = log or (lambda *_: None)
        self._pan = self._target_pan = initial_pan_deg
        self._tilt = self._target_tilt = initial_tilt_deg
        self._last_t = clock()
        self.gate_open = False

    def _advance(self):
        now = self._clock()
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        self._pan = self._step(self._pan, self._target_pan, self.pan_speed * dt)
        self._tilt = self._step(self._tilt, self._target_tilt, self.tilt_speed * dt)

    @staticmethod
    def _step(current, target, max_delta):
        diff = target - current
        if abs(diff) <= max_delta:
            return target
        return current + (max_delta if diff > 0 else -max_delta)

    def move_to(self, pan_deg, tilt_deg):
        self._advance()
        pan = _clamp(pan_deg, config.PAN_MIN_DEG, config.PAN_MAX_DEG)
        tilt = _clamp(tilt_deg, config.TILT_MIN_DEG, config.TILT_MAX_DEG)
        if (pan, tilt) != (pan_deg, tilt_deg):
            self._log(f"[Mock] 可動範囲外の指令を制限しました: ({pan_deg:.1f}, {tilt_deg:.1f}) -> ({pan:.1f}, {tilt:.1f})")
        self._target_pan, self._target_tilt = pan, tilt
        self._log(f"[Mock] move_to pan={pan:.1f} tilt={tilt:.1f} (現在 pan={self._pan:.1f} tilt={self._tilt:.1f})")

    def get_angles(self):
        self._advance()
        return self._pan, self._tilt

    def open_gate(self):
        self.gate_open = True
        self._log("[Mock] 開閉部: 開")

    def close_gate(self):
        self.gate_open = False
        self._log("[Mock] 開閉部: 閉")

    def stop(self):
        self._advance()
        self._target_pan, self._target_tilt = self._pan, self._tilt
        self._log(f"[Mock] 停止 pan={self._pan:.1f} tilt={self._tilt:.1f}")


class AK45Launcher(Launcher):
    """CubeMars AK45-10 ×2(パン・チルト)+ 開閉部の実機ドライバ(未実装)。

    TODO(実機確認後に実装):
    - 通信: CANインターフェース(USB-CANアダプタの種類、python-can の bustype/channel/bitrate)
    - CAN ID: パン用・チルト用モーターそれぞれのID
    - 制御モード: サーボモードの位置制御 / MITモードのどちらを使うか、速度・加速度の上限
    - 角度の符号とゼロ点: モーターの正方向と、パン右が+・チルト上が+ との対応、
      電源投入時のゼロ点合わせ(機械的な原点の決め方)、減速比(AK45-10 は 10:1)
    - 角度の読み取り: フィードバックフレームから現在角を取り出す方法と更新周期
    - 開閉部: 何で動かすか(サーボ / ソレノイド等)、制御方法、開・閉それぞれの指令
    - stop(): 緊急停止時にトルクを抜くのか、その場で保持するのか
    いずれもブロックせずにすぐ返すこと(映像ループを止めないため)。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AK45Launcher は未実装です。MockLauncher を使ってください。")

    def move_to(self, pan_deg, tilt_deg):
        raise NotImplementedError

    def get_angles(self):
        raise NotImplementedError

    def open_gate(self):
        raise NotImplementedError

    def close_gate(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
