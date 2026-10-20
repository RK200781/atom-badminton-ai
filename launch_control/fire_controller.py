# -*- coding: utf-8 -*-
"""
人の立ち位置から「いつ・どこへ撃つか」を決める状態機械。

毎フレーム update(時刻, 足元座標 or None) を呼ぶ。足元座標はカメラ座標系の (x, z) [cm]
(realtime_pose_detect.py の estimate_floor_position() の出力)。そのフレームで人が
見えていなければ None を渡す。

状態遷移:
  IDLE ─(人が見えた)→ TRACKING ─(立ち位置が落ち着いた・狙える)→ AIMING
  AIMING ─(目標角に到達・人が見えている)→ FIRING ─(開閉部を閉じる)→ COOLDOWN
  COOLDOWN ─(待ち時間経過)→ TRACKING(見失っていれば IDLE)
  AIMING ─(人が動いた・到達タイムアウト)→ TRACKING
  TRACKING / AIMING ─(見失った)→ IDLE

安全のため、静止判定と発射は「そのフレームで人が見えている時」にしか行わない。
過去1秒の履歴だけで判定すると、人が消えた直後に誰もいない所へ撃ってしまうため。
一方、数フレームの見失い(LOST_TIMEOUT_SEC 未満)では状態を保つので、すぐ戻れば普通に撃てる。
"""

import csv
import math
import statistics
import time
from collections import deque
from pathlib import Path

from . import aiming, config


IDLE = "IDLE"
TRACKING = "TRACKING"
AIMING = "AIMING"
FIRING = "FIRING"
COOLDOWN = "COOLDOWN"

AIM_STATUS_MESSAGES = {
    aiming.STATUS_TOO_FAR: "Too far - please come closer",
    aiming.STATUS_TOO_CLOSE: "Too close - please step back",
    aiming.STATUS_OUT_OF_PAN: "Out of range - please move toward the center",
}

SHOT_FIELDNAMES = [
    "shot_id", "timestamp",
    "person_x_cm", "person_z_cm",   # カメラ座標系での立ち位置(静止判定の中央値)
    "target_x_cm", "target_z_cm",   # 発射機座標系での狙い位置
    "distance_cm", "pan_deg", "tilt_deg",
    "landing_x_cm", "landing_z_cm",  # 着地点(後で軌道CSVなどから埋める)
]


class ShotLogger:
    """発射ごとの記録を CSV に追記する。shot_id は既存ファイルの続きから振る。"""

    def __init__(self, path: Path = config.SHOTS_CSV):
        self.path = Path(path)
        self._next_id = self._read_last_id() + 1

    def _read_last_id(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            ids = [int(row["shot_id"]) for row in csv.DictReader(f) if row.get("shot_id")]
        return max(ids, default=0)

    def log(self, timestamp, person_xz, solution: aiming.AimSolution) -> int:
        shot_id = self._next_id
        self._next_id += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(SHOT_FIELDNAMES)
            writer.writerow([
                shot_id, f"{timestamp:.3f}",
                f"{person_xz[0]:.1f}", f"{person_xz[1]:.1f}",
                f"{solution.target_x_cm:.1f}", f"{solution.target_z_cm:.1f}",
                f"{solution.distance_cm:.1f}", f"{solution.pan_deg:.2f}", f"{solution.tilt_deg:.2f}",
                "", "",
            ])
        return shot_id


class FireController:
    def __init__(self, launcher, shot_logger: ShotLogger = None, wall_clock=time.time, log=print):
        """
        launcher: launcher.Launcher の実装
        shot_logger: None なら記録しない
        wall_clock: shots.csv に書く時刻(軌道CSVと突き合わせられるよう実時刻を使う)
        """
        self.launcher = launcher
        self.shot_logger = shot_logger
        self._wall_clock = wall_clock
        self._log = log or (lambda *_: None)

        self.state = IDLE
        self.message = "No player"
        self.solution = None      # 直近に計算した AimSolution(表示用)
        self.shot_count = 0
        self.last_shot_id = None

        self._history = deque()   # 見えていたフレームの (t, x, z)
        self._last_seen_t = None
        self._tracking_since = None
        self._aim_point = None    # 狙いを決めた時の立ち位置(カメラ座標系)
        self._aim_started_t = None
        self._fire_started_t = None
        self._cooldown_started_t = None

    # ---- 内部ヘルパ ------------------------------------------------------

    def _set_state(self, state, t):
        if state != self.state:
            self._log(f"[FireController] t={t:.2f} {self.state} -> {state}")
            self.state = state
        if state == TRACKING:
            self._tracking_since = t

    def _median_xz(self, points):
        return (
            statistics.median(p[1] for p in points),
            statistics.median(p[2] for p in points),
        )

    def _still_position(self, t):
        """直近 STILL_WINDOW_SEC の立ち位置が落ち着いていれば中央値を返す。"""
        if t - self._tracking_since < config.STILL_WINDOW_SEC:
            return None
        if len(self._history) < config.STILL_MIN_SAMPLES:
            return None
        mx, mz = self._median_xz(self._history)
        spread = max(math.hypot(x - mx, z - mz) for _, x, z in self._history)
        if spread > config.STILL_RADIUS_CM:
            return None
        return mx, mz

    def _arrived(self):
        pan, tilt = self.launcher.get_angles()
        return (
            abs(pan - self.solution.pan_deg) <= config.ARRIVAL_TOLERANCE_DEG
            and abs(tilt - self.solution.tilt_deg) <= config.ARRIVAL_TOLERANCE_DEG
        )

    # ---- メイン ----------------------------------------------------------

    def update(self, t: float, person_xz):
        """1フレーム分の判断を行う。t は単調増加の時刻 [秒]、person_xz は (x, z) [cm] か None。"""
        visible = person_xz is not None
        if visible:
            self._history.append((t, person_xz[0], person_xz[1]))
            self._last_seen_t = t
        while self._history and self._history[0][0] < t - config.STILL_WINDOW_SEC:
            self._history.popleft()
        lost = self._last_seen_t is None or t - self._last_seen_t > config.LOST_TIMEOUT_SEC

        # 発射中・待機中は人の有無にかかわらず時間で進める
        if self.state == FIRING:
            if t - self._fire_started_t >= config.GATE_OPEN_SEC:
                self.launcher.close_gate()
                self._cooldown_started_t = t
                self.message = "Cooldown"
                self._set_state(COOLDOWN, t)
            return self.state

        if self.state == COOLDOWN:
            if t - self._cooldown_started_t < config.COOLDOWN_SEC:
                return self.state
            if lost:
                self._go_idle(t)
            else:
                self.message = "Hold still..."
                self._set_state(TRACKING, t)
            return self.state

        if lost:
            if self.state != IDLE:
                self._go_idle(t)
            return self.state

        if self.state == IDLE:
            if visible:
                self.message = "Hold still..."
                self._set_state(TRACKING, t)
            return self.state

        # ここから先の判断はすべて「今のフレームで見えている」ことが前提
        if not visible:
            return self.state

        if self.state == TRACKING:
            self._update_tracking(t)
        elif self.state == AIMING:
            self._update_aiming(t, person_xz)
        return self.state

    def _go_idle(self, t):
        self._history.clear()
        self.solution = None
        self.message = "No player"
        self._set_state(IDLE, t)

    def _update_tracking(self, t):
        still = self._still_position(t)
        if still is None:
            self.message = "Hold still..."
            return

        self.solution = aiming.solve_aim(*still)
        if not self.solution.ok:
            self.message = AIM_STATUS_MESSAGES[self.solution.status]
            return

        self.launcher.move_to(self.solution.pan_deg, self.solution.tilt_deg)
        self._aim_point = still
        self._aim_started_t = t
        self.message = "Aiming..."
        self._set_state(AIMING, t)

    def _update_aiming(self, t, person_xz):
        recent = list(self._history)[-config.REAIM_MEDIAN_SAMPLES:]
        cx, cz = self._median_xz(recent)
        if math.hypot(cx - self._aim_point[0], cz - self._aim_point[1]) > config.REAIM_DISTANCE_CM:
            self.message = "Player moved - re-aiming"
            self._set_state(TRACKING, t)
            return

        if not self._arrived():
            if t - self._aim_started_t > config.ARRIVAL_TIMEOUT_SEC:
                self.launcher.stop()
                self.message = "Aim timeout - retrying"
                self._set_state(TRACKING, t)
            return

        self._fire(t)

    def _fire(self, t):
        self.launcher.open_gate()
        self._fire_started_t = t
        self.shot_count += 1
        if self.shot_logger is not None:
            self.last_shot_id = self.shot_logger.log(self._wall_clock(), self._aim_point, self.solution)
        self.message = "FIRE!"
        self._log(
            f"[FireController] 発射 #{self.shot_count}: pan={self.solution.pan_deg:.1f} "
            f"tilt={self.solution.tilt_deg:.1f} distance={self.solution.distance_cm:.0f}cm"
        )
        self._set_state(FIRING, t)
