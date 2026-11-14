# -*- coding: utf-8 -*-
"""
人の立ち位置から、発射機の yaw・pitch を決める(自動版)。学習もここで行う。

座標(カメラが基準。vision.py と同じ):右 +X、奥 +Z [cm]、床の上の位置。
発射口はカメラの真後ろ LAUNCHER_MUZZLE_BEHIND_CAMERA_CM、左右のずれ無し、正面は同じ向き。

    yaw   = atan2(X, Z + 後ろのずれ) + 学習した yaw のずれ          (yaw はプラスが右)
    pitch = −(飛距離が「発射口から人まで」になる上向きの角度)        (pitch はマイナスが上)

学習(shots.csv から毎回計算し直す):
  - 発射の速さ:強さごとに、入力された「実際の飛距離」に一番合う速さを探す(ballistics.fit_speed)。
    記録が無い強さは、初期値に「記録がある強さで、初期値から何倍になったか」の中央値を掛ける。
  - yaw のずれ:「発射機が向いていた方向」と「実際に落ちた方向」の差の平均(新しい LEARN_RECENT 件)。
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ballistics
import config
from shots import Shot

log = logging.getLogger("launcher.aiming")

LEARN_RECENT = 30   # 学習に使う、新しい方からの件数(発射機の状態が変わっていくことがあるため)


def person_from_muzzle(X_cm: float, Z_cm: float):
    """人の立ち位置 → (発射口から見た方向 yaw [deg], 発射口からの水平距離 [cm])。"""
    forward = Z_cm + config.LAUNCHER_MUZZLE_BEHIND_CAMERA_CM
    return math.degrees(math.atan2(X_cm, forward)), math.hypot(X_cm, forward)


@dataclass
class AimSolution:
    yaw_deg: float
    pitch_deg: Optional[float]     # 強さ 0 のときは None(pitch は決められない)
    person_yaw_deg: float
    person_distance_cm: float
    target_distance_cm: float      # 狙う飛距離(人の立ち位置+AIM_LANDING_OFFSET_CM)
    predicted_distance_cm: Optional[float]
    launch_speed_ms: Optional[float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelState:
    speeds_ms: Dict[int, float]     # 強さ → 発射の速さ
    speed_counts: Dict[int, int]    # 強さ → 学習に使った件数
    yaw_bias_deg: float
    yaw_count: int


class AimModel:
    def __init__(self):
        self.state = ModelState(dict(config.INITIAL_LAUNCH_SPEED_MS), {s: 0 for s in config.INITIAL_LAUNCH_SPEED_MS},
                                0.0, 0)

    # ---- 学習 ------------------------------------------------------------------

    def learn(self, shots: List[Shot]) -> ModelState:
        speeds = dict(config.INITIAL_LAUNCH_SPEED_MS)
        counts = {s: 0 for s in speeds}
        ratios = []
        for speed in speeds:
            data = [(s.elevation_deg, s.actual_distance_cm()) for s in shots
                    if s.speed == speed and s.actual_distance_cm() is not None and s.actual_distance_cm() > 0]
            data = data[-LEARN_RECENT:]
            if not data:
                continue
            v = ballistics.fit_speed([d[0] for d in data], [d[1] for d in data])
            lo, hi = config.LAUNCH_SPEED_FIT_RANGE_MS
            if v <= lo + 0.05 or v >= hi - 0.05:
                log.warning("強さ %d の発射の速さが、探す範囲の端(%.1f m/s)になりました。入力した結果を確かめてください。",
                            speed, v)
            speeds[speed], counts[speed] = v, len(data)
            ratios.append(v / config.INITIAL_LAUNCH_SPEED_MS[speed])
        if ratios:
            r = statistics.median(ratios)
            for speed in speeds:
                if counts[speed] == 0:
                    speeds[speed] = config.INITIAL_LAUNCH_SPEED_MS[speed] * r

        biases = []
        for s in shots:
            direction = s.actual_direction_deg()
            if direction is not None:
                biases.append(s.yaw_deg - direction)
        biases = biases[-LEARN_RECENT:]
        yaw_bias = statistics.fmean(biases) if biases else 0.0
        self.state = ModelState(speeds, counts, yaw_bias, len(biases))
        return self.state

    def describe(self) -> str:
        st = self.state
        parts = [f"強さ{s}: {v:.1f}m/s({st.speed_counts[s]}件)" for s, v in sorted(st.speeds_ms.items())]
        parts.append(f"yaw のずれ {st.yaw_bias_deg:+.1f}°({st.yaw_count}件)")
        return "  ".join(parts)

    # ---- 狙い ------------------------------------------------------------------

    def solve(self, X_cm: float, Z_cm: float, speed: int) -> AimSolution:
        person_yaw, person_dist = person_from_muzzle(X_cm, Z_cm)
        warnings = []
        yaw = person_yaw + self.state.yaw_bias_deg
        if not (config.YAW_MIN_DEG <= yaw <= config.YAW_MAX_DEG):
            warnings.append(f"yaw {yaw:+.0f}° は可動範囲外(端に合わせます)")
            yaw = min(max(yaw, config.YAW_MIN_DEG), config.YAW_MAX_DEG)
        target = person_dist + config.AIM_LANDING_OFFSET_CM
        v0 = self.state.speeds_ms.get(speed)
        if v0 is None:
            return AimSolution(yaw, None, person_yaw, person_dist, target, None, None,
                               warnings + ["強さが 0 なので pitch は決められません(yaw だけ合わせます)"])
        elevation, result = ballistics.elevation_for_distance(target, v0)
        if result == ballistics.SOLVE_TOO_FAR:
            warnings.append(f"強さ {speed} では届きません(最大 約{ballistics.max_range_cm(v0) / 100:.1f}m)。強くしてください")
        elif result == ballistics.SOLVE_TOO_NEAR:
            warnings.append(f"強さ {speed} では近すぎます。弱くしてください")
        pitch = -elevation
        pitch = min(max(pitch, config.PITCH_MIN_DEG), config.PITCH_MAX_DEG)
        predicted = ballistics.table_range_cm(-pitch, v0)
        return AimSolution(yaw, pitch, person_yaw, person_dist, target, predicted, v0, warnings)

    def predicted_distance(self, pitch_deg: float, speed: int) -> Optional[float]:
        v0 = self.state.speeds_ms.get(speed)
        if v0 is None:
            return None
        d = ballistics.table_range_cm(-pitch_deg, v0)
        if math.isnan(d):
            d = ballistics.range_cm(-pitch_deg, v0)
        return None if math.isnan(d) else d
