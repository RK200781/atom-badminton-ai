# -*- coding: utf-8 -*-
"""
シャトルの飛び方の計算(自動版で使う)。

シャトルは空気抵抗が大きいので、速さの2乗に比例する抵抗を入れて計算する:
    加速度 = 重力 − (g / 終端速度²) × |速度| × 速度
発射口(床からの高さ h0)から、上向き θ [deg]、速さ v0 [m/s] で撃ったとき、
狙う高さ(AIM_TARGET_HEIGHT_CM)を下りながら通過するまでの水平距離を「飛距離」とする。

計算は、いくつもの角度をまとめて numpy で進める(1回 数ms)。
"""

import functools
import math
from typing import Optional, Sequence, Tuple

import numpy as np

import config

DT_SEC = 0.002
MAX_TIME_SEC = 15.0
ANGLE_STEP_DEG = 0.25


def muzzle_height_cm() -> float:
    return config.CAMERA_MOUNT_HEIGHT_CM + config.LAUNCHER_MUZZLE_ABOVE_CAMERA_CM


def ranges_cm(elevations_deg: Sequence[float], speeds_ms, h0_cm: Optional[float] = None,
              target_height_cm: Optional[float] = None,
              terminal_velocity_ms: Optional[float] = None) -> np.ndarray:
    """角度(と速さ)の組ごとの飛距離 [cm]。speeds_ms は1つの値か、角度と同じ数の配列。

    狙う高さに届かない(発射口より高い高さを狙って届かない)組は nan。
    """
    h0 = (muzzle_height_cm() if h0_cm is None else h0_cm) / 100.0
    target = (config.AIM_TARGET_HEIGHT_CM if target_height_cm is None else target_height_cm) / 100.0
    vt = config.SHUTTLE_TERMINAL_VELOCITY_MS if terminal_velocity_ms is None else terminal_velocity_ms
    g = config.GRAVITY_MS2
    k = g / (vt * vt)

    th = np.radians(np.asarray(elevations_deg, dtype=float))
    v0 = np.broadcast_to(np.asarray(speeds_ms, dtype=float), th.shape)
    x = np.zeros_like(th)
    y = np.full_like(th, h0)
    vx = v0 * np.cos(th)
    vy = v0 * np.sin(th)
    result = np.full(th.shape, np.nan)
    active = np.ones(th.shape, dtype=bool)
    for _ in range(int(MAX_TIME_SEC / DT_SEC)):
        # 中点法(2次のルンゲ・クッタ)で1ステップ進める
        sp = np.hypot(vx, vy)
        ax, ay = -k * sp * vx, -g - k * sp * vy
        mvx, mvy = vx + ax * DT_SEC / 2, vy + ay * DT_SEC / 2
        msp = np.hypot(mvx, mvy)
        nx = x + mvx * DT_SEC
        ny = y + mvy * DT_SEC
        nvx = vx + (-k * msp * mvx) * DT_SEC
        nvy = vy + (-g - k * msp * mvy) * DT_SEC
        # 下りながら狙う高さを通過したら、その間を直線で補間して記録する
        crossed = active & (y >= target) & (ny < target)
        if crossed.any():
            frac = (y[crossed] - target) / (y[crossed] - ny[crossed])
            result[crossed] = (x[crossed] + frac * (nx[crossed] - x[crossed])) * 100.0
            active &= ~crossed
        x, y, vx, vy = nx, ny, nvx, nvy
        # 狙う高さより下で下り始めたら、もう届かない(狙う高さが発射口より高いとき)
        active &= ~((vy < 0) & (y < target))
        if not active.any():
            break
    return result


def range_cm(elevation_deg: float, speed_ms: float) -> float:
    return float(ranges_cm([elevation_deg], speed_ms)[0])


@functools.lru_cache(maxsize=64)
def _table(speed_ms: float, h0_cm: float, target_height_cm: float, vt: float,
           min_deg: float, max_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.arange(min_deg, max_deg + 1e-9, ANGLE_STEP_DEG)
    return angles, ranges_cm(angles, speed_ms, h0_cm, target_height_cm, vt)


def range_table(speed_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """(角度の配列, 飛距離の配列)。速さごとに覚えておく。"""
    return _table(round(float(speed_ms), 4), muzzle_height_cm(), float(config.AIM_TARGET_HEIGHT_CM),
                  float(config.SHUTTLE_TERMINAL_VELOCITY_MS),
                  float(config.AIM_ELEVATION_MIN_DEG), float(config.AIM_ELEVATION_MAX_DEG))


SOLVE_OK = "ok"
SOLVE_TOO_FAR = "too_far"    # 一番遠くまで飛ぶ角度でも届かない → その角度で撃つ
SOLVE_TOO_NEAR = "too_near"  # 一番低い角度でも飛びすぎる → その角度で撃つ


def elevation_for_distance(distance_cm: float, speed_ms: float) -> Tuple[float, str]:
    """飛距離が distance_cm になる上向きの角度 [deg] と、結果の種類を返す。

    同じ飛距離になる角度は低い弾道と高い弾道の2つあるが、低い方(届くまでが速い)を使う。
    """
    angles, ranges = range_table(speed_ms)
    valid = ~np.isnan(ranges)
    if not valid.any():
        return float(angles[-1]), SOLVE_TOO_FAR
    angles, ranges = angles[valid], ranges[valid]
    peak = int(np.argmax(ranges))
    low_angles, low_ranges = angles[:peak + 1], ranges[:peak + 1]
    # 低い弾道の側は、角度が上がるほど遠くへ飛ぶ(単調増加)
    low_ranges = np.maximum.accumulate(low_ranges)
    if distance_cm > low_ranges[-1]:
        return float(low_angles[-1]), SOLVE_TOO_FAR
    if distance_cm < low_ranges[0]:
        return float(low_angles[0]), SOLVE_TOO_NEAR
    return float(np.interp(distance_cm, low_ranges, low_angles)), SOLVE_OK


def table_range_cm(elevation_deg: float, speed_ms: float) -> float:
    """range_cm() の速い版(表から補間)。表の角度の範囲外なら nan。"""
    angles, ranges = range_table(speed_ms)
    if not (angles[0] <= elevation_deg <= angles[-1]):
        return math.nan
    return float(np.interp(elevation_deg, angles, ranges))


def max_range_cm(speed_ms: float) -> float:
    _, ranges = range_table(speed_ms)
    return float(np.nanmax(ranges)) if not np.isnan(ranges).all() else math.nan


def fit_speed(elevations_deg: Sequence[float], distances_cm: Sequence[float],
              bounds=None, iterations: int = 40) -> float:
    """実際の飛距離に一番合う発射の速さ [m/s] を探す(誤差の2乗の和が最小になる値。黄金分割探索)。"""
    lo, hi = bounds or config.LAUNCH_SPEED_FIT_RANGE_MS
    el = np.asarray(elevations_deg, dtype=float)
    d = np.asarray(distances_cm, dtype=float)

    def cost(v):
        r = ranges_cm(el, v)
        r = np.where(np.isnan(r), 0.0, r)
        return float(np.sum((r - d) ** 2))

    ratio = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, e = b - ratio * (b - a), a + ratio * (b - a)
    fc, fe = cost(c), cost(e)
    for _ in range(iterations):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - ratio * (b - a)
            fc = cost(c)
        else:
            a, c, fc = c, e, fe
            e = a + ratio * (b - a)
            fe = cost(e)
    return (a + b) / 2
