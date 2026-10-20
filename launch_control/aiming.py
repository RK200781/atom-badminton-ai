# -*- coding: utf-8 -*-
"""
人の立ち位置(カメラ座標系)から、発射機のパン・チルト角を計算する。

カメラ座標系: realtime_pose_detect.py の estimate_floor_position() の出力。
  原点はカメラのレンズ真下の床、右が +x、カメラの向き(水平成分)が +z、単位cm。
発射機座標系: config.CameraMount の説明を参照(右が +x、発射機の正面が +z)。
"""

import math
from dataclasses import dataclass

from . import config


STATUS_OK = "ok"
STATUS_TOO_FAR = "too_far"        # 最大距離より遠い
STATUS_TOO_CLOSE = "too_close"    # 最小仰角での距離より近い
STATUS_OUT_OF_PAN = "out_of_pan"  # 首振りの範囲外


@dataclass(frozen=True)
class AimSolution:
    status: str
    target_x_cm: float   # 発射機座標系での狙い位置
    target_z_cm: float
    distance_cm: float
    pan_deg: float       # 範囲外でも計算値を入れておく(表示用)
    tilt_deg: float = None  # status が ok の時だけ値が入る

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def camera_to_launcher(x_cam: float, z_cam: float, mount: config.CameraMount = None):
    """カメラ座標系の床上の点を、発射機座標系に変換する。"""
    if mount is None:
        mount = config.CAMERA_MOUNT
    yaw = math.radians(mount.yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # カメラの +z(前)は発射機座標系で (sin yaw, cos yaw)、+x(右)は (cos yaw, -sin yaw)
    x = mount.offset_x_cm + x_cam * cos_y + z_cam * sin_y
    z = mount.offset_z_cm - x_cam * sin_y + z_cam * cos_y
    return x, z


def pan_angle_deg(x: float, z: float) -> float:
    """発射機座標系の点へ向くパン角。右が +。"""
    return math.degrees(math.atan2(x, z))


def ascending_branch(table):
    """対応表のうち、最大距離までの単調増加部分(低い弾道)を返す。

    強さ固定だと同じ距離に低い弾道・高い弾道の2つの仰角があるため、
    低い弾道だけを使って逆引きを一意にする。
    """
    if len(table) < 2:
        raise ValueError("仰角→距離の対応表には2点以上必要です")
    tilts = [t for t, _ in table]
    if tilts != sorted(tilts) or len(set(tilts)) != len(tilts):
        raise ValueError("仰角→距離の対応表は仰角の昇順(重複なし)で並べてください")

    distances = [d for _, d in table]
    peak = distances.index(max(distances))
    branch = list(table[:peak + 1])
    if len(branch) < 2:
        raise ValueError("対応表の最小仰角で距離が最大になっており、逆引きできません")
    for (_, d0), (_, d1) in zip(branch, branch[1:]):
        if d1 <= d0:
            raise ValueError("対応表の最大距離までの部分が単調増加になっていません")
    return branch


def tilt_for_distance(distance_cm: float, table=None):
    """距離から仰角を線形補間で逆引きする。

    戻り値: (status, tilt_deg)。範囲外なら tilt_deg は None。
    """
    if table is None:
        table = config.TILT_TO_DISTANCE_TABLE
    branch = ascending_branch(table)

    if distance_cm < branch[0][1]:
        return STATUS_TOO_CLOSE, None
    if distance_cm > branch[-1][1]:
        return STATUS_TOO_FAR, None

    for (t0, d0), (t1, d1) in zip(branch, branch[1:]):
        if d0 <= distance_cm <= d1:
            tilt = t0 + (t1 - t0) * (distance_cm - d0) / (d1 - d0)
            break

    # 対応表が可動範囲より広い場合に備える
    if tilt < config.TILT_MIN_DEG:
        return STATUS_TOO_CLOSE, None
    if tilt > config.TILT_MAX_DEG:
        return STATUS_TOO_FAR, None
    return STATUS_OK, tilt


def solve_aim(x_cam: float, z_cam: float, mount: config.CameraMount = None, table=None) -> AimSolution:
    """カメラ座標系の立ち位置から、狙うべきパン・チルト角を求める。"""
    x, z = camera_to_launcher(x_cam, z_cam, mount)
    distance = math.hypot(x, z)
    pan = pan_angle_deg(x, z)

    if not (config.PAN_MIN_DEG <= pan <= config.PAN_MAX_DEG):
        return AimSolution(STATUS_OUT_OF_PAN, x, z, distance, pan)

    status, tilt = tilt_for_distance(distance, table)
    return AimSolution(status, x, z, distance, pan, tilt)
