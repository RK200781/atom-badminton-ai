#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
カメラ・実機なしで、発射判断の一連の流れを確認するシミュレーション。

人の動き(カメラ座標系、cm):
  1. 右前方 (150, 400) で静止      → 狙って撃つ(ときどき数フレーム見失う)
  2. 左 (-100, 350) へ歩いて静止   → 狙い直して撃つ
  3. 遠く (0, 900) へ下がって静止  → too_far で撃たない
  4. いなくなる                    → IDLE に戻る

時計を差し替えて実時間を待たずに進める。記録は一時フォルダの shots.csv に書く
(本番の shots/shots.csv は汚さない)。

    python -m launch_control.demo_sim
"""

import random
import tempfile
from pathlib import Path

from .fire_controller import FireController, ShotLogger
from .launcher import MockLauncher

FPS = 30.0  # realtime_pose_detect.py は2フレームに1回推論するので、60fps/2 ≒ 30Hz


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def lerp(a, b, r):
    return (a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r)


# (開始時刻, 終了時刻, 位置を返す関数 or None=いない)
RIGHT_FRONT = (150.0, 400.0)
LEFT = (-100.0, 350.0)
FAR = (0.0, 900.0)
SCENARIO = [
    (0.0, 5.0, lambda r: RIGHT_FRONT),
    (5.0, 8.0, lambda r: lerp(RIGHT_FRONT, LEFT, r)),
    (8.0, 14.0, lambda r: LEFT),
    (14.0, 16.0, lambda r: lerp(LEFT, FAR, r)),
    (16.0, 20.0, lambda r: FAR),
    (20.0, 23.0, None),
]
# MediaPipe が数フレーム見失う区間(短時間なので状態は保たれる)
DROPOUTS = [(1.2, 1.3), (9.5, 9.6)]


def person_at(t, rng):
    for start, end, pos in SCENARIO:
        if start <= t < end:
            if pos is None:
                return None
            if any(a <= t < b for a, b in DROPOUTS):
                return None
            x, z = pos((t - start) / (end - start))
            # 立ち位置推定のノイズ(静止判定 30cm に収まる程度)
            return x + rng.gauss(0, 4), z + rng.gauss(0, 6)
    return None


def main():
    rng = random.Random(0)
    clock = SimClock()
    out_dir = Path(tempfile.mkdtemp(prefix="launch_demo_"))
    launcher = MockLauncher(clock=clock)
    controller = FireController(
        launcher, shot_logger=ShotLogger(out_dir / "shots.csv"), wall_clock=clock,
    )

    last_message = None
    end_t = SCENARIO[-1][1]
    while clock.t < end_t:
        person = person_at(clock.t, rng)
        controller.update(clock.t, person)
        if controller.message != last_message:
            print(f"  t={clock.t:5.2f}  [{controller.state}] {controller.message}")
            last_message = controller.message
        clock.t += 1.0 / FPS

    print(f"\n発射回数: {controller.shot_count}")
    print(f"記録: {out_dir / 'shots.csv'}")
    print((out_dir / "shots.csv").read_text(encoding="utf-8") if controller.shot_count else "(なし)")


if __name__ == "__main__":
    main()
