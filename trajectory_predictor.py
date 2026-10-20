#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シャトルコック軌道のリアルタイム予測ロジック

realtime_detect.py からフレームごとに呼び出し、直近の検出点だけを使って
等加速度運動モデル x(t) = x0 + v0*t + 0.5*a*t^2 を X, Y, Z それぞれに
フィットし直し、数百ミリ秒先の位置を予測する。

smooth_and_predict.py(記録済みCSVに対するオフライン処理)と同じモデルだが、
リアルタイム用に以下の点を変えている。

- 軌道全体ではなく、直近 window_sec 秒・最大 max_points 点だけを使う
  (シャトルは空気抵抗が大きく、長い区間を1本の放物線で近似できないため)
- フィットは np.polyfit(線形最小二乗)で行う(curve_fitより軽く、毎フレーム
  呼んでもボトルネックにならない)
- 点数が少ないうちは1次式(等速直線運動)で近似し、加速度の推定が暴れるのを防ぐ
- Z(奥行き)は見かけサイズから逆算しており、X/Yに比べてノイズが1桁大きい
  (記録済みデータで連続フレーム間の|dZ|中央値 6〜8cm、|dX|は 0.5〜0.8cm)。
  2次式で外挿すると予測が大きく暴れるため、Zはデフォルトで1次式までに制限する
  (カメラが水平なら重力はY方向にかかるので、Zの加速度は空気抵抗分のみ)
- 検出が途切れた・座標が大きくジャンプした場合は「別の軌道」とみなして
  バッファをリセットする

座標系は realtime_detect.py と同じカメラ座標系(X: 右, Y: 下, Z: 奥, 単位cm)。
カメラの設置高さ・俯角はまだ実測していないため、床面(落下地点)の判定は
「カメラ座標系で Y がある値になる時刻」として簡易的に扱う(floor_y_cm)。

このファイルを直接実行すると、trajectories/ 内の記録済みCSVをフレーム順に
再生して予測精度を評価する(カメラが使えないWSL上でも動作確認できる)。

    python trajectory_predictor.py                       # trajectories/*.csv 全て
    python trajectory_predictor.py trajectories/xxx.csv  # 指定したCSVのみ
"""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np


# ---- 設定値(デフォルト) ---------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_DIR = BASE_DIR / "trajectories"

DEFAULT_WINDOW_SEC = 0.5           # フィットに使う直近の時間幅
DEFAULT_MAX_POINTS = 30            # フィットに使う最大点数(60fpsで0.5秒分)
DEFAULT_Z_MAX_DEGREE = 1           # Zのフィット次数の上限(ノイズが大きいため)
DEFAULT_MIN_POINTS_QUADRATIC = 6   # これ以上の点数があれば2次式でフィットする
DEFAULT_MAX_GAP_SEC = 0.3          # 検出がこれ以上途切れたら別の軌道とみなす
DEFAULT_CONFIDENCE_THRESHOLD = 0.3   # smooth_and_predict.py と同じ
DEFAULT_JUMP_THRESHOLD_CM = 100.0    # smooth_and_predict.py と同じ
# ジャンプとして弾いた点がこの回数連続したら、外れ値ではなく
# 新しい軌道が始まったとみなしてバッファをリセットする
DEFAULT_MAX_CONSECUTIVE_REJECTS = 3


class RealtimeTrajectoryPredictor:
    """直近の検出点から放物運動モデルをフィットし、未来の位置を予測する。"""

    def __init__(
        self,
        window_sec: float = DEFAULT_WINDOW_SEC,
        max_points: int = DEFAULT_MAX_POINTS,
        min_points_quadratic: int = DEFAULT_MIN_POINTS_QUADRATIC,
        z_max_degree: int = DEFAULT_Z_MAX_DEGREE,
        max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        jump_threshold_cm: float = DEFAULT_JUMP_THRESHOLD_CM,
        max_consecutive_rejects: int = DEFAULT_MAX_CONSECUTIVE_REJECTS,
    ):
        self.window_sec = window_sec
        self.min_points_quadratic = min_points_quadratic
        self.z_max_degree = z_max_degree
        self.max_gap_sec = max_gap_sec
        self.confidence_threshold = confidence_threshold
        self.jump_threshold_cm = jump_threshold_cm
        self.max_consecutive_rejects = max_consecutive_rejects

        self._points = deque(maxlen=max_points)  # [(t, X, Y, Z), ...]
        self._consecutive_rejects = 0
        # フィット結果: 軸ごとの多項式係数(np.polyval形式、時刻は t_ref 基準)
        self._coeffs = None
        self._t_ref = None

    # ---- 状態 -------------------------------------------------------------

    def reset(self):
        self._points.clear()
        self._consecutive_rejects = 0
        self._coeffs = None
        self._t_ref = None

    @property
    def n_points(self) -> int:
        return len(self._points)

    @property
    def fit_degree(self):
        """現在のX/Yのフィット次数(1: 等速直線, 2: 等加速度)。未フィットなら None。"""
        return None if self._coeffs is None else len(self._coeffs[0]) - 1

    def is_active(self, t_now: float) -> bool:
        """予測に使えるフィット結果があり、最後の検出から途切れすぎていないか。"""
        return (
            self._coeffs is not None
            and t_now - self._points[-1][0] <= self.max_gap_sec
        )

    # ---- 更新 -------------------------------------------------------------

    def update(self, t: float, x: float, y: float, z: float, confidence: float) -> bool:
        """検出点を1つ追加してフィットし直す。採用した場合は True を返す。"""
        if confidence < self.confidence_threshold:
            return False

        if self._points:
            t_last, x_last, y_last, z_last = self._points[-1]
            if t - t_last > self.max_gap_sec:
                # 検出が途切れていたので、前の軌道とは別物として扱う
                self.reset()
            else:
                dist = np.sqrt((x - x_last) ** 2 + (y - y_last) ** 2 + (z - z_last) ** 2)
                if dist > self.jump_threshold_cm:
                    self._consecutive_rejects += 1
                    if self._consecutive_rejects < self.max_consecutive_rejects:
                        return False
                    # 外れ値が続く = 実際に別の場所へ移った、とみなす
                    self.reset()

        self._consecutive_rejects = 0
        self._points.append((t, x, y, z))

        # 古すぎる点を捨てる(時間窓)
        while self._points and t - self._points[0][0] > self.window_sec:
            self._points.popleft()

        self._fit()
        return True

    def _fit(self):
        n = len(self._points)
        if n < 3:
            self._coeffs = None
            return

        arr = np.array(self._points, dtype=float)
        t_ref = arr[-1, 0]  # 最新時刻を基準にする(外挿時の数値誤差を抑える)
        t_rel = arr[:, 0] - t_ref
        if np.ptp(t_rel) <= 0:
            self._coeffs = None
            return

        degree = 2 if n >= self.min_points_quadratic else 1
        degrees = (degree, degree, min(degree, self.z_max_degree))
        self._coeffs = [
            np.polyfit(t_rel, arr[:, axis], deg) for axis, deg in zip((1, 2, 3), degrees)
        ]
        self._t_ref = t_ref

    # ---- 予測 -------------------------------------------------------------

    def predict(self, t):
        """時刻 t(スカラーまたは配列)における予測位置 (X, Y, Z) を返す。未フィットなら None。"""
        if self._coeffs is None:
            return None
        t_rel = np.asarray(t, dtype=float) - self._t_ref
        return tuple(np.polyval(c, t_rel) for c in self._coeffs)

    def predict_path(self, t_start: float, horizon_sec: float, steps: int = 20):
        """t_start から horizon_sec 秒先までの予測軌道を (steps, 3) の配列で返す。"""
        if self._coeffs is None:
            return None
        ts = np.linspace(t_start, t_start + horizon_sec, steps)
        return np.stack(self.predict(ts), axis=1)

    def time_to_reach_y(self, y_target: float, t_now: float):
        """t_now 以降で Y が y_target に達する最初の時刻を返す。達しないなら None。

        カメラ座標系は Y 下向きなので、カメラが水平に設置されていれば
        y_target = カメラの床からの高さ で床への落下時刻になる
        (俯角がある場合はこの近似は成り立たない。実機組み立て後に要対応)。
        """
        if self._coeffs is None:
            return None
        poly = np.array(self._coeffs[1], dtype=float)
        poly[-1] -= y_target
        roots = np.roots(poly)
        t_now_rel = t_now - self._t_ref
        candidates = [
            r.real for r in roots
            if abs(r.imag) < 1e-9 and r.real >= t_now_rel
        ]
        if not candidates:
            return None
        return min(candidates) + self._t_ref


# =========================================================================
# オフライン評価: 記録済みCSVをフレーム順に再生して予測精度を確認する
# =========================================================================

def load_trajectory_csv(path: Path):
    """realtime_detect.py の出力CSVを (timestamp, X, Y, Z, confidence) の配列で返す。"""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                float(row["timestamp"]), float(row["X_cm"]), float(row["Y_cm"]),
                float(row["Z_cm"]), float(row["confidence"]),
            ))
    rows.sort(key=lambda r: r[0])
    return np.array(rows, dtype=float)


def evaluate_file(path: Path, horizons, tolerance_sec: float):
    """各フレームで horizons 秒先を予測し、実際にその時刻付近で検出された位置との
    誤差を集計する。比較用に「最後の位置のまま動かない」と仮定した場合の誤差も出す。
    """
    data = load_trajectory_csv(path)
    if len(data) < 4:
        print(f"\n### {path.name}: データ点数が少ないためスキップ ###")
        return

    times = data[:, 0]
    predictor = RealtimeTrajectoryPredictor()
    errors = {h: [] for h in horizons}
    baseline_errors = {h: [] for h in horizons}

    for i, (t, x, y, z, conf) in enumerate(data):
        if not predictor.update(t, x, y, z, conf):
            continue
        for h in horizons:
            # 予測対象時刻に最も近い、実際の検出点を探す
            j = int(np.argmin(np.abs(times - (t + h))))
            if j <= i or abs(times[j] - (t + h)) > tolerance_sec:
                continue
            pred = predictor.predict(t + h)
            if pred is None:
                continue
            actual = data[j, 1:4]
            errors[h].append(np.linalg.norm(np.array(pred) - actual))
            baseline_errors[h].append(np.linalg.norm(np.array([x, y, z]) - actual))

    duration = times[-1] - times[0]
    print(f"\n### {path.name} ({len(data)} 件, {duration:.2f}秒) ###")
    print("  予測先    件数   予測誤差(中央値/平均)   静止仮定の誤差(中央値/平均)")
    for h in horizons:
        if not errors[h]:
            print(f"  +{h:.2f}s   評価可能な点なし")
            continue
        e = np.array(errors[h])
        b = np.array(baseline_errors[h])
        print(
            f"  +{h:.2f}s  {len(e):5d}   {np.median(e):8.2f} / {e.mean():8.2f} cm"
            f"      {np.median(b):8.2f} / {b.mean():8.2f} cm"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="記録済み軌道CSVを再生し、リアルタイム予測の精度を評価する",
    )
    parser.add_argument(
        "csv_files", nargs="*", type=Path,
        help="評価対象のCSV(省略時は trajectories/*.csv のうち _smoothed 以外の全て)",
    )
    parser.add_argument(
        "--horizons", type=float, nargs="+", default=[0.1, 0.2, 0.3],
        help="何秒先を予測するか(複数指定可、デフォルト: 0.1 0.2 0.3)",
    )
    parser.add_argument(
        "--tolerance-sec", type=float, default=0.03,
        help="予測時刻と実際の検出時刻のずれをどこまで許容して比較するか(デフォルト: 0.03秒)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.csv_files:
        csv_paths = args.csv_files
    else:
        csv_paths = sorted(
            p for p in TRAJECTORY_DIR.glob("*.csv") if not p.stem.endswith("_smoothed")
        )
    if not csv_paths:
        print("エラー: 評価対象のCSVファイルがありません。")
        sys.exit(1)

    for path in csv_paths:
        if not path.exists():
            print(f"[警告] {path} が見つかりません。スキップします。")
            continue
        evaluate_file(path, args.horizons, args.tolerance_sec)


if __name__ == "__main__":
    main()
