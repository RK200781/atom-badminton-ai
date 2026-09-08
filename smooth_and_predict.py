#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シャトルコック軌道の平滑化・予測ロジック

trajectories/ フォルダ内のCSV(realtime_detect.py の出力形式:
timestamp, X_cm, Y_cm, Z_cm, u_px, v_px, width_px, height_px, confidence)
を読み込み、

1. 前処理・フィルタリング
   - confidence が低い行を除外
   - 前後フレームと比べて座標が異常にジャンプしている行を除外
     (visualize_trajectory.py の閾値ロジックを流用)
2. 平滑化
   - 移動平均 または Savitzky-Golay フィルタ(scipy)で X, Y, Z を平滑化
3. 物理モデルへのフィッティング
   - 平滑化後のデータに対し、放物運動(重力のみ)モデル
     x(t) = x0 + vx0*t + 0.5*ax*t^2 を X, Y, Z それぞれに最小二乗フィット
4. 予測・出力
   - フィッティングしたモデルで任意時刻の位置を予測する関数を用意
   - 生データ・平滑化後データ・フィッティング軌道を1つの3Dグラフに重ねて
     trajectories/plots/smoothed_trajectory_<日時>.png に保存
   - フィッティング精度(誤差・RMSE)をターミナルに表示

元データ(生CSV)は書き換えず、平滑化結果は新しいCSVとして保存する。
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 表示用ディスプレイがない環境でも画像保存できるようにする
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D投影を有効にするために必要)

from scipy.optimize import curve_fit

# scipyのSavitzky-Golayフィルタは任意機能として扱う(未インストールでも
# 移動平均だけは動くようにする)
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False


# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_DIR = BASE_DIR / "trajectories"
PLOTS_DIR = TRAJECTORY_DIR / "plots"

FIELDNAMES = [
    "timestamp", "X_cm", "Y_cm", "Z_cm", "u_px", "v_px",
    "width_px", "height_px", "confidence",
]

# 異常値判定のしきい値(visualize_trajectory.py と同じ考え方の定数)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3   # これ未満は「信頼度が低い」として除外
JUMP_DISTANCE_THRESHOLD_CM = 100.0   # 連続する検出間でこれ以上動いたら「ジャンプ」として除外

# 重力加速度 [cm/s^2] (Y軸が鉛直上向きという前提。実際の座標系に応じて
# フィッティング結果のay自体が重力加速度に近い値になるはずなので、
# ここでは物理定数として使うのではなく、あくまで参考表示に用いる)
GRAVITY_CM_S2 = 980.665


# =========================================================================
# 1. 入出力
# =========================================================================

def load_csv(path: Path) -> dict:
    """CSV1つを読み込み、列名 -> numpy配列 の dict にして返す。"""
    columns = {name: [] for name in FIELDNAMES}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [name for name in FIELDNAMES if name not in (reader.fieldnames or [])]
        if missing:
            print(f"エラー: {path.name}: 想定した列がありません: {missing}")
            sys.exit(1)

        for row in reader:
            for name in FIELDNAMES:
                columns[name].append(float(row[name]))

    data = {name: np.array(values, dtype=float) for name, values in columns.items()}

    # timestamp順に並べ替えておく(検出順=時刻順のはずだが念のため)
    order = np.argsort(data["timestamp"])
    for name in FIELDNAMES:
        data[name] = data[name][order]

    return data


def save_smoothed_csv(data: dict, path: Path):
    """平滑化後のデータを新しいCSVとして保存する(元データは書き換えない)。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "X_cm_smooth", "Y_cm_smooth", "Z_cm_smooth"])
        for t, x, y, z in zip(data["timestamp"], data["X_cm"], data["Y_cm"], data["Z_cm"]):
            writer.writerow([t, x, y, z])


# =========================================================================
# 2. 前処理・フィルタリング
# =========================================================================

def filter_low_confidence(data: dict, threshold: float) -> dict:
    """confidence が threshold 未満の行を除外する。"""
    mask = data["confidence"] >= threshold
    filtered = {name: data[name][mask] for name in FIELDNAMES}
    n_removed = len(data["confidence"]) - len(filtered["confidence"])
    print(f"  信頼度フィルタ(< {threshold}): {n_removed} 件を除外")
    return filtered


def filter_jumps(data: dict, jump_threshold_cm: float) -> dict:
    """
    前後フレームと比較して座標が異常にジャンプしている行を除外する。

    visualize_trajectory.py の analyze_anomalies() と同じ考え方で、
    連続する検出間のユークリッド距離が閾値を超える点を「外れ値」とみなす。
    片側だけの飛び(1点だけ大きくずれて戻ってくる)を想定し、
    前の点との距離・次の点との距離の両方が閾値を超える点を除外する
    (単に前との距離だけで判定すると、ジャンプの後に来る正常な点まで
    連鎖的に除外されてしまうため)。
    """
    n = len(data["timestamp"])
    if n < 3:
        print("  ジャンプフィルタ: 行数が少ないためスキップ")
        return data

    dX = np.diff(data["X_cm"])
    dY = np.diff(data["Y_cm"])
    dZ = np.diff(data["Z_cm"])
    dist = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)  # dist[i] = 点i と 点i+1 の距離

    keep = np.ones(n, dtype=bool)
    # 点i (先頭・末尾を除く) は、前との距離 dist[i-1] と 次との距離 dist[i]
    # の両方が閾値超なら「その点だけが飛んでいる」とみなして除外する
    for i in range(1, n - 1):
        if dist[i - 1] > jump_threshold_cm and dist[i] > jump_threshold_cm:
            keep[i] = False

    filtered = {name: data[name][keep] for name in FIELDNAMES}
    n_removed = n - int(keep.sum())
    print(f"  ジャンプフィルタ(> {jump_threshold_cm}cm): {n_removed} 件を除外")
    return filtered


# =========================================================================
# 3. 平滑化
# =========================================================================

def smooth_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """単純移動平均で平滑化する。端は窓を縮めて(利用可能な範囲だけで)平均する。"""
    n = len(values)
    if window <= 1 or n == 0:
        return values.copy()

    half = window // 2
    smoothed = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = values[lo:hi].mean()
    return smoothed


def smooth_savgol(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golayフィルタで平滑化する(scipy必須)。"""
    if not HAS_SCIPY_SIGNAL:
        raise RuntimeError("scipy.signal が利用できません。--method moving_average を使ってください。")

    n = len(values)
    # savgol_filterは window_length が奇数かつ polyorder より大きい必要がある
    win = window if window % 2 == 1 else window + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= polyorder:
        win = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    if win > n:
        # データ点数が足りない場合は移動平均にフォールバック
        print("  [警告] データ点数が少ないためSavitzky-Golayの窓を確保できません。移動平均で代替します。")
        return smooth_moving_average(values, window)

    return savgol_filter(values, window_length=win, polyorder=polyorder)


def smooth_trajectory(data: dict, method: str, window: int) -> dict:
    """X, Y, Z それぞれに指定した手法の平滑化を適用する。"""
    smoothed = dict(data)  # timestamp等はそのままコピー
    for axis in ("X_cm", "Y_cm", "Z_cm"):
        if method == "moving_average":
            smoothed[axis] = smooth_moving_average(data[axis], window)
        elif method == "savgol":
            smoothed[axis] = smooth_savgol(data[axis], window)
        else:
            raise ValueError(f"未知の平滑化手法: {method}")
    return smoothed


# =========================================================================
# 4. 物理モデルへのフィッティング(放物運動: 等加速度運動)
# =========================================================================

def quadratic_model(t, x0, v0, a):
    """等加速度運動の位置モデル: x(t) = x0 + v0*t + 0.5*a*t^2"""
    return x0 + v0 * t + 0.5 * a * t * t


def fit_axis(t: np.ndarray, values: np.ndarray):
    """1つの軸(X, Y, Zいずれか)の時系列に quadratic_model を最小二乗フィットする。"""
    # 初期値: 位置は最初の値、速度は前半の平均変化率、加速度は0からスタート
    x0_guess = values[0]
    v0_guess = (values[-1] - values[0]) / (t[-1] - t[0]) if t[-1] != t[0] else 0.0
    a_guess = 0.0
    popt, _ = curve_fit(quadratic_model, t, values, p0=[x0_guess, v0_guess, a_guess])
    return popt  # (x0, v0, a)


def fit_parabolic_trajectory(data: dict):
    """X, Y, Z それぞれに放物運動モデルをフィットし、パラメータをまとめて返す。"""
    t = data["timestamp"] - data["timestamp"][0]  # 数値安定のため時刻を先頭基準に
    params = {}
    for axis in ("X_cm", "Y_cm", "Z_cm"):
        params[axis] = fit_axis(t, data[axis])
    return params, data["timestamp"][0]


def predict_position(params: dict, t0: float, t):
    """
    フィッティングしたモデルを使って、任意の時刻 t における予測位置 (X, Y, Z) を計算する。

    t: 元データと同じ絶対時刻(秒, unix time相当)のスカラーまたは配列
    戻り値: (X, Y, Z) のタプル(各要素はtと同じ形状)
    """
    t = np.asarray(t, dtype=float)
    t_rel = t - t0
    x = quadratic_model(t_rel, *params["X_cm"])
    y = quadratic_model(t_rel, *params["Y_cm"])
    z = quadratic_model(t_rel, *params["Z_cm"])
    return x, y, z


def report_fit(name: str, params: dict, t0: float, data: dict):
    """フィッティング結果(初期位置・初速度・加速度)とRMSEをターミナルに表示する。"""
    print(f"\n=== フィッティング結果: {name} ===")
    for axis, label in (("X_cm", "X"), ("Y_cm", "Y"), ("Z_cm", "Z")):
        x0, v0, a = params[axis]
        print(
            f"  {label}(t) = {x0:.3f} + {v0:.3f}*t + 0.5*({a:.3f})*t^2   "
            f"[初期位置={x0:.3f}cm, 初速度={v0:.3f}cm/s, 加速度={a:.3f}cm/s^2]"
        )

    # 実測値(平滑化後)とモデル予測値のRMSE
    x_pred, y_pred, z_pred = predict_position(params, t0, data["timestamp"])
    errors = {
        "X": data["X_cm"] - x_pred,
        "Y": data["Y_cm"] - y_pred,
        "Z": data["Z_cm"] - z_pred,
    }
    print("  --- フィッティング精度(平滑化後データとの誤差) ---")
    for label, err in errors.items():
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        print(f"  {label}: RMSE={rmse:.3f}cm, MAE={mae:.3f}cm, max|err|={np.max(np.abs(err)):.3f}cm")

    dist_err = np.sqrt(errors["X"] ** 2 + errors["Y"] ** 2 + errors["Z"] ** 2)
    print(f"  3D距離誤差: RMSE={np.sqrt(np.mean(dist_err ** 2)):.3f}cm")


# =========================================================================
# 5. グラフ出力
# =========================================================================

def plot_result(raw: dict, smoothed: dict, params: dict, t0: float, output_path: Path):
    """生データ・平滑化後データ・フィッティング軌道を1つの3Dグラフに重ねて保存する。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        raw["X_cm"], raw["Y_cm"], raw["Z_cm"],
        s=15, color="lightgray", label="raw", alpha=0.6,
    )
    ax.plot(
        smoothed["X_cm"], smoothed["Y_cm"], smoothed["Z_cm"],
        color="tab:blue", linewidth=1.5, marker="o", markersize=3, label="smoothed",
    )

    # フィッティング曲線は平滑化データの時刻範囲を細かく刻んで描画する
    t_fine = np.linspace(smoothed["timestamp"][0], smoothed["timestamp"][-1], 200)
    x_fit, y_fit, z_fit = predict_position(params, t0, t_fine)
    ax.plot(x_fit, y_fit, z_fit, color="tab:red", linewidth=2, label="parabolic fit")

    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_zlabel("Z [cm]")
    ax.set_title("Shuttlecock trajectory: raw vs smoothed vs parabolic fit")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# main
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="シャトルコック軌道の平滑化・放物運動フィッティング・予測",
    )
    parser.add_argument(
        "csv_files", nargs="*", type=Path,
        help="処理対象のCSVファイル(省略時は trajectories/*.csv 全て)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"この値未満のconfidenceの行を除外する(デフォルト: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--jump-threshold-cm", type=float, default=JUMP_DISTANCE_THRESHOLD_CM,
        help=f"連続検出間でこれ以上動いたらジャンプとみなして除外する(デフォルト: {JUMP_DISTANCE_THRESHOLD_CM}cm)",
    )
    parser.add_argument(
        "--method", choices=["moving_average", "savgol"], default="moving_average",
        help="平滑化手法(デフォルト: moving_average)",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="平滑化の window size(デフォルト: 5)",
    )
    return parser.parse_args()


def process_one(path: Path, args, run_timestamp: str):
    print(f"\n### {path.name} を処理します ###")
    raw = load_csv(path)
    print(f"  読み込み: {len(raw['timestamp'])} 件")
    if len(raw["timestamp"]) < 4:
        print("  データ点数が少なすぎるためスキップします(4件未満)。")
        return

    # 1. 前処理・フィルタリング
    filtered = filter_low_confidence(raw, args.confidence_threshold)
    filtered = filter_jumps(filtered, args.jump_threshold_cm)
    if len(filtered["timestamp"]) < 4:
        print("  フィルタ後のデータ点数が少なすぎるためスキップします(4件未満)。")
        return

    # 2. 平滑化
    smoothed = smooth_trajectory(filtered, args.method, args.window)

    # 平滑化結果を新しいCSVとして保存(元データは変更しない)
    smoothed_csv_path = TRAJECTORY_DIR / f"{path.stem}_smoothed.csv"
    save_smoothed_csv(smoothed, smoothed_csv_path)
    print(f"  平滑化結果を保存: {smoothed_csv_path.relative_to(BASE_DIR)}")

    # 3. 放物運動モデルへのフィッティング
    params, t0 = fit_parabolic_trajectory(smoothed)
    report_fit(path.stem, params, t0, smoothed)

    # 4. 予測・グラフ出力
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"smoothed_trajectory_{run_timestamp}_{path.stem}.png"
    plot_result(filtered, smoothed, params, t0, plot_path)
    print(f"  グラフを保存: {plot_path.relative_to(BASE_DIR)}")


def main():
    args = parse_args()

    if args.csv_files:
        csv_paths = args.csv_files
    else:
        if not TRAJECTORY_DIR.exists():
            print(f"エラー: {TRAJECTORY_DIR} が見つかりません。")
            sys.exit(1)
        csv_paths = sorted(TRAJECTORY_DIR.glob("*.csv"))
        csv_paths = [p for p in csv_paths if not p.stem.endswith("_smoothed")]

    if not csv_paths:
        print("エラー: 処理対象のCSVファイルがありません。")
        sys.exit(1)

    if args.method == "savgol" and not HAS_SCIPY_SIGNAL:
        print("エラー: scipyが見つからないため --method savgol は使用できません。")
        sys.exit(1)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in csv_paths:
        if not path.exists():
            print(f"[警告] {path} が見つかりません。スキップします。")
            continue
        process_one(path, args, run_timestamp)


if __name__ == "__main__":
    main()
