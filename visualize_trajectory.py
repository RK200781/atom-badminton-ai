#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シャトルコック軌道の可視化スクリプト

trajectories/ フォルダ内の全CSV(realtime_detect.py の出力)を読み込み、
- 3D散布図・軌跡線(複数ファイルは1つの図に重ねて色分け)
- X, Y, Z の時系列変化
- 見かけサイズ(width, height)の時系列変化
- confidence(信頼度)の時系列変化
をそれぞれ画像として trajectories/plots/ に保存する。

併せて、明らかに異常な行(信頼度が極端に低い、連続する検出間で座標が
大きくジャンプしている)の件数・割合を統計情報としてターミナルに表示する。

注記: グラフ内の文言は、日本語フォントが入っていない環境でも文字化け
(豆腐)せずに表示できるよう、英語で統一している。
"""

from pathlib import Path
import csv
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 表示用ディスプレイがない環境でも画像保存できるようにする
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D投影を有効にするために必要)

# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAJECTORY_DIR = BASE_DIR / "trajectories"
PLOTS_DIR = TRAJECTORY_DIR / "plots"

FIELDNAMES = [
    "timestamp", "X_cm", "Y_cm", "Z_cm", "u_px", "v_px",
    "width_px", "height_px", "confidence",
]

# 異常値判定のしきい値(後で調整しやすいよう定数化)
LOW_CONFIDENCE_THRESHOLD = 0.3       # これ未満は「信頼度が低い」とみなす
JUMP_DISTANCE_THRESHOLD_CM = 100.0   # 連続する検出間でこれ以上動いたら「ジャンプ」とみなす


def find_trajectory_csvs():
    """trajectories/ 内の *.csv をファイル名順(=実行日時順)に返す。"""
    if not TRAJECTORY_DIR.exists():
        print(f"エラー: {TRAJECTORY_DIR} が見つかりません。")
        print("先に realtime_detect.py を実行してCSVを生成してください。")
        sys.exit(1)

    csv_paths = sorted(TRAJECTORY_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"エラー: {TRAJECTORY_DIR} にCSVファイルが見つかりません。")
        sys.exit(1)

    return csv_paths


def load_csv(path: Path):
    """CSV1つを読み込み、列名 -> numpy配列 の dict にして返す。"""
    columns = {name: [] for name in FIELDNAMES}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [name for name in FIELDNAMES if name not in (reader.fieldnames or [])]
        if missing:
            print(f"[警告] {path.name}: 想定した列がありません: {missing} — スキップします。")
            return None

        for row in reader:
            for name in FIELDNAMES:
                columns[name].append(float(row[name]))

    return {name: np.array(values, dtype=float) for name, values in columns.items()}


def analyze_anomalies(name: str, data: dict):
    """信頼度が低い行、連続検出間で大きくジャンプしている箇所を集計して表示する。"""
    n = len(data["confidence"])
    if n == 0:
        print(f"  [{name}] データが空です。")
        return

    low_conf_mask = data["confidence"] < LOW_CONFIDENCE_THRESHOLD
    n_low_conf = int(low_conf_mask.sum())

    n_intervals = max(n - 1, 0)
    n_jumps = 0
    max_jump = 0.0
    if n_intervals > 0:
        dX = np.diff(data["X_cm"])
        dY = np.diff(data["Y_cm"])
        dZ = np.diff(data["Z_cm"])
        dist = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2)
        jump_mask = dist > JUMP_DISTANCE_THRESHOLD_CM
        n_jumps = int(jump_mask.sum())
        max_jump = float(dist.max())

    print(f"  [{name}] 件数={n}")
    print(
        f"    信頼度 < {LOW_CONFIDENCE_THRESHOLD}: "
        f"{n_low_conf} 件 ({100 * n_low_conf / n:.1f}%)"
    )
    if n_intervals > 0:
        print(
            f"    連続検出間のジャンプ(> {JUMP_DISTANCE_THRESHOLD_CM}cm): "
            f"{n_jumps} 件 ({100 * n_jumps / n_intervals:.1f}% of {n_intervals} intervals, "
            f"最大 {max_jump:.1f}cm)"
        )
    else:
        print("    連続検出間のジャンプ: 判定に必要な行数(2件以上)がありません")


def plot_3d_combined(all_data, output_path: Path):
    """全CSVのX,Y,Z軌跡を1つの3D図に重ねて描画する。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    for i, (name, data) in enumerate(all_data):
        color = cmap(i % 10)
        ax.plot(
            data["X_cm"], data["Y_cm"], data["Z_cm"],
            marker="o", markersize=3, linewidth=1,
            color=color, label=name,
        )

    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_zlabel("Z [cm]")
    ax.set_title("Shuttlecock 3D Trajectory")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xyz_timeseries(name: str, data: dict, output_path: Path):
    """X, Y, Z それぞれの時系列変化を3段のグラフで描画する。"""
    idx = np.arange(len(data["X_cm"]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    specs = [("X_cm", "tab:red"), ("Y_cm", "tab:green"), ("Z_cm", "tab:blue")]

    for ax, (col, color) in zip(axes, specs):
        ax.plot(idx, data[col], color=color, marker=".", markersize=3, linewidth=1)
        ax.set_ylabel(f"{col} [cm]")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Detection index (sequence order)")
    fig.suptitle(f"X, Y, Z over time - {name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_size_timeseries(name: str, data: dict, output_path: Path):
    """見かけサイズ(width, height)の時系列変化を描画する。"""
    idx = np.arange(len(data["width_px"]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, data["width_px"], label="width_px", color="tab:orange", linewidth=1)
    ax.plot(idx, data["height_px"], label="height_px", color="tab:purple", linewidth=1)
    ax.set_xlabel("Detection index (sequence order)")
    ax.set_ylabel("pixels")
    ax.set_title(f"Apparent size (width/height) over time - {name}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confidence_timeseries(name: str, data: dict, output_path: Path):
    """confidence(信頼度)の時系列変化を描画する。"""
    idx = np.arange(len(data["confidence"]))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, data["confidence"], color="tab:blue", marker=".", markersize=3, linewidth=1)
    ax.axhline(
        LOW_CONFIDENCE_THRESHOLD, color="red", linestyle="--", linewidth=1,
        label=f"threshold = {LOW_CONFIDENCE_THRESHOLD}",
    )
    ax.set_xlabel("Detection index (sequence order)")
    ax.set_ylabel("confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Confidence over time - {name}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    csv_paths = find_trajectory_csvs()

    print(f"{len(csv_paths)} 個のCSVファイルを読み込みます(ファイル名順):")
    all_data = []
    for path in csv_paths:
        data = load_csv(path)
        if data is None:
            continue
        all_data.append((path.stem, data))
        print(f"  - {path.name} ({len(data['timestamp'])} 件)")

    if not all_data:
        print("エラー: 読み込めるCSVがありませんでした。")
        sys.exit(1)

    print("\n=== 異常値の統計情報 ===")
    for name, data in all_data:
        analyze_anomalies(name, data)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nグラフを生成しています... (保存先: {PLOTS_DIR})")

    # 3D軌跡(複数ファイルは1つの図に重ねる)
    combined_3d_path = PLOTS_DIR / "trajectory_3d_all.png"
    plot_3d_combined(all_data, combined_3d_path)
    print(f"  保存: {combined_3d_path.relative_to(BASE_DIR)}")

    # ファイルごとの時系列グラフ
    for name, data in all_data:
        xyz_path = PLOTS_DIR / f"{name}_xyz_timeseries.png"
        plot_xyz_timeseries(name, data, xyz_path)
        print(f"  保存: {xyz_path.relative_to(BASE_DIR)}")

        size_path = PLOTS_DIR / f"{name}_size_timeseries.png"
        plot_size_timeseries(name, data, size_path)
        print(f"  保存: {size_path.relative_to(BASE_DIR)}")

        conf_path = PLOTS_DIR / f"{name}_confidence_timeseries.png"
        plot_confidence_timeseries(name, data, conf_path)
        print(f"  保存: {conf_path.relative_to(BASE_DIR)}")

    print(f"\n完了しました。グラフは {PLOTS_DIR} に保存されています。")


if __name__ == "__main__":
    main()
