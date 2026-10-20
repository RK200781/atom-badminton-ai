#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイムシャトルコック検出・距離推定スクリプト

Webカメラ映像に対してYOLOモデルで推論し、検出されたシャトルコックの
バウンディングボックスから、カメラキャリブレーション結果を用いて
実空間上の座標 (X, Y, Z) [cm] を推定する。

推定した (X, Y, Z) はフレームごとに記録し、終了時に
trajectories/shuttle_trajectory_<実行日時>.csv として出力する
(過去の実行結果を上書きせず、実行ごとに別ファイルとして残す)。

さらに、直近の検出点に放物運動モデルをフィットして数百ミリ秒先の位置を
リアルタイムに予測し(trajectory_predictor.py)、予測軌道を映像に重ねて
表示する。予測結果は trajectories/predictions/shuttle_prediction_<実行日時>.csv
に保存する。--floor-y-cm を指定すると、Yがその値に達する時刻・位置を
落下地点として予測・表示する。

'q' キーで終了。
"""

from pathlib import Path
import argparse
import csv
import datetime
import sys
import tempfile
import time

import platform

import cv2
import numpy as np
import yaml

from trajectory_predictor import RealtimeTrajectoryPredictor

# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CALIBRATION_YAML = BASE_DIR / "camera_calibration.yaml"
# 学習済みモデルの重み
# shuttlecock_v2: adwproj + pradyumna + 自前撮影データ(experiment)を
# 統合した datasets/merged で学習したモデル。baseline(adwproj+pradyumnaのみ)
# より Precision/Recall/mAP50 が向上しているため、こちらを主軸として使う。
MODEL_PATH = BASE_DIR / "runs" / "shuttlecock_v2" / "weights" / "best.pt"

# 自動検出したい、あるいは接続を確認したいカメラの名前(部分一致・大文字小文字無視)。
CAMERA_NAME_KEYWORD = "Elgato Facecam 4K"

# 名前による自動検出に失敗した場合、インデックスを総当たりで探す際の上限
# (0 ... MAX_CAMERA_PROBE_INDEX-1 を試す)
MAX_CAMERA_PROBE_INDEX = 5

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
# Elgato Facecam 4K の対応解像度/fps表によると、720pはコーデック問わず
# 60fpsに対応しているため、明示的に60fpsを要求する。
DESIRED_FPS = 60

# シャトルコックの実サイズ(直径, cm)。後で調整しやすいよう定数化。
SHUTTLE_DIAMETER_CM = 6.35

CONFIDENCE_THRESHOLD = 0.25  # YOLO推論時の信頼度しきい値

# 軌道CSVの出力先ディレクトリ。実行ごとに実行日時をファイル名に含めて保存し、
# 過去の実行結果を上書きしないようにする。
TRAJECTORY_OUTPUT_DIR = BASE_DIR / "trajectories"
# 予測結果CSVの出力先。visualize_trajectory.py / smooth_and_predict.py は
# trajectories/*.csv を軌道CSVとして読み込むため、列構成の違う予測CSVは
# サブフォルダに分けて置く。
PREDICTION_OUTPUT_DIR = TRAJECTORY_OUTPUT_DIR / "predictions"

# 何秒先までの軌道を予測・表示するか(--predict-horizon で変更可能)
DEFAULT_PREDICT_HORIZON_SEC = 0.3
# 落下地点予測を何秒先まで探すか(これより先に床に届く予測は表示しない)
MAX_LANDING_SEARCH_SEC = 3.0

# YOLOのPredictorが内部的に作る保存先ディレクトリ(未使用でも初期化時に
# パス解決される)。BASE_DIR配下(\\wsl.localhost\... のネットワークパス)を
# 使うと、resolve()がネットワーク越しになり毎フレーム遅延・フリーズの
# 原因になることがあるため、Windowsローカルの一時フォルダに固定する。
YOLO_SCRATCH_DIR = Path(tempfile.gettempdir()) / "atom_badminton_yolo_runs"

# 表示ウィンドウの名前。namedWindow / imshow / getWindowProperty で必ずこの
# 定数だけを使う(別名を渡すと別ウィンドウとして生成されてしまうため)。
WINDOW_NAME = "Shuttlecock Realtime Detection"


def load_calibration(yaml_path: Path):
    """camera_calibration.yaml から fx, fy, cx, cy と校正時の解像度を読み込む。"""
    if not yaml_path.exists():
        print(f"エラー: キャリブレーションファイルが見つかりません: {yaml_path}")
        print("先に calibrate_camera.py を実行してください。")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        cm = data["camera_matrix"]
        fx, fy, cx, cy = cm["fx"], cm["fy"], cm["cx"], cm["cy"]
        calib_width = data["image_resolution"]["width"]
        calib_height = data["image_resolution"]["height"]
    except (KeyError, TypeError) as e:
        print(f"エラー: {yaml_path} の形式が想定と異なります ({e})")
        sys.exit(1)

    return fx, fy, cx, cy, calib_width, calib_height


def list_camera_device_names():
    """接続されているカメラのデバイス名一覧を取得する。

    Windows上で pygrabber (DirectShowのラッパー) がインストールされていれば、
    それを使ってカメラ名の一覧を取得する。取得できない場合(非Windows、
    pygrabber未インストール、取得エラー)は None を返す
    (呼び出し側はインデックス総当たりにフォールバックする)。

    戻り値のリストは、インデックス番号 == cv2.VideoCapture に渡すインデックス
    に対応している(pygrabberのFilterGraphの仕様による)。
    """
    if platform.system() != "Windows":
        return None

    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print(
            "[情報] pygrabber がインストールされていないため、カメラ名による"
            "自動検出はできません(pip install pygrabber で有効になります)。"
            "インデックスの総当たりにフォールバックします。"
        )
        return None

    try:
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception as e:
        print(f"[警告] カメラ名一覧の取得に失敗しました ({e})。インデックス総当たりにフォールバックします。")
        return None


def find_camera_index_by_name(keyword: str):
    """デバイス名に keyword(部分一致・大文字小文字無視)を含むカメラの
    インデックスを返す。見つからない場合、または名前取得ができない場合は None。
    """
    names = list_camera_device_names()
    if names is None:
        return None

    print("検出されたカメラデバイス:")
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")

    keyword_lower = keyword.lower()
    for idx, name in enumerate(names):
        if keyword_lower in name.lower():
            print(f'-> "{keyword}" に一致するカメラを自動選択しました: '
                  f'index={idx} ("{name}")')
            return idx

    print(f'[警告] "{keyword}" を含むカメラは見つかりませんでした。')
    return None


def probe_available_camera_indices(max_index: int):
    """0 から max_index-1 まで実際に開いてみて、開けたインデックスの一覧を返す。
    各インデックスにつき VideoCapture は開いたら即座に release するため、
    複数のキャプチャが同時に開いたままになることはない。
    """
    available = []
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def prompt_user_to_select_camera(candidate_indices, device_names):
    """利用可能なカメラのインデックス一覧をユーザーに提示し、選択させる。"""
    print("利用可能なカメラが複数見つかりました。使用するカメラを選択してください:")
    for idx in candidate_indices:
        if device_names is not None and idx < len(device_names):
            label = device_names[idx]
        else:
            label = f"デバイス {idx}(名前不明)"
        print(f"  [{idx}] {label}")

    while True:
        choice = input("インデックスを入力してください: ").strip()
        if choice.isdigit() and int(choice) in candidate_indices:
            return int(choice)
        print(f"無効な入力です。次の中から選んでください: {candidate_indices}")


def resolve_camera_index(name_keyword: str, max_probe_index: int) -> int:
    """使用するカメラのインデックスを決定する。

    1. まず name_keyword を含む名前のカメラを自動検出できないか試す
       (Windows + pygrabber がある場合のみ)。
    2. 自動検出できなければ、インデックス 0..max_probe_index-1 を実際に
       開いてみて、利用可能なものだけを候補にする。
    3. 候補が1つならそれを自動選択、複数あればユーザーに選ばせる。
    """
    idx = find_camera_index_by_name(name_keyword)
    if idx is not None:
        return idx

    print(f"インデックス 0〜{max_probe_index - 1} の範囲でカメラを探索します...")
    available = probe_available_camera_indices(max_probe_index)

    if not available:
        print("エラー: 利用可能なカメラが1台も見つかりませんでした。")
        print("カメラが接続されているか、他のアプリで使用中でないか確認してください。")
        sys.exit(1)

    if len(available) == 1:
        print(f"カメラが1台だけ見つかりました。index={available[0]} を使用します。")
        return available[0]

    device_names = list_camera_device_names()
    return prompt_user_to_select_camera(available, device_names)


def open_camera(device_index: int, desired_width: int, desired_height: int, desired_fps: int, debug_fps: bool = False):
    """Webカメラを開き、可能なら desired_width x desired_height に設定する。
    設定できなかった場合は実際に得られた解像度をそのまま使う(フォールバック)。

    VideoCapture は常にこの関数の中でのみ、1回だけ生成する
    (このプロセス内で他に cv2.VideoCapture を呼び出す場所はない)。
    解像度フォールバック時も、既存の cap をそのまま使い続けるだけで、
    2つ目の VideoCapture を開くことはしない。

    戻り値: (cap, actual_width, actual_height)
    """
    # Windows では既定のバックエンド(MSMF)だと、Elgato Facecam 4K のような
    # 配信向けカメラでOS側のカメラフレームサーバーが独自のプレビュー
    # ウィンドウを生成し、cv2.imshow のウィンドウと合わせて2つ表示される
    # ことがある。CAP_DSHOW を明示指定することでこれを避ける。
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # CAP_DSHOW非対応の環境向けフォールバック。
            # 必ず先に release() してから既定バックエンドで開き直す
            # (openしたままの cap を放置して2つ目を開くことはしない)。
            cap.release()
            cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"エラー: カメラ(デバイス {device_index})を開けませんでした。")
        print("カメラが接続されているか、他のアプリで使用中でないか確認してください。")
        sys.exit(1)

    # MJPGを明示指定する。既定(非圧縮フォーマット)のままだと高解像度時に
    # USB転送帯域がボトルネックになり、キャプチャ自体のfpsが大きく落ちる
    # カメラがあるため、対応していれば圧縮フォーマットに切り替える
    # (非対応のカメラでは無視されるだけで害はない)。
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    # バッファに古いフレームが溜まって遅延が蓄積しないよう、可能な限り
    # バッファサイズを1にする(非対応の環境では無視されるだけで害はない)。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 実際に設定された解像度を確認するため、1フレーム読み込む
    ok, frame = cap.read()
    if not ok or frame is None:
        print("エラー: カメラからフレームを取得できませんでした。")
        cap.release()
        sys.exit(1)

    actual_height, actual_width = frame.shape[:2]
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if (actual_width, actual_height) != (desired_width, desired_height):
        print(
            f"[警告] カメラが希望解像度 {desired_width}x{desired_height} に対応していません。"
            f" 実際の解像度 {actual_width}x{actual_height} にフォールバックします。"
        )

    fps_hint = " (--debug-fps を付けると実際のキャプチャ間隔を確認できます)" if not debug_fps else ""
    print(f"カメラが報告するfps: {actual_fps:.1f}{fps_hint}")

    return cap, actual_width, actual_height


def project_to_image(points_xyz, fx, fy, cx, cy):
    """カメラ座標系の点列 (N, 3) [cm] を画像座標 (u, v) に投影する。
    カメラより手前(Z <= 0)の点は投影できないため除外する。
    """
    pts = np.asarray(points_xyz, dtype=float)
    pts = pts[pts[:, 2] > 0]
    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy
    return np.stack([u, v], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="リアルタイムシャトルコック検出・距離推定スクリプト",
    )
    parser.add_argument(
        "--debug-fps", action="store_true",
        help="capture/推論の所要時間・実測fpsを実行中30フレームごとにターミナルへ表示する"
             "(ボトルネック調査用。通常運用では不要)",
    )
    parser.add_argument(
        "--predict-horizon", type=float, default=DEFAULT_PREDICT_HORIZON_SEC,
        help=f"何秒先までの軌道を予測・表示するか(デフォルト: {DEFAULT_PREDICT_HORIZON_SEC})",
    )
    parser.add_argument(
        "--floor-y-cm", type=float, default=None,
        help="カメラ座標系(Y下向き)で床面とみなすYの値[cm]。指定すると落下地点を予測する。"
             "カメラが水平ならカメラの床からの高さに相当する(俯角がある場合は未対応)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- キャリブレーション読み込み ------------------------------------
    fx, fy, cx, cy, calib_width, calib_height = load_calibration(CALIBRATION_YAML)
    print(f"キャリブレーション値を読み込みました (解像度 {calib_width}x{calib_height}):")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ---- モデル読み込み --------------------------------------------------
    if not MODEL_PATH.exists():
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        print("train_yolo.py で学習を行い、weights/best.pt を用意してください。")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("エラー: ultralytics パッケージがインストールされていません。")
        print("  pip install ultralytics")
        sys.exit(1)

    print(f"モデルを読み込み中: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False

    if use_cuda:
        model.to("cuda")
        print(f"GPU推論を使用します: {torch.cuda.get_device_name(0)}")
    else:
        print("[情報] CUDAが利用できないため、CPUで推論します(fpsが低くなる場合があります)。")

    # ---- カメラ起動 -------------------------------------------------------
    camera_index = resolve_camera_index(CAMERA_NAME_KEYWORD, MAX_CAMERA_PROBE_INDEX)
    cap, actual_width, actual_height = open_camera(
        camera_index, DESIRED_WIDTH, DESIRED_HEIGHT, DESIRED_FPS, debug_fps=args.debug_fps
    )

    # フォールバックが発生した場合、キャリブレーション時の解像度からの比率で
    # fx, fy, cx, cy をスケーリングする
    if (actual_width, actual_height) != (calib_width, calib_height):
        scale_x = actual_width / calib_width
        scale_y = actual_height / calib_height
        fx_use = fx * scale_x
        fy_use = fy * scale_y
        cx_use = cx * scale_x
        cy_use = cy * scale_y
        print(
            f"カメラパラメータを解像度比率でスケーリングしました "
            f"(scale_x={scale_x:.4f}, scale_y={scale_y:.4f}):"
        )
        print(f"  fx={fx_use:.2f}, fy={fy_use:.2f}, cx={cx_use:.2f}, cy={cy_use:.2f}")
    else:
        fx_use, fy_use, cx_use, cy_use = fx, fy, cx, cy

    print(f"カメラ解像度: {actual_width}x{actual_height}")
    print("'q'(または Esc)キーで終了します。"
          "反応しない場合は映像ウィンドウをクリックしてフォーカスしてから押してください。")

    trajectory = []  # [(timestamp, X, Y, Z, u, v, w, h, confidence), ...]

    predictor = RealtimeTrajectoryPredictor()
    # [(timestamp, horizon, pred_X, pred_Y, pred_Z, fit_degree, n_points,
    #   landing_timestamp, landing_X, landing_Y, landing_Z), ...]
    predictions = []

    # ウィンドウは起動時に1度だけ作成する。ループ内では作成済みの
    # このウィンドウに対して imshow するだけで、新規ウィンドウは作られない。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # fps計測用(直近30フレームの平均をターミナルに表示する)
    frame_count = 0
    timing_window = []
    TIMING_WINDOW_SIZE = 30

    try:
        while True:
            t_capture_start = time.time() if args.debug_fps else None
            ok, frame = cap.read()
            # 検出時刻はキャプチャ直後に取る(推論後に取ると推論時間分だけ
            # 時刻がずれ、軌道予測の速度・タイミングに誤差が乗るため)
            t_frame = time.time()
            t_capture_end = t_frame if args.debug_fps else None
            if not ok or frame is None:
                print("[警告] フレームを取得できませんでした。終了します。")
                break

            # 推論(リアルタイム性優先: verbose抑制、必要最低限の後処理)
            t_infer_start = time.time() if args.debug_fps else None
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                device=0 if use_cuda else "cpu",
                # save_dirの解決がネットワークパス越しにならないよう、
                # ローカルの一時フォルダを明示指定する(実際には保存しない)。
                project=str(YOLO_SCRATCH_DIR),
                name="realtime",
                exist_ok=True,
            )
            t_infer_end = time.time() if args.debug_fps else None
            result = results[0]

            if args.debug_fps:
                frame_count += 1
                timing_window.append(
                    (t_capture_end - t_capture_start, t_infer_end - t_infer_start)
                )
            if args.debug_fps and len(timing_window) >= TIMING_WINDOW_SIZE:
                capture_avg = sum(c for c, _ in timing_window) / len(timing_window)
                infer_avg = sum(i for _, i in timing_window) / len(timing_window)
                total_avg = capture_avg + infer_avg
                print(
                    f"[fps計測] capture={capture_avg*1000:.1f}ms "
                    f"infer={infer_avg*1000:.1f}ms "
                    f"合計={total_avg*1000:.1f}ms (~{1.0/total_avg:.1f}fps)"
                )
                timing_window.clear()

            best_box = None
            best_conf = -1.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                # 全検出のバウンディングボックスを描画
                for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
                    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{conf:.2f}", (x1i, max(0, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                # 最も信頼度が高い検出を採用
                best_idx = int(np.argmax(confs))
                best_conf = float(confs[best_idx])
                best_box = boxes_xyxy[best_idx]

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                # 見かけサイズは幅・高さの大きい方(検出角度によるブレを軽減)を使用
                apparent_size_px = max(w, h)

                if apparent_size_px > 0:
                    # Z = f * 実サイズ / 見かけサイズ
                    # fx, fy が近い値であるため、平均値をfとして使用
                    f_avg = (fx_use + fy_use) / 2.0
                    Z = f_avg * SHUTTLE_DIAMETER_CM / apparent_size_px

                    # 逆投影で X, Y を算出
                    X = (u - cx_use) * Z / fx_use
                    Y = (v - cy_use) * Z / fy_use

                    timestamp = t_frame
                    trajectory.append((timestamp, X, Y, Z, u, v, w, h, best_conf))
                    predictor.update(timestamp, X, Y, Z, best_conf)

                    # 最も信頼度の高い検出は赤枠で強調
                    cv2.rectangle(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 0, 255), 3,
                    )

                    info_lines = [
                        f"u,v=({u:.1f},{v:.1f})  w,h=({w:.1f},{h:.1f})px",
                        f"X={X:.1f}cm Y={Y:.1f}cm Z={Z:.1f}cm  conf={best_conf:.2f}",
                    ]
                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            frame, line, (10, 40 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                        )

                    print(
                        f"u={u:.1f} v={v:.1f} w={w:.1f} h={h:.1f}px  "
                        f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}cm  conf={best_conf:.2f}"
                    )

            # ---- 軌道予測(検出が一瞬途切れたフレームでも、直近のフィットで予測を続ける)
            now = time.time()
            if predictor.is_active(now):
                path_xyz = predictor.predict_path(now, args.predict_horizon)
                path_uv = project_to_image(path_xyz, fx_use, fy_use, cx_use, cy_use)
                if len(path_uv) >= 2:
                    cv2.polylines(
                        frame, [path_uv.astype(np.int32).reshape(-1, 1, 2)],
                        isClosed=False, color=(255, 255, 0), thickness=2,
                    )
                pX, pY, pZ = path_xyz[-1]

                landing = None
                if args.floor_y_cm is not None:
                    t_land = predictor.time_to_reach_y(args.floor_y_cm, now)
                    if t_land is not None and t_land - now <= MAX_LANDING_SEARCH_SEC:
                        landing = (t_land, *predictor.predict(t_land))

                pred_lines = [
                    f"pred +{args.predict_horizon:.2f}s: X={pX:.1f} Y={pY:.1f} Z={pZ:.1f}cm"
                    f"  (deg={predictor.fit_degree}, n={predictor.n_points})",
                ]
                if landing is not None:
                    t_land, lX, lY, lZ = landing
                    pred_lines.append(
                        f"landing in {t_land - now:.2f}s: X={lX:.1f} Z={lZ:.1f}cm"
                    )
                    land_uv = project_to_image([(lX, lY, lZ)], fx_use, fy_use, cx_use, cy_use)
                    if len(land_uv) == 1:
                        cv2.drawMarker(
                            frame, tuple(land_uv[0].astype(int)), (255, 0, 255),
                            markerType=cv2.MARKER_TILTED_CROSS, markerSize=30, thickness=3,
                        )
                for i, line in enumerate(pred_lines):
                    cv2.putText(
                        frame, line, (10, 100 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
                    )

                predictions.append(
                    (now, args.predict_horizon, pX, pY, pZ,
                     predictor.fit_degree, predictor.n_points)
                    + (landing if landing is not None else ("", "", "", ""))
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            # Caps Lock等で大文字 'Q' になる場合や、キー配列の都合で 'q' が
            # 押しづらい場合に備え、Esc キーでも終了できるようにする。
            # (waitKeyはウィンドウがアクティブ(フォーカスされている)でないと
            #  キー入力を拾えないので、反応しない場合はまず映像ウィンドウを
            #  クリックしてからキーを押すこと。)
            if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
                print("終了キーが押されたため終了します。")
                break

            # ウィンドウが閉じられた場合も終了する
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        # Ctrl+Cで終了した場合、汚いtracebackを出さずに終了する
        # (finallyブロックでのcap.release()・CSV保存は変わらず実行される)。
        print("\n[情報] Ctrl+Cが押されたため終了します。")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ---- 軌道をCSVに保存(実行日時付きファイル名、過去分は上書きしない) -------
        if trajectory:
            TRAJECTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = TRAJECTORY_OUTPUT_DIR / f"shuttle_trajectory_{run_timestamp}.csv"

            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "X_cm", "Y_cm", "Z_cm", "u_px", "v_px",
                     "width_px", "height_px", "confidence"]
                )
                writer.writerows(trajectory)
            print(f"軌道データを保存しました: {output_csv} ({len(trajectory)} 件)")
        else:
            print("シャトルコックが検出されなかったため、CSVは出力しませんでした。")

        # ---- 予測結果をCSVに保存 ------------------------------------------------
        if predictions:
            PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = PREDICTION_OUTPUT_DIR / f"shuttle_prediction_{run_timestamp}.csv"

            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "horizon_sec", "pred_X_cm", "pred_Y_cm", "pred_Z_cm",
                     "fit_degree", "n_points", "landing_timestamp",
                     "landing_X_cm", "landing_Y_cm", "landing_Z_cm"]
                )
                writer.writerows(predictions)
            print(f"予測データを保存しました: {output_csv} ({len(predictions)} 件)")


if __name__ == "__main__":
    main()
