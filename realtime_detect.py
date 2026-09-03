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

'q' キーで終了。
"""

from pathlib import Path
import csv
import datetime
import sys
import time

import platform

import cv2
import numpy as np
import yaml

# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CALIBRATION_YAML = BASE_DIR / "camera_calibration.yaml"
# 学習済みモデルの重み
# (依頼文中のパス runs/detect/runs/shuttlecock_baseline/weights/best.pt は
#  実際には存在しなかったため、実在するパスに変更しています)
MODEL_PATH = BASE_DIR / "runs" / "shuttlecock_baseline" / "weights" / "best.pt"

# 自動検出したい、あるいは接続を確認したいカメラの名前(部分一致・大文字小文字無視)。
CAMERA_NAME_KEYWORD = "Elgato Facecam 4K"

# 名前による自動検出に失敗した場合、インデックスを総当たりで探す際の上限
# (0 ... MAX_CAMERA_PROBE_INDEX-1 を試す)
MAX_CAMERA_PROBE_INDEX = 5

DESIRED_WIDTH = 3840
DESIRED_HEIGHT = 2160

# シャトルコックの実サイズ(直径, cm)。後で調整しやすいよう定数化。
SHUTTLE_DIAMETER_CM = 6.35

CONFIDENCE_THRESHOLD = 0.25  # YOLO推論時の信頼度しきい値

# 軌道CSVの出力先ディレクトリ。実行ごとに実行日時をファイル名に含めて保存し、
# 過去の実行結果を上書きしないようにする。
TRAJECTORY_OUTPUT_DIR = BASE_DIR / "trajectories"

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


def open_camera(device_index: int, desired_width: int, desired_height: int):
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # 実際に設定された解像度を確認するため、1フレーム読み込む
    ok, frame = cap.read()
    if not ok or frame is None:
        print("エラー: カメラからフレームを取得できませんでした。")
        cap.release()
        sys.exit(1)

    actual_height, actual_width = frame.shape[:2]

    if (actual_width, actual_height) != (desired_width, desired_height):
        print(
            f"[警告] カメラが希望解像度 {desired_width}x{desired_height} に対応していません。"
            f" 実際の解像度 {actual_width}x{actual_height} にフォールバックします。"
        )

    return cap, actual_width, actual_height


def main():
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

    # ---- カメラ起動 -------------------------------------------------------
    camera_index = resolve_camera_index(CAMERA_NAME_KEYWORD, MAX_CAMERA_PROBE_INDEX)
    cap, actual_width, actual_height = open_camera(
        camera_index, DESIRED_WIDTH, DESIRED_HEIGHT
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
    print("'q' キーで終了します。")

    trajectory = []  # [(timestamp, X, Y, Z, u, v, w, h, confidence), ...]

    # ウィンドウは起動時に1度だけ作成する。ループ内では作成済みの
    # このウィンドウに対して imshow するだけで、新規ウィンドウは作られない。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[警告] フレームを取得できませんでした。終了します。")
                break

            # 推論(リアルタイム性優先: verbose抑制、必要最低限の後処理)
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            result = results[0]

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

                    timestamp = time.time()
                    trajectory.append((timestamp, X, Y, Z, u, v, w, h, best_conf))

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

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("'q' が押されたため終了します。")
                break

            # ウィンドウが閉じられた場合も終了する
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

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


if __name__ == "__main__":
    main()
