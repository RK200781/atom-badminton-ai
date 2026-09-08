#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイム人物検出・立ち位置算出スクリプト

Webカメラ映像に対して MediaPipe Pose で人物の骨格(関節点)を検出し、
左右の足首(LEFT_ANKLE, RIGHT_ANKLE)の中点を「利用者の立ち位置」として
推定する。

カメラの自動検出・解像度フォールバックのロジックは realtime_detect.py と
同じもの(Elgato Facecam 4K を名前で自動検出、失敗時はインデックス総当たり)
をそのまま流用している。

推定した立ち位置はフレームごとに記録し、終了時に
person_positions/person_position_<実行日時>.csv として出力する
(realtime_detect.py の trajectories/ と同じ考え方)。

'q' キーで終了。

注: 現行バージョンのMediaPipeでは旧来の `mediapipe.solutions.pose` API は
廃止されており、Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`)
を使用している。このAPIはモデルファイル(.task)を明示的に必要とするため、
初回実行時に weights/pose_landmarker_lite.task が無ければ自動ダウンロード
する。

--- 座標変換についての注意 ---------------------------------------------
シャトルコック検出(realtime_detect.py)では「シャトルの既知の直径」を
使って見かけサイズから距離Zを逆算できたが、人物の場合は姿勢や体格に
よって見かけサイズが大きくばらつくため、同じ方法では信頼できるZを
求められない。

そのため、このMVPでは以下の簡易ロジックにとどめている:
  1. 画像上の足首中点 (u, v) を求める
  2. カメラの設置高さ・俯角(見下ろし角度)を「仮の定数」として置き、
     床面(Y=0の水平面)とカメラ光線の交点を計算することで、
     大まかな X, Z(奥行き)を求める(床面投影)

この「カメラ設置高さ・俯角」は現状では実測せず仮の値を CAMERA_HEIGHT_CM /
CAMERA_TILT_DEG として置いているだけなので、精度は保証されない。
将来的な拡張案:
  - カメラの設置高さ・俯角を実測してキャリブレーションする
  - あるいは、シャトル発射位置とコートの既知の寸法(コートライン等)を
    使って、より正確な床面ホモグラフィを求める
  - 複数カメラでのステレオ視、深度センサーの併用
などにより、このモジュールの `estimate_floor_position()` を正確な実装に
差し替えられるよう、関数として分離してある。
"""

from pathlib import Path
import csv
import datetime
import math
import sys
import time
import urllib.request

import platform

import cv2
import numpy as np
import yaml

# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CALIBRATION_YAML = BASE_DIR / "camera_calibration.yaml"

# MediaPipe Pose Landmarker のモデルファイル。
# 新しいバージョンのMediaPipe(Tasks API)では、旧来の
# `mediapipe.solutions.pose` は廃止されており、モデルファイル(.task)を
# 明示的に読み込む方式になっている。リポジトリには含めず、初回実行時に
# 未取得なら自動ダウンロードする(weights/ 以下は *.pt 等と同様バイナリの
# ため .gitignore で除外)。
POSE_MODEL_PATH = BASE_DIR / "weights" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# 自動検出したい、あるいは接続を確認したいカメラの名前(部分一致・大文字小文字無視)。
# realtime_detect.py と同じ値。
CAMERA_NAME_KEYWORD = "Elgato Facecam 4K"

# 名前による自動検出に失敗した場合、インデックスを総当たりで探す際の上限
# (0 ... MAX_CAMERA_PROBE_INDEX-1 を試す)
MAX_CAMERA_PROBE_INDEX = 5

DESIRED_WIDTH = 3840
DESIRED_HEIGHT = 2160

# --- 床面投影のための仮のカメラ設置パラメータ(要実測・要調整) -----------
# カメラのレンズ中心の床からの高さ [cm]
CAMERA_HEIGHT_CM = 150.0
# カメラの俯角(水平から下向きに何度傾けて設置しているか)[degree]
# 0度 = 水平、正の値 = 下向き(見下ろし)
CAMERA_TILT_DEG = 15.0

# MediaPipeの処理負荷軽減のため、何フレームに1回推論するか
# (1 = 毎フレーム推論、2 = 1フレームおきに推論、など)
POSE_INFERENCE_INTERVAL = 2

# MediaPipe Poseの検出・追跡の信頼度しきい値
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# 立ち位置ログの出力先ディレクトリ。実行ごとに実行日時をファイル名に含めて
# 保存し、過去の実行結果を上書きしないようにする。
POSITION_OUTPUT_DIR = BASE_DIR / "person_positions"

# 表示ウィンドウの名前。namedWindow / imshow / getWindowProperty で必ずこの
# 定数だけを使う(別名を渡すと別ウィンドウとして生成されてしまうため)。
WINDOW_NAME = "Person Pose / Standing Position Detection"


# =========================================================================
# カメラ選択・起動(realtime_detect.py と同じロジックのコピー)
# =========================================================================

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


# =========================================================================
# キャリブレーション読み込み
# =========================================================================

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


# =========================================================================
# 足首座標 -> 立ち位置(床面投影)
# =========================================================================

def estimate_floor_position(u, v, fx, fy, cx, cy,
                             camera_height_cm=CAMERA_HEIGHT_CM,
                             camera_tilt_deg=CAMERA_TILT_DEG):
    """
    画像上の足首中点 (u, v) から、簡易的な床面投影で大まかな (X, Z) を求める。

    シャトルのように既知サイズの物体ではないため、Zを直接逆算することは
    できない。代わりに「カメラの設置高さ camera_height_cm・俯角
    camera_tilt_deg が既知である」と仮定し、カメラから (u, v) 方向に伸ばした
    視線ベクトルが、床面(カメラ設置点の真下 camera_height_cm 下にある
    水平面)と交わる点を計算する。

    カメラ座標系(この関数内だけの一時的な定義):
      - 光軸(俯角0の状態)を基準に、X: 右方向, Y: 下方向, Z: 前方(奥行き)
      - 俯角 camera_tilt_deg だけ下向きに回転させた姿勢で設置されている

    戻り値: (X_cm, Z_cm, valid)
      X_cm: カメラを基準にした水平方向の位置(左右)
      Z_cm: カメラからの奥行き距離
      valid: 視線が床と交わらない(水平より上を向いている等)場合 False

    注意: これはあくまでMVPの簡易ロジックであり、camera_height_cm /
    camera_tilt_deg は実測値ではなく仮置きの定数。正確な立ち位置が必要に
    なった場合は、この関数を実測キャリブレーション済みの床面ホモグラフィ
    などに差し替えること。
    """
    # ピンホールカメラモデルで、画像座標 (u, v) を「カメラ光軸方向を+Zとした
    # 正規化視線ベクトル」に逆投影する
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    ray_camera = np.array([x_norm, y_norm, 1.0])

    # カメラは俯角 camera_tilt_deg だけ下向きに傾けて設置されているとして、
    # 視線ベクトルをワールド座標系(Y: 上方向, Z: 水平前方)に回転する。
    # カメラ座標系のY(下方向)・Z(前方)を、俯角分だけX軸まわりに回転させる。
    tilt_rad = math.radians(camera_tilt_deg)
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)

    x_world = ray_camera[0]
    # カメラ座標系の下向きYと前向きZを、下向き俯角分だけ回転してワールド系へ
    y_world = -(ray_camera[1] * cos_t - ray_camera[2] * sin_t)  # 上向きを正にするため符号反転
    z_world = ray_camera[1] * sin_t + ray_camera[2] * cos_t

    # 視線が水平より上(y_world >= 0)を向いている場合、床(y=-camera_height_cm)
    # とは交わらない
    if y_world >= -1e-9:
        return None, None, False

    # カメラ位置(高さ camera_height_cm)から視線方向に進み、
    # 床(ワールドY = -camera_height_cm)に到達するスケール t を求める
    # camera_height_cm + t * y_world = 0  ->  t = -camera_height_cm / y_world
    t = -camera_height_cm / y_world
    if t <= 0:
        return None, None, False

    X_cm = t * x_world
    Z_cm = t * z_world
    return float(X_cm), float(Z_cm), True


# =========================================================================
# main
# =========================================================================

def ensure_pose_model(model_path: Path, model_url: str):
    """MediaPipe Pose Landmarker のモデルファイルが無ければダウンロードする。"""
    if model_path.exists():
        return
    print(f"Pose Landmarkerのモデルファイルが見つからないためダウンロードします: {model_url}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as e:
        print(f"エラー: モデルファイルのダウンロードに失敗しました ({e})")
        print(f"手動で {model_url} を取得し、{model_path} に配置してください。")
        sys.exit(1)
    print(f"ダウンロード完了: {model_path}")


def main():
    # ---- MediaPipe読み込み -------------------------------------------------
    # 現行バージョンのMediaPipeでは旧来の `mediapipe.solutions.pose` API は
    # 廃止されており、Tasks API (`mediapipe.tasks.python.vision`) を使う。
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("エラー: mediapipe パッケージがインストールされていません。")
        print("  pip install mediapipe")
        sys.exit(1)

    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URL)

    pose_landmark_enum = mp_vision.PoseLandmark
    pose_connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
    pose_landmark_style = mp_vision.drawing_styles.get_default_pose_landmarks_style()

    # ---- キャリブレーション読み込み ------------------------------------
    fx, fy, cx, cy, calib_width, calib_height = load_calibration(CALIBRATION_YAML)
    print(f"キャリブレーション値を読み込みました (解像度 {calib_width}x{calib_height}):")
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(
        f"床面投影の仮定パラメータ: camera_height={CAMERA_HEIGHT_CM}cm, "
        f"camera_tilt={CAMERA_TILT_DEG}deg (未実測。実測次第で調整すること)"
    )

    # ---- カメラ起動 -------------------------------------------------------
    camera_index = resolve_camera_index(CAMERA_NAME_KEYWORD, MAX_CAMERA_PROBE_INDEX)
    cap, actual_width, actual_height = open_camera(
        camera_index, DESIRED_WIDTH, DESIRED_HEIGHT
    )

    # フォールバックが発生した場合、キャリブレーション時の解像度からの比率で
    # fx, fy, cx, cy をスケーリングする(realtime_detect.pyと同じ考え方)
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

    positions = []  # [(timestamp, u, v, X_cm, Z_cm), ...]

    # ウィンドウは起動時に1度だけ作成する。ループ内では作成済みの
    # このウィンドウに対して imshow するだけで、新規ウィンドウは作られない。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count = 0
    # 間引き推論時、推論しなかったフレームでも直近の検出結果を描画し続けるために保持する
    last_landmarks = None

    # Tasks APIのPoseLandmarkerを作成する。RunningMode.VIDEOは、
    # 単調増加するタイムスタンプ(ミリ秒)を渡して1フレームずつ同期的に
    # 処理するモード(Webカメラのようなライブ映像でも、フレームを順番に
    # 処理していく用途であればVIDEOモードで問題ない)。
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
        num_poses=1,  # 立ち位置算出は1人を想定
    )

    with mp_vision.PoseLandmarker.create_from_options(pose_options) as landmarker:
        try:
            start_time = time.monotonic()
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[警告] フレームを取得できませんでした。終了します。")
                    break

                frame_count += 1
                h, w = frame.shape[:2]

                # 処理負荷軽減のため、POSE_INFERENCE_INTERVAL フレームに1回だけ推論する
                run_inference = (frame_count % POSE_INFERENCE_INTERVAL == 0)

                if run_inference:
                    # MediaPipeはRGB画像を期待するため変換する
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    # VIDEOモードは単調増加のタイムスタンプ(ミリ秒)が必須
                    timestamp_ms = int((time.monotonic() - start_time) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    # 複数人検出時も先頭(=最も信頼度の高い)1人分だけを使う
                    last_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

                if last_landmarks is not None:
                    # 骨格全体を描画
                    mp_vision.drawing_utils.draw_landmarks(
                        frame,
                        last_landmarks,
                        pose_connections,
                        landmark_drawing_spec=pose_landmark_style,
                    )

                    left_ankle = last_landmarks[pose_landmark_enum.LEFT_ANKLE]
                    right_ankle = last_landmarks[pose_landmark_enum.RIGHT_ANKLE]

                    # MediaPipeのランドマークは正規化座標(0.0〜1.0)なので、
                    # 実ピクセル座標に変換する
                    left_u, left_v = left_ankle.x * w, left_ankle.y * h
                    right_u, right_v = right_ankle.x * w, right_ankle.y * h

                    # 両足首の中点を「立ち位置」の画像座標とする
                    u = (left_u + right_u) / 2.0
                    v = (left_v + right_v) / 2.0

                    # 足首点を強調して描画
                    for px, py in [(left_u, left_v), (right_u, right_v)]:
                        cv2.circle(frame, (int(px), int(py)), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (int(u), int(v)), 10, (0, 0, 255), -1)

                    # 床面投影で大まかな (X, Z) を算出(既知サイズが無いためZ算出は近似)
                    X_cm, Z_cm, valid = estimate_floor_position(
                        u, v, fx_use, fy_use, cx_use, cy_use
                    )

                    if valid:
                        timestamp = time.time()
                        positions.append((timestamp, u, v, X_cm, Z_cm))

                        info_lines = [
                            f"ankle mid u,v=({u:.1f},{v:.1f})px",
                            f"standing pos (approx): X={X_cm:.1f}cm Z={Z_cm:.1f}cm",
                        ]
                    else:
                        info_lines = [
                            f"ankle mid u,v=({u:.1f},{v:.1f})px",
                            "standing pos: N/A (line of sight above horizon)",
                        ]

                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            frame, line, (10, 40 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
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

            # ---- 立ち位置ログをCSVに保存(実行日時付きファイル名) -----------
            if positions:
                POSITION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_csv = POSITION_OUTPUT_DIR / f"person_position_{run_timestamp}.csv"

                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["timestamp", "ankle_mid_u_px", "ankle_mid_v_px", "X_cm", "Z_cm"]
                    )
                    writer.writerows(positions)
                print(f"立ち位置データを保存しました: {output_csv} ({len(positions)} 件)")
            else:
                print("人物が検出されなかったため、CSVは出力しませんでした。")


if __name__ == "__main__":
    main()
