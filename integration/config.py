# -*- coding: utf-8 -*-
"""
統合版の設定値。

値はノートブック Untitled-2.ipynb と stattle.ino に合わせてある
(コメントの「セルN:行」はノートブック、「ino:行」は stattle.ino の位置)。
"""

# ============================================================
# 接続(COM 番号は PC や USB の挿し口で変わるので、main.py の引数でも上書きできる)
# ============================================================

# CubeMars モーター: USB-CAN(slcan)。セル2:58〜62
CAN_INTERFACE = "slcan"
CAN_CHANNEL = "COM6@115200"   # slcan アダプタの COM ポート@シリアル速度
CAN_BITRATE = 1_000_000       # CAN バスのビットレート 1Mbps

# Arduino(ローラーと装填): セル4:24〜32、ino:90
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_SEC = 0.1      # readline の待ち時間

# ============================================================
# モーター(CubeMars AK45-10 V3.0)
# ============================================================

# モーターID。ノートブックでは 104 を ROLL と呼んでいたが、統合版では pitch(縦)と呼ぶ。セル2:10〜11
YAW_ID = 105     # 横。プラスが右(実機で確認済み)
PITCH_ID = 104   # 縦。マイナスが上、0° で水平(実機で確認済み)

# 機械的な原点 [deg](モーターの絶対角度)。セル2:14〜15
YAW_ORIGIN_DEG = 145.0
PITCH_ORIGIN_DEG = 165.0

# 可動範囲 [deg](原点からの角度)。セル2:18〜22
YAW_MIN_DEG = -60.0
YAW_MAX_DEG = 60.0
PITCH_MIN_DEG = -50.0
PITCH_MAX_DEG = 30.0

# 起動時の目標角 [deg](原点からの角度)。ノートブックと同じく原点。
# 保持ループが始まると、モーターはすぐにこの角度へ動き出す。
INITIAL_YAW_DEG = 0.0
INITIAL_PITCH_DEG = 0.0

# 出力軸の最大回転速度 [rpm]。セル2:25
MAX_OUTPUT_RPM = 50.0

# 減速比と極対数。セル2:28〜29
GEAR_RATIO = 10.0
POLE_PAIRS = 14

# CubeMars の CAN コマンド。セル2:32、セル3:43
CAN_PACKET_SET_POS_SPD = 6     # Position-Speed Loop の指令
CAN_FEEDBACK_FUNCTION_ID = 0x29  # 状態フィードバック

# 位置を保持するループ。セル3:587〜591(head_hold_loop の既定値)
#
# 速度・加速度の単位(2026-11-13 実機で確認済み):
#   ノートブックのコメントでは「送信値 = ERPM / 10」となっているが、コードは 10 で割らずに
#   ERPM をそのまま送っている(0.5rpm → speed_cmd=70)。実機で動く速さが想定どおりだったので、
#   10 で割らない今の方式で正しい。
HOLD_SPEED_RPM = 0.5          # yaw・pitch 共通 [rpm]
HOLD_ACCEL_RPM_S = 20.0       # [rpm/s]
HOLD_INTERVAL_SEC = 0.05      # 50ms ごとに目標角を送り直す

# W/A/S/D を1回押したときに動かす角度 [deg]。セル4:9
KEY_STEP_DEG = 2.0

# 現在角(フィードバック)がこの時間届かなければ「不明」として扱う
ANGLE_STALE_SEC = 0.5

# ============================================================
# Arduino(stattle.ino)
# ============================================================

# ポートを開くと Arduino がリセットされる(Uno / Nano の場合)ので、その間待つ
ARDUINO_RESET_WAIT_SEC = 2.0

# シャトル検知の値(Arduino が約10msごとに送る 1/0)が、この時間以内に届いたものなら有効とみなす
SHUTTLE_FRESH_SEC = 0.2

# 強さを変えてから、ローラーが回り始める(回転が落ち着く)までの待ち時間(仮の値。実機で調整)
ROLLER_SPINUP_SEC = 1.5

# 前の発射から、次の発射までに空ける時間(仮の値。実機で調整)
FIRE_INTERVAL_SEC = 2.0

# 発射の指令を送ってから、シャトル検知が 1→0 に変わるのを待つ時間(仮の値)
FIRE_CONFIRM_SEC = 1.0

# Arduino は、シャトルを検知しない状態が 10 秒続くとローラーを一時的に止め、
# シャトルを入れた瞬間に元の強さで回し直す(ino:51、ino:119〜121)。
# Python 側でもこれを真似て、10 秒以上検知がなかった後にシャトルが入ったら、
# そこからもう一度 ROLLER_SPINUP_SEC 待つ。
ARDUINO_NO_SHUTTLE_TIMEOUT_SEC = 10.0

# 強さの段階と、Arduino での出力(ino:231〜240)。表示用。
SPEED_LABELS = {0: "停止", 1: "弱(20%)", 2: "中(50%)", 3: "強(100%)"}

# ============================================================
# カメラ(シャトル検出:realtime_detect.py、人の検出:realtime_pose_detect.py と同じ値)
# ============================================================

from pathlib import Path as _Path

# リポジトリのルート。学習済みモデルやキャリブレーションのように、今後も更新されるデータは
# コピーせず、元の場所を参照する(コードは integration/ の外から import しない)。
REPO_DIR = _Path(__file__).resolve().parent.parent

CALIBRATION_YAML = REPO_DIR / "camera_calibration.yaml"                     # calibrate_camera.py が作る
YOLO_MODEL_PATH = REPO_DIR / "runs" / "shuttlecock_v2" / "weights" / "best.pt"  # train_yolo.py が作る
POSE_MODEL_PATH = REPO_DIR / "weights" / "pose_landmarker_lite.task"       # 無ければ自動ダウンロード
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# CSV の保存先。単品スクリプトと同じ場所・同じ形式にして、既存の分析ツールでそのまま読めるようにする
# (どちらも Git の管理外)。
SHUTTLE_CSV_DIR = REPO_DIR / "trajectories"         # shuttle_trajectory_<日時>.csv
PERSON_CSV_DIR = REPO_DIR / "person_positions"      # person_position_<日時>.csv

# カメラの選択と設定
CAMERA_NAME_KEYWORD = "Elgato Facecam 4K"  # この名前を含むカメラを自動で選ぶ(Windows + pygrabber のとき)
MAX_CAMERA_PROBE_INDEX = 5                 # 名前で選べないとき、0〜4 番を試す
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# シャトルの検出(YOLO)
SHUTTLE_DIAMETER_CM = 6.35       # 見かけの大きさから距離 Z を出すための実際の直径
YOLO_CONFIDENCE = 0.25

# 人の検出(MediaPipe Pose)
POSE_INFERENCE_INTERVAL = 2      # 何フレームに1回推論するか
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
ANKLE_VISIBILITY_THRESHOLD = 0.5
ANKLE_PRESENCE_THRESHOLD = 0.5
SHOULDER_VISIBILITY_THRESHOLD = 0.5
SHOULDER_PRESENCE_THRESHOLD = 0.5

# 足首が見えないときは、肩の中点がこの高さ(床から)にあると仮定して立ち位置を出す(仮の値。身長で変わる)
SHOULDER_HEIGHT_CM = 140.0

# カメラの設置
#   高さ:床から約 20cm(2026-11-13 にユーザーが測った値)。
#   向き:約 19° 上向き。2026-11-13 の録画(detection_test/recordings/rec_20261113_191721_2m.mp4)で、
#   垂直な柱が上に行くほど寄って見える度合い(消失点)から求めた値(7 フレームで 19.1〜22.1°、中央値 19.2°)。
#   最初に聞いた「35°」は、レンズの向き(光軸)ではない角度を測っていたらしい。
#   この向きだと画面の一番下は水平より約 6° 下を見ているので、カメラから約 1.1m 以上離れた人の足首が映り、
#   足首から立ち位置を出せる。それより近い、または足首を見失ったときは肩の高さから出す。
CAMERA_MOUNT_HEIGHT_CM = 20.0
CAMERA_TILT_DEG = -19.0          # 水平から下向きを正とする角度。上向き 19° なので −19

# 発射口の位置(カメラから見て。2026-11-13 にユーザーが測った値)。左右のずれは無く、発射機の正面(yaw 0°)と
# カメラの正面は同じ向き。自動版(auto_main.py)で使う。手動版では使わない。
LAUNCHER_MUZZLE_BEHIND_CAMERA_CM = 25.0   # カメラより後ろ(水平方向)
LAUNCHER_MUZZLE_ABOVE_CAMERA_CM = 115.0   # カメラより上(床からは約 135cm)

# ============================================================
# 画面
# ============================================================

WINDOW_NAME = "Launcher Control"
CAMERA_WINDOW_NAME = "Camera"

# ウィンドウの位置と大きさ [px]。2つが重なって片方が隠れないよう、横に並べて開く。
# 操作画面は幅 960px なので、カメラ映像はその右隣に置く。画面に収まらなければ値を変える。
WINDOW_POSITION = (0, 0)
CAMERA_WINDOW_POSITION = (970, 0)
CAMERA_WINDOW_SIZE = (854, 480)   # 開いたときの大きさ(16:9)。マウスで変えられる
UI_REFRESH_MS = 30            # cv2.waitKey の待ち時間

# ============================================================
# 自動版(auto_main.py)。手動版(main.py)では使わない
# ============================================================

# ---- シャトルの飛び方(ballistics.py) ----
GRAVITY_MS2 = 9.81
# シャトルの終端速度 [m/s]。空気抵抗は速さの2乗に比例するとして計算する(一般的な値 6.5〜7)
SHUTTLE_TERMINAL_VELOCITY_MS = 6.8
# 狙う高さ [cm]。0 なら「床に落ちる位置」が人の立ち位置になるように狙う
AIM_TARGET_HEIGHT_CM = 0.0
# 落ちる位置を、人の立ち位置からこの距離だけ奥(+)/手前(−)にずらして狙う [cm]
AIM_LANDING_OFFSET_CM = 0.0

# 強さ(1/2/3)ごとの、発射の速さ [m/s] の初期値。
# **仮の値**(実測なし。10/31 の動画で、強いほど・上向きほど遠くへ落ちていた順番だけ合わせてある)。
# 発射の後にターミナルで「1m足りない」などと入力すると、その記録から強さごとに合わせ直す(learning)。
INITIAL_LAUNCH_SPEED_MS = {1: 7.0, 2: 9.0, 3: 11.5}
LAUNCH_SPEED_FIT_RANGE_MS = (2.0, 40.0)   # 合わせ直すときに探す範囲

# 自動で狙うときに使う上下の角度の範囲 [deg](水平から上向きが +。pitch はマイナスが上なので pitch = −この値)
AIM_ELEVATION_MIN_DEG = -5.0
AIM_ELEVATION_MAX_DEG = 45.0

# ---- 自動で狙う(auto_controller.py) ----
PERSON_SMOOTHING_SEC = 0.6      # 人の立ち位置は、この時間の中央値を使う(揺れでモーターがガタつかないように)
PERSON_LOST_SEC = 1.0           # この時間、人を見失ったら、その場で止まる(新しい目標を出さない)
AIM_DEADBAND_DEG = 0.5          # 目標角の変化がこれより小さければ送り直さない
AIM_SETTLED_DEG = 1.0           # 現在角と目標角の差がこれ以下なら「狙いが合った」とする

# ---- 自動で発射する(F キーでオン。起動時はオフ) ----
AUTO_FIRE_STABLE_SEC = 1.5      # 人がこの時間ほぼ止まっていたら撃つ
AUTO_FIRE_STABLE_CM = 30.0      # 「止まっている」とみなす立ち位置の揺れ(この時間の最大−最小)
AUTO_FIRE_INTERVAL_SEC = 5.0    # 自動発射の間隔(前の発射からこの時間は撃たない)

# ---- 記録(shots.py、session_video.py) ----
# 発射ごとの記録(CSV)と、セッションの録画。人が映る・人の立ち位置を含むので Git の管理外(.gitignore の shots/)。
SHOTS_DIR = _Path(__file__).resolve().parent / "shots"
SHOTS_CSV = SHOTS_DIR / "shots.csv"       # 全セッション分を1つのファイルに追記する。学習はこのファイルから行う
SESSION_VIDEO_FPS = 30.0                  # 録画ファイルに書く fps(実際の間隔は *_frames.csv に記録する)
