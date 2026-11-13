# -*- coding: utf-8 -*-
"""
検出テスト(detection_test/)の設定値。

カメラの値は手動版(integration/config.py)と同じにしてある。学習済みモデルのように
今後も更新されるデータはコピーせず、元の場所を参照する。
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent

# ---- 参照するデータ(コピーしない) ----
YOLO_MODEL_PATH = REPO_DIR / "runs" / "shuttlecock_v2" / "weights" / "best.pt"   # train_yolo.py が作る
CALIBRATION_YAML = REPO_DIR / "camera_calibration.yaml"                         # camera.py が参照する

# ---- カメラ(integration/config.py と同じ) ----
CAMERA_NAME_KEYWORD = "Elgato Facecam 4K"
MAX_CAMERA_PROBE_INDEX = 5
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# ---- 録画(record.py) ----
RECORDINGS_DIR = BASE_DIR / "recordings"   # Git の管理外(人が映るため)
RECORD_FOURCC = "mp4v"                     # .mp4 で保存する
WINDOW_NAME = "Detection Test - Recorder"

# ---- 比較(compare.py) ----
RESULTS_DIR = BASE_DIR / "results"          # Git の管理外
DEFAULT_IMGSZ = [640, 960, 1280]            # YOLO が処理する大きさ(学習時は 640)
DEFAULT_CONF = [0.25, 0.15]                 # 信頼度のしきい値(手動版は 0.25)
# 検出した枠の大きさ(長い辺 [px])の区分。小さいシャトルがどれだけ取れているかを見る
SIZE_BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 10_000)]
