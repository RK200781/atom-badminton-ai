from pathlib import Path

from ultralytics import YOLO

# 事前学習済みのYOLOv8n(nano)モデルを読み込む
model = YOLO("yolov8n.pt")

# project は絶対パスで指定する。
# ultralytics は project が相対パスだと settings.json の runs_dir(既に .../runs) の
# 配下に "runs/detect/<project>" のような形で入れ子にしてしまうため。
PROJECT_DIR = Path(__file__).resolve().parent / "runs"

# 統合データセットで転移学習
results = model.train(
    data="datasets/merged/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,             # GPU(RTX3050)を使う指定
    project=PROJECT_DIR,  # 学習結果の保存先フォルダ (runs/shuttlecock_baseline)
    name="shuttlecock_baseline"
)
