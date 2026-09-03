from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# 学習済みモデルを読み込む
model = YOLO(BASE_DIR / "runs/shuttlecock_baseline/weights/best.pt")

# project は絶対パスで指定する。
# ultralytics は project が相対パスだと settings.json の runs_dir(既に .../runs) の
# 配下に "runs/detect/<project>" のような形で入れ子にしてしまうため。
PROJECT_DIR = BASE_DIR / "runs"

# テストデータセットに対して推論し、検出結果を画像として保存する
results = model.predict(
    source=BASE_DIR / "datasets/merged/test/images",
    save=True,
    project=PROJECT_DIR,   # 保存先フォルダ (runs/predict_test)
    name="predict_test",
    device=0,
)
