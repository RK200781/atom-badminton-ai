import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="シャトルコック検出モデルの転移学習")
    parser.add_argument(
        "--name", default="shuttlecock_v2",
        help="学習結果の保存先フォルダ名(runs/<name>)。既存の学習結果を"
             "上書きしないよう、baselineとは別名を指定すること"
             "(デフォルト: shuttlecock_v2)。",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="学習エポック数(デフォルト: 50。datasets/merged/data.yaml 統合時の"
             "baselineと同じ値)。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 事前学習済みのYOLOv8n(nano)モデルを読み込む
    model = YOLO("yolov8n.pt")

    # project は絶対パスで指定する。
    # ultralytics は project が相対パスだと settings.json の runs_dir(既に .../runs) の
    # 配下に "runs/detect/<project>" のような形で入れ子にしてしまうため。
    PROJECT_DIR = Path(__file__).resolve().parent / "runs"

    # 統合データセットで転移学習
    results = model.train(
        data="datasets/merged/data.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device=0,             # GPU(RTX3050)を使う指定
        project=PROJECT_DIR,  # 学習結果の保存先フォルダ (runs/<name>)
        name=args.name,
    )
    return results


if __name__ == "__main__":
    main()
