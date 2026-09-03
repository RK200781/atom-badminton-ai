from pathlib import Path
import os
import shutil

from dotenv import load_dotenv
from roboflow import Roboflow

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MERGED_DIR = DATASETS_DIR / "merged"

SPLITS = ["train", "valid", "test"]

# train_yolo.py / predict_yolo.py が参照する datasets/merged/data.yaml の内容
MERGED_DATA_YAML = """train: train/images
val: valid/images
test: test/images
nc: 1
names: ['shuttlecock']
"""


def download_datasets():
    """.envファイルからAPIキーを読み込む"""
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=api_key)

    # データセット1: AdwProj(911枚)
    project1 = rf.workspace("adwproj").project("shuttlecock-yu8ra")
    version1 = project1.version(6)
    version1.download("yolov8", location=str(DATASETS_DIR / "adwproj"))

    # データセット2: pradyumna(226枚)
    project2 = rf.workspace("pradyumna-r1baa").project("shuttlecock-m9ihi")
    version2 = project2.version(1)
    version2.download("yolov8", location=str(DATASETS_DIR / "pradyumna"))

    print("ダウンロード完了")

    return [DATASETS_DIR / "adwproj", DATASETS_DIR / "pradyumna"]


def merge_datasets(source_dirs):
    """複数のデータセット(train/valid/testそれぞれにimages・labelsを持つ形式)を
    datasets/merged/ に統合する。

    train_yolo.py / predict_yolo.py はこの datasets/merged/ を直接参照するため、
    ここで images・labels をコピーし、data.yaml も併せて生成する。

    再実行しても常に最新のソースデータセットの内容と一致するよう、
    datasets/merged/ は毎回作り直す(既存の内容は削除される)。
    """
    print(f"\nデータセットを {MERGED_DIR} に統合します...")

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    total_copied = 0
    total_skipped = 0

    for split in SPLITS:
        dst_images_dir = MERGED_DIR / split / "images"
        dst_labels_dir = MERGED_DIR / split / "labels"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        seen_filenames = set()
        split_copied = 0
        split_skipped = 0

        for src_dir in source_dirs:
            src_images_dir = src_dir / split / "images"
            src_labels_dir = src_dir / split / "labels"

            if not src_images_dir.exists():
                print(f"  [警告] {src_images_dir} が見つかりません。スキップします。")
                continue

            for image_path in sorted(src_images_dir.iterdir()):
                if not image_path.is_file():
                    continue

                # 重複ファイル名がないか確認する。
                # (Roboflowのエクスポートはファイル名にハッシュが含まれるため
                #  通常は衝突しないが、念のため検出して警告し、コピーはスキップする)
                if image_path.name in seen_filenames:
                    print(
                        f"  [警告] ファイル名が重複しています。スキップします: "
                        f"{src_dir.name}/{split}/images/{image_path.name}"
                    )
                    split_skipped += 1
                    continue
                seen_filenames.add(image_path.name)

                shutil.copy2(image_path, dst_images_dir / image_path.name)

                # 対応するラベルファイル(同じファイル名、拡張子だけ.txt)をコピーする
                label_path = src_labels_dir / (image_path.stem + ".txt")
                if label_path.exists():
                    shutil.copy2(label_path, dst_labels_dir / label_path.name)
                else:
                    print(
                        f"  [警告] ラベルファイルが見つかりません: "
                        f"{src_dir.name}/{split}/labels/{label_path.name}"
                    )

                split_copied += 1

        print(f"  {split}: {split_copied} 枚統合(重複スキップ {split_skipped} 枚)")
        total_copied += split_copied
        total_skipped += split_skipped

    (MERGED_DIR / "data.yaml").write_text(MERGED_DATA_YAML, encoding="utf-8")

    print(f"統合完了: 合計 {total_copied} 枚(重複スキップ {total_skipped} 枚)")
    print(f"data.yaml を作成しました: {MERGED_DIR / 'data.yaml'}")


if __name__ == "__main__":
    downloaded_dirs = download_datasets()
    merge_datasets(downloaded_dirs)
