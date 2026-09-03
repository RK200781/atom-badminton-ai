# atom-badminton-ai

チャレンジアトム向け「伴走型Physical AI」バドミントンシャトル発射機プロジェクトの一部として、
カメラ映像からシャトルコックを検出し、実空間上の3次元位置 (X, Y, Z) を推定する仕組みを開発しています。
YOLOによる物体検出とカメラキャリブレーションを組み合わせ、Webカメラだけでシャトルの飛来位置を
リアルタイムに推定することを目指しています。

## リポジトリの管理方針

このリポジトリでは、以下の2つの方針でコミット対象を絞っています(詳細は `.gitignore` 参照)。

1. **再現可能なものはコミットしない**
   仮想環境、学習済みモデルの重み、ダウンロード可能なデータセットなどは、スクリプトを
   再実行すれば手元で再構築できるため、リポジトリには含めていません。
2. **秘密情報・個人情報はコミットしない**
   APIキーや、撮影した本人が写り込む可能性のある画像・動画などは、プライバシー・
   セキュリティ上の理由でリポジトリには含めていません。

## ディレクトリ構成

### スクリプト

| ファイル | 説明 |
|---|---|
| `calibrate_camera.py` | `images/calibration/` 内のチェッカーボード画像(内部コーナー9x6、1マス25mm)からカメラキャリブレーションを行い、カメラ行列(fx, fy, cx, cy)・歪み係数・再投影誤差を算出して `camera_calibration.yaml` に保存する |
| `train_yolo.py` | 事前学習済みの `yolov8n.pt` を読み込み、`datasets/merged/data.yaml` を使ってシャトルコック検出モデルを転移学習する(GPU使用、50エポック)。結果は `runs/shuttlecock_baseline/` に保存される |
| `predict_yolo.py` | 学習済みモデル(`runs/shuttlecock_baseline/weights/best.pt`)で `datasets/merged/test/images` に対して推論し、検出結果画像を `runs/predict_test/` に保存する |
| `realtime_detect.py` | `camera_calibration.yaml` の値と学習済みモデルを使い、Webカメラ映像からシャトルコックをリアルタイム検出して距離(X, Y, Z)を推定・表示する。実行結果は `trajectories/` にCSVとして保存される |
| `visualize_trajectory.py` | `trajectories/` 内の軌道CSVを読み込み、3D軌跡・時系列グラフ(X/Y/Z、見かけサイズ、confidence)を生成して `trajectories/plots/` に保存する。異常値(低confidence、大きな座標ジャンプ)の統計もターミナルに表示する |
| `download_datasets.py` | Roboflow上のシャトルコック検出用データセット2種を `datasets/adwproj/`, `datasets/pradyumna/` にダウンロードし(`.env` の `ROBOFLOW_API_KEY` を使用)、続けて両者を `datasets/merged/` に統合する(train/valid/testごとにimages・labelsをコピーし、ファイル名の重複がないか確認、`data.yaml` を生成) |

### 設定・データファイル

| ファイル | 説明 |
|---|---|
| `camera_calibration.yaml` | `calibrate_camera.py` の出力。カメラ行列(fx, fy, cx, cy)、歪み係数、校正時の画像解像度などを含む。`realtime_detect.py` がこれを読み込んで距離推定に使う |
| `requirements.txt` | pipの依存パッケージ一覧(`pip freeze` 形式)。学習・推論に必要な主要パッケージ(torch, ultralytics, opencv-python など)に加え、`download_datasets.py` が使う `python-dotenv` と `roboflow`、およびそれらの依存パッケージも含む |
| `.env` | `ROBOFLOW_API_KEY` を保持する秘密情報ファイル。Git管理外のため、各自作成が必要 |
| `.gitignore` | Git管理から除外するファイル・フォルダのルール |

### .gitignore で除外されているフォルダ

| フォルダ | 内容 |
|---|---|
| `venv/`, `venv_win/` | Python仮想環境(それぞれLinux/WSL用、Windows用)。`requirements.txt` から再構築できるため管理しない |
| `datasets/` | `download_datasets.py` でダウンロードした学習用データセット(`adwproj/`, `pradyumna/`)と、それらを統合した `merged/`。容量が大きく、再取得可能なため管理しない |
| `runs/` | `train_yolo.py` の学習結果(重み `weights/best.pt`、学習ログ `results.csv` など)や `predict_yolo.py` の推論結果画像。再実行すれば生成できるため管理しない |
| `images/` | カメラキャリブレーション用に撮影した画像(`images/calibration/`)。撮影者が写り込む可能性があるため管理しない |
| `trajectories/` | `realtime_detect.py` の実行結果(軌道CSV)と `visualize_trajectory.py` が生成するグラフ(`trajectories/plots/`)。実行のたびに増える計測データのため管理しない |
| `weights/` | `*.pt` にマッチするため除外対象。現時点で `weights/yolo26n.pt` が置かれているが、現行スクリプトからは参照されていない |
| `__pycache__/` | Pythonの実行キャッシュ |

## 新しい環境でのセットアップ手順

### 1. リポジトリをclone

```bash
git clone https://github.com/RK200781/atom-badminton-ai.git
cd atom-badminton-ai
```

### 2. Python仮想環境の作成・有効化

**WSL / Linux の場合:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows の場合:**

```powershell
python -m venv venv_win
venv_win\Scripts\Activate.ps1
```

### 3. ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 4. `.env` ファイルの作成

リポジトリ直下に `.env` ファイルを作成し、自分のRoboflow APIキーを設定してください
(このファイルはGit管理外なので、各自で作成する必要があります)。

```
ROBOFLOW_API_KEY=あなたのAPIキー
```

### 5. データセットの取得

```bash
python download_datasets.py
```

`datasets/adwproj/` と `datasets/pradyumna/` にデータセットがダウンロードされ、続けて
`datasets/merged/`(`train_yolo.py` / `predict_yolo.py` が参照する統合データセット)が
自動的に作成されます。統合処理はスクリプト内で以下を行います。

- train/valid/testそれぞれについて、2つのデータセットのimages・labelsを
  `datasets/merged/<split>/images`, `labels` にコピー
- コピー時にファイル名の重複がないか確認し、重複があれば警告してスキップ
- `datasets/merged/data.yaml`(`nc: 1`, `names: ['shuttlecock']`)を生成

再実行するたびに `datasets/merged/` は作り直されるため、常にダウンロード済みの
最新データセットの内容と一致します。

### 6. モデルの学習(または学習済みモデルの共有)

GPU環境がある場合は、自分で学習できます。

```bash
python train_yolo.py
```

学習結果は `runs/shuttlecock_baseline/weights/best.pt` に保存されます。

GPU環境がない、またはすぐに `realtime_detect.py` を試したい場合は、チームメンバーから
学習済みの `best.pt` を共有してもらい、同じパス `runs/shuttlecock_baseline/weights/best.pt`
に配置してください。

### 7. カメラキャリブレーション

`camera_calibration.yaml` は撮影に使ったカメラの機種・解像度に紐づいた値です。
**別のカメラを使う場合は、必ず再キャリブレーションしてください。**

1. 9x6の内部コーナー(1マス25mm)のチェッカーボードを、使用するカメラで複数枚(20枚前後推奨)撮影する
2. 撮影した画像を `images/calibration/` に配置する
3. 以下を実行する

```bash
python calibrate_camera.py
```

`camera_calibration.yaml` が新しい値で上書き保存されます。

### 8. リアルタイム検出の実行

```bash
python realtime_detect.py
```

**Windows側での実行を推奨します。** WSLはUSBカメラに直接アクセスできないため、
`realtime_detect.py` はWSL上では動作しません(詳しくは次の「既知の注意点」を参照)。

実行中は検出結果がウィンドウに表示され、`q` キーで終了できます。終了時に
`trajectories/shuttle_trajectory_<実行日時>.csv` として軌道データが保存されます。

```bash
# 保存された軌道データを可視化する場合
python visualize_trajectory.py
```

## 既知の注意点

- **開発はWSL、実行はWindowsという役割分担にしています。** WSL環境はUSBカメラ
  (`/dev/video*`)に直接アクセスできないため、`realtime_detect.py` によるリアルタイム
  検出はWindows側(`venv_win`)で実行する必要があります。コードの編集やモデルの学習は
  WSL側で行い、カメラを使う実行だけWindows側で行ってください。
- **カメラを変更した場合は、`calibrate_camera.py` で再キャリブレーションが必要です。**
  `camera_calibration.yaml` の値(fx, fy, cx, cy、歪み係数)は特定のカメラ・解像度に
  対して算出されたものであり、別の機種のカメラにはそのまま使えません。
- `roboflow` パッケージの依存関係により、`opencv-python` に加えて
  `opencv-python-headless` もインストールされ、`numpy` のバージョンが
  やや古いもの(2.3.5)に固定されています。動作確認はこの組み合わせで
  行っていますが、パッケージを更新する際は依存関係の衝突に注意してください。
