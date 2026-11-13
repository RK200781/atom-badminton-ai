# 検出テスト(detection_test/)

離れた位置の小さいシャトルを YOLO がどれだけ検出できるかを、本番と同じカメラの映像で確かめるためのツール。
手動版(`integration/`)とは別のフォルダで、カメラだけを使う(発射機はつながなくてよい)。

1. `record.py` で、本番と同じ設定のカメラ映像を、何も描き込まずに録画する
2. `compare.py` で、その録画に対して YOLO の設定(imgsz と信頼度のしきい値)を変えながら検出率を比べる
3. 結果を見て、手動版の設定を変えるか、モデルを学習し直すかを決める

## ファイル

| ファイル | 内容 | 元にしたもの |
|---|---|---|
| `record.py` | 録画ツール | 新規 |
| `compare.py` | 設定ごとの検出率の比較 | 新規 |
| `camera.py` | カメラの選択・起動(1280x720・60fps、DirectShow、MJPG、解像度の設定し直し・開き直し) | `integration/camera.py` のコピー(ロガー名だけ変更) |
| `config.py` | 設定値 | 新規(カメラの値は `integration/config.py` と同じ) |
| `tests/` | テスト | 新規 |

- コードはこのフォルダの中で完結している(外のコードは import しない)。
- 学習済みモデル `runs/shuttlecock_v2/weights/best.pt` は、コピーせず元の場所を参照する(`config.YOLO_MODEL_PATH`)。
- 録画(`recordings/`)と結果(`results/`)は Git の管理外(人が映るため)。

## 1. 録画する(Windows の PowerShell)

```powershell
cd \\wsl.localhost\Ubuntu-22.04\home\kr\atom-badminton-ai
venv_win\Scripts\Activate.ps1
python detection_test\record.py --label 2m
```

- 録画ウィンドウが前面のときに **V かスペースで録画の開始・停止**、**Q で終了**。押すたびに別のファイルになる。
- `--label` はファイル名に付く目印(`2m`、`3m`、`flight` など)。場面ごとに起動し直してラベルを変えると、後で比べやすい。
- 保存先:`detection_test\recordings\rec_<日時>_<ラベル>.mp4` と、各フレームの時刻 `..._timestamps.csv`。
- 手動版(`integration\main.py`)が動いているとカメラを開けないので、先に終了しておく。
- 起動時のログに `カメラを開きました: index=0 1280x720` と出ることを確認する(本番と同じ解像度で撮るため)。

**撮っておくとよい場面**(1本あたり10〜20秒で十分)
- カメラから 2m・3m・4m の位置で、シャトルを手に持って立つ(人とシャトルが同時に映る場面)
- 発射機からシャトルを飛ばす
- シャトルが映っていない場面(誤検出がどれくらい出るかを見るため)

## 2. 比べる

```powershell
python detection_test\compare.py                                   # recordings\ の動画をすべて
python detection_test\compare.py detection_test\recordings\rec_X.mp4   # 指定した動画だけ
python detection_test\compare.py --imgsz 640 1280 --conf 0.25 0.1     # 比べる設定を変える
python detection_test\compare.py --every 3                         # 3フレームに1枚だけ使う(速く終わる)
```

既定では imgsz(YOLO が処理する大きさ)を 640・960・1280、信頼度のしきい値を 0.25・0.15 にして、
その全部の組み合わせ(6通り)を比べる。手動版は今 imgsz=640・しきい値 0.25。

**出力の例**

```
■ rec_20261113_191356_2m.mp4(9 秒)
  設定                       検出率   検出/全体  枠(中央値)    0-30px   30-60px  60-120px    120px-       YOLO
  imgsz=640 conf=0.25        42.5%     68/160        11px        64         0         0         4    11.1 ms
  imgsz=1280 conf=0.15       76.9%    123/160         8px       111         0         8         4    14.2 ms
  設定によって結果が違ったフレーム: 83
```

- **検出率**:シャトルらしいものが1つ以上見つかったフレームの割合。正解のラベルが無いので、**誤検出も数に入る**。
- **0-30px などの列**:見つかった枠の大きさ(長い辺)ごとの数。小さいシャトルがどれだけ取れているかが分かる。
- **YOLO**:1枚あたりの処理時間。本番のカメラは 60fps(1枚 16.7ms)なので、これより十分小さければよい
  (人の検出などもあるので、目安は 12ms 程度まで)。
- 結果は `detection_test\results\<日時>\` に保存する。
  - `summary.csv`:動画ごと・設定ごとの集計
  - `diff\`:設定によって検出できたりできなかったりしたフレームの画像(最大30枚/本。`--save-diff` で変更)。
    各設定の枠が色分けされ、左上にどの設定で見つかったか(信頼度)が書いてある。
    **しきい値を下げて増えた分が本物のシャトルか、誤検出かを、ここで目で確かめる。**

## 結果の使い方

| 結果 | 次にすること |
|---|---|
| imgsz を上げる、またはしきい値を下げると、小さいシャトルが本物として取れるようになる | 手動版の `integration/config.py` の設定を変える(imgsz の設定は手動版にまだ無いので追加が必要) |
| しきい値を下げると誤検出ばかり増える、または imgsz を上げても取れない | このカメラで撮った画像を学習データに加えて、学習し直す(録画をそのまま画像の素材にできる) |

## テスト(カメラ・GPU 不要)

```bash
python -m unittest discover -s detection_test/tests
```
