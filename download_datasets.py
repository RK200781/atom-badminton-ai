import os
from dotenv import load_dotenv
from roboflow import Roboflow

# .envファイルからAPIキーを読み込む
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)

# データセット1: AdwProj(911枚)
project1 = rf.workspace("adwproj").project("shuttlecock-yu8ra")
version1 = project1.version(6)
dataset1 = version1.download("yolov8", location="datasets/adwproj")

# データセット2: pradyumna(226枚)
project2 = rf.workspace("pradyumna-r1baa").project("shuttlecock-m9ihi")
version2 = project2.version(1)
dataset2 = version2.download("yolov8", location="datasets/pradyumna")

print("ダウンロード完了")

