# -*- coding: utf-8 -*-
"""
シャトル発射機の「判断と行動」部分。

- config.py          : 設定値(カメラ取り付け位置、可動範囲、しきい値、仰角→距離の対応表)
- aiming.py          : カメラ座標 → 発射機座標の変換と、パン・チルト角の計算
- launcher.py        : 発射機ハードウェアの抽象化(MockLauncher / AK45Launcher)
- fire_controller.py : 人の立ち位置から「いつ・どこへ撃つか」を決める状態機械
- overlay.py         : 状態をカメラ映像に重ねて表示する(cv2を使うのはここだけ)
- demo_sim.py        : カメラ・実機なしで一連の流れを確認するシミュレーション
"""
