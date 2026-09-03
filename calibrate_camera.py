#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
カメラキャリブレーションスクリプト

images/calibration/ 内のチェッカーボード画像から、
カメラ行列(fx, fy, cx, cy)と歪み係数を算出し、
camera_calibration.yaml として保存する。

前提:
- チェッカーボードの内部コーナー数: 9x6(横9個×縦6個)
- 1マスの実測サイズ: 25mm(2.5cm)
"""

from pathlib import Path
import sys
import datetime

import cv2
import numpy as np
import yaml

# ---- 設定値 -----------------------------------------------------------
CHECKERBOARD = (9, 6)  # 内部コーナー数 (横, 縦)
SQUARE_SIZE_MM = 25.0  # 1マスの実測サイズ [mm]

IMAGES_DIR = Path(__file__).resolve().parent / "images" / "calibration"
OUTPUT_YAML = Path(__file__).resolve().parent / "camera_calibration.yaml"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# 再投影誤差が大きすぎる(ブレ・ピント不良など)ため、キャリブレーションから
# 除外する画像ファイル名。前回の実行結果(3px以上)をもとに手動指定。
EXCLUDED_FILENAMES = {
    "WIN_20260903_14_38_39_Pro.jpg",  # 再投影誤差 6.62px
    "WIN_20260903_14_40_17_Pro.jpg",  # 再投影誤差 3.61px
}


def imread_unicode(path: Path):
    """日本語やスペースを含むパスでも安全に画像を読み込む。

    cv2.imread は環境によってはマルチバイトパスを扱えないことがあるため、
    np.fromfile + cv2.imdecode を使う。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"  [エラー] 画像読み込み失敗: {path.name} ({e})")
        return None


def main():
    if not IMAGES_DIR.exists():
        print(f"エラー: 画像フォルダが見つかりません: {IMAGES_DIR}")
        sys.exit(1)

    all_image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not all_image_paths:
        print(f"エラー: {IMAGES_DIR} に画像ファイル(jpg/png)が見つかりません。")
        sys.exit(1)

    image_paths = [p for p in all_image_paths if p.name not in EXCLUDED_FILENAMES]
    excluded_present = [p.name for p in all_image_paths if p.name in EXCLUDED_FILENAMES]

    print(f"対象画像フォルダ: {IMAGES_DIR}")
    print(f"画像ファイル数(全体): {len(all_image_paths)}")
    if excluded_present:
        print(f"手動除外した画像: {len(excluded_present)} 枚")
        for name in excluded_present:
            print(f"  - {name}")
    print(f"キャリブレーション対象の画像ファイル数: {len(image_paths)}")
    print(f"チェッカーボード内部コーナー数: {CHECKERBOARD[0]} x {CHECKERBOARD[1]}")
    print(f"1マスの実測サイズ: {SQUARE_SIZE_MM} mm")
    print("-" * 60)

    # 3D世界座標系でのチェッカーボードのコーナー座標を用意
    # (0,0,0), (1,0,0), ..., (8,5,0) に実寸(mm)を掛ける
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []  # 3D点(実世界座標、複数画像分)
    imgpoints = []  # 2D点(画像上のコーナー座標、複数画像分)

    success_files = []
    failed_files = []
    image_size = None  # (width, height)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            failed_files.append(path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)
        elif (gray.shape[1], gray.shape[0]) != image_size:
            print(f"  [警告] 解像度が他の画像と異なります: {path.name} "
                  f"({gray.shape[1]}x{gray.shape[0]})")

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            success_files.append(path.name)
            print(f"  [OK] {path.name}")
        else:
            failed_files.append(path.name)
            print(f"  [検出失敗] {path.name}")

    print("-" * 60)
    print(f"検出成功: {len(success_files)} 枚 / 検出失敗: {len(failed_files)} 枚")

    if failed_files:
        print("\n検出できなかった画像一覧:")
        for name in failed_files:
            print(f"  - {name}")

    if len(objpoints) < 3:
        print("\nエラー: キャリブレーションに十分な画像がありません"
              "(コーナー検出成功が3枚未満)。")
        sys.exit(1)

    # ---- キャリブレーション実行 ----------------------------------------
    print("\nカメラキャリブレーションを実行中...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    print("\n=== キャリブレーション結果 ===")
    print(f"画像解像度: {image_size[0]} x {image_size[1]}")
    print(f"fx = {fx:.4f}")
    print(f"fy = {fy:.4f}")
    print(f"cx = {cx:.4f}")
    print(f"cy = {cy:.4f}")
    print(f"歪み係数 (k1, k2, p1, p2, k3, ...): {dist_coeffs.ravel().tolist()}")
    print(f"cv2.calibrateCamera が返したRMS再投影誤差: {ret:.4f}")

    # ---- 各画像ごとの再投影誤差(RMS, pixel)を計算 --------------------------
    # cv2.calibrateCamera が返す ret(全体RMS)と同じスケールになるよう、
    # 「二乗誤差の合計 / 点数」の平方根で統一する。
    per_image_errors = []
    total_squared_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints_reproj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        detected = imgpoints[i].reshape(-1, 2)
        reprojected = imgpoints_reproj.reshape(-1, 2)
        diff = detected - reprojected
        squared_error = float(np.sum(diff ** 2))
        n_points = len(reprojected)

        rms_error = np.sqrt(squared_error / n_points)
        per_image_errors.append((success_files[i], float(rms_error)))

        total_squared_error += squared_error
        total_points += n_points

    mean_error = float(np.sqrt(total_squared_error / total_points))

    print("\n=== 画像ごとの再投影誤差 (pixel) ===")
    for name, err in per_image_errors:
        print(f"  {name}: {err:.4f}")
    print(f"\n全体の平均再投影誤差 (RMS): {mean_error:.4f} pixel")

    # ---- 結果をYAMLに保存 ----------------------------------------------
    result = {
        "calibration_date": datetime.datetime.now().isoformat(),
        "checkerboard_inner_corners": {
            "cols": CHECKERBOARD[0],
            "rows": CHECKERBOARD[1],
        },
        "square_size_mm": SQUARE_SIZE_MM,
        "image_resolution": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "num_images_used": len(success_files),
        "num_images_failed": len(failed_files),
        "failed_images": failed_files,
        "manually_excluded_images": excluded_present,
        "camera_matrix": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "full_matrix": camera_matrix.tolist(),
        },
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
        "reprojection_error": {
            "rms_overall": mean_error,
            "cv2_calibrate_camera_ret": float(ret),
            "per_image": [
                {"file": name, "error": err} for name, err in per_image_errors
            ],
        },
    }

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)

    print(f"\n結果を保存しました: {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
