#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キャリブレーション用チェッカーボード画像の撮影スクリプト

realtime_detect.py / realtime_pose_detect.py と全く同じ設定(カメラ選択、
DirectShow、MJPG、1280x720・60fps)でカメラを開き、チェッカーボードを撮影して
images/calibration_720p/ に保存する。

Windowsのカメラアプリなど別の方法で撮影すると、検出時とは別のモード
(解像度・画角が異なる)で撮影されることがあり、fx などがずれる原因になる。
検出時と同じ映像でキャリブレーションするために、撮影は必ずこのスクリプトで行う。

操作:
  スペース : 現在のフレームを保存する(チェッカーボードが見つかっている時のみ)
  q / Esc  : 終了

撮影後は以下でキャリブレーションする:
  python calibrate_camera.py --images-dir images/calibration_720p
"""

from pathlib import Path
import argparse
import datetime

import cv2
import numpy as np

from calibrate_camera import CHECKERBOARD
from realtime_detect import (
    CAMERA_NAME_KEYWORD,
    DESIRED_FPS,
    DESIRED_HEIGHT,
    DESIRED_WIDTH,
    MAX_CAMERA_PROBE_INDEX,
    open_camera,
    resolve_camera_index,
)

# ---- 設定値 -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "images" / "calibration_720p"

WINDOW_NAME = "Calibration Capture"

# プレビュー中のチェッカーボード検出は毎フレーム行うと重いため、
# 縮小画像で何フレームかに1回だけ行う(保存する画像は縮小しない)
PREVIEW_DETECT_INTERVAL = 3
PREVIEW_DETECT_SCALE = 0.5

TARGET_IMAGE_COUNT = 20  # README手順7の推奨枚数


def find_board_preview(frame):
    """プレビュー用に縮小画像でチェッカーボードを探す。見つかれば元画像座標のコーナーを返す。"""
    small = cv2.resize(frame, None, fx=PREVIEW_DETECT_SCALE, fy=PREVIEW_DETECT_SCALE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None
    return corners / PREVIEW_DETECT_SCALE


def save_image(path: Path, frame) -> bool:
    """日本語やネットワークパス(\\\\wsl.localhost\\...)でも保存できるよう imencode + tofile で書き出す。"""
    ok, buf = cv2.imencode(path.suffix, frame)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="検出スクリプトと同じカメラ設定でキャリブレーション用画像を撮影する",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"画像の保存先(デフォルト: {DEFAULT_OUTPUT_DIR.relative_to(BASE_DIR)})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_index = resolve_camera_index(CAMERA_NAME_KEYWORD, MAX_CAMERA_PROBE_INDEX)
    cap, width, height = open_camera(camera_index, DESIRED_WIDTH, DESIRED_HEIGHT, DESIRED_FPS)
    print(f"カメラ解像度: {width}x{height}")
    if (width, height) != (DESIRED_WIDTH, DESIRED_HEIGHT):
        print("[警告] 検出スクリプトが要求する解像度と異なります。"
              "検出時も同じ解像度にフォールバックしているか確認してください。")

    existing = sorted(output_dir.glob("*.png"))
    print(f"保存先: {output_dir}(既存 {len(existing)} 枚)")
    print("スペースで保存、q / Esc で終了します。"
          "反応しない場合は映像ウィンドウをクリックしてから押してください。")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    saved_count = len(existing)
    saved_outlines = []  # 撮影済みのボード位置(画面の四隅まで撮れているか確認する用)
    corners = None
    frame_count = 0
    flash_until = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[警告] フレームを取得できませんでした。終了します。")
                break

            if frame_count % PREVIEW_DETECT_INTERVAL == 0:
                corners = find_board_preview(frame)
            frame_count += 1

            # 保存するのは描画前のフレーム
            display = frame.copy()
            for outline in saved_outlines:
                cv2.polylines(display, [outline], isClosed=True, color=(255, 200, 0), thickness=1)
            if corners is not None:
                cv2.drawChessboardCorners(display, CHECKERBOARD, corners.astype(np.float32), True)

            status = "board: FOUND (space to save)" if corners is not None else "board: not found"
            color = (0, 255, 0) if corners is not None else (0, 0, 255)
            lines = [
                status,
                f"saved: {saved_count} / {TARGET_IMAGE_COUNT}  ({width}x{height})",
            ]
            for i, line in enumerate(lines):
                y = 40 + 30 * i
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            now = cv2.getTickCount() / cv2.getTickFrequency()
            if now < flash_until:
                cv2.rectangle(display, (0, 0), (width - 1, height - 1), (255, 255, 255), 12)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
                break
            if key == ord(" "):
                if corners is None:
                    print("[情報] チェッカーボードが見つかっていないため保存しませんでした。")
                    continue
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = output_dir / f"calib_{width}x{height}_{timestamp}.png"
                if save_image(path, frame):
                    saved_count += 1
                    flash_until = now + 0.2
                    hull = cv2.convexHull(corners.astype(np.int32))
                    saved_outlines.append(hull)
                    print(f"  保存しました ({saved_count} 枚目): {path.name}")
                else:
                    print(f"[エラー] 保存に失敗しました: {path}")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\n[情報] Ctrl+Cが押されたため終了します。")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n撮影済み: {saved_count} 枚({output_dir})")
    print("次のコマンドでキャリブレーションできます:")
    print(f"  python calibrate_camera.py --images-dir {output_dir}")


if __name__ == "__main__":
    main()
