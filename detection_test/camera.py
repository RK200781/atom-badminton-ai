# -*- coding: utf-8 -*-
"""
カメラの選択・起動と、キャリブレーション値の読み込み。

integration/camera.py のコピー(ロガー名だけ変更)。検出テスト用のフォルダの中で完結させるため、
import せずにコピーしている。手動版と同じ開き方(DirectShow、MJPG、1280x720・60fps、
解像度が反映されないときの設定し直し・開き直し)で、録画した映像と本番の映像を同じ条件にする。
"""

import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import cv2

import config

log = logging.getLogger("detection_test.camera")


class CameraError(RuntimeError):
    """カメラやキャリブレーションが使えないとき。"""


# ============================================================
# キャリブレーション
# ============================================================

@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """解像度が違うときに、比率で fx, fy, cx, cy をスケーリングする(realtime_detect.py と同じ)。"""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        log.warning("キャリブレーション(%dx%d)とカメラ(%dx%d)の解像度が違うので、比率でスケーリングしました。",
                    self.width, self.height, width, height)
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


def load_calibration(yaml_path: Path = config.CALIBRATION_YAML) -> Intrinsics:
    import yaml
    if not Path(yaml_path).exists():
        raise CameraError(f"キャリブレーションファイルが見つかりません: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        cm = data["camera_matrix"]
        intr = Intrinsics(float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"]),
                          int(data["image_resolution"]["width"]), int(data["image_resolution"]["height"]))
    except (KeyError, TypeError, ValueError) as e:
        raise CameraError(f"{yaml_path} の形式が想定と違います ({e})") from e
    log.info("キャリブレーション値を読み込みました(%dx%d): fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
             intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    return intr


# ============================================================
# カメラの選択
# ============================================================

def _backend():
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def list_camera_device_names():
    """カメラ名の一覧(Windows + pygrabber のときだけ)。取れなければ None。"""
    if platform.system() != "Windows":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        log.info("pygrabber が無いので、カメラ名での自動選択はできません(pip install pygrabber)。")
        return None
    try:
        return FilterGraph().get_input_devices()
    except Exception as e:
        log.warning("カメラ名の一覧を取得できませんでした: %s", e)
        return None


def find_camera_index_by_name(keyword: str):
    names = list_camera_device_names()
    if names is None:
        return None
    for idx, name in enumerate(names):
        log.info("カメラ [%d] %s", idx, name)
    for idx, name in enumerate(names):
        if keyword.lower() in name.lower():
            log.info('"%s" に一致するカメラを選びました: index=%d (%s)', keyword, idx, name)
            return idx
    log.warning('"%s" を含むカメラは見つかりませんでした。', keyword)
    return None


def probe_available_camera_indices(max_index: int):
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, _backend())
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def prompt_user_to_select_camera(candidates, device_names):
    print("使えるカメラが複数見つかりました。使うカメラを選んでください:")
    for idx in candidates:
        label = device_names[idx] if device_names is not None and idx < len(device_names) else f"デバイス {idx}(名前不明)"
        print(f"  [{idx}] {label}")
    while True:
        choice = input("番号を入力してください: ").strip()
        if choice.isdigit() and int(choice) in candidates:
            return int(choice)
        print(f"無効な入力です。次の中から選んでください: {candidates}")


def resolve_camera_index(keyword: str = config.CAMERA_NAME_KEYWORD,
                         max_probe_index: int = config.MAX_CAMERA_PROBE_INDEX) -> int:
    """名前で選ぶ → だめなら 0〜max-1 番を試す → 1台ならそれ、複数ならコンソールで選ばせる。"""
    idx = find_camera_index_by_name(keyword)
    if idx is not None:
        return idx
    log.info("0〜%d 番のカメラを探します...", max_probe_index - 1)
    available = probe_available_camera_indices(max_probe_index)
    if not available:
        raise CameraError("使えるカメラが1台も見つかりませんでした(接続と、他のアプリで使用中でないかを確認)")
    if len(available) == 1:
        log.info("カメラが1台だけ見つかりました。index=%d を使います。", available[0])
        return available[0]
    return prompt_user_to_select_camera(available, list_camera_device_names())


# ============================================================
# カメラを開く
# ============================================================

def _open_capture(device_index: int):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"カメラ(デバイス {device_index})を開けませんでした(他のアプリで使用中でないか確認)")
    return cap


def _configure(cap, width: int, height: int, fps: int) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _frame_size(cap):
    """1フレーム読んで (幅, 高さ) を返す。読めなければ None。"""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    return w, h


def open_camera(device_index: int, width: int = config.CAMERA_WIDTH, height: int = config.CAMERA_HEIGHT,
                fps: int = config.CAMERA_FPS):
    """カメラを開いて解像度などを設定する。(cap, 実際の幅, 実際の高さ) を返す。

    Windows では DirectShow を使う(既定の MSMF だと Elgato のカメラで別のプレビュー窓が出ることがある)。
    MJPG を指定して USB の帯域不足による fps 低下を避け、バッファを1にして遅延が溜まらないようにする。

    DirectShow では解像度の設定が一度で反映されないことがある(同じカメラで 1280x720 になったり
    960x540 になったりした)。違う大きさで開いたら、設定し直す → カメラを開き直す、の順に試し、
    それでもだめならその大きさで動かす(キャリブレーション値は比率でスケーリングされる)。
    """
    cap = _open_capture(device_index)
    _configure(cap, width, height, fps)
    size = _frame_size(cap)

    if size is not None and size != (width, height):
        log.warning("カメラが %dx%d で開いたので、%dx%d に設定し直します。", *size, width, height)
        _configure(cap, width, height, fps)
        size = _frame_size(cap)
    if size is not None and size != (width, height):
        log.warning("まだ %dx%d なので、カメラを開き直します。", *size)
        cap.release()
        cap = _open_capture(device_index)
        _configure(cap, width, height, fps)
        size = _frame_size(cap)

    if size is None:
        cap.release()
        raise CameraError("カメラからフレームを取得できませんでした")
    actual_w, actual_h = size
    if size != (width, height):
        log.warning("カメラを %dx%d にできなかったため、%dx%d で動かします"
                    "(Elgato Camera Hub を閉じる、USB を抜き差しする、で直ることがあります)。",
                    width, height, actual_w, actual_h)
    log.info("カメラを開きました: index=%d %dx%d(カメラが報告する fps: %.1f)",
             device_index, actual_w, actual_h, cap.get(cv2.CAP_PROP_FPS))
    return cap, actual_w, actual_h


def open_video_file(path: Path):
    """動画ファイルをカメラの代わりに開く(実機のカメラなしで確認する用)。"""
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok or frame is None:
        cap.release()
        raise CameraError(f"動画ファイルを開けませんでした: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h, w = frame.shape[:2]
    log.info("動画ファイルをカメラの代わりに使います: %s (%dx%d)", path, w, h)
    return cap, w, h
