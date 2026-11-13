# -*- coding: utf-8 -*-
"""
シャトルの検出(YOLO)と人の検出(MediaPipe Pose)。

- シャトル: realtime_detect.py のモデル読み込み・推論・距離推定・描画を書き直したもの。
  軌道の予測(trajectory_predictor.py)は入れていない。
- 人: realtime_pose_detect.py の MediaPipe の準備・推論・立ち位置の推定(estimate_floor_position)・
  描画を書き直したもの。launch_control(発射の判断)は入れていない。
"""

import logging
import math
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from camera import CameraError, Intrinsics

log = logging.getLogger("launcher.vision")

# MediaPipe Pose のランドマーク番号(mp_vision.PoseLandmark の値と同じ)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# YOLO が結果の保存先を決めるときに、ネットワークパス越しにならないよう一時フォルダを指定する
# (実際には保存しない)。realtime_detect.py と同じ。
YOLO_SCRATCH_DIR = Path(tempfile.gettempdir()) / "atom_badminton_yolo_runs"


# ============================================================
# シャトル
# ============================================================

@dataclass
class ShuttleObservation:
    timestamp: float   # time.time()(CSV 用)
    X: float           # [cm] カメラ座標系(右 +X、下 +Y、奥 +Z)
    Y: float
    Z: float
    u: float           # 枠の中心 [px]
    v: float
    w: float           # 枠の幅・高さ [px]
    h: float
    confidence: float

    def csv_row(self):
        return (self.timestamp, self.X, self.Y, self.Z, self.u, self.v, self.w, self.h, self.confidence)


SHUTTLE_CSV_HEADER = ["timestamp", "X_cm", "Y_cm", "Z_cm", "u_px", "v_px", "width_px", "height_px", "confidence"]


def shuttle_position(box, intr: Intrinsics) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """枠 (x1, y1, x2, y2) から (X, Y, Z, u, v, w, h) を求める。realtime_detect.py と同じ計算。

    Z = f × 実際の直径 / 見かけの大きさ(幅・高さの大きい方)、X・Y は逆投影。
    """
    x1, y1, x2, y2 = (float(a) for a in box)
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    size = max(w, h)
    if size <= 0:
        return None
    f_avg = (intr.fx + intr.fy) / 2.0
    Z = f_avg * config.SHUTTLE_DIAMETER_CM / size
    X = (u - intr.cx) * Z / intr.fx
    Y = (v - intr.cy) * Z / intr.fy
    return X, Y, Z, u, v, w, h


class ShuttleDetector:
    def __init__(self, model_path: Path = config.YOLO_MODEL_PATH):
        if not Path(model_path).exists():
            raise CameraError(f"YOLO のモデルが見つかりません: {model_path}(train_yolo.py で学習して用意する)")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise CameraError("ultralytics が入っていません(pip install ultralytics)") from e
        log.info("YOLO のモデルを読み込んでいます: %s", model_path)
        self._model = YOLO(str(model_path))
        try:
            import torch
            self._use_cuda = torch.cuda.is_available()
        except ImportError:
            self._use_cuda = False
        if self._use_cuda:
            self._model.to("cuda")
            log.info("YOLO は GPU で推論します: %s", torch.cuda.get_device_name(0))
        else:
            log.info("CUDA が使えないため、YOLO は CPU で推論します(fps が下がることがあります)。")

    def detect(self, frame):
        """(枠の配列 N×4, 信頼度の配列 N) を返す。"""
        result = self._model.predict(
            frame, conf=config.YOLO_CONFIDENCE, verbose=False,
            device=0 if self._use_cuda else "cpu",
            project=str(YOLO_SCRATCH_DIR), name="realtime", exist_ok=True,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


def draw_shuttle(frame, boxes, confs, best: Optional[ShuttleObservation], best_box=None) -> None:
    """全部の枠を緑、採用した枠を赤で描き、位置を左上に出す(realtime_detect.py と同じ見た目)。"""
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if best is None:
        return
    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    lines = [
        f"shuttle u,v=({best.u:.1f},{best.v:.1f})  w,h=({best.w:.1f},{best.h:.1f})px",
        f"X={best.X:.1f}cm Y={best.Y:.1f}cm Z={best.Z:.1f}cm  conf={best.confidence:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# ============================================================
# 人
# ============================================================

def estimate_position_on_plane(u, v, fx, fy, cx, cy, plane_height_cm,
                               camera_height_cm=None, camera_tilt_deg=None):
    """画像上の点 (u, v) を、床から plane_height_cm の高さの水平面に投影して、大まかな (X, Z) を求める。

    realtime_pose_detect.py の estimate_floor_position()(床=高さ0 に投影)を、任意の高さの面に広げたもの。
    カメラの設置高さと俯角が分かっている前提で、カメラから (u, v) 方向の視線がその面と交わる点を求める。
    面がカメラより低ければ下向きの視線、高ければ上向きの視線だけが交わる。
    戻り値: (X_cm, Z_cm, valid)。交わらないときは (None, None, False)。
    """
    if camera_height_cm is None:
        camera_height_cm = config.CAMERA_MOUNT_HEIGHT_CM
    if camera_tilt_deg is None:
        camera_tilt_deg = config.CAMERA_TILT_DEG
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    tilt = math.radians(camera_tilt_deg)
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    x_world = ray[0]
    y_world = -(ray[1] * cos_t + ray[2] * sin_t)   # 上向きが正
    z_world = -ray[1] * sin_t + ray[2] * cos_t
    if abs(y_world) <= 1e-9:
        return None, None, False
    t = (plane_height_cm - camera_height_cm) / y_world
    if t <= 0:
        return None, None, False
    return float(t * x_world), float(t * z_world), True


def estimate_floor_position(u, v, fx, fy, cx, cy, camera_height_cm=None, camera_tilt_deg=None):
    """足首の中点 (u, v) を床に投影する(realtime_pose_detect.py の estimate_floor_position() と同じ結果)。"""
    return estimate_position_on_plane(u, v, fx, fy, cx, cy, 0.0, camera_height_cm, camera_tilt_deg)


METHOD_ANKLE = "ankle"        # 足首の中点を床に投影した
METHOD_SHOULDER = "shoulder"  # 肩の中点を、肩の高さ(SHOULDER_HEIGHT_CM と仮定)の面に投影した


@dataclass
class PersonObservation:
    timestamp: float   # time.time()(CSV 用)
    u: float           # 使った点(足首か肩の中点)[px]
    v: float
    X: float           # 立ち位置 [cm](カメラが基準。右 +X、奥 +Z)
    Z: float
    method: str = METHOD_ANKLE

    def csv_row(self):
        return (self.timestamp, self.u, self.v, self.X, self.Z, self.method)


# 先頭の5列は realtime_pose_detect.py と同じ。method が shoulder の行では、u・v は肩の中点。
PERSON_CSV_HEADER = ["timestamp", "ankle_mid_u_px", "ankle_mid_v_px", "X_cm", "Z_cm", "method"]

# 立ち位置を出せなかった理由(画面表示用)
PERSON_NONE = "人を検出していません"
PERSON_NOT_VISIBLE = "足首も肩もはっきり見えていません"
PERSON_NO_INTERSECTION = "足首・肩を通る視線から位置を出せません(カメラの向きを確認)"


def _midpoint_if_reliable(landmarks, left_idx, right_idx, vis_threshold, presence_threshold, width, height):
    left, right = landmarks[left_idx], landmarks[right_idx]
    if not all(lm.visibility >= vis_threshold and lm.presence >= presence_threshold for lm in (left, right)):
        return None
    return (left.x + right.x) / 2.0 * width, (left.y + right.y) / 2.0 * height


def person_position(landmarks, width: int, height: int, intr: Intrinsics, timestamp: float):
    """骨格から立ち位置を求める。(PersonObservation か None, 出せなかった理由か None) を返す。

    1. 両足首がはっきり見えていて(visibility・presence がしきい値以上。realtime_pose_detect.py と同じ)、
       床と交わるなら、足首の中点を床に投影する。
    2. そうでなければ、両肩がはっきり見えていれば、肩の中点を肩の高さ(SHOULDER_HEIGHT_CM と仮定)の
       水平面に投影する。動いているうちに足首を見失ったときや、カメラが上向きで足首が映らないとき用。
    """
    if landmarks is None:
        return None, PERSON_NONE
    seen_any = False
    candidates = (
        (METHOD_ANKLE, LEFT_ANKLE, RIGHT_ANKLE,
         config.ANKLE_VISIBILITY_THRESHOLD, config.ANKLE_PRESENCE_THRESHOLD, 0.0),
        (METHOD_SHOULDER, LEFT_SHOULDER, RIGHT_SHOULDER,
         config.SHOULDER_VISIBILITY_THRESHOLD, config.SHOULDER_PRESENCE_THRESHOLD, config.SHOULDER_HEIGHT_CM),
    )
    for method, left_idx, right_idx, vis, presence, plane_height in candidates:
        mid = _midpoint_if_reliable(landmarks, left_idx, right_idx, vis, presence, width, height)
        if mid is None:
            continue
        seen_any = True
        u, v = mid
        X, Z, valid = estimate_position_on_plane(u, v, intr.fx, intr.fy, intr.cx, intr.cy, plane_height)
        if valid:
            return PersonObservation(timestamp, u, v, X, Z, method), None
    return None, (PERSON_NO_INTERSECTION if seen_any else PERSON_NOT_VISIBLE)


def ensure_pose_model(model_path: Path = config.POSE_MODEL_PATH, url: str = config.POSE_MODEL_URL) -> None:
    if Path(model_path).exists():
        return
    log.info("MediaPipe のモデルが無いのでダウンロードします: %s", url)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        raise CameraError(f"MediaPipe のモデルをダウンロードできませんでした({e})。{url} を {model_path} に置いてください") from e


class PoseDetector:
    def __init__(self, model_path: Path = config.POSE_MODEL_PATH):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            raise CameraError("mediapipe が入っていません(pip install mediapipe)") from e
        ensure_pose_model(model_path)
        assert mp_vision.PoseLandmark.LEFT_ANKLE == LEFT_ANKLE and mp_vision.PoseLandmark.RIGHT_ANKLE == RIGHT_ANKLE
        self._mp = mp
        self._vision = mp_vision
        self._connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        self._style = mp_vision.drawing_styles.get_default_pose_landmarks_style()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        log.info("MediaPipe Pose を準備しました: %s", model_path)

    def detect(self, frame, timestamp_ms: int):
        """1人分の骨格(ランドマークのリスト)か None を返す。timestamp_ms は単調増加であること。"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def draw(self, frame, landmarks) -> None:
        self._vision.drawing_utils.draw_landmarks(frame, landmarks, self._connections,
                                                  landmark_drawing_spec=self._style)

    def close(self) -> None:
        self._landmarker.close()


def draw_person(frame, landmarks, person: Optional[PersonObservation], note: Optional[str]) -> None:
    """使った点(足首か肩)と立ち位置を描き、位置をシャトルの表示の下に出す。"""
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    if person is not None:
        pair = (LEFT_ANKLE, RIGHT_ANKLE) if person.method == METHOD_ANKLE else (LEFT_SHOULDER, RIGHT_SHOULDER)
        for idx in pair:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 8, (0, 255, 255), -1)
        cv2.circle(frame, (int(person.u), int(person.v)), 10, (0, 0, 255), -1)
        source = ("ankles" if person.method == METHOD_ANKLE
                  else f"shoulders, assumed {config.SHOULDER_HEIGHT_CM:.0f}cm high")
        lines = [f"person u,v=({person.u:.1f},{person.v:.1f})px ({source})",
                 f"standing pos (approx): X={person.X:.1f}cm Z={person.Z:.1f}cm"]
    elif note == PERSON_NOT_VISIBLE:
        lines = ["standing pos: N/A (ankles/shoulders not reliably visible)"]
    else:
        lines = ["standing pos: N/A (cannot project ankles/shoulders)"]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 110 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
