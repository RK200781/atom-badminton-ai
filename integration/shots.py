# -*- coding: utf-8 -*-
"""
発射ごとの記録(自動版)。

発射するたびに1行を作り、ターミナルで入力した結果(feedback.py)をその行に書き足す。
全セッション分を shots/shots.csv の1つのファイルにまとめ、学習(aiming.py)はこのファイルから行う。
変更のたびにファイル全体を書き直す(件数は多くても数千なので十分速い)。
"""

import csv
import logging
import math
import os
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

from feedback import Feedback

log = logging.getLogger("launcher.shots")


@dataclass
class Shot:
    shot_id: int
    session: str                 # セッション(起動)ごとの日時
    time: str                    # 発射した時刻
    trigger: str                 # "key"(1 キー)/ "auto"(自動発射)
    speed: int                   # 強さ 1〜3
    yaw_deg: float               # 発射したときの yaw(現在角。分からなければ目標角)
    pitch_deg: float             # 発射したときの pitch(同上。マイナスが上)
    auto_aim: bool               # 自動で狙っていたか
    person_X_cm: Optional[float] = None    # 人の立ち位置(カメラ基準。人がいなければ空)
    person_Z_cm: Optional[float] = None
    person_method: str = ""
    person_distance_cm: Optional[float] = None  # 発射口から人の立ち位置まで(水平)
    person_yaw_deg: Optional[float] = None      # 発射口から見た人の方向
    launch_speed_ms: Optional[float] = None     # このとき使っていた発射の速さ(学習の値)
    yaw_bias_deg: Optional[float] = None        # このとき使っていた yaw のずれ(学習の値)
    predicted_distance_cm: Optional[float] = None  # 計算で予想した飛距離
    video: str = ""              # セッションの録画ファイル名
    video_frame: Optional[int] = None  # 発射した瞬間のフレーム番号(0 始まり)
    # ---- ターミナルで入力した結果 ----
    feedback_text: str = ""
    distance_error_cm: Optional[float] = None   # 人の立ち位置より 奥 +、手前 −
    landing_distance_cm: Optional[float] = None  # 発射口から落ちた位置まで
    lateral_cm: Optional[float] = None           # 人の立ち位置より 右 +、左 −(発射機から見て)

    @property
    def elevation_deg(self) -> float:
        return -self.pitch_deg

    def actual_distance_cm(self) -> Optional[float]:
        """実際に落ちた位置までの距離(発射口から)。分からなければ None。"""
        if self.landing_distance_cm is not None:
            return self.landing_distance_cm
        if self.distance_error_cm is not None and self.person_distance_cm is not None:
            return self.person_distance_cm + self.distance_error_cm
        return None

    def actual_direction_deg(self) -> Optional[float]:
        """実際に落ちた方向(発射口から見た yaw)。分からなければ None。

        左右のずれは、人がいたときは人の立ち位置から、いなかったときは発射機が向いていた方向からのずれとする。
        """
        if self.lateral_cm is None:
            return None
        reference = self.person_yaw_deg if self.person_yaw_deg is not None else self.yaw_deg
        distance = self.actual_distance_cm() or self.person_distance_cm
        if not distance or distance <= 0:
            return None
        return reference + math.degrees(math.atan2(self.lateral_cm, distance))


_FIELDS = [f.name for f in fields(Shot)]
_TYPES = {f.name: str(f.type) for f in fields(Shot)}   # 例 "<class 'int'>"、"typing.Optional[float]"


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
    return str(value)


def _from_text(name: str, text: str):
    t = _TYPES[name]
    if text == "":
        return "" if t == "<class 'str'>" else None
    if "bool" in t:
        return text == "1"
    if "int" in t:
        return int(float(text))
    if "float" in t:
        return float(text)
    return text


class ShotLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._shots: List[Shot] = []
        self._undo_stack = []   # (shot_id, 入力前の値)
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    self._shots.append(Shot(**{k: _from_text(k, row.get(k, "")) for k in _FIELDS}))
                except Exception as e:
                    log.warning("発射の記録の1行を読めませんでした(飛ばします): %s", e)
        log.info("発射の記録を読み込みました: %d 件(%s)", len(self._shots), self.path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(_FIELDS)
            for s in self._shots:
                w.writerow([_to_text(getattr(s, k)) for k in _FIELDS])
        os.replace(tmp, self.path)

    # ---- 操作 ------------------------------------------------------------------

    def next_id(self) -> int:
        with self._lock:
            return max((s.shot_id for s in self._shots), default=0) + 1

    def add(self, shot: Shot) -> None:
        with self._lock:
            self._shots.append(shot)
            self._save()

    def all(self) -> List[Shot]:
        with self._lock:
            return list(self._shots)

    def get(self, shot_id: Optional[int]) -> Optional[Shot]:
        """番号の発射。None なら一番新しい発射。"""
        with self._lock:
            if not self._shots:
                return None
            if shot_id is None:
                return self._shots[-1]
            return next((s for s in self._shots if s.shot_id == shot_id), None)

    def apply_feedback(self, shot: Shot, fb: Feedback) -> None:
        """入力した結果を書き足す。入力に含まれている項目だけ上書きする(続けて入力して足していける)。"""
        with self._lock:
            before = {k: getattr(shot, k) for k in
                      ("feedback_text", "distance_error_cm", "landing_distance_cm", "lateral_cm")}
            self._undo_stack.append((shot.shot_id, before))
            shot.feedback_text = f"{shot.feedback_text} / {fb.text}" if shot.feedback_text else fb.text
            if fb.landing_distance_cm is not None:
                shot.landing_distance_cm = fb.landing_distance_cm
                shot.distance_error_cm = None
            if fb.distance_error_cm is not None:
                shot.distance_error_cm = fb.distance_error_cm
                shot.landing_distance_cm = None
            if fb.lateral_cm is not None:
                shot.lateral_cm = fb.lateral_cm
            self._save()

    def undo(self) -> Optional[int]:
        """直前の入力を取り消す。取り消した発射の番号(無ければ None)。"""
        with self._lock:
            if not self._undo_stack:
                return None
            shot_id, before = self._undo_stack.pop()
            shot = next((s for s in self._shots if s.shot_id == shot_id), None)
            if shot is None:
                return None
            for k, v in before.items():
                setattr(shot, k, v)
            self._save()
            return shot_id
