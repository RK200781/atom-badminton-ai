# -*- coding: utf-8 -*-
"""
自動版の中心:人の立ち位置に向けて狙い、(オンなら)自動で発射し、発射ごとに記録して、
ターミナルで入力された結果から学習し直す。

main ループ(auto_main.py)から UI の周期ごとに tick() を呼ぶ。モーターや Arduino への指令は
手動版と同じ Launcher(aim / fire)を通すので、手動版の安全の仕組み(発射の条件、停止、可動範囲)はそのまま効く。
"""

import collections
import datetime
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import config
from aiming import AimModel, AimSolution
from feedback import HELP_TEXT, parse_command
from shots import Shot, ShotLog

log = logging.getLogger("launcher.auto")

REASON_AIM_OFF = "自動狙いがオフ(M キー)"
REASON_PERSON_LOST = "人を見失っています"
REASON_PERSON_MOVING = "人が動いています"
REASON_NO_PITCH = "強さが 0 なので pitch が決まっていません"
REASON_OUT_OF_RANGE = "届かない・近すぎる距離です"
REASON_NOT_SETTLED = "狙いを合わせている途中です"
REASON_MOTORS_STOPPED = "モーターの保持ループが止まっています"


@dataclass
class AutoStatus:
    aim_enabled: bool
    fire_enabled: bool
    person: Optional[tuple]              # ならした立ち位置 (X, Z) [cm]。見失っていれば None
    person_age_sec: Optional[float]      # 最後に人を見てからの秒数
    solution: Optional[AimSolution]
    settled: bool
    auto_fire_reasons: List[str]         # 自動発射しない理由(空なら条件がそろっている)
    last_shot: Optional[Shot]
    model_text: str
    video_name: Optional[str]
    recent_shots: List[Shot] = field(default_factory=list)


class AutoController:
    def __init__(self, launcher, shot_log: ShotLog, model: Optional[AimModel] = None, session: Optional[str] = None,
                 video=None, clock=time.monotonic):
        self.launcher = launcher
        self.shot_log = shot_log
        self.model = model or AimModel()
        self.session = session or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video = video
        self._clock = clock
        self.aim_enabled = False
        self.fire_enabled = False
        self._history = collections.deque()   # (時刻, X, Z)
        self._last_person_ts = None
        self._last_method = ""
        self._last_seen = None
        self._last_fire = None
        self._solution: Optional[AimSolution] = None
        self._smoothed = None
        self.model.learn(self.shot_log.all())
        log.info("学習した値: %s", self.model.describe())

    # ---- モードの切り替え ----------------------------------------------------------

    def toggle_aim(self) -> None:
        self.aim_enabled = not self.aim_enabled
        if not self.aim_enabled:
            self.fire_enabled = False
        log.info("自動狙い: %s", "オン" if self.aim_enabled else "オフ")

    def toggle_fire(self) -> None:
        if not self.fire_enabled and not self.aim_enabled:
            log.warning("自動発射は、自動狙い(M キー)をオンにしてから使えます。")
            return
        self.fire_enabled = not self.fire_enabled
        log.info("自動発射: %s", "オン" if self.fire_enabled else "オフ")

    def disable(self, why: str) -> None:
        if self.aim_enabled or self.fire_enabled:
            self.aim_enabled = self.fire_enabled = False
            log.info("自動狙い・自動発射をオフにしました(%s)。", why)

    # ---- 毎周期 ----------------------------------------------------------------

    def _update_person(self, person, now: float) -> None:
        if person is not None and person.timestamp != self._last_person_ts:
            self._last_person_ts = person.timestamp
            self._last_seen = now
            self._last_method = person.method
            self._history.append((now, person.X, person.Z))
        keep = max(config.PERSON_SMOOTHING_SEC, config.AUTO_FIRE_STABLE_SEC) + 0.5
        while self._history and now - self._history[0][0] > keep:
            self._history.popleft()
        lost = self._last_seen is None or now - self._last_seen > config.PERSON_LOST_SEC
        recent = [(x, z) for t, x, z in self._history if now - t <= config.PERSON_SMOOTHING_SEC]
        if lost or not recent:
            self._smoothed = None
        else:
            self._smoothed = (statistics.median(x for x, _ in recent), statistics.median(z for _, z in recent))

    def _person_stable(self, now: float) -> bool:
        window = [(t, x, z) for t, x, z in self._history if now - t <= config.AUTO_FIRE_STABLE_SEC]
        # 窓の大部分をカバーするだけの記録があること(見つけた直後は「止まっている」と言えない)
        if not window or now - window[0][0] < config.AUTO_FIRE_STABLE_SEC * 0.8:
            return False
        xs, zs = [w[1] for w in window], [w[2] for w in window]
        return math.hypot(max(xs) - min(xs), max(zs) - min(zs)) <= config.AUTO_FIRE_STABLE_CM

    @staticmethod
    def _settled(motors) -> bool:
        if motors.current_yaw is None or motors.current_pitch is None:
            return False
        return (abs(motors.current_yaw - motors.target_yaw) <= config.AIM_SETTLED_DEG
                and abs(motors.current_pitch - motors.target_pitch) <= config.AIM_SETTLED_DEG)

    def tick(self, person) -> AutoStatus:
        """person: カメラの最新の立ち位置(PersonObservation。古い・無いときは None)。"""
        now = self._clock()
        self._update_person(person, now)
        status = self.launcher.status()
        speed = status.feeder.speed
        self._solution = self.model.solve(*self._smoothed, speed) if self._smoothed else None

        sol = self._solution
        if self.aim_enabled and sol is not None and status.motors.running:
            pitch = sol.pitch_deg if sol.pitch_deg is not None else status.motors.target_pitch
            if (abs(sol.yaw_deg - status.motors.target_yaw) > config.AIM_DEADBAND_DEG
                    or abs(pitch - status.motors.target_pitch) > config.AIM_DEADBAND_DEG):
                self.launcher.aim(sol.yaw_deg, pitch)
                status = self.launcher.status()

        settled = self._settled(status.motors)
        reasons = self._auto_fire_reasons(now, status, sol, settled)
        if self.fire_enabled and not reasons:
            self.fire("auto")
        return self.status(now, settled, reasons)

    def _auto_fire_reasons(self, now, status, sol, settled) -> List[str]:
        reasons = []
        if not self.aim_enabled:
            reasons.append(REASON_AIM_OFF)
        if not status.motors.running:
            reasons.append(REASON_MOTORS_STOPPED)
        if self._smoothed is None:
            reasons.append(REASON_PERSON_LOST)
        elif not self._person_stable(now):
            reasons.append(REASON_PERSON_MOVING)
        if sol is not None:
            if sol.pitch_deg is None:
                reasons.append(REASON_NO_PITCH)
            elif any("届きません" in w or "近すぎ" in w for w in sol.warnings):
                reasons.append(REASON_OUT_OF_RANGE)
        if self.aim_enabled and not settled:
            reasons.append(REASON_NOT_SETTLED)
        if self._last_fire is not None and now - self._last_fire < config.AUTO_FIRE_INTERVAL_SEC:
            reasons.append(f"前の発射から {config.AUTO_FIRE_INTERVAL_SEC:.0f} 秒たっていません")
        reasons.extend(status.fire_reasons)
        return reasons

    def status(self, now=None, settled=None, reasons=None) -> AutoStatus:
        now = self._clock() if now is None else now
        shots = [s for s in self.shot_log.all() if s.session == self.session]
        return AutoStatus(
            aim_enabled=self.aim_enabled,
            fire_enabled=self.fire_enabled,
            person=self._smoothed,
            person_age_sec=None if self._last_seen is None else now - self._last_seen,
            solution=self._solution,
            settled=bool(settled),
            auto_fire_reasons=list(reasons or []),
            last_shot=shots[-1] if shots else None,
            model_text=self.model.describe(),
            video_name=self.video.name if self.video is not None and self.video.recording else None,
            recent_shots=shots[-4:],
        )

    # ---- 発射 ------------------------------------------------------------------

    def fire(self, trigger: str) -> Optional[Shot]:
        """発射して、成功したら記録する。trigger: "key"(1 キー)/ "auto"。"""
        status = self.launcher.status()
        frame = self.video.frame_count if self.video is not None and self.video.recording else None
        ok, _ = self.launcher.fire()
        if not ok:
            if trigger == "auto":  # 条件は確かめてあるはずだが、念のため連続で試さないようにする
                self._last_fire = self._clock()
            return None
        self._last_fire = self._clock()
        m = status.motors
        speed = status.feeder.speed
        yaw = m.current_yaw if m.current_yaw is not None else m.target_yaw
        pitch = m.current_pitch if m.current_pitch is not None else m.target_pitch
        shot = Shot(
            shot_id=self.shot_log.next_id(), session=self.session,
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            trigger=trigger, speed=speed, yaw_deg=yaw, pitch_deg=pitch, auto_aim=self.aim_enabled,
            launch_speed_ms=self.model.state.speeds_ms.get(speed),
            yaw_bias_deg=self.model.state.yaw_bias_deg,
            predicted_distance_cm=self.model.predicted_distance(pitch, speed),
            video=self.video.name if frame is not None else "", video_frame=frame,
        )
        sol = self._solution
        if self._smoothed is not None and sol is not None:
            shot.person_X_cm, shot.person_Z_cm = self._smoothed
            shot.person_method = self._last_method
            shot.person_distance_cm = sol.person_distance_cm
            shot.person_yaw_deg = sol.person_yaw_deg
        self.shot_log.add(shot)
        where = (f"人まで {shot.person_distance_cm / 100:.2f}m" if shot.person_distance_cm is not None
                 else "人なし")
        pred = (f"予想 {shot.predicted_distance_cm / 100:.2f}m" if shot.predicted_distance_cm is not None
                else "予想なし")
        log.info("発射 #%d(%s)強さ %d、yaw %+.1f°、pitch %+.1f°、%s、%s。結果をターミナルに入力してください。",
                 shot.shot_id, "自動" if trigger == "auto" else "1 キー", speed, yaw, pitch, where, pred)
        return shot

    # ---- ターミナルの入力 ---------------------------------------------------------

    def handle_line(self, line: str) -> None:
        cmd = parse_command(line)
        if cmd.kind == "empty":
            return
        if cmd.kind == "help":
            for text in HELP_TEXT.splitlines():
                log.info("%s", text)
            return
        if cmd.kind == "undo":
            shot_id = self.shot_log.undo()
            if shot_id is None:
                log.warning("取り消す入力がありません。")
                return
            self._relearn(f"#{shot_id} の直前の入力を取り消しました")
            return
        shot = self.shot_log.get(cmd.shot_id)
        if shot is None:
            log.warning("%s。入力は使いませんでした: %s",
                        "まだ発射していません" if cmd.shot_id is None else f"#{cmd.shot_id} の発射はありません",
                        line.strip())
            return
        fb = cmd.feedback
        self.shot_log.apply_feedback(shot, fb)
        if not fb.usable:
            log.info("#%d にメモとして保存しました(学習には使いません): %s", shot.shot_id, fb.text)
            return
        if fb.distance_error_cm is not None and shot.person_distance_cm is None:
            log.warning("#%d は人がいないときの発射なので、「〜足りない」は使えません。"
                        "巻き尺で測って「飛距離4.5m」のように入力してください。", shot.shot_id)
        self._relearn(f"#{shot.shot_id}: {fb.summary()}")

    def _relearn(self, what: str) -> None:
        self.model.learn(self.shot_log.all())
        log.info("%s → 学習し直しました: %s", what, self.model.describe())
