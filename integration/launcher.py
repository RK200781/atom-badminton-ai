# -*- coding: utf-8 -*-
"""
発射機全体の操作:モーター(motors.py)と Arduino(feeder.py)をまとめる。

    aim(yaw, pitch)   目標角を指定する(原点からの角度 [deg])
    nudge(dyaw, dpitch) 今の目標角から少し動かす
    set_speed(段階)    ローラーの強さ 0〜3
    fire()            条件がそろっていれば発射する
    stop()            ローラー停止+その場で保持+強さを設定し直すまで発射不可
    restart_motors()  通信エラーで止まったモーターの保持ループを再開する
    shutdown()        終了処理(必ず呼ぶ)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from feeder import Feeder, FeederSnapshot
from motors import MotorController, MotorState

log = logging.getLogger("launcher")

LOCKED_REASON = "停止中です(2/3/4キーで強さを設定し直すと発射できます)"


@dataclass
class LauncherStatus:
    motors: MotorState
    feeder: FeederSnapshot
    shuttle: Optional[bool]          # 最新のシャトル検知(古い・不明なら None)
    locked: bool                     # stop() の後で、まだ強さを設定し直していない
    fire_reasons: List[str]          # 発射できない理由(空なら発射できる)


class Launcher:
    def __init__(self, motors: MotorController, feeder: Feeder):
        self.motors = motors
        self.feeder = feeder
        self._locked = False
        self._shut_down = False

    def start(self) -> None:
        """Arduino を開き(リセット待ちを含む)、モーターの保持ループを始める。"""
        self.feeder.open()
        self.motors.start()

    # ---- 操作 --------------------------------------------------------------

    def aim(self, yaw_deg: float, pitch_deg: float) -> None:
        self.motors.set_target(yaw_deg, pitch_deg)

    def nudge(self, d_yaw_deg: float, d_pitch_deg: float) -> None:
        self.motors.nudge(d_yaw_deg, d_pitch_deg)

    def set_speed(self, speed: int) -> None:
        try:
            self.feeder.set_speed(speed)
        except Exception as e:
            log.error("強さを設定できませんでした: %s", e)
            return
        if speed in (1, 2, 3) and self._locked:
            self._locked = False
            log.info("強さを設定し直したので、発射できる状態に戻しました。")

    def fire(self) -> Tuple[bool, List[str]]:
        if self._locked:
            log.warning("発射しませんでした: %s", LOCKED_REASON)
            return False, [LOCKED_REASON]
        try:
            return self.feeder.fire(hold_running=self.motors.running)
        except Exception as e:
            log.error("発射の指令を送れませんでした: %s", e)
            return False, [f"送信エラー: {e}"]

    def stop(self) -> None:
        """ローラーを止め、モーターをその場で保持し、強さを設定し直すまで発射を受け付けない。"""
        self._locked = True
        try:
            self.feeder.set_speed(0)
        except Exception as e:
            log.error("ローラーを止める指令を送れませんでした: %s", e)
        self.motors.hold_here()
        log.warning("停止しました(ローラー停止・その場で保持・発射不可)。"
                    "装填用モーターはシリアルからは止められません。")

    def restart_motors(self) -> None:
        try:
            self.motors.restart()
        except Exception as e:
            log.error("モーターの通信を再開できませんでした: %s", e)

    def shutdown(self) -> None:
        """終了処理。"0,0" を送る → 保持ループを止めて CAN を閉じる → シリアルを閉じる。

        途中でエラーが起きても残りの手順は必ず行う。何度呼んでもよい。
        """
        if self._shut_down:
            return
        self._shut_down = True
        log.info("終了処理を始めます。")
        for name, step in (
            ("ローラー停止(0,0)", self.feeder.stop_roller),
            ("保持ループ停止・CAN 終了", self.motors.shutdown),
            ("シリアルを閉じる", lambda: self.feeder.close(send_stop=False)),
        ):
            try:
                step()
            except Exception as e:
                log.error("終了処理「%s」でエラー: %s", name, e)
        log.info("終了処理が終わりました。")

    # ---- 状態 --------------------------------------------------------------

    def status(self) -> LauncherStatus:
        hold_running = self.motors.running
        reasons = self.feeder.block_reasons(hold_running)
        if self._locked:
            reasons = [LOCKED_REASON] + reasons
        return LauncherStatus(
            motors=self.motors.state(),
            feeder=self.feeder.snapshot(),
            shuttle=self.feeder.shuttle_detected(),
            locked=self._locked,
            fire_reasons=reasons,
        )
