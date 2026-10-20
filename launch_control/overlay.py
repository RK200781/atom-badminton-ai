# -*- coding: utf-8 -*-
"""
発射機の状態をカメラ映像に重ねて表示する。

cv2.putText は日本語を描画できないため、表示はすべて英語。
"""

import cv2

from . import fire_controller as fc


STATE_COLORS = {  # BGR
    fc.IDLE: (180, 180, 180),
    fc.TRACKING: (0, 255, 255),
    fc.AIMING: (0, 165, 255),
    fc.FIRING: (0, 0, 255),
    fc.COOLDOWN: (255, 200, 0),
}


def draw_launch_status(frame, controller, launcher):
    """画面左下に状態・角度・メッセージを、FIRING 中は中央に大きく「FIRE!」を描画する。"""
    h, w = frame.shape[:2]
    color = STATE_COLORS.get(controller.state, (255, 255, 255))
    pan, tilt = launcher.get_angles()

    lines = [f"[{controller.state}] {controller.message}"]
    sol = controller.solution
    angle_line = f"pan={pan:.1f} tilt={tilt:.1f}"
    if sol is not None:
        target_tilt = f"{sol.tilt_deg:.1f}" if sol.tilt_deg is not None else "-"
        angle_line += f"  target pan={sol.pan_deg:.1f} tilt={target_tilt}  dist={sol.distance_cm:.0f}cm"
    lines.append(angle_line)
    lines.append(f"shots={controller.shot_count}")

    for i, line in enumerate(reversed(lines)):
        y = h - 20 - 32 * i
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if controller.state == fc.FIRING:
        text = "FIRE!"
        scale, thickness = 4.0, 10
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = ((w - tw) // 2, (h + th) // 2)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 8)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness)
