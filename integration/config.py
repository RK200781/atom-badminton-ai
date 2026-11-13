# -*- coding: utf-8 -*-
"""
統合版の設定値。

値はノートブック Untitled-2.ipynb と stattle.ino に合わせてある
(コメントの「セルN:行」はノートブック、「ino:行」は stattle.ino の位置)。
"""

# ============================================================
# 接続(COM 番号は PC や USB の挿し口で変わるので、main.py の引数でも上書きできる)
# ============================================================

# CubeMars モーター: USB-CAN(slcan)。セル2:58〜62
CAN_INTERFACE = "slcan"
CAN_CHANNEL = "COM6@115200"   # slcan アダプタの COM ポート@シリアル速度
CAN_BITRATE = 1_000_000       # CAN バスのビットレート 1Mbps

# Arduino(ローラーと装填): セル4:24〜32、ino:90
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_SEC = 0.1      # readline の待ち時間

# ============================================================
# モーター(CubeMars AK45-10 V3.0)
# ============================================================

# モーターID。ノートブックでは 104 を ROLL と呼んでいたが、統合版では pitch(縦)と呼ぶ。セル2:10〜11
YAW_ID = 105     # 横。プラスが右(実機で確認済み)
PITCH_ID = 104   # 縦。マイナスが上、0° で水平(実機で確認済み)

# 機械的な原点 [deg](モーターの絶対角度)。セル2:14〜15
YAW_ORIGIN_DEG = 145.0
PITCH_ORIGIN_DEG = 165.0

# 可動範囲 [deg](原点からの角度)。セル2:18〜22
YAW_MIN_DEG = -60.0
YAW_MAX_DEG = 60.0
PITCH_MIN_DEG = -50.0
PITCH_MAX_DEG = 30.0

# 起動時の目標角 [deg](原点からの角度)。ノートブックと同じく原点。
# 保持ループが始まると、モーターはすぐにこの角度へ動き出す。
INITIAL_YAW_DEG = 0.0
INITIAL_PITCH_DEG = 0.0

# 出力軸の最大回転速度 [rpm]。セル2:25
MAX_OUTPUT_RPM = 50.0

# 減速比と極対数。セル2:28〜29
GEAR_RATIO = 10.0
POLE_PAIRS = 14

# CubeMars の CAN コマンド。セル2:32、セル3:43
CAN_PACKET_SET_POS_SPD = 6     # Position-Speed Loop の指令
CAN_FEEDBACK_FUNCTION_ID = 0x29  # 状態フィードバック

# 位置を保持するループ。セル3:587〜591(head_hold_loop の既定値)
#
# 速度・加速度の単位(2026-11-13 実機で確認済み):
#   ノートブックのコメントでは「送信値 = ERPM / 10」となっているが、コードは 10 で割らずに
#   ERPM をそのまま送っている(0.5rpm → speed_cmd=70)。実機で動く速さが想定どおりだったので、
#   10 で割らない今の方式で正しい。
HOLD_SPEED_RPM = 0.5          # yaw・pitch 共通 [rpm]
HOLD_ACCEL_RPM_S = 20.0       # [rpm/s]
HOLD_INTERVAL_SEC = 0.05      # 50ms ごとに目標角を送り直す

# W/A/S/D を1回押したときに動かす角度 [deg]。セル4:9
KEY_STEP_DEG = 2.0

# 現在角(フィードバック)がこの時間届かなければ「不明」として扱う
ANGLE_STALE_SEC = 0.5

# ============================================================
# Arduino(stattle.ino)
# ============================================================

# ポートを開くと Arduino がリセットされる(Uno / Nano の場合)ので、その間待つ
ARDUINO_RESET_WAIT_SEC = 2.0

# シャトル検知の値(Arduino が約10msごとに送る 1/0)が、この時間以内に届いたものなら有効とみなす
SHUTTLE_FRESH_SEC = 0.2

# 強さを変えてから、ローラーが回り始める(回転が落ち着く)までの待ち時間(仮の値。実機で調整)
ROLLER_SPINUP_SEC = 1.5

# 前の発射から、次の発射までに空ける時間(仮の値。実機で調整)
FIRE_INTERVAL_SEC = 2.0

# 発射の指令を送ってから、シャトル検知が 1→0 に変わるのを待つ時間(仮の値)
FIRE_CONFIRM_SEC = 1.0

# Arduino は、シャトルを検知しない状態が 10 秒続くとローラーを一時的に止め、
# シャトルを入れた瞬間に元の強さで回し直す(ino:51、ino:119〜121)。
# Python 側でもこれを真似て、10 秒以上検知がなかった後にシャトルが入ったら、
# そこからもう一度 ROLLER_SPINUP_SEC 待つ。
ARDUINO_NO_SHUTTLE_TIMEOUT_SEC = 10.0

# 強さの段階と、Arduino での出力(ino:231〜240)。表示用。
SPEED_LABELS = {0: "停止", 1: "弱(20%)", 2: "中(50%)", 3: "強(100%)"}

# ============================================================
# 画面
# ============================================================

WINDOW_NAME = "Launcher Control"
UI_REFRESH_MS = 30            # cv2.waitKey の待ち時間
