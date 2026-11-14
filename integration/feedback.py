# -*- coding: utf-8 -*-
"""
発射の結果をターミナルに文章で入力したものを読み取る(自動版の学習用)。

例(左右は「発射機から見て」):
    1m足りない / 50cm手前 / あと1.5メートル     → 人の立ち位置より手前に落ちた(距離の誤差 −)
    2m超えた / 30センチ奥 / 1m飛びすぎ         → 人の立ち位置より奥に落ちた(距離の誤差 +)
    右に30cm / 20cm左                          → 人の立ち位置より右(+)/左(−)に落ちた
    飛距離4.5m / 4m50地点                       → 発射口から落ちた位置までの距離(巻き尺で測ったとき)
    ちょうど / ぴったり / OK                     → 距離も左右も合っていた
    1m足りなくて右に30cm、                      → 両方まとめて書いてもよい

単位が無い数字は、20 未満ならメートル、20 以上ならセンチとみなす。
「少し足りない」のように数字が無いものは、学習には使わない(メモとして残す)。
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

SHORT_WORDS = ["足りな", "足らな", "たりな", "たらな", "届かな", "届いてな", "とどかな", "とどいてな",
               "短", "手前", "ショート", "あと"]
LONG_WORDS = ["超え", "越え", "こえ", "オーバー", "飛びすぎ", "とびすぎ", "飛び過ぎ", "行き過ぎ", "いきすぎ",
              "長", "奥", "後ろ", "うしろ", "向こう"]
RIGHT_WORDS = ["右", "みぎ"]
LEFT_WORDS = ["左", "ひだり"]
ABSOLUTE_WORDS = ["飛距離", "地点", "まで飛", "のところ", "距離"]
OK_WORDS = ["ちょうど", "ぴったり", "ok", "ナイス", "完璧", "合って", "あって", "いい感じ"]

_NUMBER = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|センチ|m|メートル|メーター)?(?:(?<=m|ル|ー)(\d+)(?![\d.]|\s*(?:cm|センチ|m|メ)))?"
)
# 「1m 50cm」「1メートル50センチ」→「150cm」
_M_AND_CM = re.compile(r"(\d+)\s*(?:m|メートル|メーター)\s*(\d+)\s*(?:cm|センチ)")
_CLAUSE_SPLIT = re.compile(r"[、。,;/\n]")


@dataclass
class Feedback:
    text: str
    distance_error_cm: Optional[float] = None   # 人の立ち位置より 奥 +、手前 −
    landing_distance_cm: Optional[float] = None  # 発射口から落ちた位置まで(絶対)
    lateral_cm: Optional[float] = None           # 人の立ち位置より 右 +、左 −(発射機から見て)
    unparsed: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return any(v is not None for v in (self.distance_error_cm, self.landing_distance_cm, self.lateral_cm))

    def summary(self) -> str:
        parts = []
        if self.landing_distance_cm is not None:
            parts.append(f"飛距離 {self.landing_distance_cm / 100:.2f}m")
        if self.distance_error_cm is not None:
            e = self.distance_error_cm
            parts.append("距離 ちょうど" if e == 0 else f"{abs(e) / 100:.2f}m {'奥' if e > 0 else '手前'}")
        if self.lateral_cm is not None:
            e = self.lateral_cm
            parts.append("左右 ちょうど" if e == 0 else f"{'右' if e > 0 else '左'}に {abs(e):.0f}cm")
        text = "、".join(parts) if parts else "読み取れる数字がありません(メモとして保存)"
        if self.unparsed and parts:
            text += f"(読み取れなかった部分: {' / '.join(self.unparsed)})"
        return text


def _to_cm(value: str, unit: Optional[str], extra: Optional[str]) -> float:
    v = float(value)
    if unit in ("cm", "センチ"):
        return v
    if unit in ("m", "メートル", "メーター"):
        cm = v * 100.0
        if extra:  # 「1m50」→ 150cm
            cm += float(extra) * (10.0 if len(extra) == 1 else 1.0)
        return cm
    return v * 100.0 if v < 20 else v


def _nearest(clause: str, start: int, end: int, words) -> Optional[int]:
    """数字(start〜end)から一番近い、words のどれかの出現までの文字数。無ければ None。"""
    best = None
    for w in words:
        for m in re.finditer(re.escape(w), clause):
            d = (start - m.end()) if m.end() <= start else max(0, m.start() - end)
            if best is None or d < best:
                best = d
    return best


def parse_feedback(text: str) -> Feedback:
    norm = unicodedata.normalize("NFKC", text).lower().strip()
    norm = _M_AND_CM.sub(lambda m: f"{int(m.group(1)) * 100 + int(m.group(2))}cm", norm)
    fb = Feedback(text=text.strip())
    for clause in (c.strip() for c in _CLAUSE_SPLIT.split(norm)):
        if not clause:
            continue
        numbers = list(_NUMBER.finditer(clause))
        if not numbers:
            if any(w in clause for w in OK_WORDS):
                if fb.distance_error_cm is None and fb.landing_distance_cm is None:
                    fb.distance_error_cm = 0.0
                if fb.lateral_cm is None:
                    fb.lateral_cm = 0.0
            else:
                fb.unparsed.append(clause)
            continue
        for i, m in enumerate(numbers):
            # 数字ごとに、前の数字の後ろから次の数字の前までを見て、一番近い言葉で意味を決める
            lo = numbers[i - 1].end() if i > 0 else 0
            hi = numbers[i + 1].start() if i + 1 < len(numbers) else len(clause)
            ctx = clause[lo:hi]
            s, e = m.start() - lo, m.end() - lo
            cm = _to_cm(m.group(1), m.group(2), m.group(3))
            dist = {k: _nearest(ctx, s, e, words) for k, words in (
                ("short", SHORT_WORDS), ("long", LONG_WORDS), ("right", RIGHT_WORDS),
                ("left", LEFT_WORDS), ("abs", ABSOLUTE_WORDS))}
            found = {k: d for k, d in dist.items() if d is not None}
            if not found:
                fb.unparsed.append(clause[m.start():m.end()])
                continue
            kind = min(found, key=lambda k: (found[k], ["right", "left", "abs", "short", "long"].index(k)))
            if kind == "short":
                fb.distance_error_cm = -cm
            elif kind == "long":
                fb.distance_error_cm = cm
            elif kind == "right":
                fb.lateral_cm = cm
            elif kind == "left":
                fb.lateral_cm = -cm
            else:
                fb.landing_distance_cm = cm
    return fb


# ---- ターミナルのコマンド ------------------------------------------------------

UNDO_WORDS = ("取り消し", "とりけし", "取消", "undo")
HELP_WORDS = ("help", "?", "?", "ヘルプ")
_SHOT_PREFIX = re.compile(r"^\s*#\s*(\d+)\s*")


@dataclass
class Command:
    kind: str                      # "feedback" / "undo" / "help" / "empty"
    shot_id: Optional[int] = None  # 「#3 1m足りない」のように番号を指定したとき
    feedback: Optional[Feedback] = None


def parse_command(line: str) -> Command:
    norm = unicodedata.normalize("NFKC", line).strip()
    if not norm:
        return Command("empty")
    shot_id = None
    m = _SHOT_PREFIX.match(norm)
    if m:
        shot_id = int(m.group(1))
        norm = norm[m.end():]
    low = norm.lower()
    if low in HELP_WORDS:
        return Command("help")
    if any(low.startswith(w) for w in UNDO_WORDS):
        return Command("undo", shot_id)
    return Command("feedback", shot_id, parse_feedback(norm))


HELP_TEXT = """\
発射の結果を入力してください(Enter で確定。直前の発射に付きます。左右は発射機から見て):
  1m足りない / 50cm手前        人の立ち位置より手前に落ちた
  2m超えた / 30cm奥            人の立ち位置より奥に落ちた
  右に30cm / 20cm左            人の立ち位置より右/左に落ちた
  飛距離4.5m                   発射口から落ちた位置まで(巻き尺で測ったとき。人がいなくても使える)
  ちょうど                     合っていた
  #3 1m足りない                3番の発射に付ける
  取り消し                     直前に入力した結果を取り消す
"""
