# -*- coding: utf-8 -*-
"""
検出結果の CSV 記録。

単品スクリプト(realtime_detect.py / realtime_pose_detect.py)と同じ場所・ファイル名・列で保存する。
単品スクリプトは終了時にまとめて書き出すが、統合版は途中で異常終了しても残るよう、
1行ずつ追記して約1秒ごとにディスクへ書き出す。1件も無ければファイルは作らない。
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("launcher.records")


class CsvRecorder:
    FLUSH_INTERVAL_SEC = 1.0

    def __init__(self, directory: Path, prefix: str, header: Sequence[str], run_stamp: str, label: str):
        self.path = Path(directory) / f"{prefix}_{run_stamp}.csv"
        self._header = list(header)
        self._label = label
        self._file = None
        self._writer = None
        self._last_flush = 0.0
        self.count = 0

    def add(self, row: Sequence) -> None:
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._header)
            log.info("%sの記録を始めました: %s", self._label, self.path)
        self._writer.writerow(row)
        self.count += 1
        now = time.monotonic()
        if now - self._last_flush >= self.FLUSH_INTERVAL_SEC:
            self._file.flush()
            self._last_flush = now

    def close(self) -> Optional[Path]:
        if self._file is None:
            log.info("%sは1件も無かったので、CSV は作りませんでした。", self._label)
            return None
        self._file.close()
        self._file = None
        log.info("%sを保存しました: %s(%d 件)", self._label, self.path, self.count)
        return self.path
