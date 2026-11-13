#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
録画した動画で、YOLO の設定(imgsz と信頼度のしきい値)の組み合わせごとに、シャトルを検出できた
フレームの割合を比べる。小さい(遠い)シャトルがどれだけ取れるかを見るため、検出した枠の大きさごとにも数える。

    python detection_test/compare.py                       # recordings/ の動画をすべて
    python detection_test/compare.py recordings/rec_X.mp4  # 指定した動画だけ
    python detection_test/compare.py --imgsz 640 1280 --conf 0.25 0.1 --every 2

結果は detection_test/results/<日時>/ に保存する。
    summary.csv   動画ごと・設定ごとの集計
    diff/         設定によって検出できたりできなかったりしたフレームの画像(各設定の枠を色分けして描く)。
                  見つかったものが本当にシャトルか(誤検出でないか)を目で確かめる用

注意:正解のラベルが無いので「検出率」は「何か見つかったフレームの割合」。誤検出も数に入るので、
diff/ の画像で中身を確かめること。
"""

import argparse
import csv
import datetime
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 画像に枠を描くときの色(設定の順に使う。BGR)
COLORS = [(0, 255, 0), (0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


@dataclass(frozen=True)
class Setting:
    imgsz: int
    conf: float

    @property
    def label(self) -> str:
        return f"imgsz={self.imgsz} conf={self.conf:g}"


@dataclass
class Stats:
    frames: int = 0
    detected: int = 0
    sizes: List[float] = field(default_factory=list)   # 採用した(一番信頼度の高い)枠の長い辺 [px]
    confidences: List[float] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.detected / self.frames if self.frames else 0.0

    def bucket_counts(self, buckets=None) -> List[int]:
        buckets = buckets or config.SIZE_BUCKETS
        return [sum(1 for s in self.sizes if lo <= s < hi) for lo, hi in buckets]


@dataclass
class Result:
    stats: Dict[Setting, Stats]
    ms_per_frame: Dict[int, float]   # imgsz ごとの YOLO の平均処理時間
    disagreements: int


def best_detection(boxes, confs, conf: float) -> Optional[Tuple[np.ndarray, float]]:
    """しきい値 conf 以上の中で、一番信頼度の高い枠 (box, 信頼度) を返す。無ければ None。"""
    if len(confs) == 0:
        return None
    keep = np.asarray(confs) >= conf
    if not keep.any():
        return None
    idx = int(np.argmax(np.where(keep, confs, -1.0)))
    return np.asarray(boxes[idx]), float(confs[idx])


def evaluate(frames: Iterable[Tuple[int, np.ndarray]],
             detect: Callable[[np.ndarray, int, float], Tuple[np.ndarray, np.ndarray]],
             settings: List[Setting],
             on_disagreement: Optional[Callable] = None, warmup: bool = True) -> Result:
    """フレームごとに、imgsz ごとに1回だけ推論して(しきい値は一番低いもの)、各設定の結果を数える。

    warmup: 最初のフレームで、各 imgsz を一度ずつ時間を測らずに推論しておく
    (GPU の初回の準備時間が、最初に処理した imgsz の処理時間に入ってしまうのを避ける)。
    """
    imgsz_list = sorted({s.imgsz for s in settings})
    min_conf = {sz: min(s.conf for s in settings if s.imgsz == sz) for sz in imgsz_list}
    stats = {s: Stats() for s in settings}
    total_ms = {sz: 0.0 for sz in imgsz_list}
    n_frames = 0
    disagreements = 0
    for index, frame in frames:
        if warmup and n_frames == 0:
            for sz in imgsz_list:
                detect(frame, sz, min_conf[sz])
        n_frames += 1
        raw = {}
        for sz in imgsz_list:
            t0 = time.perf_counter()
            raw[sz] = detect(frame, sz, min_conf[sz])
            total_ms[sz] += (time.perf_counter() - t0) * 1000
        found = {}
        for s in settings:
            best = best_detection(*raw[s.imgsz], s.conf)
            found[s] = best
            st = stats[s]
            st.frames += 1
            if best is not None:
                box, c = best
                st.detected += 1
                st.sizes.append(float(max(box[2] - box[0], box[3] - box[1])))
                st.confidences.append(c)
        if len({b is not None for b in found.values()}) > 1:
            disagreements += 1
            if on_disagreement is not None:
                on_disagreement(index, frame, found)
    return Result(stats, {sz: total_ms[sz] / n_frames if n_frames else 0.0 for sz in imgsz_list}, disagreements)


def iter_video(path: Path, every: int = 1, max_frames: Optional[int] = None):
    """動画のフレームを (フレーム番号, 画像) で返す。every フレームに1枚だけ使う。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {path}")
    index, used = 0, 0
    try:
        while max_frames is None or used < max_frames:
            if index % every != 0:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1
            used += 1
    finally:
        cap.release()


def draw_disagreement(frame, found: Dict[Setting, Optional[tuple]], settings: List[Setting]) -> np.ndarray:
    """各設定の枠を色分けして描き、左上に設定ごとの結果(見つかったか)を書く。"""
    img = frame.copy()
    for i, s in enumerate(settings):
        color = COLORS[i % len(COLORS)]
        best = found[s]
        if best is not None:
            (x1, y1, x2, y2), c = best
            pad = 3 * i  # 同じ枠が重ならないよう少しずつ広げる
            cv2.rectangle(img, (int(x1) - pad, int(y1) - pad), (int(x2) + pad, int(y2) + pad), color, 2)
        mark = f"found {best[1]:.2f}" if best is not None else "none"
        text = f"{s.label}: {mark}"
        cv2.putText(img, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(img, text, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


class YoloDetector:
    def __init__(self, model_path: Path = config.YOLO_MODEL_PATH):
        from ultralytics import YOLO
        import torch
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO のモデルが見つかりません: {model_path}")
        self._model = YOLO(str(model_path))
        self._device = 0 if torch.cuda.is_available() else "cpu"
        print(f"YOLO: {model_path}({'GPU: ' + torch.cuda.get_device_name(0) if self._device == 0 else 'CPU'})")

    def __call__(self, frame, imgsz: int, conf: float):
        r = self._model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=self._device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def format_table(title: str, result: Result, settings: List[Setting]) -> str:
    bucket_names = [f"{lo}-{hi}px" if hi < 10_000 else f"{lo}px-" for lo, hi in config.SIZE_BUCKETS]
    lines = [title,
             f"  {'設定':<24}{'検出率':>8}{'検出/全体':>12}{'枠(中央値)':>12}  " + "  ".join(f"{b:>9}" for b in bucket_names)
             + f"  {'YOLO':>9}"]
    for s in settings:
        st = result.stats[s]
        med = f"{statistics.median(st.sizes):.0f}px" if st.sizes else "-"
        buckets = "  ".join(f"{n:>9}" for n in st.bucket_counts())
        lines.append(f"  {s.label:<24}{st.rate:>8.1%}{f'{st.detected}/{st.frames}':>12}{med:>12}  {buckets}"
                     f"  {result.ms_per_frame[s.imgsz]:>6.1f} ms")
    lines.append(f"  設定によって結果が違ったフレーム: {result.disagreements}")
    return "\n".join(lines)


def collect_videos(paths: List[Path]) -> List[Path]:
    videos = []
    for p in paths:
        if p.is_dir():
            videos += sorted(p.glob("*.mp4"))
        else:
            videos.append(p)
    return videos


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLO の設定ごとに、シャトルの検出率を比べる")
    parser.add_argument("videos", nargs="*", type=Path, default=[config.RECORDINGS_DIR],
                        help="動画ファイルかフォルダ(既定: detection_test/recordings/)")
    parser.add_argument("--imgsz", nargs="+", type=int, default=config.DEFAULT_IMGSZ, help="YOLO が処理する大きさ")
    parser.add_argument("--conf", nargs="+", type=float, default=config.DEFAULT_CONF, help="信頼度のしきい値")
    parser.add_argument("--every", type=int, default=1, help="何フレームに1枚使うか(時間を短くしたいとき)")
    parser.add_argument("--max-frames", type=int, default=None, help="動画1本あたりに使う最大フレーム数")
    parser.add_argument("--save-diff", type=int, default=30,
                        help="設定によって結果が違ったフレームの画像を、動画1本あたり何枚まで保存するか(0 で保存しない)")
    parser.add_argument("--out", type=Path, default=None, help="結果の保存先(既定: detection_test/results/<日時>)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    videos = collect_videos(args.videos)
    if not videos:
        print(f"動画が見つかりません: {[str(p) for p in args.videos]}(先に record.py で録画する)")
        return 1
    settings = [Setting(sz, c) for sz in args.imgsz for c in sorted(args.conf, reverse=True)]
    out_dir = args.out or config.RESULTS_DIR / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "diff").mkdir(parents=True, exist_ok=True)
    detect = YoloDetector()

    rows = []
    for video in videos:
        saved = [0]

        def save(index, frame, found, video=video):
            if saved[0] >= args.save_diff:
                return
            cv2.imwrite(str(out_dir / "diff" / f"{video.stem}_f{index:06d}.jpg"), draw_disagreement(frame, found, settings))
            saved[0] += 1

        t0 = time.perf_counter()
        result = evaluate(iter_video(video, args.every, args.max_frames), detect, settings, save)
        print()
        print(format_table(f"■ {video.name}({time.perf_counter() - t0:.0f} 秒)", result, settings))
        for s in settings:
            st = result.stats[s]
            rows.append([video.name, s.imgsz, s.conf, st.frames, st.detected, f"{st.rate:.4f}",
                         f"{statistics.median(st.sizes):.1f}" if st.sizes else "",
                         *st.bucket_counts(), f"{result.ms_per_frame[s.imgsz]:.2f}"])

    bucket_cols = [f"size_{lo}_{hi}" if hi < 10_000 else f"size_{lo}_plus" for lo, hi in config.SIZE_BUCKETS]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "imgsz", "conf", "frames", "detected", "rate", "median_box_px",
                         *bucket_cols, "yolo_ms_per_frame"])
        writer.writerows(rows)
    print(f"\n結果を保存しました: {out_dir}(summary.csv、diff/ に違いが出たフレームの画像)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
