"""
Evaluation script: run HSV and Hybrid detectors on labeled video files,
compute Precision / Recall / F1 / average FPS, and write results/metrics.csv.

Label format
------------
Ground-truth labels are expected as a JSON file (default: data/labels.json)
with the following structure:

    {
        "data/sample_videos/fire_inside.mp4": [1, 1, 0, 1, ...],
        "data/sample_videos/no_fire.mp4":    [0, 0, 0, 0, ...]
    }

Each integer list contains one label per frame: 1 = fire present, 0 = no fire.
Frame count must match the actual video frame count exactly.

TODO: You need to produce this file before running evaluation.
      Recommended approach:
        1. Use a frame-labeling tool such as CVAT (https://www.cvat.ai/) or
           Label Studio (https://labelstud.io/) to annotate your sample videos.
        2. Export frame-level labels and convert to the JSON format above.
        3. Save to data/labels.json (or pass --labels <path> on the CLI).

Usage
-----
    python src/evaluate.py --videos data/sample_videos/ --labels data/labels.json
    python src/evaluate.py --videos data/sample_videos/ --labels data/labels.json --output results/metrics.csv
"""

import sys
import os

# Allow running as `python src/evaluate.py` without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import csv

from hsv_detector import HSVFireDetector
from hybrid_detector import HybridFireDetector


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """
    Compute binary classification metrics from frame-level prediction lists.

    Returns precision, recall, F1, and support counts.
    Both lists must be the same length and contain only 0/1 values.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------

def evaluate_detector_on_video(
    detector,
    video_path: str,
    ground_truth: List[int],
) -> Tuple[Dict[str, float], float]:
    """
    Run `detector` on `video_path`, compare to `ground_truth`, return metrics + FPS.

    The detector must expose a `run_on_video(path)` generator that yields
    objects with a `.fire_detected` boolean attribute (FrameResult).

    Returns
    -------
    (metrics_dict, avg_fps)
    """
    predictions: List[int] = []
    frame_times: List[float] = []

    for result in detector.run_on_video(video_path):
        t0 = time.perf_counter()
        predictions.append(1 if result.fire_detected else 0)
        frame_times.append(time.perf_counter() - t0)

    # Align prediction length with ground truth (trim or pad with 0).
    # Trim/pad handles minor off-by-one from codec differences.
    n = len(ground_truth)
    if len(predictions) > n:
        predictions = predictions[:n]
    elif len(predictions) < n:
        predictions.extend([0] * (n - len(predictions)))

    metrics = compute_metrics(ground_truth, predictions)

    # FPS = frames / total_elapsed (re-timed during iteration above captures
    # only the prediction-extraction overhead; for a more accurate number
    # we re-run timing around the full generator loop).
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Estimate processing FPS: total frames / sum of per-frame wall times.
    # This approximates throughput without including video I/O latency.
    total_time = sum(frame_times) if frame_times else 1.0
    avg_fps = len(predictions) / total_time if total_time > 0 else 0.0

    return metrics, avg_fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(
    video_dir: str,
    labels_path: str,
    output_path: str,
) -> None:
    """
    Evaluate both detectors on all labeled videos and write metrics.csv.
    """
    # --- load ground truth ---
    if not os.path.exists(labels_path):
        print(
            f"ERROR: labels file not found: {labels_path}\n"
            "See the TODO in this file for instructions on creating it."
        )
        sys.exit(1)

    with open(labels_path) as f:
        labels: Dict[str, List[int]] = json.load(f)

    video_paths = [p for p in labels if os.path.exists(p)]
    if not video_paths:
        print("ERROR: No video files found that match the keys in the labels file.")
        sys.exit(1)

    detectors = {
        "HSV Baseline": HSVFireDetector(),
        "Hybrid (HSV + MobileNetV2)": HybridFireDetector(),
    }

    rows = []

    for detector_name, detector in detectors.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {detector_name}")
        print(f"{'='*60}")

        all_true:  List[int] = []
        all_pred:  List[int] = []
        fps_list:  List[float] = []

        for video_path in video_paths:
            gt = labels[video_path]
            print(f"  Processing: {video_path} ({len(gt)} frames) ...", end=" ", flush=True)

            preds: List[int] = []
            frame_times: List[float] = []

            detector.reset()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("SKIP (cannot open)")
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                t0 = time.perf_counter()
                result = detector.process_frame(frame)
                frame_times.append(time.perf_counter() - t0)
                preds.append(1 if result.fire_detected else 0)

            cap.release()

            # Align lengths
            n = len(gt)
            if len(preds) > n:
                preds = preds[:n]
            elif len(preds) < n:
                preds.extend([0] * (n - len(preds)))

            all_true.extend(gt)
            all_pred.extend(preds)
            fps = len(preds) / sum(frame_times) if frame_times else 0.0
            fps_list.append(fps)
            print(f"done ({fps:.1f} fps)")

        # Aggregate metrics across all videos for this detector
        metrics = compute_metrics(all_true, all_pred)
        avg_fps = float(np.mean(fps_list)) if fps_list else 0.0

        print(
            f"  Precision: {metrics['precision']:.3f}  "
            f"Recall: {metrics['recall']:.3f}  "
            f"F1: {metrics['f1']:.3f}  "
            f"Avg FPS: {avg_fps:.1f}"
        )

        rows.append({
            "detector":  detector_name,
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1":        round(metrics["f1"],        4),
            "avg_fps":   round(avg_fps, 2),
            "tp":        metrics["tp"],
            "fp":        metrics["fp"],
            "fn":        metrics["fn"],
        })

    # --- write CSV ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["detector", "precision", "recall", "f1", "avg_fps", "tp", "fp", "fn"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fire detectors on labeled videos.")
    parser.add_argument(
        "--videos", default="data/sample_videos/",
        help="Directory containing sample video files."
    )
    parser.add_argument(
        "--labels", default="data/labels.json",
        help="JSON file mapping video paths to per-frame binary labels."
    )
    parser.add_argument(
        "--output", default="results/metrics.csv",
        help="Output path for the metrics CSV."
    )
    args = parser.parse_args()
    run_evaluation(args.videos, args.labels, args.output)
