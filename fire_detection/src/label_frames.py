"""
Lightweight interactive frame labeler.

Controls
--------
  F  — label current frame as FIRE (1)
  N  — label current frame as NO FIRE (0)
  S  — skip frame (excluded from labels.json)
  Q  — quit and save all labels collected so far

Labels are written to data/labels.json in the format expected by evaluate.py:
    {
        "data/sample_videos/fire_indoor.mp4": [1, 1, 0, ...],
        ...
    }

Usage
-----
    # Label one video
    python src/label_frames.py --video data/sample_videos/fire_indoor.mp4

    # Label multiple videos back-to-back
    python src/label_frames.py --video data/sample_videos/fire_indoor.mp4 data/sample_videos/no_fire_sunset.mp4

    # Resume: existing labels.json is loaded and merged, so you can stop and continue
    python src/label_frames.py --video data/sample_videos/fire_indoor.mp4 --output data/labels.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import cv2

# How many frames to skip between labeling prompts.
# 15 means you label every 15th frame (2x/second at 30fps).
# Lower = more labels + more work. Raise to go faster.
FRAME_STRIDE = 15


def label_video(video_path: str, existing_labels: dict) -> list:
    """
    Open a video and ask the user to label frames interactively.
    Returns a list of per-frame labels (one entry per frame in the video).
    Frames not manually reviewed are filled with -1 and then interpolated
    from the nearest labeled neighbor before saving.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\n{'='*60}")
    print(f"Video : {video_path}")
    print(f"Frames: {total_frames}  |  FPS: {fps:.1f}")
    print(f"You will be shown every {FRAME_STRIDE}th frame (~{total_frames//FRAME_STRIDE} keyframes).")
    print("Controls:  F = fire    N = no fire    S = skip    Q = quit & save")
    print(f"{'='*60}")

    # Pre-fill with -1 (unlabeled)
    labels = [-1] * total_frames
    frame_idx = 0
    quit_early = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only show every FRAME_STRIDE-th frame to the user
        if frame_idx % FRAME_STRIDE == 0:
            display = cv2.resize(frame, (700, 500))

            # Progress overlay
            progress = f"Frame {frame_idx+1}/{total_frames}  |  F=fire  N=no fire  S=skip  Q=quit"
            cv2.putText(display, progress, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display, progress, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

            cv2.namedWindow("Label Frames", cv2.WINDOW_NORMAL)
            cv2.imshow("Label Frames", display)
            cv2.setWindowProperty("Label Frames", cv2.WND_PROP_TOPMOST, 1)

            # Poll with waitKey(1) in a tight loop — more reliable on macOS
            # than blocking waitKey(0)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('f'):
                    # Mark this frame AND the next FRAME_STRIDE-1 frames as fire
                    for i in range(frame_idx, min(frame_idx + FRAME_STRIDE, total_frames)):
                        labels[i] = 1
                    print(f"  [{frame_idx:>5}] FIRE")
                    break
                elif key == ord('n'):
                    for i in range(frame_idx, min(frame_idx + FRAME_STRIDE, total_frames)):
                        labels[i] = 0
                    print(f"  [{frame_idx:>5}] NO FIRE")
                    break
                elif key == ord('s'):
                    print(f"  [{frame_idx:>5}] skipped")
                    break
                elif key == ord('q'):
                    print("\nQuitting early — saving labels collected so far.")
                    quit_early = True
                    break

        if quit_early:
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # --- fill unlabeled frames by forward-fill then backward-fill ---
    # Any skipped or unvisited frame gets the label of the nearest labeled neighbor.
    # This is a reasonable assumption for fire videos (fire doesn't flicker in/out
    # frame-by-frame; a short run of unlabeled frames shares its neighbors' class).
    filled = labels[:]
    last = None
    for i, v in enumerate(filled):
        if v != -1:
            last = v
        elif last is not None:
            filled[i] = last

    # backward pass for any leading unlabeled frames
    last = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] != -1:
            last = filled[i]
        elif last is not None:
            filled[i] = last

    # If the entire video was skipped, default to 0
    filled = [v if v != -1 else 0 for v in filled]

    labeled = sum(1 for v in labels if v != -1)
    print(f"  Labeled {labeled} keyframes → interpolated to {total_frames} total frames.")
    return filled


def main():
    global FRAME_STRIDE
    parser = argparse.ArgumentParser(description="Interactive frame labeler for fire detection.")
    parser.add_argument("--video", nargs="+", required=True,
                        help="Path(s) to video file(s) to label.")
    parser.add_argument("--output", default="data/labels.json",
                        help="Output path for labels JSON (default: data/labels.json).")
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE,
                        help=f"Show every Nth frame (default: {FRAME_STRIDE}).")
    args = parser.parse_args()

    FRAME_STRIDE = args.stride

    # Load existing labels so re-runs merge rather than overwrite
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        print(f"Loaded existing labels from {args.output} ({len(existing)} video(s) already labeled).")

    for video_path in args.video:
        if not os.path.exists(video_path):
            print(f"WARNING: File not found, skipping: {video_path}")
            continue

        if video_path in existing:
            print(f"SKIP: {video_path} already labeled ({len(existing[video_path])} frames). "
                  "Delete its entry from labels.json to re-label.")
            continue

        frame_labels = label_video(video_path, existing)
        if frame_labels:
            existing[video_path] = frame_labels

        # Save after every video so progress isn't lost if you quit mid-session
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Saved → {args.output}")

    print(f"\nDone. {len(existing)} video(s) in {args.output}.")


if __name__ == "__main__":
    main()
