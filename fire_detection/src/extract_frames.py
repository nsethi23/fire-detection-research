"""
Extract frames from labeled videos into a folder structure for MobileNetV2 training.

Output structure:
    frames/
        fire/
            fire_indoor_0001.jpg
            fire_indoor_0016.jpg
            ...
        no_fire/
            sunset_0001.jpg
            ...

Usage
-----
    python src/extract_frames.py --labels data/labels.json --output frames/ --stride 5
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import cv2

def extract_frames(labels_path: str, output_dir: str, stride: int):
    with open(labels_path) as f:
        labels = json.load(f)

    fire_dir    = os.path.join(output_dir, "fire")
    no_fire_dir = os.path.join(output_dir, "no_fire")
    os.makedirs(fire_dir,    exist_ok=True)
    os.makedirs(no_fire_dir, exist_ok=True)

    fire_count = no_fire_count = 0

    for video_path, frame_labels in labels.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"WARNING: Cannot open {video_path}, skipping.")
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]
        frame_idx = 0
        print(f"Extracting: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0 and frame_idx < len(frame_labels):
                label = frame_labels[frame_idx]
                resized = cv2.resize(frame, (224, 224))

                if label == 1:
                    out_path = os.path.join(fire_dir, f"{base}_{frame_idx:05d}.jpg")
                    cv2.imwrite(out_path, resized)
                    fire_count += 1
                else:
                    out_path = os.path.join(no_fire_dir, f"{base}_{frame_idx:05d}.jpg")
                    cv2.imwrite(out_path, resized)
                    no_fire_count += 1

            frame_idx += 1

        cap.release()
        print(f"  Done — fire: {fire_count}  no_fire: {no_fire_count} so far")

    print(f"\nExtraction complete.")
    print(f"  frames/fire/    : {fire_count} images")
    print(f"  frames/no_fire/ : {no_fire_count} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels.json")
    parser.add_argument("--output", default="frames/")
    parser.add_argument("--stride", type=int, default=5,
                        help="Save every Nth frame (default 5 = 6fps from 30fps video)")
    args = parser.parse_args()
    extract_frames(args.labels, args.output, args.stride)
