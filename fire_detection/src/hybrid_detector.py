"""
Hybrid fire detector: HSV pre-filter + MobileNetV2 classifier.

Architecture rationale
----------------------
Running a neural network on every frame of a video stream is expensive.
Fire is typically absent from most frames, so a cheap color-based filter can
gate the expensive classifier:

  ┌─────────────┐    hsv_pixel_count       ┌──────────────────────┐
  │ HSV Detector│ ─── < HSV_GATE_THRESHOLD ──▶  SKIP (not fire)    │
  │  (fast)     │                           └──────────────────────┘
  │             │    hsv_pixel_count
  │             │ ─── ≥ HSV_GATE_THRESHOLD ──▶ ┌────────────────┐
  └─────────────┘                               │ MobileNetV2    │
                                                │ classifier     │
                                                │ (authoritative)│
                                                └────────────────┘

This keeps average inference cost proportional to *alarm rate* rather than
total frame count.  In low-fire-frequency footage, 90%+ of frames are
disposed of cheaply by the HSV stage.

The HSV_GATE_THRESHOLD should be set *lower* than the HSV-only detection
threshold to preserve recall: we would rather over-trigger the NN than
miss a real fire.  A value around 30–50% of the standalone HSV threshold
works well empirically.
"""

import cv2
import numpy as np
from typing import Tuple

from hsv_detector import HSVFireDetector, FrameResult, Detection


# ---------------------------------------------------------------------------
# TODO: replace this stub with your real MobileNetV2 model loader.
#
# Steps:
#   1. Fine-tune MobileNetV2 on a labeled fire/no-fire frame dataset.
#      A good starting point: the FLAME dataset (Shamsoshoara et al., 2021)
#      https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
#   2. Save the Keras model:  model.save("models/mobilenetv2_fire.keras")
#   3. Set MODEL_PATH below and remove the stub.
#
# Expected model I/O:
#   - Input:  (1, 224, 224, 3)  float32, pixel values in [0, 1]
#   - Output: (1, 1)            sigmoid probability — fire likelihood
# ---------------------------------------------------------------------------
MODEL_PATH = "models/mobilenetv2_fire.keras"  # TODO: update this path


def _load_model(model_path: str):
    """
    Load a Keras model from disk.  Returns None if the file doesn't exist so
    the rest of the pipeline can fail gracefully with a clear error message.
    """
    import os
    if not os.path.exists(model_path):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def _preprocess_for_mobilenet(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Resize and normalize a BGR frame for MobileNetV2.

    MobileNetV2 expects 224×224 RGB input with pixel values scaled to [0, 1].
    Note: tf.keras.applications.mobilenet_v2.preprocess_input expects [-1, 1];
    we use simple [0,1] normalization here to match typical fine-tuning setups.
    Adjust if your training preprocessing differed.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    return resized.astype("float32") / 255.0


class HybridFireDetector:
    """
    Two-stage fire detector: HSV pre-filter → MobileNetV2 classifier.

    Parameters
    ----------
    hsv_gate_threshold : int
        HSV pixel count above which the MobileNetV2 classifier is invoked.
        Should be lower than HSVFireDetector.pixel_count_threshold to avoid
        silently dropping potentially true positives before the NN sees them.
        Default 5 000 is ~33% of the standalone HSV threshold of 15 000.
    nn_confidence_threshold : float
        Sigmoid output of the NN above which a frame is declared fire.
        0.5 is the natural decision boundary; raise it to reduce false alarms.
    model_path : str
        Path to the saved Keras model file.
    hsv_kwargs : dict
        Keyword arguments forwarded to HSVFireDetector.__init__.
    """

    def __init__(
        self,
        hsv_gate_threshold: int = 5000,
        nn_confidence_threshold: float = 0.5,
        model_path: str = MODEL_PATH,
        hsv_kwargs: dict = None,
    ):
        self.hsv_gate_threshold = hsv_gate_threshold
        self.nn_confidence_threshold = nn_confidence_threshold

        self._hsv = HSVFireDetector(**(hsv_kwargs or {}))
        self._model = _load_model(model_path)

        if self._model is None:
            # TODO: once you have trained weights, remove this warning.
            print(
                f"[HybridDetector] WARNING: model not found at '{model_path}'. "
                "All frames that pass the HSV gate will be classified as fire "
                "until a real model is provided.  See the TODO in hybrid_detector.py."
            )

    def reset(self) -> None:
        """Reset inter-frame state (call between videos)."""
        self._hsv.reset()

    def process_frame(self, frame: np.ndarray, annotate: bool = False) -> FrameResult:
        """
        Process one BGR frame through the two-stage pipeline.

        Stage 1 — HSV filter (always runs, O(pixels)):
          Computes the fire-color pixel count.  If it is below hsv_gate_threshold,
          return immediately with fire_detected=False — no NN call needed.

        Stage 2 — MobileNetV2 classifier (conditional, O(224²·model_depth)):
          Runs only when the HSV gate fires.  Its sigmoid output is the
          authoritative fire/no-fire decision; the HSV bounding boxes are
          kept for spatial annotation but do not override the NN verdict.

        Returns
        -------
        FrameResult with fire_detected reflecting the NN decision (or HSV
        fallback if no model is loaded).
        """
        hsv_result = self._hsv.process_frame(frame, annotate=annotate)

        # --- Stage 1 gate ---
        # If too few fire-colored pixels, skip the NN entirely.
        if hsv_result.hsv_pixel_count < self.hsv_gate_threshold:
            # Overwrite fire_detected to False: even if HSV contours triggered,
            # we trust the pixel-count gate as a more conservative signal here.
            hsv_result.fire_detected = False
            return hsv_result

        # --- Stage 2: NN classifier ---
        if self._model is None:
            # Fallback: no model available, trust HSV result as-is.
            # TODO: remove this branch once MODEL_PATH is populated.
            return hsv_result

        patch = _preprocess_for_mobilenet(frame)
        patch_batch = np.expand_dims(patch, axis=0)  # (1, 224, 224, 3)
        confidence = float(self._model.predict(patch_batch, verbose=0)[0][0])

        fire_detected = confidence >= self.nn_confidence_threshold

        # Re-use HSV detections for bounding boxes even when NN overrides verdict.
        # The boxes show *where* the color signature is; the NN says *whether* it's fire.
        if annotate and hsv_result.annotated_frame is not None:
            label = f"Fire ({confidence:.2f})" if fire_detected else f"No Fire ({confidence:.2f})"
            color = (0, 0, 255) if fire_detected else (0, 200, 0)
            cv2.putText(
                hsv_result.annotated_frame,
                label,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        hsv_result.fire_detected = fire_detected
        return hsv_result

    def run_on_video(self, video_path: str, annotate: bool = False):
        """
        Generator yielding FrameResult for every frame of a video file.

        Mirrors HSVFireDetector.run_on_video so both detectors share the same
        calling convention in evaluate.py.
        """
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self.process_frame(frame, annotate=annotate)
        finally:
            cap.release()
