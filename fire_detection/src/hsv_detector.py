"""
HSV-based fire detector — baseline approach.

Detects fire using two complementary signals:
  1. Color: pixels whose hue/saturation/value fall in a fire-like orange-yellow range.
  2. Motion: frame-to-frame difference in HSV space to filter static warm-colored objects
     (e.g. incandescent lights, sunlit walls) that would otherwise cause false positives.

Both masks are AND-ed so a region must be *both* fire-colored *and* moving to trigger.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    """A single fire detection bounding box within one frame."""
    x: int
    y: int
    w: int
    h: int
    contour_area: float


@dataclass
class FrameResult:
    """All information produced by processing a single frame."""
    fire_detected: bool
    hsv_pixel_count: int          # raw count of color-matched pixels (before motion gate)
    detections: List[Detection] = field(default_factory=list)
    annotated_frame: np.ndarray = None  # BGR frame with boxes drawn (optional)


class HSVFireDetector:
    """
    Rule-based fire detector using HSV color segmentation + motion gating.

    Parameters
    ----------
    frame_size : tuple[int, int]
        (width, height) to resize every input frame to before processing.
        Keeping this fixed makes pixel-count thresholds meaningful across sources.
    blur_kernel : int
        Side length of the Gaussian blur kernel (must be odd).  Blur reduces
        sensor noise and small highlights that mimic fire color.
    hsv_lower : list[int]
        Lower HSV bound [H, S, V] for the fire color range.
    hsv_upper : list[int]
        Upper HSV bound [H, S, V] for the fire color range.
        Default hue window 18-35 covers orange-to-yellow flames.
        Narrow S/V floors (85+) exclude washed-out or dark regions.
    motion_threshold : int
        Pixel-intensity difference (in any HSV channel) required to call a
        pixel "moving".  25 is a practical sweet-spot: low enough to catch
        slow fire growth, high enough to ignore camera sensor jitter.
    contour_min_area : int
        Minimum contour area in pixels to report as a detection.  Filters
        tiny color blobs from reflections and compression artifacts.
    pixel_count_threshold : int
        If the total number of fire-colored pixels exceeds this, the frame
        is marked fire_detected=True regardless of contour analysis.
        Also re-exported for use by HybridDetector as a pre-filter gate.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (700, 500),
        blur_kernel: int = 21,
        hsv_lower: List[int] = None,
        hsv_upper: List[int] = None,
        motion_threshold: int = 25,
        contour_min_area: int = 400,
        pixel_count_threshold: int = 15000,
    ):
        self.frame_size = frame_size
        self.blur_kernel = (blur_kernel, blur_kernel)
        self.hsv_lower = np.array(hsv_lower or [18, 85, 85], dtype="uint8")
        self.hsv_upper = np.array(hsv_upper or [35, 255, 255], dtype="uint8")
        self.motion_threshold = motion_threshold
        self.contour_min_area = contour_min_area
        self.pixel_count_threshold = pixel_count_threshold

        # Previous-frame HSV is kept as state so the caller doesn't need to
        # manage it; reset between videos via reset().
        self._prev_hsv: np.ndarray = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear inter-frame state.  Call this when switching to a new video."""
        self._prev_hsv = None

    def process_frame(self, frame: np.ndarray, annotate: bool = False) -> FrameResult:
        """
        Process a single BGR frame and return a FrameResult.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from cv2.VideoCapture.
        annotate : bool
            If True, draw bounding boxes onto a copy of the resized frame
            and attach it to FrameResult.annotated_frame.

        Returns
        -------
        FrameResult
        """
        # --- pre-processing ---
        resized = cv2.resize(frame, self.frame_size)
        blurred = cv2.GaussianBlur(resized, self.blur_kernel, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # --- color mask ---
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        hsv_pixel_count = cv2.countNonZero(color_mask)

        # --- motion mask ---
        # On the first frame there is no previous frame; skip motion gating.
        if self._prev_hsv is None:
            combined_mask = color_mask
        else:
            diff = cv2.absdiff(self._prev_hsv, hsv)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(
                gray_diff, self.motion_threshold, 255, cv2.THRESH_BINARY
            )
            # AND: require both fire color AND pixel-level motion.
            # This rejects warm-colored static objects (lamps, sunlit walls).
            combined_mask = cv2.bitwise_and(color_mask, motion_mask)

        self._prev_hsv = hsv.copy()

        # --- contour detection ---
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.contour_min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(x=x, y=y, w=w, h=h, contour_area=area))

        fire_detected = bool(detections) or (hsv_pixel_count > self.pixel_count_threshold)

        # --- optional annotation ---
        annotated = None
        if annotate:
            annotated = resized.copy()
            for det in detections:
                cv2.rectangle(
                    annotated,
                    (det.x, det.y),
                    (det.x + det.w, det.y + det.h),
                    (0, 255, 0),
                    2,
                )
            if fire_detected:
                cv2.putText(
                    annotated,
                    "Fire Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        return FrameResult(
            fire_detected=fire_detected,
            hsv_pixel_count=hsv_pixel_count,
            detections=detections,
            annotated_frame=annotated,
        )

    def run_on_video(self, video_path: str, annotate: bool = False):
        """
        Generator that yields FrameResult for every frame of a video file.

        Usage
        -----
        detector = HSVFireDetector()
        for result in detector.run_on_video("fire.mp4"):
            if result.fire_detected:
                ...
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
