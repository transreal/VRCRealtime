"""Local full-body candidate detection for VRChat owner search.

The detector uses OpenCV Zoo's MediaPipe Pose person detector.  It runs only
on local VRChat captures; frames and detections never leave the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

import VRCRealtime_assets as assets


@dataclass(frozen=True)
class AvatarDetection:
    bbox: tuple[int, int, int, int]
    score: float
    hip_center: tuple[float, float]
    shoulder_center: tuple[float, float]


class MediaPipePersonDetector:
    """Small OpenCV-DNN wrapper for the 224px MediaPipe person model."""

    INPUT_SIZE = 224

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.30,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"person detector model not found: {path}")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.net = cv2.dnn.readNet(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.anchors = self._generate_anchors()

    MODEL_DIRECTORY = "person_detection_mediapipe"
    MODEL_FILENAME = "person_detection_mediapipe_2023mar.onnx"

    @classmethod
    def default_model_path(cls) -> Path:
        """Resolve the model through the SourceVault_vision asset roots.

        Raises MissingAsset (a FileNotFoundError) naming every path searched,
        so an unconfigured machine gets an actionable message rather than a
        bare "file not found" for a path it never expected to exist.
        """

        found = assets.find_vision_model(cls.MODEL_DIRECTORY, cls.MODEL_FILENAME)
        if found is not None:
            return found
        raise assets.MissingAsset(
            f"person detector model ({cls.MODEL_FILENAME})",
            [root / cls.MODEL_DIRECTORY / cls.MODEL_FILENAME
             for root in assets.vision_search_path()],
        )

    @classmethod
    def _generate_anchors(cls) -> NDArray[np.float32]:
        """Generate the MediaPipe SSD anchors used by the published model."""

        num_layers = 5
        min_scale = 0.1484375
        max_scale = 0.75
        strides = (8, 16, 32, 32, 32)

        def layer_scale(layer: int) -> float:
            if num_layers == 1:
                return (min_scale + max_scale) * 0.5
            return min_scale + (max_scale - min_scale) * layer / (num_layers - 1)

        anchors: list[tuple[float, float]] = []
        layer = 0
        while layer < num_layers:
            last_same_stride = layer
            scales: list[float] = []
            while (
                last_same_stride < num_layers
                and strides[last_same_stride] == strides[layer]
            ):
                scale = layer_scale(last_same_stride)
                scales.append(scale)
                next_scale = (
                    1.0
                    if last_same_stride == num_layers - 1
                    else layer_scale(last_same_stride + 1)
                )
                scales.append(float(np.sqrt(scale * next_scale)))
                last_same_stride += 1
            stride = strides[layer]
            feature_size = int(np.ceil(cls.INPUT_SIZE / stride))
            for y in range(feature_size):
                centre_y = (y + 0.5) / feature_size
                for x in range(feature_size):
                    centre_x = (x + 0.5) / feature_size
                    for _scale in scales:
                        # The model uses fixed-size anchors. Width and height
                        # are therefore not needed for decoding.
                        anchors.append((centre_x, centre_y))
            layer = last_same_stride
        result = np.asarray(anchors, dtype=np.float32)
        if result.shape != (2254, 2):
            raise RuntimeError(f"unexpected person-detector anchors: {result.shape}")
        return result

    def detect(self, frame_rgb: NDArray[np.uint8]) -> list[AvatarDetection]:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"unexpected person detector frame: {frame_rgb.shape}")
        image_height, image_width = frame_rgb.shape[:2]
        ratio = min(
            self.INPUT_SIZE / max(image_height, 1),
            self.INPUT_SIZE / max(image_width, 1),
        )
        resized_width = max(1, int(round(image_width * ratio)))
        resized_height = max(1, int(round(image_height * ratio)))
        resized = cv2.resize(
            frame_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        pad_left = (self.INPUT_SIZE - resized_width) // 2
        pad_top = (self.INPUT_SIZE - resized_height) // 2
        pad_right = self.INPUT_SIZE - resized_width - pad_left
        pad_bottom = self.INPUT_SIZE - resized_height - pad_top
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        normalized = (padded.astype(np.float32) / 255.0 - 0.5) * 2.0
        blob = np.transpose(normalized, (2, 0, 1))[None, ...]
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        regression = next(output for output in outputs if output.shape[-1] == 12)[0]
        logits = next(output for output in outputs if output.shape[-1] == 1)[0, :, 0]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -100, 100)))
        selected = np.flatnonzero(scores >= self.score_threshold)
        if selected.size == 0:
            return []

        scale = float(max(image_width, image_height))
        pad_original = np.asarray(
            [pad_left / max(ratio, 1.0e-8), pad_top / max(ratio, 1.0e-8)],
            dtype=np.float32,
        )
        landmarks = regression[selected, 4:].reshape(-1, 4, 2)
        landmarks = landmarks / self.INPUT_SIZE
        landmarks += self.anchors[selected, None, :]
        landmarks *= scale
        landmarks -= pad_original[None, None, :]

        boxes: list[list[int]] = []
        valid_scores: list[float] = []
        valid_landmarks: list[NDArray[np.float32]] = []
        for score, points in zip(scores[selected], landmarks):
            hip = points[0]
            full_body_point = points[1]
            shoulder = points[2]
            upper_body_point = points[3]
            full_radius = float(np.linalg.norm(hip - full_body_point))
            upper_radius = float(np.linalg.norm(shoulder - upper_body_point))
            radius = max(full_radius, upper_radius * 1.45)
            if radius < 12.0:
                continue
            left = int(round(hip[0] - radius))
            top = int(round(hip[1] - radius))
            right = int(round(hip[0] + radius))
            bottom = int(round(hip[1] + radius))
            left = max(0, min(image_width - 1, left))
            top = max(0, min(image_height - 1, top))
            right = max(left + 1, min(image_width, right))
            bottom = max(top + 1, min(image_height, bottom))
            width = right - left
            height = bottom - top
            if width < 20 or height < 40:
                continue
            boxes.append([left, top, width, height])
            valid_scores.append(float(score))
            valid_landmarks.append(points)
        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(
            boxes,
            valid_scores,
            self.score_threshold,
            self.nms_threshold,
        )
        results: list[AvatarDetection] = []
        for raw_index in np.asarray(keep).reshape(-1):
            index = int(raw_index)
            points = valid_landmarks[index]
            results.append(
                AvatarDetection(
                    bbox=tuple(boxes[index]),
                    score=valid_scores[index],
                    hip_center=(float(points[0, 0]), float(points[0, 1])),
                    shoulder_center=(float(points[2, 0]), float(points[2, 1])),
                )
            )
        return sorted(results, key=lambda detection: detection.score, reverse=True)

    def detect_multiscale(
        self,
        frame_rgb: NDArray[np.uint8],
    ) -> list[AvatarDetection]:
        """Detect distant people using the full frame and overlapping strips.

        A 16:9 VRChat capture is heavily reduced when it is letterboxed into
        the model's square 224px input.  Three overlapping, full-height strips
        make a distant avatar roughly 1.6 times larger without losing the
        vertical body context needed by the MediaPipe detector.  Results are
        mapped back to full-frame coordinates and deduplicated with NMS.
        """

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"unexpected person detector frame: {frame_rgb.shape}")
        image_height, image_width = frame_rgb.shape[:2]
        detections = list(self.detect(frame_rgb))
        if image_width < 320 or image_height < 180:
            return detections

        strip_width = min(
            image_width,
            max(224, int(round(image_width * 0.60))),
        )
        max_left = max(0, image_width - strip_width)
        strip_lefts = sorted({0, max_left // 2, max_left})
        for left in strip_lefts:
            strip = frame_rgb[:, left : left + strip_width]
            for detection in self.detect(strip):
                x, y, width, height = detection.bbox
                detections.append(
                    AvatarDetection(
                        bbox=(x + left, y, width, height),
                        score=detection.score,
                        hip_center=(
                            detection.hip_center[0] + left,
                            detection.hip_center[1],
                        ),
                        shoulder_center=(
                            detection.shoulder_center[0] + left,
                            detection.shoulder_center[1],
                        ),
                    )
                )
        if not detections:
            return []

        boxes = [list(detection.bbox) for detection in detections]
        scores = [float(detection.score) for detection in detections]
        keep = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            max(0.05, self.score_threshold * 0.80),
            self.nms_threshold,
        )
        merged = [detections[int(index)] for index in np.asarray(keep).reshape(-1)]
        return sorted(merged, key=lambda detection: detection.score, reverse=True)
