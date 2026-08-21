"""Local visual-attention gate and shared VRChat frame source.

VRC Pilot continuously captures the VRChat window. OpenCV uses the frames
locally for the attention gate, while :meth:`snapshot` makes the most recent
frame available to explicitly enabled owner-follow and on-demand vision paths.
No frame is sent externally by this class itself.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from vrcpilot.capture import CaptureLoop

#: The aspect the size thresholds are expressed in.  A Unity camera keeps its
#: *vertical* field of view fixed and lets the horizontal one follow the aspect
#: ratio, so "how big does a face look at 2 m" is constant as a fraction of
#: frame HEIGHT, not width.  The options are historically written as a fraction
#: of width, so they are converted through this reference aspect: at 16:9 the
#: numbers mean exactly what they always meant, and at any other aspect they
#: now mean the same *angle* instead of silently changing.
REFERENCE_ASPECT = 16.0 / 9.0

#: Anything darker than this everywhere along a row/column is letterbox.
LETTERBOX_LEVEL = 8.0


def content_rect(
    frame: NDArray[np.uint8], *, level: float = LETTERBOX_LEVEL
) -> tuple[int, int, int, int]:
    """The rendered image inside the captured window, as ``(x, y, w, h)``.

    VRChat in borderless fullscreen renders at its own resolution and letter-
    boxes that into the window, so on a portrait monitor the capture is (for
    example) 2560x2880 carrying picture only in the middle 2560x1440.
    Measuring "fraction of the frame" or "is it centred" against the window
    then answers a question nobody asked.

    Falls back to the whole frame whenever the borders are not a clean dark
    band: a wrong crop would be far worse than no crop.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        return (0, 0, int(frame.shape[1]), int(frame.shape[0]))
    height, width = frame.shape[:2]
    # A coarse subsample is enough and keeps this off the critical path: the
    # bars are hundreds of pixels thick, never one.
    probe = frame[::8, ::8]
    brightest = probe.max(axis=2)
    rows = np.where(brightest.max(axis=1) > level)[0]
    cols = np.where(brightest.max(axis=0) > level)[0]
    if rows.size == 0 or cols.size == 0:
        return (0, 0, width, height)
    top = int(rows[0]) * 8
    bottom = min(height, (int(rows[-1]) + 1) * 8)
    left = int(cols[0]) * 8
    right = min(width, (int(cols[-1]) + 1) * 8)
    if bottom - top < height * 0.2 or right - left < width * 0.2:
        return (0, 0, width, height)
    return (left, top, right - left, bottom - top)


class VisualAttentionGate:
    """Detect a centred, frontal, visibly speaking VRChat avatar."""

    def __init__(
        self,
        *,
        fps: float,
        pid: int | None,
        center_tolerance: float,
        min_face_fraction: float,
        motion_threshold: float,
        audio_rms_threshold: float,
        max_stereo_imbalance: float,
        hangover_ms: int,
        require_nameplate_pulse: bool,
        nameplate_blue_threshold: float,
        state_callback: Callable[..., None] | None = None,
    ) -> None:
        self.fps = fps
        self.pid = pid
        self.center_tolerance = center_tolerance
        self.min_face_fraction = min_face_fraction
        self.motion_threshold = motion_threshold
        self.audio_rms_threshold = audio_rms_threshold
        self.max_stereo_imbalance = max_stereo_imbalance
        self.hangover_seconds = hangover_ms / 1000.0
        self.require_nameplate_pulse = require_nameplate_pulse
        self.nameplate_blue_threshold = nameplate_blue_threshold
        self.state_callback = state_callback
        #: Detection runs on the rendered image scaled to this height.
        #: Scaling by height rather than by the window's width keeps the
        #: working image identical whatever the letterbox does.
        self.working_height = 480
        self.last_content_rect: tuple[int, int, int, int] | None = None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("OpenCV frontal-face cascade could not be loaded")

        self.lock = threading.Lock()
        self.capture_loop: CaptureLoop | None = None
        self.previous_mouth: NDArray[np.uint8] | None = None
        self.previous_upper: NDArray[np.uint8] | None = None
        self.previous_face: tuple[int, int, int, int] | None = None
        self.motion_hits: deque[bool] = deque(maxlen=3)
        self.stable_face_frames = 0
        self.last_face_seen = 0.0
        self.mouth_active_until = 0.0
        self.nameplate_active_until = 0.0
        self.audio_open_until = 0.0
        self.attention_candidate_until = 0.0
        self.last_mouth_motion = 0.0
        self.last_mouth_delta = 0.0
        self.last_upper_delta = 0.0
        self.last_audio_rms = 0.0
        self.last_stereo_balance = 0.0
        self.last_face_box: tuple[int, int, int, int] | None = None
        self.last_nameplate_box: tuple[int, int, int, int] | None = None
        self.last_nameplate_blue_ratio = 0.0
        self.reported_open = False
        self.latest_frame: NDArray[np.uint8] | None = None
        self.latest_frame_at = 0.0

    def start(self) -> None:
        if self.capture_loop is not None:
            return
        loop = CaptureLoop(
            self._on_frame,
            fps=self.fps,
            frame_timeout=2.0,
            pid=self.pid,
        )
        loop.start()
        self.capture_loop = loop
        self._report(visualGateCaptureRunning=True, visualGateLastError=None)

    def close(self) -> None:
        loop = self.capture_loop
        self.capture_loop = None
        if loop is not None:
            try:
                loop.close()
            except Exception as exc:
                self._report(
                    visualGateCaptureRunning=False,
                    visualGateLastError=f"{type(exc).__name__}: {exc}",
                )
                return
        self._report(visualGateCaptureRunning=False)

    def observe_audio(self, chunk: NDArray[np.float32]) -> None:
        """Correlate local audio energy with the latest mouth motion."""
        if chunk.size == 0:
            return
        audio = np.asarray(chunk, dtype=np.float32)
        if audio.ndim == 1:
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            balance = 0.0
        else:
            left = audio[:, 0]
            right = audio[:, 1] if audio.shape[1] > 1 else left
            left_rms = float(np.sqrt(np.mean(np.square(left), dtype=np.float64)))
            right_rms = float(np.sqrt(np.mean(np.square(right), dtype=np.float64)))
            rms = (left_rms + right_rms) * 0.5
            balance = (right_rms - left_rms) / max(left_rms + right_rms, 1.0e-9)

        now = time.monotonic()
        with self.lock:
            self.last_audio_rms = rms
            self.last_stereo_balance = balance
            face_recent = now - self.last_face_seen <= max(0.35, 2.5 / self.fps)
            mouth_recent = now <= self.mouth_active_until
            nameplate_recent = now <= self.nameplate_active_until
            visual_speech = (
                nameplate_recent if self.require_nameplate_pulse else mouth_recent
            )
            attention_candidate = (
                face_recent
                and (nameplate_recent or mouth_recent)
                and rms >= self.audio_rms_threshold
                and abs(balance) <= self.max_stereo_imbalance
            )
            if attention_candidate:
                # This weaker signal is used only to pause owner-follow motion.
                # Audio is still withheld until the stricter mouth/nameplate
                # correlation below opens the actual conversation gate.
                self.attention_candidate_until = max(
                    self.attention_candidate_until,
                    now + max(0.8, 3.0 / self.fps),
                )
            if (
                face_recent
                and visual_speech
                and rms >= self.audio_rms_threshold
                and abs(balance) <= self.max_stereo_imbalance
            ):
                self.audio_open_until = max(
                    self.audio_open_until, now + self.hangover_seconds
                )
            is_open = face_recent and now <= self.audio_open_until
            changed = is_open != self.reported_open
            self.reported_open = is_open
        if changed:
            self._report(
                visualGateOpen=is_open,
                visualGateAudioRMS=round(rms, 5),
                visualGateStereoBalance=round(balance, 4),
            )

    def attention_candidate(self) -> bool:
        """Return whether a centred visible face may currently be addressing us."""

        now = time.monotonic()
        with self.lock:
            face_recent = now - self.last_face_seen <= max(0.35, 2.5 / self.fps)
            return face_recent and now <= self.attention_candidate_until

    def allows_audio(self) -> bool:
        now = time.monotonic()
        with self.lock:
            face_recent = now - self.last_face_seen <= max(0.35, 2.5 / self.fps)
            return face_recent and now <= self.audio_open_until

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "faceDetected": self.last_face_box is not None,
                "faceBox": list(self.last_face_box) if self.last_face_box else None,
                "nameplateBox": (
                    list(self.last_nameplate_box)
                    if self.last_nameplate_box
                    else None
                ),
                "nameplateBlueRatio": self.last_nameplate_blue_ratio,
                "nameplateSpeaking": time.monotonic() <= self.nameplate_active_until,
                "mouthMotion": self.last_mouth_motion,
                "mouthDelta": self.last_mouth_delta,
                "upperDelta": self.last_upper_delta,
                "audioRMS": self.last_audio_rms,
                "stereoBalance": self.last_stereo_balance,
                "attentionCandidate": (
                    time.monotonic() <= self.attention_candidate_until
                ),
                "open": self.reported_open,
            }

    def snapshot(self, *, max_age_seconds: float = 2.0) -> NDArray[np.uint8] | None:
        """Return a detached copy of the freshest VRChat frame, if recent."""

        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be > 0")
        now = time.monotonic()
        with self.lock:
            frame = self.latest_frame
            captured_at = self.latest_frame_at
            if frame is None or now - captured_at > max_age_seconds:
                return None
            return frame.copy()

    def _report(self, **updates: Any) -> None:
        if self.state_callback is not None:
            self.state_callback(**updates)

    def _on_frame(self, frame: NDArray[np.uint8]) -> None:
        try:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"unexpected capture frame shape: {frame.shape}")
            with self.lock:
                self.latest_frame = np.asarray(frame, dtype=np.uint8).copy()
                self.latest_frame_at = time.monotonic()
            self._process_frame(frame)
        except Exception as exc:
            self._report(visualGateLastError=f"{type(exc).__name__}: {exc}")

    def _process_frame(self, frame: NDArray[np.uint8]) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"unexpected capture frame shape: {frame.shape}")
        height, width = frame.shape[:2]
        content_x, content_y, content_w, content_h = content_rect(frame)
        with self.lock:
            self.last_content_rect = (content_x, content_y, content_w, content_h)
        content = frame[
            content_y : content_y + content_h, content_x : content_x + content_w
        ]
        scale = min(1.0, self.working_height / max(content_h, 1))
        small = cv2.resize(
            content,
            (
                max(1, int(round(content_w * scale))),
                max(1, int(round(content_h * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        gray_small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # min_face_fraction is a fraction of a 16:9 width; going through the
        # height keeps it the same angle on any aspect (see REFERENCE_ASPECT).
        min_size = max(
            30,
            int(round(gray_small.shape[0] * REFERENCE_ASPECT * self.min_face_fraction)),
        )
        detected = self.face_detector.detectMultiScale(
            gray_small,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(min_size, min_size),
        )
        face = self._select_central_face(
            detected,
            gray_small.shape[1],
            gray_small.shape[0],
        )
        now = time.monotonic()
        if face is None:
            with self.lock:
                if now - self.last_face_seen > 0.5:
                    self.previous_mouth = None
                    self.previous_upper = None
                    self.previous_face = None
                    self.motion_hits.clear()
                    self.stable_face_frames = 0
                self.last_face_box = None
                self.last_nameplate_box = None
                self.last_nameplate_blue_ratio = 0.0
                self.last_mouth_motion = 0.0
            self._report(
                visualGateContentRect=[content_x, content_y, content_w, content_h],
                visualGateFaceFraction=self._largest_face_fraction(
                    detected, gray_small.shape[0]
                ),
                visualGateFaceFractionRequired=round(self.min_face_fraction, 4),
                visualGateFaceDetected=False,
                visualGateFaceBox=None,
                visualGateNameplateBox=None,
                visualGateNameplateBlueRatio=0.0,
                visualGateNameplateSpeaking=False,
                visualGateMouthMotion=0.0,
            )
            return

        # Back to full-frame coordinates: the mouth ROI and the nameplate
        # search below both work on `frame`, not on the crop.
        x, y, face_width, face_height = (
            int(round(value / scale)) for value in face
        )
        x += content_x
        y += content_y
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        face_width = max(1, min(face_width, width - x))
        face_height = max(1, min(face_height, height - y))
        with self.lock:
            previous_face = self.previous_face
        if self._same_face(previous_face, (x, y, face_width, face_height)):
            assert previous_face is not None
            # Haar boxes jitter by a few pixels on stylised avatars. Smooth
            # that jitter before extracting a small mouth ROI so idle head
            # motion is not mistaken for speech.
            x = int(round(previous_face[0] * 0.72 + x * 0.28))
            y = int(round(previous_face[1] * 0.72 + y * 0.28))
            face_width = int(round(previous_face[2] * 0.72 + face_width * 0.28))
            face_height = int(round(previous_face[3] * 0.72 + face_height * 0.28))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        nameplate_box, nameplate_blue_ratio = self._nameplate_metrics(
            frame,
            gray,
            (x, y, face_width, face_height),
        )
        mouth = self._normalized_roi(
            gray,
            x + int(0.27 * face_width),
            y + int(0.55 * face_height),
            int(0.46 * face_width),
            int(0.28 * face_height),
            (64, 32),
        )
        upper = self._normalized_roi(
            gray,
            x + int(0.20 * face_width),
            y + int(0.18 * face_height),
            int(0.60 * face_width),
            int(0.28 * face_height),
            (64, 32),
        )

        with self.lock:
            stable = self._same_face(self.previous_face, (x, y, face_width, face_height))
            self.stable_face_frames = self.stable_face_frames + 1 if stable else 1
            if stable and self.previous_mouth is not None and self.previous_upper is not None:
                mouth_delta = float(
                    np.mean(cv2.absdiff(mouth, self.previous_mouth), dtype=np.float64)
                )
                upper_delta = float(
                    np.mean(cv2.absdiff(upper, self.previous_upper), dtype=np.float64)
                )
                motion = max(0.0, mouth_delta - 0.85 * upper_delta) / 255.0
            else:
                motion = 0.0
                mouth_delta = 0.0
                upper_delta = 0.0
                self.motion_hits.clear()
            self.motion_hits.append(motion >= self.motion_threshold)
            if self.stable_face_frames >= 2 and sum(self.motion_hits) >= 2:
                self.mouth_active_until = now + max(0.25, 2.5 / self.fps)
            if nameplate_blue_ratio >= self.nameplate_blue_threshold:
                self.nameplate_active_until = now + max(0.30, 2.5 / self.fps)
            self.previous_mouth = mouth
            self.previous_upper = upper
            self.previous_face = (x, y, face_width, face_height)
            self.last_face_box = (x, y, face_width, face_height)
            self.last_nameplate_box = nameplate_box
            self.last_nameplate_blue_ratio = nameplate_blue_ratio
            self.last_face_seen = now
            self.last_mouth_motion = motion
            self.last_mouth_delta = mouth_delta / 255.0
            self.last_upper_delta = upper_delta / 255.0
        self._report(
            visualGateContentRect=[content_x, content_y, content_w, content_h],
            # Reported on the hit path too: knowing how much headroom the size
            # criterion has is what tells you whether the gate is about to
            # start missing when somebody stands one step further back.
            visualGateFaceFraction=round(
                face_width / (REFERENCE_ASPECT * max(content_h, 1)), 4
            ),
            visualGateFaceFractionRequired=round(self.min_face_fraction, 4),
            visualGateFaceDetected=True,
            visualGateFaceBox=[x, y, face_width, face_height],
            visualGateNameplateBox=(list(nameplate_box) if nameplate_box else None),
            visualGateNameplateBlueRatio=round(nameplate_blue_ratio, 6),
            visualGateNameplateSpeaking=(now <= self.nameplate_active_until),
            visualGateMouthMotion=round(motion, 5),
        )

    def _nameplate_metrics(
        self,
        frame: NDArray[np.uint8],
        gray: NDArray[np.uint8],
        face: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        """Locate the dark nameplate above a face and measure its blue pulse."""
        image_height, image_width = gray.shape[:2]
        face_x, face_y, face_width, face_height = face
        centre_x = face_x + face_width // 2
        box_width = max(48, int(round(face_width * 1.35)))
        box_height = max(28, int(round(face_width * 0.72)))
        x = max(0, min(centre_x - box_width // 2, image_width - box_width))
        max_y = min(
            image_height - box_height,
            max(0, face_y - int(round(face_height * 0.60))),
        )
        if box_width > image_width or box_height > image_height:
            return None, 0.0

        step = max(3, box_height // 12)
        best_score = 0.0
        best_blue_ratio = 0.0
        best_y: int | None = None
        for candidate_y in range(0, max_y + 1, step):
            gray_roi = gray[
                candidate_y : candidate_y + box_height,
                x : x + box_width,
            ]
            dark_ratio = float(np.mean(gray_roi < 170))
            if dark_ratio < 0.45:
                continue
            hsv = cv2.cvtColor(
                frame[
                    candidate_y : candidate_y + box_height,
                    x : x + box_width,
                ],
                cv2.COLOR_RGB2HSV,
            )
            blue = (
                (hsv[:, :, 0] >= 75)
                & (hsv[:, :, 0] <= 135)
                & (hsv[:, :, 1] >= 65)
                & (hsv[:, :, 2] >= 105)
            )
            blue_ratio = float(np.mean(blue))
            if blue_ratio < self.nameplate_blue_threshold * 0.5:
                continue
            # A speaking nameplate is a mostly dark horizontal plate with a
            # small, saturated blue pulsing border. Broad blue world scenery
            # fails the dark-area condition and is not selected.
            score = blue_ratio * np.sqrt(dark_ratio)
            if score > best_score:
                best_score = score
                best_blue_ratio = blue_ratio
                best_y = candidate_y
        if best_y is None:
            return None, 0.0

        box = (x, best_y, box_width, box_height)
        return box, best_blue_ratio

    def _largest_face_fraction(self, detected: Any, small_height: int) -> float:
        """Biggest face seen, as a min_face_fraction-comparable number.

        Without this a gate that never opens looks identical to a gate that
        misses by ten percent -- which is exactly how a conversational-distance
        owner was silently ignored (measured 0.063 against a 0.07 threshold,
        2026-08-21).  Reporting the measurement makes the next near-miss a
        number instead of a mystery.
        """

        widest = 0
        for raw in detected:
            widest = max(widest, int(raw[2]))
        if widest == 0 or small_height <= 0:
            return 0.0
        return round(widest / (REFERENCE_ASPECT * small_height), 4)

    def _select_central_face(
        self,
        detected: Any,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for raw in detected:
            x, y, face_width, face_height = (int(value) for value in raw)
            centre_x = x + face_width * 0.5
            centre_y = y + face_height * 0.5
            horizontal_distance = abs(centre_x - width * 0.5) / max(width, 1)
            if horizontal_distance > self.center_tolerance:
                continue
            if not 0.15 <= centre_y / max(height, 1) <= 0.78:
                continue
            # Same conversion as min_size: an angle, not a width fraction.
            if face_width < self.min_face_fraction * REFERENCE_ASPECT * max(height, 1):
                continue
            score = face_width * face_height * (1.0 - horizontal_distance)
            candidates.append((score, (x, y, face_width, face_height)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _normalized_roi(
        gray: NDArray[np.uint8],
        x: int,
        y: int,
        width: int,
        height: int,
        output_size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        image_height, image_width = gray.shape[:2]
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))
        roi = gray[y : y + height, x : x + width]
        resized = cv2.resize(roi, output_size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(resized, (3, 3), 0)

    @staticmethod
    def _same_face(
        previous: tuple[int, int, int, int] | None,
        current: tuple[int, int, int, int],
    ) -> bool:
        if previous is None:
            return False
        px, py, pw, ph = previous
        x, y, width, height = current
        centre_shift = np.hypot(
            (x + width * 0.5) - (px + pw * 0.5),
            (y + height * 0.5) - (py + ph * 0.5),
        )
        scale_change = max(width / max(pw, 1), pw / max(width, 1))
        return centre_shift <= max(width, height) * 0.28 and scale_change <= 1.35
