"""Geometric owner-formation controller for the VRChat bridge.

Goal: find the owner's nameplate, walk up to the owner, move to the owner's
left-rear, face (roughly) the same direction as the owner and stop with the
owner held at the right edge of the view.  When the owner walks away, follow
and re-form when they stop.

Everything here is local: VRChat frames, OCR text and pose landmarks never
leave the machine.  Only OSC axes and relative mouse motion are emitted.

Measured facts this design relies on (live, 2026-08-19, desktop VRChat):

* Relative mouse motion turns the avatar deterministically:
  3600 counts per 360 degrees of yaw (``YAW_COUNTS_PER_DEGREE``).  OSC
  LookLeft/LookRight buttons turn by an unpredictable angle and are only a
  fallback when the window cannot be focused.
* The desktop camera is a 60-degree-vertical pinhole: focal length about
  0.496 x frame width (HFOV ~90 degrees).  Nameplate x therefore converts to
  a bearing exactly.
* VRChat nameplates keep an almost constant *screen* size regardless of
  distance, so nameplate height is useless as a range sensor.  Range comes
  from body landmarks (OpenCV Zoo MediaPipe pose model): torso length in
  pixels vs. the model's metric torso length gives metres directly.
* The pose model's yaw is reliable when the face is visible (front
  hemisphere), compressed towards 0 for back-oblique views and noisy at the
  profile.  Yaw odometry (mouse counts) is exact, so heading is fused:
  measurement when trustworthy, odometry in between, owner motion direction
  whenever the owner walks.
* OSC ``Vertical``/``Horizontal`` axes move at about 2.0 m/s per unit axis.
"""

from __future__ import annotations

import math
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

import VRCRealtime_assets as assets

RGBImage = NDArray[np.uint8]

# --------------------------------------------------------------------------
# Calibration constants (see module docstring)
# --------------------------------------------------------------------------

#: Relative mouse counts per degree of yaw (3600 per full turn, measured).
YAW_COUNTS_PER_DEGREE = 10.0
#: Counts from the bottom pitch clamp up to a level camera (measured).
LEVEL_FROM_BOTTOM_COUNTS = 880.0
#: Counts spanning the whole pitch range; used to reach the bottom clamp.
PITCH_RANGE_COUNTS = 1800.0
#: Camera focal length as a fraction of the frame width (60 deg vertical FOV
#: at 16:9 -> f = 637 px for a 1284 px wide capture).
FOCAL_LENGTH_FRACTION = 637.0 / 1284.0
#: Avatar walking speed per unit OSC axis (m/s).
WALK_SPEED_MPS_PER_AXIS = 2.0
#: Height of the VRChat window title bar included in captures.
DEFAULT_TITLE_BAR_PX = 45
#: Mouse counts per degree of pitch (about 1800 counts over 180 degrees).
PITCH_COUNTS_PER_DEGREE = PITCH_RANGE_COUNTS / 180.0
#: Default height of the nameplate text above the camera eye line (metres);
#: refined online against the landmark range at mid distance.
DEFAULT_NAMEPLATE_HEIGHT_M = 0.6

# BlazePose landmark indices used here.
LM_NOSE = 0
LM_LEFT_EAR, LM_RIGHT_EAR = 7, 8
LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER = 11, 12
LM_LEFT_HIP, LM_RIGHT_HIP = 23, 24


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to (-180, 180]."""

    value = math.fmod(float(angle) + 180.0, 360.0)
    if value < 0.0:
        value += 360.0
    return value - 180.0


def bearing_from_x(x_px: float, frame_width: int) -> float:
    """Horizontal angle (deg, right positive) of an image column."""

    focal = FOCAL_LENGTH_FRACTION * max(frame_width, 1)
    return math.degrees(math.atan((float(x_px) - frame_width / 2.0) / focal))


def x_from_bearing(bearing_deg: float, frame_width: int) -> float:
    focal = FOCAL_LENGTH_FRACTION * max(frame_width, 1)
    return frame_width / 2.0 + focal * math.tan(math.radians(bearing_deg))


def circular_mean_degrees(
    angles: Sequence[float], weights: Sequence[float] | None = None
) -> float | None:
    if not angles:
        return None
    if weights is None:
        weights = [1.0] * len(angles)
    sx = sum(w * math.cos(math.radians(a)) for a, w in zip(angles, weights))
    sy = sum(w * math.sin(math.radians(a)) for a, w in zip(angles, weights))
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:
        return None
    return math.degrees(math.atan2(sy, sx))


def normalise_name(text: str) -> str:
    normalised = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalised if ch.isalnum())


def title_bar_height(frame: RGBImage) -> int:
    """Rows of the window title bar at the top of a capture (0 if none)."""

    if frame.ndim != 3 or frame.shape[0] < 70:
        return 0
    rows = frame[:70].mean(axis=(1, 2))
    if rows[2] < 200:
        return 0
    for index in range(8, 70):
        if rows[index] < 200:
            return int(index)
    return DEFAULT_TITLE_BAR_PX


# --------------------------------------------------------------------------
# Protocols for injected dependencies
# --------------------------------------------------------------------------


class OCRLike(Protocol):
    def recognize(self, image: RGBImage) -> Sequence[Any]: ...


class OSCControllerLike(Protocol):
    def vertical(self, value: float) -> None: ...

    def horizontal(self, value: float) -> None: ...

    def look_left(self, active: bool = True) -> None: ...

    def look_right(self, active: bool = True) -> None: ...

    def chatbox(self, text: str, *, send: bool = True, sfx: bool = True) -> None: ...


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class FormationConfig:
    owner_account: str
    #: Distance from the owner at which to hold the formation (metres).
    follow_distance_m: float = 1.8
    #: How far behind the owner's pure left the slot sits (degrees).  0 = the
    #: bot stands exactly beside the owner, 90 = directly behind.
    rear_angle_deg: float = 35.0
    #: Where the owner should sit in the settled view (degrees right of the
    #: view centre).  33 deg is about x = 0.82 of the frame width.
    owner_bearing_deg: float = 33.0
    #: Perception loop rate cap.
    tracking_fps: float = 4.0
    #: Seconds without any owner observation before reacquisition starts.
    lost_seconds: float = 1.5
    #: Seconds of reacquisition (small turns toward the last bearing) before
    #: a full search sweep.
    reacquire_seconds: float = 4.0
    #: OSC axis magnitude used for walking/strafing.
    move_axis: float = 0.45
    #: OCR engine parameters (RapidOCR); tracking uses a small crop, search
    #: the whole frame.
    ocr_box_threshold: float = 0.35
    ocr_text_score: float = 0.35
    #: Search sweep step (degrees) and settle time before the OCR pass.
    search_step_deg: float = 30.0
    search_settle_seconds: float = 0.35
    #: Minimum distance; walking closer than this backs off.
    min_distance_m: float = 1.0
    #: Level the camera (absolute) when the controller starts following.
    level_camera_on_start: bool = True
    #: When the room search has completed this many full sweeps without the
    #: owner, check the VRChat menu (instance user list) and, if the owner is
    #: in another instance, join them there.  0 disables the menu check.
    menu_locate_after_sweeps: int = 2
    #: Minimum seconds between two menu checks.
    menu_locate_cooldown_seconds: float = 240.0
    #: Between empty sweeps, walk a couple of metres toward the most open
    #: heading instead of spinning on the spot (stuck-detected; bounded).
    search_patrol: bool = True
    search_patrol_seconds: float = 2.0
    search_patrol_max_moves: int = 6
    #: An unsuccessful search episode longer than this ends in "waiting":
    #: stand still, watch for the owner, re-check the menu periodically.
    idle_after_seconds: float = 360.0
    #: While waiting (owner offline / unreachable), re-run the menu check
    #: this often.
    idle_recheck_seconds: float = 300.0

    @property
    def target_theta_deg(self) -> float:
        """Owner heading relative to the ray bot->owner at the slot.

        0 = owner faces away from the bot, +/-180 = owner faces the bot,
        +90 = owner faces the bot's right.  At the owner's left-rear the
        owner faces away and to the bot's left.
        """

        return wrap_degrees(self.rear_angle_deg - 90.0)

    @property
    def heading_offset_deg(self) -> float:
        """Bot heading minus owner heading once settled (right positive)."""

        return 90.0 - self.rear_angle_deg - self.owner_bearing_deg


@dataclass
class NameplateObservation:
    text: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str  # "crop" | "full"


@dataclass
class BodyObservation:
    conf: float
    yaw_deg: float  # owner heading relative to camera z axis (see module doc)
    hip_yaw_deg: float
    torso_px: float
    torso_m: float
    distance_m: float
    bbox: tuple[float, float, float, float]
    shoulder_centre: tuple[float, float]
    hip_centre: tuple[float, float]
    landmarks: NDArray[np.float32] = field(repr=False)
    #: A frontal face was detected in the head region.  The landmark model
    #: reads some back views as front views (hair); without a face such a
    #: reading is only trusted as a profile-quality measurement.
    face_visible: bool = False


@dataclass
class Observation:
    time: float
    frame_shape: tuple[int, int]
    nameplate: NameplateObservation | None
    body: BodyObservation | None
    #: Bearing of the owner (deg, right positive), from the nameplate when
    #: present, else the body centre.
    bearing_deg: float | None
    distance_m: float | None
    yaw_odometry_deg: float
    #: Range from the nameplate's height in a level view (metres), when the
    #: plate was read.  Coarse but immune to the landmark model's close-range
    #: failures.
    plate_distance_m: float | None = None
    pose_distance_m: float | None = None

    @property
    def seen(self) -> bool:
        return self.bearing_deg is not None


# --------------------------------------------------------------------------
# Actuators
# --------------------------------------------------------------------------


class YawActuator:
    """Deterministic yaw via relative mouse motion with an OSC fallback."""

    def __init__(
        self,
        *,
        pid: int | None,
        osc: OSCControllerLike | None,
        relative_move: Callable[[int, int], None] | None = None,
    ) -> None:
        self.pid = pid
        self.osc = osc
        self._relative_move = relative_move
        self._last_focus_check = 0.0
        self._residue = 0.0
        self.total_yaw_deg = 0.0
        self.mouse_available = True
        self.last_error: str | None = None
        self.lock = threading.Lock()

    def _send_relative(self, dx: int, dy: int) -> None:
        if self._relative_move is not None:
            self._relative_move(dx, dy)
            return
        from vrcpilot.controls import ensure_target
        from vrcpilot.controls.mouse.windows import pydirectinput

        now = time.monotonic()
        if now - self._last_focus_check > 1.0:
            ensure_target(pid=self.pid)
            self._last_focus_check = now
        previous_pause = getattr(pydirectinput, "PAUSE", 0.1)
        pydirectinput.PAUSE = 0.004
        try:
            span = max(abs(dx), abs(dy))
            count = max(1, -(-span // 6))
            sent_x = sent_y = 0
            for index in range(1, count + 1):
                step_x = int(round(dx * index / count)) - sent_x
                step_y = int(round(dy * index / count)) - sent_y
                sent_x += step_x
                sent_y += step_y
                if step_x or step_y:
                    pydirectinput.moveRel(step_x, step_y, relative=True)
        finally:
            pydirectinput.PAUSE = previous_pause

    def turn(self, degrees: float) -> float:
        """Turn by ``degrees`` (right positive).  Returns degrees applied."""

        with self.lock:
            wanted = float(degrees) * YAW_COUNTS_PER_DEGREE + self._residue
            counts = int(round(wanted))
            self._residue = wanted - counts
            if counts == 0:
                return 0.0
            try:
                self._send_relative(counts, 0)
                self.mouse_available = True
                self.last_error = None
            except Exception as exc:
                self.mouse_available = False
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._osc_turn(counts / YAW_COUNTS_PER_DEGREE)
            applied = counts / YAW_COUNTS_PER_DEGREE
            self.total_yaw_deg += applied
            return applied

    def _osc_turn(self, degrees: float) -> None:
        """Fallback: OSC look buttons (angle is only approximate)."""

        if self.osc is None:
            return
        seconds = min(1.4, abs(degrees) / 165.0)
        if seconds < 0.03:
            return
        button = self.osc.look_right if degrees > 0 else self.osc.look_left
        button(True)
        try:
            time.sleep(seconds)
        finally:
            button(False)

    def pitch(self, counts: float) -> None:
        with self.lock:
            try:
                self._send_relative(0, int(round(counts)))
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def level_camera(self) -> None:
        """Absolute camera levelling via the bottom pitch clamp."""

        self.pitch(PITCH_RANGE_COUNTS * 1.5)
        time.sleep(0.05)
        self.pitch(-LEVEL_FROM_BOTTOM_COUNTS)


class MotionActuator:
    """OSC translation axes with dead-reckoning integration."""

    def __init__(
        self, osc: OSCControllerLike | None, *, clock: Callable[[], float] | None = None
    ) -> None:
        self.osc = osc
        self._clock = clock or time.monotonic
        self.lock = threading.Lock()
        self.forward = 0.0
        self.strafe = 0.0
        self._since = self._clock()
        self._pulse_timer: threading.Timer | None = None
        self._pulse_until = 0.0
        #: Body-frame displacement accumulated since the last ``consume``.
        self._pending: list[tuple[float, float, float]] = []

    def set_axes(self, forward: float, strafe: float) -> None:
        forward = float(max(-1.0, min(1.0, forward)))
        strafe = float(max(-1.0, min(1.0, strafe)))
        with self.lock:
            if self._pulse_timer is not None:
                self._pulse_timer.cancel()
                self._pulse_timer = None
                self._pulse_until = 0.0
            now = self._clock()
            self._flush_locked(now)
            changed = (
                abs(forward - self.forward) > 1e-6 or abs(strafe - self.strafe) > 1e-6
            )
            self.forward, self.strafe = forward, strafe
            if changed and self.osc is not None:
                self.osc.vertical(forward)
                self.osc.horizontal(strafe)

    def stop(self, *, force: bool = False) -> None:
        with self.lock:
            if self._pulse_timer is not None:
                self._pulse_timer.cancel()
                self._pulse_timer = None
                self._pulse_until = 0.0
            now = self._clock()
            self._flush_locked(now)
            was_moving = abs(self.forward) > 1e-6 or abs(self.strafe) > 1e-6
            self.forward = self.strafe = 0.0
            if (was_moving or force) and self.osc is not None:
                self.osc.vertical(0.0)
                self.osc.horizontal(0.0)

    @property
    def moving(self) -> bool:
        return abs(self.forward) > 1e-6 or abs(self.strafe) > 1e-6

    def pulse(self, forward: float, strafe: float, seconds: float) -> None:
        """Move for a bounded time, then stop (used for fine orbit steps)."""

        seconds = max(0.05, min(0.6, float(seconds)))
        self.set_axes(forward, strafe)
        timer = threading.Timer(seconds, self._pulse_end)
        timer.daemon = True
        with self.lock:
            self._pulse_timer = timer
            self._pulse_until = self._clock() + seconds
        timer.start()

    def _pulse_end(self) -> None:
        with self.lock:
            if self._pulse_until <= 0.0 or self._clock() < self._pulse_until - 0.01:
                return
        self.stop()

    def pulse_remaining(self) -> float:
        with self.lock:
            if self._pulse_until <= 0.0:
                return 0.0
            return max(0.0, self._pulse_until - self._clock())

    def expire_pulse(self) -> None:
        """Deterministic pulse end for callers that drive the clock themselves."""

        with self.lock:
            due = self._pulse_until > 0.0 and self._clock() >= self._pulse_until
        if due:
            self.stop()

    def _flush_locked(self, now: float) -> None:
        dt = max(0.0, now - self._since)
        self._since = now
        if dt > 0.0 and (abs(self.forward) > 1e-6 or abs(self.strafe) > 1e-6):
            self._pending.append((self.forward, self.strafe, dt))

    def consume(self) -> list[tuple[float, float, float]]:
        """Return (forward_axis, strafe_axis, seconds) segments since last call."""

        with self.lock:
            self._flush_locked(self._clock())
            segments = self._pending
            self._pending = []
            return segments


# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------


class NameplateFinder:
    """Locate the owner nameplate with RapidOCR (crop tracking + full sweeps)."""

    CROP_WIDTH = 340
    CROP_HEIGHT = 200

    def __init__(
        self,
        owner_account: str,
        *,
        crop_ocr_factory: Callable[[], OCRLike] | None = None,
        full_ocr_factory: Callable[[], OCRLike] | None = None,
        box_threshold: float = 0.35,
        text_score: float = 0.35,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.owner_normalised = normalise_name(owner_account)
        self._clock = clock or time.monotonic
        self._crop_ocr_factory = crop_ocr_factory
        self._full_ocr_factory = full_ocr_factory
        self.box_threshold = box_threshold
        self.text_score = text_score
        self._crop_ocr: OCRLike | None = None
        self._full_ocr: OCRLike | None = None
        self.last_box: tuple[int, int, int, int] | None = None
        self.last_full_ocr_at = 0.0
        self.last_full_ocr_seconds = 0.0
        self.last_crop_ocr_seconds = 0.0
        self.last_texts: list[str] = []

    def set_owner(self, owner_account: str) -> None:
        self.owner_normalised = normalise_name(owner_account)
        self.last_box = None

    def _make_engine(self, *, crop: bool) -> OCRLike:
        factory = self._crop_ocr_factory if crop else self._full_ocr_factory
        if factory is not None:
            return factory()
        from vrcpilot.ocr import RapidOCREngine

        params = {
            "Det.box_thresh": self.box_threshold,
            "Global.text_score": self.text_score,
            "Global.log_level": "warning",
            "Det.limit_type": "max",
            # A 340x200 crop is upscaled about 2.5x (10 px nameplate text ->
            # 25 px); the full frame 1.5x.  Both measured to read the plate.
            "Det.limit_side_len": 860 if crop else 1920,
        }
        return RapidOCREngine(params=params)

    @property
    def crop_ocr(self) -> OCRLike:
        if self._crop_ocr is None:
            self._crop_ocr = self._make_engine(crop=True)
        return self._crop_ocr

    @property
    def full_ocr(self) -> OCRLike:
        if self._full_ocr is None:
            self._full_ocr = self._make_engine(crop=False)
        return self._full_ocr

    def match(
        self, words: Sequence[Any], offset: tuple[int, int] = (0, 0)
    ) -> tuple[str, float, tuple[int, int, int, int]] | None:
        owner = self.owner_normalised
        best: tuple[float, str, float, tuple[int, int, int, int]] | None = None
        texts: list[str] = []
        for word in words:
            text = str(word.text)
            texts.append(text)
            confidence = float(word.confidence)
            if confidence < 0.30:
                continue
            compact = normalise_name(text)
            if not compact:
                continue
            if compact == owner:
                score = confidence + 1.0
            elif len(owner) >= 3 and (
                owner in compact or (len(compact) >= 4 and compact in owner)
            ):
                score = confidence + 0.8
            elif (
                min(len(owner), len(compact)) >= 3
                and SequenceMatcher(None, owner, compact).ratio() >= 0.62
            ):
                score = confidence + 0.5 * SequenceMatcher(None, owner, compact).ratio()
            else:
                continue
            x, y, w, h = word.bbox
            box = (int(x + offset[0]), int(y + offset[1]), int(w), int(h))
            if best is None or score > best[0]:
                best = (score, text, confidence, box)
        self.last_texts = texts[:16]
        if best is None:
            return None
        return best[1], best[2], best[3]

    def find(
        self,
        frame: RGBImage,
        *,
        predicted_centre: tuple[float, float] | None = None,
        allow_full: bool = True,
        full_min_interval: float = 0.8,
        before_full: Callable[[], None] | None = None,
    ) -> NameplateObservation | None:
        height, width = frame.shape[:2]
        centres: list[tuple[float, float]] = []
        if predicted_centre is not None:
            centres.append(predicted_centre)
        if self.last_box is not None:
            x, y, w, h = self.last_box
            centre = (x + w / 2.0, y + h / 2.0)
            if (
                not centres
                or abs(centres[0][0] - centre[0]) > 30
                or abs(centres[0][1] - centre[1]) > 30
            ):
                centres.append(centre)
        for cx, cy in centres[:2]:
            x0 = int(max(0, min(width - self.CROP_WIDTH, cx - self.CROP_WIDTH / 2.0)))
            y0 = int(max(0, min(height - self.CROP_HEIGHT, cy - self.CROP_HEIGHT / 2.0)))
            crop = frame[y0 : y0 + self.CROP_HEIGHT, x0 : x0 + self.CROP_WIDTH]
            if crop.size == 0:
                continue
            started = time.monotonic()
            words = self.crop_ocr.recognize(crop)
            self.last_crop_ocr_seconds = time.monotonic() - started
            found = self.match(words, (x0, y0))
            if found is not None:
                text, confidence, box = found
                self.last_box = box
                return NameplateObservation(text, confidence, box, "crop")
        if not allow_full:
            return None
        now = self._clock()
        if now - self.last_full_ocr_at < full_min_interval:
            return None
        self.last_full_ocr_at = now
        if before_full is not None:
            before_full()
        started = time.monotonic()
        words = self.full_ocr.recognize(frame)
        self.last_full_ocr_seconds = time.monotonic() - started
        found = self.match(words)
        if found is None:
            self.last_box = None
            return None
        text, confidence, box = found
        self.last_box = box
        return NameplateObservation(text, confidence, box, "full")


class BodyPoseEstimator:
    """OpenCV-DNN wrapper for the OpenCV Zoo MediaPipe pose landmark model."""

    INPUT_SIZE = 256

    def __init__(
        self, model_path: str | Path | None = None, *, net: Any | None = None
    ) -> None:
        if net is not None:
            self.net = net
        else:
            path = Path(model_path) if model_path is not None else self.default_model_path()
            if not path.is_file():
                raise FileNotFoundError(f"pose model not found: {path}")
            self.net = cv2.dnn.readNet(str(path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.last_bbox: tuple[float, float, float, float] | None = None
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            self.face_cascade = None

    def face_in_head_region(self, frame: RGBImage, body: BodyObservation) -> bool:
        """Run the frontal-face cascade only around the landmark head."""

        if self.face_cascade is None:
            return False
        lm = body.landmarks
        nose = lm[LM_NOSE, :2]
        ears = float(np.linalg.norm(lm[LM_LEFT_EAR, :2] - lm[LM_RIGHT_EAR, :2]))
        # Measured: the face of a 2 m avatar spans ~90 px for a 150 px torso,
        # so a torso-sized square around the nose contains it with margin.
        side = max(60.0, ears * 2.5, body.torso_px * 1.0)
        height, width = frame.shape[:2]
        x0 = int(max(0, nose[0] - side / 2.0))
        y0 = int(max(0, nose[1] - side / 2.0))
        x1 = int(min(width, nose[0] + side / 2.0))
        y1 = int(min(height, nose[1] + side / 2.0))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return False
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        minimum = max(20, int(round(side * 0.25)))
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=3, minSize=(minimum, minimum)
        )
        return len(faces) > 0

    MODEL_DIRECTORY = "pose_estimation_mediapipe"
    MODEL_FILENAME = "pose_estimation_mediapipe_2023mar.onnx"

    @classmethod
    def default_model_path(cls) -> Path:
        """Resolve the model through the SourceVault_vision asset roots."""

        found = assets.find_vision_model(cls.MODEL_DIRECTORY, cls.MODEL_FILENAME)
        if found is not None:
            return found
        raise assets.MissingAsset(
            f"pose model ({cls.MODEL_FILENAME})",
            [root / cls.MODEL_DIRECTORY / cls.MODEL_FILENAME
             for root in assets.vision_search_path()],
        )

    def infer(
        self, frame: RGBImage, roi: tuple[int, int, int]
    ) -> BodyObservation | None:
        """Run the landmark model on a square ROI ``(x0, y0, side)``."""

        x0, y0, side = int(roi[0]), int(roi[1]), int(roi[2])
        if side < 32:
            return None
        height, width = frame.shape[:2]
        x1, y1 = x0 + side, y0 + side
        pad_left, pad_top = max(0, -x0), max(0, -y0)
        pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)
        crop = frame[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
        if crop.size == 0:
            return None
        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        blob = cv2.resize(
            crop, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA
        )
        blob = blob.astype(np.float32) / 255.0
        self.net.setInput(blob[None])
        outputs = self.net.forward(self.output_names)
        landmarks = world = None
        conf = 0.0
        for output in outputs:
            if output.size == 195:
                landmarks = output.reshape(39, 5).astype(np.float32).copy()
            elif output.size == 117:
                world = output.reshape(39, 3).astype(np.float32).copy()
            elif output.size == 1:
                conf = float(output.reshape(-1)[0])
        if landmarks is None or world is None:
            return None
        landmarks[:, 3:] = 1.0 / (1.0 + np.exp(-np.clip(landmarks[:, 3:], -30, 30)))
        scale = side / float(self.INPUT_SIZE)
        landmarks[:, 0] = landmarks[:, 0] * scale + x0
        landmarks[:, 1] = landmarks[:, 1] * scale + y0
        left = world[LM_LEFT_SHOULDER] - world[LM_RIGHT_SHOULDER]
        yaw = math.degrees(math.atan2(float(left[2]), -float(left[0])))
        left_hip = world[LM_LEFT_HIP] - world[LM_RIGHT_HIP]
        hip_yaw = math.degrees(math.atan2(float(left_hip[2]), -float(left_hip[0])))
        shoulder_centre = (
            landmarks[LM_LEFT_SHOULDER, :2] + landmarks[LM_RIGHT_SHOULDER, :2]
        ) / 2.0
        hip_centre = (landmarks[LM_LEFT_HIP, :2] + landmarks[LM_RIGHT_HIP, :2]) / 2.0
        torso_px = float(np.linalg.norm(shoulder_centre - hip_centre))
        shoulder_world = (world[LM_LEFT_SHOULDER] + world[LM_RIGHT_SHOULDER]) / 2.0
        hip_world = (world[LM_LEFT_HIP] + world[LM_RIGHT_HIP]) / 2.0
        torso_m = float(np.linalg.norm(shoulder_world - hip_world))
        focal = FOCAL_LENGTH_FRACTION * float(width)
        distance = focal * torso_m / max(torso_px, 1e-3)
        points = landmarks[:33]
        visible = points[:, 3] > 0.5
        chosen = points[visible] if int(visible.sum()) >= 4 else points
        bbox = (
            float(chosen[:, 0].min()),
            float(chosen[:, 1].min()),
            float(chosen[:, 0].max()),
            float(chosen[:, 1].max()),
        )
        return BodyObservation(
            conf=conf,
            yaw_deg=yaw,
            hip_yaw_deg=hip_yaw,
            torso_px=torso_px,
            torso_m=torso_m,
            distance_m=distance,
            bbox=bbox,
            shoulder_centre=(float(shoulder_centre[0]), float(shoulder_centre[1])),
            hip_centre=(float(hip_centre[0]), float(hip_centre[1])),
            landmarks=landmarks,
        )

    @staticmethod
    def roi_from_bbox(
        bbox: tuple[float, float, float, float],
        enlarge: float = 1.35,
        min_side: float = 96.0,
    ) -> tuple[int, int, int]:
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side = max(min_side, max(x1 - x0, y1 - y0) * enlarge)
        return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), int(round(side))

    @staticmethod
    def roi_from_nameplate(
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        side_fraction: float = 0.75,
    ) -> tuple[int, int, int]:
        x, y, w, h = box
        height = frame_shape[0]
        side = int(round(height * side_fraction))
        return int(round(x + w / 2.0 - side / 2.0)), int(round(y + h * 1.2)), side

    def track(
        self,
        frame: RGBImage,
        *,
        nameplate_box: tuple[int, int, int, int] | None,
        min_conf: float = 0.5,
    ) -> BodyObservation | None:
        """Estimate the owner's body below the nameplate / near the last body."""

        candidates: list[tuple[int, int, int]] = []
        if self.last_bbox is not None:
            candidates.append(self.roi_from_bbox(self.last_bbox))
        if nameplate_box is not None:
            candidates.append(self.roi_from_nameplate(nameplate_box, frame.shape))
            candidates.append(
                self.roi_from_nameplate(nameplate_box, frame.shape, side_fraction=1.05)
            )
        best: BodyObservation | None = None
        for roi in candidates:
            result = self.infer(frame, roi)
            if result is None or result.conf < min_conf:
                continue
            refined = self.infer(frame, self.roi_from_bbox(result.bbox))
            if refined is not None and refined.conf >= result.conf - 0.05:
                result = refined
            if nameplate_box is not None:
                # The body must hang below its own nameplate; reject a track
                # that drifted onto somebody else.
                nx = nameplate_box[0] + nameplate_box[2] / 2.0
                if abs(result.shoulder_centre[0] - nx) > max(80.0, result.torso_px * 1.2):
                    continue
            if best is None or result.conf > best.conf:
                best = result
            if best.conf > 0.9:
                break
        self.last_bbox = best.bbox if best is not None else None
        if best is not None and abs(wrap_degrees(best.yaw_deg)) >= 100.0:
            best.face_visible = self.face_in_head_region(frame, best)
        return best


# --------------------------------------------------------------------------
# Owner state estimation (world frame = bot yaw frame at start)
# --------------------------------------------------------------------------


class OwnerStateEstimator:
    """Fuse nameplate bearing, body range, pose yaw and odometry."""

    def __init__(self, *, lost_seconds: float) -> None:
        self.lost_seconds = lost_seconds
        self.reset()

    def reset(self) -> None:
        self.last_seen = 0.0
        self.bearing_deg: float | None = None
        self.distance_m: float | None = None
        self.distance_ema: float | None = None
        self.owner_position: tuple[float, float] | None = None
        self.owner_heading: float | None = None
        self.heading_confidence = 0.0
        self.heading_outliers = 0
        self.outlier_times: deque[float] = deque(maxlen=8)
        self.last_pose_yaw: float | None = None
        self.last_pose_conf = 0.0
        #: Reliability band of the measurement that anchored the heading:
        #: 2 = face visible, 1 = profile, 0 = back view.
        self.anchor_band = -1
        self.motion_heading: float | None = None
        self.motion_heading_at = 0.0
        self.owner_speed = 0.0
        self.owner_velocity = (0.0, 0.0)
        self.owner_moving = False
        self.moving_since = 0.0
        self.still_since = 0.0
        self.position_history: deque[tuple[float, float, float]] = deque(maxlen=24)
        self.distance_samples: deque[float] = deque(maxlen=3)

    # -- geometry helpers ------------------------------------------------
    @staticmethod
    def owner_position_from(
        bot_position: tuple[float, float],
        bot_yaw: float,
        bearing: float,
        distance: float,
    ) -> tuple[float, float]:
        angle = math.radians(bot_yaw + bearing)
        return (
            bot_position[0] + distance * math.sin(angle),
            bot_position[1] + distance * math.cos(angle),
        )

    def theta_relative(self, bot_yaw: float) -> float | None:
        """Owner heading relative to the ray bot->owner (deg, right positive)."""

        if self.owner_heading is None or self.bearing_deg is None:
            return None
        return wrap_degrees(self.owner_heading - (bot_yaw + self.bearing_deg))

    @staticmethod
    def pose_band(measured_yaw: float, face_visible: bool = True) -> int:
        magnitude = abs(wrap_degrees(measured_yaw))
        if magnitude >= 120.0:
            return 2 if face_visible else 1
        if magnitude >= 60.0:
            return 1
        return 0

    @staticmethod
    def pose_reliability(measured_yaw: float, face_visible: bool = True) -> float:
        """Trust in a pose yaw for fusion.

        Measured live: with the face visible (|yaw| >= 120) the model is good
        to ~10 deg; at the profile it is noisy; for back-oblique views it is
        compressed towards 0 (truth 46 read as 4) and must not pull the
        odometry-propagated estimate.  Back views still serve as an initial
        anchor and for the gross-outlier test.
        """

        magnitude = abs(wrap_degrees(measured_yaw))
        if magnitude >= 120.0:
            return 1.0 if face_visible else 0.4
        if magnitude >= 60.0:
            return 0.4
        return 0.0

    # -- updates -----------------------------------------------------------
    def update(
        self,
        observation: Observation,
        *,
        bot_position: tuple[float, float],
        bot_yaw: float,
        bot_moving: bool,
        pose_gate: str = "all",
        holding: bool = False,
    ) -> None:
        """Fold one observation in.

        ``pose_gate`` = ``"front_only"`` ignores pose yaw unless the face is
        visible (used while holding the formation, where the view of the
        owner is back-oblique and the model's bias would drift the estimate).
        ``holding`` smooths range harder and demands a clearer displacement
        before declaring the owner in motion: the back-oblique landmark
        range jitters by +-15 % and a static owner must not look like a
        walking one.
        """

        if not observation.seen:
            return
        now = observation.time
        assert observation.bearing_deg is not None
        self.bearing_deg = observation.bearing_deg
        distance = observation.distance_m
        if distance is not None:
            stale = self.distance_ema is None or now - self.last_seen > self.lost_seconds
            if stale:
                self.distance_samples.clear()
            self.distance_samples.append(distance)
            # Median of the last three pose ranges: single-frame failures of
            # the landmark model (profile views) are wild but isolated.
            distance = float(sorted(self.distance_samples)[len(self.distance_samples) // 2])
            if stale:
                self.distance_ema = distance
            else:
                # Reject a single wild jump (profile-view failures) but let a
                # persistent change through.
                alpha = 0.25 if holding else 0.45
                if (
                    abs(distance - self.distance_ema) > 0.45 * self.distance_ema
                    and not bot_moving
                ):
                    self.distance_ema = self.distance_ema * 0.85 + distance * 0.15
                else:
                    self.distance_ema = self.distance_ema * (1.0 - alpha) + distance * alpha
            self.distance_m = self.distance_ema
        if self.distance_m is not None:
            position = self.owner_position_from(
                bot_position, bot_yaw, self.bearing_deg, self.distance_m
            )
            self.owner_position = position
            self.position_history.append((now, position[0], position[1]))
            self._update_motion(now, bot_moving or holding)
        body = observation.body
        if body is not None and body.conf >= 0.5:
            if pose_gate != "front_only" or (
                abs(wrap_degrees(body.yaw_deg)) >= 120.0 and body.face_visible
            ):
                self._update_pose_heading(body, bot_yaw, now)
            else:
                self.last_pose_yaw = body.yaw_deg
                self.last_pose_conf = body.conf
        self.last_seen = now

    def _update_motion(self, now: float, bot_moving: bool = False) -> None:
        history = [item for item in self.position_history if now - item[0] <= 2.0]
        if len(history) < 5 or history[-1][0] - history[0][0] < 1.0:
            return
        # Least-squares velocity over the window: robust to the +-10 % range
        # jitter of the landmark model that a first/last difference is not.
        ts = np.asarray([item[0] for item in history])
        xs = np.asarray([item[1] for item in history])
        ys = np.asarray([item[2] for item in history])
        tc = ts - ts.mean()
        denom = float((tc * tc).sum()) or 1e-9
        vx = float((tc * (xs - xs.mean())).sum() / denom)
        vy = float((tc * (ys - ys.mean())).sum() / denom)
        speed = math.hypot(vx, vy)
        span = speed * float(ts[-1] - ts[0])
        self.owner_speed = speed
        self.owner_velocity = (vx, vy)
        # While the bot itself moves, dead-reckoning error leaks into the
        # owner position; demand a clearer displacement before believing it.
        min_speed, min_dist = (0.9, 1.0) if bot_moving else (0.5, 0.5)
        if speed >= min_speed and span >= min_dist:
            if not self.owner_moving:
                self.owner_moving = True
                self.moving_since = now
            self.still_since = now
            heading = math.degrees(math.atan2(vx, vy))
            self.motion_heading = heading
            self.motion_heading_at = now
            if self.owner_heading is None:
                self.owner_heading = heading
            else:
                self.owner_heading = circular_mean_degrees(
                    [self.owner_heading, heading], [0.35, 0.65]
                )
            self.heading_confidence = max(self.heading_confidence, 0.9)
            self.heading_outliers = 0
        elif self.owner_moving and speed < 0.3:
            if self.still_since <= self.moving_since:
                self.still_since = now
            if now - self.still_since > 0.8:
                self.owner_moving = False
        elif self.owner_moving:
            self.still_since = now

    def _update_pose_heading(self, body: BodyObservation, bot_yaw: float, now: float) -> None:
        # Pose yaw is measured relative to the ray to the owner.
        measured = wrap_degrees(bot_yaw + (self.bearing_deg or 0.0) + body.yaw_deg)
        self.last_pose_yaw = body.yaw_deg
        self.last_pose_conf = body.conf
        weight = body.conf * self.pose_reliability(body.yaw_deg, body.face_visible)
        motion_fresh = now - self.motion_heading_at < 4.0 and self.motion_heading is not None
        if motion_fresh:
            weight *= 0.3
        band = self.pose_band(body.yaw_deg, body.face_visible)
        if self.owner_heading is None or self.heading_confidence < 0.15:
            # First anchor: even a back view is better than nothing, but it
            # is marked weak so a later front view can replace it.
            self.owner_heading = measured
            self.heading_confidence = max(0.25, min(0.6, weight)) * body.conf
            self.heading_outliers = 0
            self.anchor_band = band
            return
        delta = wrap_degrees(measured - self.owner_heading)
        if self.heading_confidence < 0.4 and not motion_fresh:
            # Weak anchor.  A more reliable view replaces it outright; a
            # view of the same quality is fused only when it agrees, so one
            # wild profile reading cannot drag a back-view anchor around.
            if band > self.anchor_band:
                self.owner_heading = measured
                self.heading_confidence = max(0.25, min(0.6, weight)) * body.conf
                self.anchor_band = band
                self.heading_outliers = 0
                return
            if band < self.anchor_band or abs(delta) > 45.0:
                return
        if abs(delta) > 70.0 and self.heading_confidence >= 0.4:
            # Disagreement counts only when it is persistent *and* recent:
            # four readings within 2.5 s.  Isolated misreads spread over a
            # long hold must never accumulate into a bogus re-anchor.
            self.outlier_times.append(now)
            recent = [t for t in self.outlier_times if now - t <= 2.5]
            self.heading_outliers = len(recent)
            if self.heading_outliers < 4:
                self.heading_confidence = max(0.15, self.heading_confidence - 0.05)
                return
            # Persistent disagreement: the owner turned (or the anchor was
            # wrong).  Re-anchor.
            self.owner_heading = measured
            self.heading_confidence = min(0.5, weight)
            self.heading_outliers = 0
            self.outlier_times.clear()
            self.anchor_band = band
            return
        self.heading_outliers = 0
        self.outlier_times.clear()
        if weight <= 0.0:
            return
        alpha = 0.55 * weight
        self.owner_heading = circular_mean_degrees(
            [self.owner_heading, measured], [1.0 - alpha, alpha]
        )
        self.heading_confidence = min(1.0, self.heading_confidence + 0.15 * weight)

    def predicted_bearing(
        self, bot_position: tuple[float, float], bot_yaw: float, now: float
    ) -> float | None:
        """Where a walking owner should be now (deg, right positive), or None."""

        if self.owner_position is None or not self.owner_moving or self.last_seen <= 0.0:
            return None
        dt = max(0.0, min(2.0, now - self.last_seen))
        x = self.owner_position[0] + self.owner_velocity[0] * dt - bot_position[0]
        y = self.owner_position[1] + self.owner_velocity[1] * dt - bot_position[1]
        if math.hypot(x, y) < 0.2:
            return None
        return wrap_degrees(math.degrees(math.atan2(x, y)) - bot_yaw)

    def apply_yaw(self, delta_deg: float) -> None:
        """The bot turned by ``delta_deg``: keep the last bearing consistent."""

        if self.bearing_deg is not None:
            self.bearing_deg = wrap_degrees(self.bearing_deg - delta_deg)

    @property
    def stationary(self) -> bool:
        return not self.owner_moving and self.owner_speed < 0.35


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


class OwnerFormationController:
    """Owner formation state machine (drop-in for ``OwnerFollowController``).

    Phases::

        searching   -> deterministic 30-degree yaw steps with a full-frame OCR
                       pass at each stop until the owner nameplate is read
        approaching -> centre the owner (yaw servo) and walk to the follow
                       distance (body-landmark range); also used while the
                       owner walks ("following")
        estimating  -> stand still, gather pose measurements for the heading
        orbiting    -> strafe around the owner keeping them centred until the
                       owner heading relative to the ray equals the slot value
        aligning    -> turn left so the owner sits at the right-edge bearing
        holding     -> stand still; small yaw corrections; watch for motion
        reacquiring -> owner lost briefly: look toward the last bearing
    """

    PRIVATE_OWNER_CONFIRM_TTL_SECONDS = 30.0
    PRIVATE_OWNER_VISIBLE_GRACE_SECONDS = 3.5
    PRIVATE_OWNER_CONFIRM_DISTANCE_M = 3.5
    IDENTITY_GRACE_SECONDS = 6.0
    SPEECH_RELEASE_GRACE_SECONDS = 1.0
    FORMATION_PHASES = frozenset({"estimating", "orbiting", "aligning", "holding"})

    def __init__(
        self,
        *,
        config: FormationConfig,
        initial_mode: str,
        frame_provider: Callable[..., RGBImage | None],
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
        vrchat_pid: int | None = None,
        state_callback: Callable[..., None] | None = None,
        controller: OSCControllerLike | None = None,
        crop_ocr_factory: Callable[[], OCRLike] | None = None,
        full_ocr_factory: Callable[[], OCRLike] | None = None,
        pose_factory: Callable[[], BodyPoseEstimator] | None = None,
        relative_move: Callable[[int, int], None] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        locator_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not config.owner_account.strip():
            raise ValueError("owner_account must not be empty")
        #: Builds a menu locator (``VRCRealtime_ownerjoin.OwnerLocator``-like:
        #: ``run(join=True) -> result with .outcome``) for an owner name.
        self._locator_factory = locator_factory
        self.locate_requested = False
        self.locate_in_progress = False
        self.last_locate_at = 0.0
        self.last_locate_result: dict[str, Any] | None = None
        # search episode / patrol / waiting bookkeeping
        self.search_episode_started = 0.0
        self.episode_sweeps = 0
        self.patrol_moves = 0
        self.patrol_pending = False
        self.patrol_turn_deg = 0.0
        self.blocked_headings: set[int] = set()
        self.sweep_openness: list[float] = []
        self.last_openness = 0.0
        self.waiting_since = 0.0
        self.waiting_next_observe_at = 0.0
        self.waiting_reason = ""
        self.config = config
        self.frame_provider = frame_provider
        self.osc_host = osc_host
        self.osc_port = osc_port
        self.vrchat_pid = vrchat_pid
        self.state_callback = state_callback
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.controller = controller
        self._crop_ocr_factory = crop_ocr_factory
        self._full_ocr_factory = full_ocr_factory
        self._pose_factory = pose_factory
        self._relative_move = relative_move

        self.lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.mode = self._canonical_mode(initial_mode)
        self.phase = self._idle_phase(self.mode)
        self.finder: NameplateFinder | None = None
        self.pose: BodyPoseEstimator | None = None
        self.pose_error: str | None = None
        self.yaw: YawActuator | None = None
        self.motion: MotionActuator | None = None
        self.estimator = OwnerStateEstimator(lost_seconds=config.lost_seconds)
        self.bot_position = (0.0, 0.0)
        self.runtime_ready = False
        self.camera_levelled = False

        # phase bookkeeping
        self.phase_started = 0.0
        self.search_direction = 1.0
        self.search_steps = 0
        self.search_next_observe_at = 0.0
        self.reacquire_step = 0
        self.reacquire_next_at = 0.0
        self.estimate_cycles = 0
        self.orbit_started = 0.0
        self.align_turned = False
        self.hold_deviation_cycles = 0
        self.last_observation: Observation | None = None
        self.last_observation_yaw = 0.0
        self.last_ocr_confirmed = 0.0
        self.owner_private_confirmed_at = 0.0
        self.identity_latched = False
        self.last_axes = (0.0, 0.0, 0.0)
        self.formation_completed_at = 0.0
        self.hold_facing_only = False
        self.owner_facing_us = False
        self.last_orbit_delta = 0.0
        self.settle_until = 0.0
        self.force_full_ocr = False
        self.last_actuated_axes = (0.0, 0.0)
        #: Search sweeps alternate between a level camera and one pitched up
        #: by this many mouse counts (about 20 degrees): a very close owner's
        #: nameplate sits above a level view.
        self.search_pitch_counts = 200.0
        self.search_pitched = False
        #: Current camera pitch offset from level in mouse counts (+ = up).
        self.pitch_offset_counts = 0.0
        self.nameplate_height_m = DEFAULT_NAMEPLATE_HEIGHT_M
        self.nameplate_height_samples: deque[float] = deque(maxlen=15)
        self.search_backstep_done = False

        # holds / manual turn
        self.manual_turn_deg = 0.0
        self.manual_turn_label: str | None = None
        self.attention_hold_until = 0.0
        self.conversation_hold = False
        self.speech_hold = False
        self.speech_hold_until = 0.0
        self.listener_hold = False
        self.listener_hold_until = 0.0
        self.chat_key_times: dict[str, float] = {}

        self._report(
            avatarMode=self.mode,
            ownerAccount=config.owner_account,
            ownerFollowEngine="formation",
            ownerFollowState=self.phase,
            ownerFollowDetected=False,
            ownerFollowFollowDistanceM=config.follow_distance_m,
            ownerFollowRearAngleDeg=config.rear_angle_deg,
            ownerFollowOwnerBearingDeg=config.owner_bearing_deg,
            ownerFollowHeadingOffsetDeg=round(config.heading_offset_deg, 2),
            ownerFollowTargetTheta=round(config.target_theta_deg, 2),
            ownerFollowIdentityLatched=False,
            ownerFollowTrackConfirmed=False,
            ownerFollowPrivateConfirmationCurrent=False,
            ownerFollowPrivateConfirmationAgeSeconds=None,
            ownerFollowSpeechHold=False,
            ownerFollowListenerHold=False,
            ownerFollowFacingUs=False,
        )

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_mode(mode: str) -> str:
        compact = "".join(ch for ch in str(mode).casefold() if ch.isalnum())
        if compact == "standalone":
            return "Standalone"
        if compact in {"ownerfollow", "followowner"}:
            return "OwnerFollow"
        if compact in {"presentation", "presenter", "present"}:
            return "Presentation"
        raise ValueError("mode must be Standalone, OwnerFollow, or Presentation")

    @staticmethod
    def _idle_phase(mode: str) -> str:
        if mode == "OwnerFollow":
            return "searching"
        return "presentation" if mode == "Presentation" else "standalone"

    @staticmethod
    def _frame_openness(frame: RGBImage) -> float:
        """Cheap 'open space ahead' proxy: 1 - edge density of the centre band."""

        height, width = frame.shape[:2]
        band = frame[int(height * 0.35) : int(height * 0.75), int(width * 0.3) : int(width * 0.7)]
        if band.size == 0:
            return 0.0
        gray = cv2.cvtColor(cv2.resize(band, (160, 80), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        return float(1.0 - min(1.0, edges.mean() / 255.0 * 4.0))

    def _report(self, **updates: Any) -> None:
        if self.state_callback is not None:
            self.state_callback(**updates)

    def announce_status(self, key: str, text: str, *, min_interval: float = 2.0) -> bool:
        controller = self.controller
        if controller is None or not hasattr(controller, "chatbox"):
            return False
        now = self._clock()
        with self.chat_lock:
            previous = self.chat_key_times.get(key, 0.0)
            if previous > 0.0 and now - previous < max(1.2, min_interval):
                return False
            message = str(text).strip()[:144]
            if not message:
                return False
            try:
                controller.chatbox(message, send=True, sfx=False)
            except Exception as exc:
                self._report(ownerFollowChatboxLastError=f"{type(exc).__name__}: {exc}")
                return False
            self.chat_key_times[key] = now
            if len(self.chat_key_times) > 128:
                for old in sorted(self.chat_key_times, key=self.chat_key_times.get)[:32]:
                    self.chat_key_times.pop(old, None)
        self._report(
            ownerFollowChatboxText=message,
            ownerFollowChatboxAt=time.time(),
            ownerFollowChatboxLastError=None,
        )
        return True

    # ------------------------------------------------------------------
    # public API (worker-facing, compatible with OwnerFollowController)
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name="VRCRealtimeOwnerFormation", daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self.thread = None
        self._stop_motion(force=True)
        self._report(ownerFollowState="stopped", ownerFollowDetected=False)

    def set_mode(self, mode: str) -> str:
        canonical = self._canonical_mode(mode)
        with self.lock:
            self.mode = canonical
            self._reset_cycle_locked()
        if canonical != "OwnerFollow":
            self._stop_motion(force=True)
        self._report(
            avatarMode=canonical,
            ownerFollowState=self.phase,
            ownerFollowDetected=False,
            ownerFollowIdentityLatched=False,
            ownerFollowPrivateConfirmationCurrent=False,
            ownerFollowPrivateConfirmationAgeSeconds=None,
        )
        self.wake_event.set()
        return canonical

    def set_owner_account(self, owner_account: str) -> None:
        value = owner_account.strip()
        if not value:
            raise ValueError("owner_account must not be empty")
        with self.lock:
            self.config.owner_account = value
            if self.finder is not None:
                self.finder.set_owner(value)
            self._reset_cycle_locked()
        self._stop_motion(force=True)
        self._report(ownerAccount=value, ownerFollowDetected=False)
        self.wake_event.set()

    def set_follow_distance(self, metres: float) -> float:
        value = float(metres)
        if not 0.8 <= value <= 6.0:
            raise ValueError("follow distance must be between 0.8 and 6 metres")
        with self.lock:
            self.config.follow_distance_m = value
        self._report(ownerFollowFollowDistanceM=value)
        self.wake_event.set()
        return value

    def set_target_nameplate_height(self, value: float) -> float:
        """Legacy knob: larger = closer.  Mapped onto metres (0.055 -> 1.8 m)."""

        height = float(value)
        if not 0.005 <= height <= 0.15:
            raise ValueError("target nameplate height must be between 0.005 and 0.15")
        metres = max(0.8, min(6.0, 0.099 / height))
        self.set_follow_distance(metres)
        self._report(ownerFollowTargetNameplateHeight=height)
        return height

    def set_formation(
        self,
        *,
        rear_angle_deg: float | None = None,
        owner_bearing_deg: float | None = None,
    ) -> dict[str, float]:
        with self.lock:
            if rear_angle_deg is not None:
                value = float(rear_angle_deg)
                if not 0.0 <= value <= 90.0:
                    raise ValueError("rear angle must be between 0 and 90 degrees")
                self.config.rear_angle_deg = value
            if owner_bearing_deg is not None:
                value = float(owner_bearing_deg)
                if not 10.0 <= value <= 42.0:
                    raise ValueError("owner bearing must be between 10 and 42 degrees")
                self.config.owner_bearing_deg = value
            result = {
                "rearAngleDeg": self.config.rear_angle_deg,
                "ownerBearingDeg": self.config.owner_bearing_deg,
                "headingOffsetDeg": self.config.heading_offset_deg,
                "targetTheta": self.config.target_theta_deg,
            }
            if self.mode == "OwnerFollow" and self.phase in self.FORMATION_PHASES:
                self._enter_phase_locked("estimating", self._clock())
        self._report(
            ownerFollowRearAngleDeg=result["rearAngleDeg"],
            ownerFollowOwnerBearingDeg=result["ownerBearingDeg"],
            ownerFollowHeadingOffsetDeg=round(result["headingOffsetDeg"], 2),
            ownerFollowTargetTheta=round(result["targetTheta"], 2),
        )
        self.wake_event.set()
        return result

    def request_locate(self) -> bool:
        """Run the menu check (instance users / join the owner) on the next cycle."""

        if self._locator_factory is None:
            return False
        with self.lock:
            self.locate_requested = True
        self.wake_event.set()
        return True

    def turn(self, direction: str) -> str:
        compact = "".join(ch for ch in str(direction).casefold() if ch.isalnum())
        if compact in {"left", "left90", "左", "左旋回"}:
            label, degrees = "left", -90.0
        elif compact in {"right", "right90", "右", "右旋回"}:
            label, degrees = "right", 90.0
        elif compact in {"around", "turnaround", "back", "180", "反対", "反対向き"}:
            label, degrees = "around", 180.0
        else:
            raise ValueError("direction must be left, right, or around")
        with self.lock:
            self.manual_turn_deg = degrees
            self.manual_turn_label = label
        self._stop_motion(force=True)
        self._report(
            ownerFollowState="manual_turning",
            ownerFollowManualTurn=label,
            ownerFollowManualTurnActive=True,
        )
        self.wake_event.set()
        return label

    def hold_for_attention(self, seconds: float = 1.0) -> None:
        until = self._clock() + max(0.25, float(seconds))
        with self.lock:
            self.attention_hold_until = max(self.attention_hold_until, until)
        self._stop_motion()
        self._report(
            ownerFollowState="attention_hold",
            ownerFollowAttentionHold=True,
            ownerFollowAxes=[0.0, 0.0, 0.0],
        )

    def set_conversation_active(self, active: bool) -> None:
        with self.lock:
            self.conversation_hold = bool(active)
            if not active:
                self.attention_hold_until = 0.0
        if active:
            self._stop_motion(force=True)
        self._report(
            ownerFollowConversationHold=bool(active),
            ownerFollowState=("conversation_hold" if active else self.phase),
            ownerFollowAxes=[0.0, 0.0, 0.0] if active else list(self.last_axes),
        )
        self.announce_status(
            "conversation_hold" if active else "conversation_resume",
            "💬 会話中：発話中は停止" if active else "✅ 会話終了：追尾を再開",
            min_interval=1.2,
        )
        self.wake_event.set()

    def set_speech_active(self, active: bool) -> None:
        now = self._clock()
        with self.lock:
            self.speech_hold = bool(active)
            self.speech_hold_until = 0.0 if active else now + self.SPEECH_RELEASE_GRACE_SECONDS
        if active:
            self._stop_motion(force=True)
        self._report(
            ownerFollowSpeechHold=bool(active),
            ownerFollowSpeechReleaseAt=(
                None if active else time.time() + self.SPEECH_RELEASE_GRACE_SECONDS
            ),
            ownerFollowState=("speech_hold" if active else self.phase),
            ownerFollowAxes=[0.0, 0.0, 0.0] if active else list(self.last_axes),
        )
        self.wake_event.set()

    def set_listener_active(self, active: bool) -> None:
        now = self._clock()
        with self.lock:
            self.listener_hold = bool(active)
            self.listener_hold_until = 0.0 if active else now + 0.45
        if active:
            self._stop_motion(force=True)
        self._report(
            ownerFollowListenerHold=bool(active),
            ownerFollowState=("listener_hold" if active else self.phase),
            ownerFollowAxes=[0.0, 0.0, 0.0] if active else list(self.last_axes),
        )
        self.wake_event.set()

    def private_access_status(self) -> dict[str, Any]:
        """Live owner-proximity decision for high-PL local speech.

        A nameplate OCR read within ``PRIVATE_OWNER_CONFIRM_DISTANCE_M``
        establishes a 30-second identity lease; the formation phases and a
        fresh observation are also required.
        """

        now = self._clock()
        with self.lock:
            mode = self.mode
            phase = self.phase
            last_seen = self.estimator.last_seen
            distance = self.estimator.distance_m
            latched = self.identity_latched
            confirmed_at = self.owner_private_confirmed_at
            follow_distance = self.config.follow_distance_m
        visible_grace = max(self.PRIVATE_OWNER_VISIBLE_GRACE_SECONDS, self.config.lost_seconds * 2.0)
        seen_age = now - last_seen if last_seen > 0.0 else float("inf")
        fresh = seen_age <= visible_grace
        confirmation_age = now - confirmed_at if confirmed_at > 0.0 else float("inf")
        confirmation_current = confirmation_age <= self.PRIVATE_OWNER_CONFIRM_TTL_SECONDS
        identity_confirmed = bool(latched and confirmation_current)
        formation_phase = phase in self.FORMATION_PHASES
        near = distance is not None and distance <= follow_distance * 1.7
        allowed = bool(
            mode == "OwnerFollow" and fresh and identity_confirmed and formation_phase and near
        )
        if mode != "OwnerFollow":
            reason = "presentation_mode" if mode == "Presentation" else "standalone_mode"
        elif not fresh:
            reason = "owner_not_visible"
        elif not identity_confirmed:
            reason = (
                "owner_close_confirmation_expired"
                if latched and confirmed_at > 0.0
                else "owner_identity_unconfirmed"
            )
        elif not formation_phase:
            reason = "owner_search_or_approach_incomplete"
        elif not near:
            reason = "owner_too_far"
        else:
            reason = "allowed"
        status = {
            "allowed": allowed,
            "reason": reason,
            "mode": mode,
            "phase": phase,
            "identityConfirmed": identity_confirmed,
            "identitySource": "ocr" if latched else None,
            "identityLatched": latched,
            "closeConfirmationCurrent": confirmation_current,
            "closeConfirmationAgeSeconds": (
                round(confirmation_age, 3) if confirmation_age != float("inf") else None
            ),
            "closeConfirmationTTLSeconds": self.PRIVATE_OWNER_CONFIRM_TTL_SECONDS,
            "fresh": fresh,
            "seenAgeSeconds": round(seen_age, 3) if seen_age != float("inf") else None,
            "visibleGraceSeconds": round(visible_grace, 3),
            "distanceM": round(distance, 3) if distance is not None else None,
            "nearDistanceM": round(follow_distance * 1.7, 3),
        }
        self._report(
            ownerFollowPrivateAccessAllowed=allowed,
            ownerFollowPrivateAccessReason=reason,
            ownerFollowIdentityLatched=latched,
            ownerFollowPrivateConfirmationCurrent=confirmation_current,
            ownerFollowPrivateConfirmationAgeSeconds=status["closeConfirmationAgeSeconds"],
        )
        return status

    # ------------------------------------------------------------------
    # runtime
    # ------------------------------------------------------------------
    def _ensure_runtime(self) -> None:
        if self.runtime_ready:
            return
        if self.controller is None:
            from vrcpilot import OscSender

            self.controller = OscSender(self.osc_host, self.osc_port).controller()
        if self.finder is None:
            self._report(ownerFollowState="loading_ocr")
            self.finder = NameplateFinder(
                self.config.owner_account,
                crop_ocr_factory=self._crop_ocr_factory,
                full_ocr_factory=self._full_ocr_factory,
                box_threshold=self.config.ocr_box_threshold,
                text_score=self.config.ocr_text_score,
                clock=self._clock,
            )
            # Force engine construction now so the first cycle is not slow.
            _ = self.finder.crop_ocr
            _ = self.finder.full_ocr
        if self.pose is None and self.pose_error is None:
            try:
                self.pose = (
                    self._pose_factory() if self._pose_factory is not None else BodyPoseEstimator()
                )
                self._report(ownerFollowPoseReady=True, ownerFollowPoseError=None)
            except Exception as exc:
                self.pose_error = f"{type(exc).__name__}: {exc}"
                self._report(ownerFollowPoseReady=False, ownerFollowPoseError=self.pose_error)
        if self.yaw is None:
            self.yaw = YawActuator(
                pid=self.vrchat_pid, osc=self.controller, relative_move=self._relative_move
            )
        if self.motion is None:
            self.motion = MotionActuator(self.controller, clock=self._clock)
        self.runtime_ready = True
        self._report(ownerFollowOCRReady=True, ownerFollowLastError=None)

    def _reset_cycle_locked(self) -> None:
        self.phase = self._idle_phase(self.mode)
        self.phase_started = self._clock()
        self.estimator.reset()
        self.bot_position = (0.0, 0.0)
        self.search_steps = 0
        self.search_next_observe_at = 0.0
        self.reacquire_step = 0
        self.estimate_cycles = 0
        self.align_turned = False
        self.hold_deviation_cycles = 0
        self.last_observation = None
        self.last_ocr_confirmed = 0.0
        self.owner_private_confirmed_at = 0.0
        self.identity_latched = False
        self.attention_hold_until = 0.0
        self.conversation_hold = False
        self.speech_hold = False
        self.speech_hold_until = 0.0
        self.listener_hold = False
        self.listener_hold_until = 0.0
        self.camera_levelled = False
        self.pitch_offset_counts = 0.0
        self.search_pitched = False
        self.search_episode_started = self._clock()
        self.episode_sweeps = 0
        self.patrol_moves = 0
        self.patrol_pending = False
        self.blocked_headings = set()
        self.sweep_openness = []
        self.waiting_since = 0.0
        self.waiting_next_observe_at = 0.0
        self.locate_requested = False
        if self.finder is not None:
            self.finder.last_box = None
        if self.pose is not None:
            self.pose.last_bbox = None

    def _stop_motion(self, *, force: bool = False) -> None:
        motion = self.motion
        if motion is not None:
            motion.stop(force=force)
        self.last_axes = (0.0, 0.0, 0.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                manual = self.manual_turn_deg
                mode = self.mode
            if manual != 0.0:
                try:
                    self._ensure_runtime()
                    self._apply_manual_turn()
                except Exception as exc:
                    self._stop_motion(force=True)
                    self._report(
                        ownerFollowState="error",
                        ownerFollowLastError=f"{type(exc).__name__}: {exc}",
                    )
                continue
            if mode != "OwnerFollow":
                self._stop_motion()
                self.wake_event.wait(0.25)
                self.wake_event.clear()
                continue
            cycle_started = self._clock()
            try:
                self._ensure_runtime()
                self._step(cycle_started)
            except Exception as exc:
                self._stop_motion(force=True)
                self._report(
                    ownerFollowState="error",
                    ownerFollowDetected=False,
                    ownerFollowLastError=f"{type(exc).__name__}: {exc}",
                )
                self.stop_event.wait(1.0)
            period = 1.0 / max(self.config.tracking_fps, 0.25)
            remaining = period - (self._clock() - cycle_started)
            if remaining > 0.0:
                self.stop_event.wait(remaining)
        self._stop_motion(force=True)

    def _apply_manual_turn(self) -> None:
        with self.lock:
            degrees = self.manual_turn_deg
            label = self.manual_turn_label
            self.manual_turn_deg = 0.0
            self.manual_turn_label = None
        self._stop_motion(force=True)
        assert self.yaw is not None
        self.yaw.turn(degrees)
        with self.lock:
            if self.mode == "OwnerFollow":
                # Odometry keeps the owner estimate valid across the turn; the
                # owner may simply have left the view, so reacquire first,
                # looking where the plate should now be.
                if self.estimator.bearing_deg is not None:
                    self.estimator.bearing_deg = wrap_degrees(
                        self.estimator.bearing_deg - degrees
                    )
                self._enter_phase_locked("reacquiring", self._clock())
            next_state = self.phase
        self._report(
            ownerFollowState=next_state,
            ownerFollowManualTurn=label,
            ownerFollowManualTurnActive=False,
            ownerFollowAxes=[0.0, 0.0, round(degrees, 2)],
        )

    # ------------------------------------------------------------------
    # one perception + control cycle
    # ------------------------------------------------------------------
    def _integrate_odometry(self, bot_yaw: float) -> None:
        assert self.motion is not None
        for forward_axis, strafe_axis, seconds in self.motion.consume():
            step = WALK_SPEED_MPS_PER_AXIS * seconds
            yaw = math.radians(bot_yaw)
            dx = step * (strafe_axis * math.cos(yaw) + forward_axis * math.sin(yaw))
            dy = step * (forward_axis * math.cos(yaw) - strafe_axis * math.sin(yaw))
            self.bot_position = (self.bot_position[0] + dx, self.bot_position[1] + dy)

    def _observe(self, now: float, *, allow_full_ocr: bool, bot_yaw: float) -> Observation | None:
        frame = self.frame_provider(max_age_seconds=max(1.0, self.config.lost_seconds))
        if frame is None:
            self._report(ownerFollowCaptureFresh=False)
            return None
        assert self.finder is not None
        height, width = frame.shape[:2]
        predicted: tuple[float, float] | None = None
        previous = self.last_observation
        if previous is not None and previous.nameplate is not None and previous.bearing_deg is not None:
            delta_yaw = bot_yaw - self.last_observation_yaw
            x_pred = x_from_bearing(previous.bearing_deg - delta_yaw, width)
            y_pred = previous.nameplate.box[1] + previous.nameplate.box[3] / 2.0
            predicted = (x_pred, y_pred)
        nameplate = self.finder.find(
            frame,
            predicted_centre=predicted,
            allow_full=allow_full_ocr,
            full_min_interval=0.0 if self.phase == "searching" else 0.8,
            before_full=self._before_full_ocr,
        )
        body: BodyObservation | None = None
        if self.pose is not None:
            box = nameplate.box if nameplate is not None else None
            if box is None and now - self.last_ocr_confirmed <= self.IDENTITY_GRACE_SECONDS:
                box = self.finder.last_box
            if box is not None or self.pose.last_bbox is not None:
                if box is None and now - self.last_ocr_confirmed > self.IDENTITY_GRACE_SECONDS:
                    self.pose.last_bbox = None
                else:
                    body = self.pose.track(frame, nameplate_box=box)
        bearing: float | None = None
        distance: float | None = None
        plate_distance: float | None = None
        pose_distance: float | None = None
        if nameplate is not None:
            x, y, w, h = nameplate.box
            bearing = bearing_from_x(x + w / 2.0, width)
            plate_distance = self._plate_distance(y + h / 2.0, frame)
        elif body is not None and now - self.last_ocr_confirmed <= self.IDENTITY_GRACE_SECONDS:
            bearing = bearing_from_x(body.shoulder_centre[0], width)
        if body is not None and body.conf >= 0.5 and bearing is not None:
            pose_distance = body.distance_m
        distance = self._fuse_distance(pose_distance, plate_distance, body, frame)
        observation = Observation(
            time=now,
            frame_shape=(height, width),
            nameplate=nameplate,
            body=body,
            bearing_deg=bearing,
            distance_m=distance,
            yaw_odometry_deg=bot_yaw,
            plate_distance_m=plate_distance,
            pose_distance_m=pose_distance,
        )
        if nameplate is not None:
            self.last_ocr_confirmed = now
        if self.phase in {"searching", "waiting"} and nameplate is None:
            try:
                self.last_openness = self._frame_openness(frame)
            except Exception:
                self.last_openness = 0.0
        self.last_observation = observation
        self.last_observation_yaw = bot_yaw
        return observation

    def _before_full_ocr(self) -> None:
        """A full-frame OCR pass takes up to a second: never translate meanwhile."""

        if self.motion is not None and self.motion.moving:
            self.motion.stop()
            self.last_actuated_axes = (0.0, 0.0)

    def _plate_distance(self, plate_y: float, frame: RGBImage) -> float | None:
        """Range from the nameplate's height above the eye line (level view)."""

        height, width = frame.shape[:2]
        title = title_bar_height(frame)
        horizon_y = title + (height - title) / 2.0
        focal = FOCAL_LENGTH_FRACTION * width
        # The camera is pitched up by ``pitch_offset_counts``: the scene is
        # shifted down, so the level-view row is higher by that many pixels.
        px_per_count = focal * math.pi / 180.0 / PITCH_COUNTS_PER_DEGREE
        level_y = plate_y - self.pitch_offset_counts * px_per_count
        rise = horizon_y - level_y
        if rise < 25.0:
            return None
        return focal * self.nameplate_height_m / rise

    def _fuse_distance(
        self,
        pose_distance: float | None,
        plate_distance: float | None,
        body: BodyObservation | None,
        frame: RGBImage,
    ) -> float | None:
        """Prefer the landmark range, but let the plate veto gross failures.

        The landmark model's range is good to about 10 % at 1.5-4 m, yet a
        body that overflows the frame (owner closer than ~1.2 m, or the bot
        standing inside the owner) yields metres of error; the plate height
        cannot fail that way.  Mid-range agreement calibrates the plate
        height for this avatar.
        """

        if pose_distance is not None and plate_distance is not None:
            reliable_pose = (
                body is not None
                and body.conf >= 0.9
                and 1.4 <= pose_distance <= 4.5
                and body.bbox[3] <= frame.shape[0] * 1.15
            )
            if reliable_pose:
                implied = self.nameplate_height_m * pose_distance / plate_distance
                self.nameplate_height_samples.append(implied)
                if len(self.nameplate_height_samples) >= 5:
                    ordered = sorted(self.nameplate_height_samples)
                    self.nameplate_height_m = float(ordered[len(ordered) // 2])
            ratio = pose_distance / max(plate_distance, 1e-3)
            if ratio > 1.45 or ratio < 0.55:
                return plate_distance
            return pose_distance
        if pose_distance is not None:
            return pose_distance
        return plate_distance

    def _step(self, now: float) -> None:
        assert self.yaw is not None and self.motion is not None
        with self.lock:
            phase = self.phase
            attention_hold = now < self.attention_hold_until
            conversation_hold = self.conversation_hold
            speech_hold = self.speech_hold or now < self.speech_hold_until
            listener_hold = self.listener_hold or now < self.listener_hold_until
        with self.lock:
            locate_now = self.locate_requested and not self.locate_in_progress
            patrol_now = self.patrol_pending and not locate_now
        if locate_now:
            self._run_menu_locate(now)
            return
        if patrol_now:
            self._run_patrol(now)
            return
        if self.config.level_camera_on_start and not self.camera_levelled:
            self._level_camera()
        bot_yaw = self.yaw.total_yaw_deg
        remaining = self.motion.pulse_remaining()
        if remaining > 0.0:
            self._sleep(min(0.35, remaining + 0.03))
            now = self._clock()
        self.motion.expire_pulse()
        self._integrate_odometry(bot_yaw)

        if phase == "searching" and now < self.search_next_observe_at:
            return  # let the view settle after the last search turn
        if phase == "waiting" and now < self.waiting_next_observe_at:
            return  # owner offline/unreachable: only an occasional look
        if now < self.settle_until:
            return  # a large turn was just sent; the next capture may be stale

        holding_input = attention_hold or speech_hold or listener_hold
        allow_full = (
            phase in {"searching", "reacquiring", "waiting"}
            or self.force_full_ocr
            or (now - self.estimator.last_seen > 0.6 if self.estimator.last_seen > 0 else True)
        )
        observation = self._observe(now, allow_full_ocr=allow_full, bot_yaw=bot_yaw)
        self.force_full_ocr = False
        fresh = observation is not None and observation.seen
        if observation is not None:
            self.estimator.update(
                observation,
                bot_position=self.bot_position,
                bot_yaw=bot_yaw,
                bot_moving=self.motion.moving,
                pose_gate=(
                    "front_only"
                    if phase in {"holding", "aligning"} and not self.owner_facing_us
                    else "all"
                ),
                holding=phase in {"holding", "aligning"},
            )
            if observation.nameplate is not None:
                distance = self.estimator.distance_m
                if distance is not None and distance <= self.PRIVATE_OWNER_CONFIRM_DISTANCE_M:
                    with self.lock:
                        self.identity_latched = True
                        self.owner_private_confirmed_at = now
        seen_age = now - self.estimator.last_seen if self.estimator.last_seen > 0 else float("inf")

        if holding_input or (conversation_hold and seen_age > self.config.lost_seconds):
            # While the voice pipeline is active never move; a lost owner
            # during a conversation is looked for by small turns only.
            self._stop_motion()
            hold_state = (
                "speech_hold" if speech_hold else ("listener_hold" if listener_hold else "attention_hold")
            )
            if not holding_input:
                hold_state = "conversation_hold"
            self._publish(observation, hold_state, (0.0, 0.0, 0.0), now)
            return

        forward = strafe = turn = 0.0
        with self.lock:
            # ---------------- lost handling ----------------
            if seen_age > self.config.lost_seconds and phase not in {"searching", "reacquiring", "waiting"}:
                if seen_age <= self.config.lost_seconds + self.config.reacquire_seconds and self.estimator.bearing_deg is not None:
                    self._enter_phase_locked("reacquiring", now)
                else:
                    self._enter_phase_locked("searching", now)
                phase = self.phase
            if phase in {"searching", "reacquiring", "waiting"} and seen_age <= self.config.lost_seconds:
                self._enter_phase_locked("approaching", now)
                phase = self.phase
                self.announce_status("owner_found", "✅ オーナー発見：接近します", min_interval=4.0)

            pulse_seconds = 0.0
            if phase == "searching":
                turn = self._search_locked(now)
            elif phase == "waiting":
                turn = self._waiting_locked(now)
            elif phase == "reacquiring":
                turn = self._reacquire_locked(now)
            elif not fresh:
                # Never act on a bearing that was not measured this cycle: a
                # relative yaw repeated on a stale value spins the avatar
                # away from the owner.  Keep a short translation going for
                # one missed frame, then stand still until reacquisition.
                # A *walking* owner is the exception: dead-reckon where they
                # should be and turn that way, so a side-step out of the
                # view does not become a full search.
                if seen_age <= 0.6 and phase in {"approaching", "following"}:
                    forward, strafe = self.last_actuated_axes
                turn = 0.0
                predicted = self.estimator.predicted_bearing(self.bot_position, bot_yaw, now)
                if predicted is not None and abs(predicted) > 4.0 and seen_age <= 1.5:
                    turn = max(-40.0, min(40.0, predicted))
            else:
                forward, strafe, turn, pulse_seconds = self._formation_control_locked(
                    now, phase, bot_yaw
                )
            phase = self.phase

        # ---------------- actuate ----------------
        applied_turn = 0.0
        if abs(turn) >= 0.4:
            applied_turn = self.yaw.turn(turn)
            self._after_turn(applied_turn, observation)
        if abs(forward) > 1e-6 or abs(strafe) > 1e-6:
            if pulse_seconds > 0.0:
                self.motion.pulse(forward, strafe, pulse_seconds)
            else:
                self.motion.set_axes(forward, strafe)
        else:
            self._stop_motion()
        self.last_actuated_axes = (forward, strafe) if pulse_seconds <= 0.0 else (0.0, 0.0)
        self.last_axes = (round(forward, 3), round(strafe, 3), round(applied_turn, 2))
        self._publish(observation, phase, self.last_axes, now)

    def _after_turn(self, applied_turn: float, observation: Observation | None) -> None:
        """Keep the perception priors consistent with a yaw just applied."""

        self.estimator.apply_yaw(applied_turn)
        width = observation.frame_shape[1] if observation is not None else 1284
        shift = 0.0
        if observation is not None and observation.bearing_deg is not None:
            shift = x_from_bearing(observation.bearing_deg - applied_turn, width) - x_from_bearing(
                observation.bearing_deg, width
            )
        else:
            shift = -math.radians(applied_turn) * FOCAL_LENGTH_FRACTION * width
        if self.finder is not None and self.finder.last_box is not None:
            x, y, w, h = self.finder.last_box
            self.finder.last_box = (int(round(x + shift)), y, w, h)
        if self.pose is not None and self.pose.last_bbox is not None:
            x0, y0, x1, y1 = self.pose.last_bbox
            self.pose.last_bbox = (x0 + shift, y0, x1 + shift, y1)
        if abs(applied_turn) >= 30.0:
            self.force_full_ocr = True
            self.settle_until = self._clock() + 0.25
        elif abs(applied_turn) >= 12.0:
            self.force_full_ocr = True
            self.settle_until = self._clock() + 0.15
        elif abs(applied_turn) >= 4.0:
            self.settle_until = self._clock() + 0.08

    # ------------------------------------------------------------------
    # phase behaviours (all called with self.lock held)
    # ------------------------------------------------------------------
    def _pitch_by(self, counts_up: float) -> None:
        """Pitch the camera (positive = up) and keep the offset bookkeeping."""

        if self.yaw is None or abs(counts_up) < 1.0:
            return
        self.yaw.pitch(-float(counts_up))
        self.pitch_offset_counts += float(counts_up)

    def _level_camera(self) -> None:
        if self.yaw is None:
            return
        self.yaw.level_camera()
        self.pitch_offset_counts = 0.0
        self.search_pitched = False
        self.camera_levelled = True

    def _enter_phase_locked(self, phase: str, now: float) -> None:
        if phase == self.phase:
            return
        previous = self.phase
        if phase in {"estimating", "aligning"} and abs(self.pitch_offset_counts) >= 1.0:
            # Formation phases want the level view the owner would have.
            self._pitch_by(-self.pitch_offset_counts)
            self.search_pitched = False
        self.phase = phase
        self.phase_started = now
        if phase == "searching":
            self.search_steps = 0
            self.search_next_observe_at = 0.0
            self.search_backstep_done = False
            self.sweep_openness = []
            if previous not in {"searching", "reacquiring", "waiting"}:
                self.search_episode_started = now
                self.episode_sweeps = 0
                self.patrol_moves = 0
                self.blocked_headings = set()
        elif phase == "waiting":
            self.waiting_since = now
            self.waiting_next_observe_at = now + 1.0
            self.patrol_pending = False
            self.identity_latched = False
            self.owner_private_confirmed_at = 0.0
            last_bearing = self.estimator.bearing_deg
            self.search_direction = -1.0 if (last_bearing is not None and last_bearing < 0) else 1.0
            self.identity_latched = False
            self.owner_private_confirmed_at = 0.0
        elif phase == "reacquiring":
            self.reacquire_step = 0
            self.reacquire_next_at = 0.0
        elif phase == "estimating":
            self.estimate_cycles = 0
        elif phase == "orbiting":
            self.orbit_started = now
        elif phase == "aligning":
            self.align_turned = False
        elif phase == "holding":
            self.hold_deviation_cycles = 0
            self.formation_completed_at = now

    def _search_locked(self, now: float) -> float:
        """One search step: the observation for this heading was empty."""

        cfg = self.config
        self.search_steps += 1
        self.sweep_openness.append(self.last_openness)
        if self.search_steps == 1:
            self.announce_status(
                "owner_search_started",
                "🔎 オーナー探索中：" + ("左" if self.search_direction < 0 else "右") + "回りに全周スキャン",
                min_interval=2.0,
            )
        if self.search_episode_started <= 0.0:
            self.search_episode_started = now
        if cfg.idle_after_seconds > 0 and now - self.search_episode_started > cfg.idle_after_seconds:
            self._enter_phase_locked("waiting", now)
            self.waiting_reason = "search_budget"
            self.announce_status(
                "waiting_idle",
                "🛌 オーナーが見つかりません：待機します（見かけたら接近、定期的に所在を再確認）",
                min_interval=10.0,
            )
            return 0.0
        if self.search_steps % int(round(360.0 / cfg.search_step_deg)) == 0:
            self.episode_sweeps += 1
            sweeps = self.episode_sweeps
            self.announce_status(
                f"empty_sweep_{sweeps}", f"🔎 全周スキャン {sweeps} 回完了：未発見。探索を継続", min_interval=0.0
            )
            if (
                self._locator_factory is not None
                and cfg.menu_locate_after_sweeps > 0
                and sweeps >= cfg.menu_locate_after_sweeps
                and sweeps % max(1, cfg.menu_locate_after_sweeps) == 0
                and (
                    self.last_locate_at <= 0.0
                    or now - self.last_locate_at >= cfg.menu_locate_cooldown_seconds
                )
            ):
                self.locate_requested = True
            elif cfg.search_patrol and self.patrol_moves < cfg.search_patrol_max_moves:
                # Spinning on the spot again is pointless: move toward the
                # heading that looked most open and sweep from there.
                steps_per_sweep = int(round(360.0 / cfg.search_step_deg))
                scores = self.sweep_openness[-steps_per_sweep:]
                best: tuple[float, int] | None = None
                for index, score in enumerate(scores):
                    heading = (index + 1) % steps_per_sweep  # step k observed heading k*step
                    if heading in self.blocked_headings:
                        continue
                    if best is None or score > best[0]:
                        best = (score, heading)
                if best is not None:
                    self.patrol_turn_deg = self.search_direction * best[1] * cfg.search_step_deg
                    self.patrol_pending = True
            self.sweep_openness = []
            if self.yaw is not None:
                # Odd sweeps look up for a nameplate that is too close to be
                # in a level view; even sweeps re-level absolutely.  Standing
                # inside another avatar hides the plate from any pitch, so
                # one step back is taken before the second pair of sweeps.
                if not self.search_pitched:
                    self._pitch_by(self.search_pitch_counts)
                    self.search_pitched = True
                else:
                    self._level_camera()
                    if sweeps >= 2 and not self.search_backstep_done and self.motion is not None:
                        self.search_backstep_done = True
                        self.motion.pulse(-cfg.move_axis, 0.0, 0.6)
        self.search_next_observe_at = now + cfg.search_settle_seconds + 0.05
        return self.search_direction * cfg.search_step_deg

    def _reacquire_locked(self, now: float) -> float:
        """Look toward the last bearing, then widen; escalate to search."""

        if now < self.reacquire_next_at:
            return 0.0
        elapsed = now - self.phase_started
        if elapsed > self.config.reacquire_seconds:
            self._enter_phase_locked("searching", now)
            return 0.0
        last_bearing = self.estimator.bearing_deg or 0.0
        if self.reacquire_step == 0:
            predicted = self.estimator.predicted_bearing(
                self.bot_position, self.yaw.total_yaw_deg if self.yaw else 0.0, now
            )
            if predicted is not None:
                last_bearing = predicted
        pattern = [last_bearing, -18.0, 36.0, -54.0, 72.0]
        if self.reacquire_step >= len(pattern):
            self._enter_phase_locked("searching", now)
            return 0.0
        turn = pattern[self.reacquire_step]
        if self.reacquire_step == 0 and abs(turn) < 3.0:
            turn = 0.0
        self.reacquire_step += 1
        self.reacquire_next_at = now + 0.45
        return turn

    def _distance_command(self, distance: float | None, *, gentle: bool = False) -> float:
        cfg = self.config
        if distance is None:
            return 0.0
        if distance < cfg.min_distance_m:
            return -cfg.move_axis
        error = distance - cfg.follow_distance_m
        deadband = 0.25 if not gentle else 0.35
        if abs(error) <= deadband:
            return 0.0
        magnitude = min(1.0, max(0.45, abs(error) / 1.2))
        axis = cfg.move_axis * magnitude * (1.0 if error > 0 else -0.8)
        return axis

    def _formation_control_locked(
        self, now: float, phase: str, bot_yaw: float
    ) -> tuple[float, float, float, float]:
        """Return (forward, strafe, turn_deg, pulse_seconds) for a fresh observation."""

        cfg = self.config
        est = self.estimator
        bearing = est.bearing_deg if est.bearing_deg is not None else 0.0
        distance = est.distance_m
        forward = strafe = turn = 0.0
        pulse = 0.0

        # An owner in motion is followed from wherever we are.
        moving = est.owner_moving
        if moving and phase in self.FORMATION_PHASES:
            self._enter_phase_locked("following", now)
            phase = "following"

        if phase in {"approaching", "following"}:
            if abs(bearing) > 2.0:
                turn = max(-45.0, min(45.0, bearing))
            if abs(bearing) <= 25.0:
                forward = self._distance_command(distance)
            if distance is None and self.pose_error is not None:
                # No range sensor at all: walk while the plate is centred and
                # stop when it climbs to the top of the frame.
                forward = 0.0
            settled = (
                distance is not None
                and abs(distance - cfg.follow_distance_m) <= 0.3
                and abs(bearing) <= 4.0
            )
            if settled and not moving and est.stationary and now - self.phase_started >= 0.6:
                if phase == "following":
                    self.announce_status("owner_stopped", "⏸ オーナー停止：隊形へ移動", min_interval=3.0)
                self._enter_phase_locked("estimating", now)
                phase = "estimating"
                self.estimate_cycles = 0
            else:
                return forward, strafe, turn, pulse

        if phase == "estimating":
            self.estimate_cycles += 1
            if abs(bearing) > 2.5:
                turn = max(-30.0, min(30.0, bearing))
            forward = self._distance_command(distance, gentle=True)
            elapsed = now - self.phase_started
            if est.owner_heading is not None and (self.estimate_cycles >= 3 or elapsed >= 1.2):
                self._enter_phase_locked("orbiting", now)
                phase = "orbiting"
            elif elapsed >= 4.0:
                # No usable pose: hold facing the owner rather than wander.
                self._enter_phase_locked("holding", now)
                self.hold_facing_only = True
                return forward, 0.0, turn, pulse
            else:
                return forward, 0.0, turn, pulse

        if phase == "orbiting":
            theta = est.theta_relative(bot_yaw)
            if theta is None:
                self._enter_phase_locked("estimating", now)
                return 0.0, 0.0, 0.0, 0.0
            delta = wrap_degrees(cfg.target_theta_deg - theta)
            self.last_orbit_delta = delta
            if abs(bearing) > 2.0:
                turn = max(-40.0, min(40.0, bearing))
            forward = self._distance_command(distance, gentle=True)
            timed_out = now - self.orbit_started > 45.0
            if abs(delta) <= 5.0 or timed_out:
                self._enter_phase_locked("aligning", now)
                phase = "aligning"
                strafe = 0.0
            else:
                radius = distance if distance is not None else cfg.follow_distance_m
                # Orbit in timed pulses that end before the next capture, so
                # every observation is taken standing still and a slow OCR
                # pass can never carry the avatar past the slot.  The pulse
                # covers at most the remaining angle (v / r) and 0.3 s.
                speed_mps = cfg.move_axis * WALK_SPEED_MPS_PER_AXIS
                seconds = math.radians(abs(delta)) * radius / max(speed_mps, 0.2)
                pulse = max(0.08, min(0.30, seconds * 0.85))
                strafe = cfg.move_axis if delta > 0 else -cfg.move_axis
                return forward, strafe, turn, pulse

        if phase == "aligning":
            if not self.align_turned:
                self.align_turned = True
                # After orbiting the owner is (nearly) centred; turning by
                # (bearing - desired bearing) puts them at the right edge and
                # the bot on the owner's heading (+ heading_offset).
                return 0.0, 0.0, bearing - cfg.owner_bearing_deg, 0.0
            self._enter_phase_locked("holding", now)
            self.hold_facing_only = False
            self.announce_status(
                "formation_complete", "🧭 隊形完了：オーナーの左後方で同じ方向を向いています", min_interval=5.0
            )
            phase = "holding"

        if phase == "holding":
            facing_only = getattr(self, "hold_facing_only", False)
            target_bearing = 0.0 if facing_only else cfg.owner_bearing_deg
            error = bearing - target_bearing
            distance_error = (distance - cfg.follow_distance_m) if distance is not None else 0.0
            if abs(error) > 6.0:
                turn = max(-25.0, min(25.0, error))
            large_deviation = abs(error) > 18.0 or abs(distance_error) > 0.7
            self.hold_deviation_cycles = self.hold_deviation_cycles + 1 if large_deviation else 0
            if self.hold_deviation_cycles >= 2 or moving:
                self._enter_phase_locked("following", now)
                return 0.0, 0.0, turn, 0.0
            # The owner turned: re-form after they settle -- unless they
            # turned to face us (a conversation, not a new viewing
            # direction), in which case we keep them in view where we are.
            theta = est.theta_relative(bot_yaw)
            facing_us = theta is not None and abs(theta) > 140.0
            self.owner_facing_us = facing_us
            if (
                not facing_only
                and theta is not None
                and not facing_us
                and est.heading_confidence >= 0.5
                and abs(wrap_degrees(theta - cfg.target_theta_deg)) > 40.0
                and now - self.formation_completed_at > 3.0
            ):
                self._enter_phase_locked("estimating", now)
            if facing_only and now - self.phase_started > 10.0:
                self._enter_phase_locked("estimating", now)
            return 0.0, 0.0, turn, 0.0

        return forward, strafe, turn, pulse

    def _waiting_locked(self, now: float) -> float:
        """Owner offline/unreachable: stand still, glance regularly, re-check."""

        cfg = self.config
        self.waiting_next_observe_at = now + 3.0
        if (
            self._locator_factory is not None
            and cfg.idle_recheck_seconds > 0
            and now - self.last_locate_at >= cfg.idle_recheck_seconds
        ):
            self.locate_requested = True
        return 0.0

    def _run_patrol(self, now: float) -> None:
        """Turn to the chosen heading and walk a bounded distance (stuck-aware)."""

        assert self.yaw is not None and self.motion is not None
        cfg = self.config
        with self.lock:
            self.patrol_pending = False
            turn = self.patrol_turn_deg
            self.patrol_moves += 1
        self._stop_motion()
        if abs(turn) >= 1.0:
            applied = self.yaw.turn(turn)
            self.estimator.apply_yaw(applied)
            self._sleep(0.3)
        before = self.frame_provider(max_age_seconds=2.0)
        self.motion.set_axes(cfg.move_axis, 0.0)
        self._sleep(max(0.3, cfg.search_patrol_seconds))
        self._stop_motion(force=True)
        self._sleep(0.35)
        after = self.frame_provider(max_age_seconds=2.0)
        moved = True
        if before is not None and after is not None and before.shape == after.shape:
            small_before = cv2.resize(before, (160, 90), interpolation=cv2.INTER_AREA)
            small_after = cv2.resize(after, (160, 90), interpolation=cv2.INTER_AREA)
            difference = float(np.abs(small_after.astype(np.int16) - small_before.astype(np.int16)).mean())
            moved = difference >= 4.0
        heading_index = int(round(turn / cfg.search_step_deg)) % max(1, int(round(360.0 / cfg.search_step_deg)))
        with self.lock:
            if not moved:
                self.blocked_headings.add(heading_index)
            self.search_steps = 0
            self.search_next_observe_at = self._clock() + 0.4
            self.sweep_openness = []
            self.finder.last_box = None if self.finder is not None else None
        self._report(
            ownerFollowPatrolMoves=self.patrol_moves,
            ownerFollowPatrolMoved=moved,
            ownerFollowPatrolTurnDeg=round(turn, 1),
            ownerFollowBlockedHeadings=sorted(self.blocked_headings),
        )

    # ------------------------------------------------------------------
    # menu fallback: is the owner in this instance? join them elsewhere
    # ------------------------------------------------------------------
    def _run_menu_locate(self, now: float) -> None:
        """Blocking: VRChat menu check / join, then restart the search."""

        factory = self._locator_factory
        with self.lock:
            self.locate_requested = False
            if factory is None:
                return
            self.locate_in_progress = True
            self.last_locate_at = now
            owner = self.config.owner_account
        self._stop_motion(force=True)
        self.announce_status(
            "menu_locate", "🧭 オーナーが見当たりません：メニューで所在を確認します", min_interval=10.0
        )
        self._report(ownerFollowState="locating", ownerFollowPhase="locating", ownerFollowDetected=False)
        result: dict[str, Any]
        try:
            locator = factory(owner)
            outcome = locator.run(join=True)
            result = outcome.as_dict() if hasattr(outcome, "as_dict") else dict(outcome)
        except Exception as exc:
            result = {"outcome": "error", "detail": f"{type(exc).__name__}: {exc}"}
        with self.lock:
            self.locate_in_progress = False
            self.last_locate_result = result
            self.last_locate_at = self._clock()
            # Whatever happened, the view and possibly the world changed:
            # forget the track and sweep again from a level camera.
            self.estimator.reset()
            self.bot_position = (0.0, 0.0)
            self.last_observation = None
            self.camera_levelled = False
            if self.finder is not None:
                self.finder.last_box = None
            if self.pose is not None:
                self.pose.last_bbox = None
            self.identity_latched = False
            self.owner_private_confirmed_at = 0.0
            self.last_ocr_confirmed = 0.0
            outcome = str(result.get("outcome"))
            if outcome in {"not_online", "join_unavailable"}:
                # Nobody to find here: do not spin.  Watch, and re-check later.
                self.phase = "searching"  # so _enter_phase_locked records the transition
                self._enter_phase_locked("waiting", self._clock())
                self.waiting_reason = outcome
            else:
                previous = self.phase
                self.phase = "locating"
                self._enter_phase_locked("searching", self._clock())
                if outcome in {"joined", "in_instance"}:
                    self.search_episode_started = self._clock()
                    self.episode_sweeps = 0
                    self.patrol_moves = 0
                    self.blocked_headings = set()
        outcome_text = {
            "in_instance": "✅ オーナーはこのワールドにいます：探索を続けます",
            "joined": "🚪 オーナーのインスタンスへ移動しました：探索を再開します",
            "not_online": "🛌 オーナーはオフラインのようです：待機します（定期的に再確認）",
            "join_unavailable": "⛔ オーナーのインスタンスには参加できません：待機します（定期的に再確認）",
        }.get(str(result.get("outcome")), "⚠ メニュー確認に失敗しました：探索を続けます")
        self.announce_status("menu_locate_result", outcome_text, min_interval=0.0)
        self._report(
            ownerFollowState=self.phase,
            ownerFollowPhase=self.phase,
            ownerFollowLocateOutcome=result.get("outcome"),
            ownerFollowLocateDetail=result.get("detail"),
            ownerFollowLocateResult=result,
            ownerFollowLocateAt=time.time(),
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _publish(self, observation: Observation | None, phase: str, axes: tuple[float, float, float], now: float) -> None:
        est = self.estimator
        bot_yaw = self.yaw.total_yaw_deg if self.yaw is not None else 0.0
        theta = est.theta_relative(bot_yaw)
        seen = observation is not None and observation.seen
        updates: dict[str, Any] = {
            "ownerFollowState": phase,
            "ownerFollowPhase": phase,
            "ownerFollowDetected": bool(seen),
            "ownerFollowAxes": list(axes),
            "ownerFollowBearingDeg": round(est.bearing_deg, 2) if est.bearing_deg is not None else None,
            "ownerFollowDistanceM": round(est.distance_m, 3) if est.distance_m is not None else None,
            "ownerFollowOwnerHeadingDeg": round(est.owner_heading, 2) if est.owner_heading is not None else None,
            "ownerFollowHeadingConfidence": round(est.heading_confidence, 3),
            "ownerFollowThetaRelative": round(theta, 2) if theta is not None else None,
            "ownerFollowOwnerSpeed": round(est.owner_speed, 3),
            "ownerFollowYawOdometryDeg": round(bot_yaw, 2),
            "ownerFollowBotPosition": [round(self.bot_position[0], 3), round(self.bot_position[1], 3)],
            "ownerFollowIdentityLatched": self.identity_latched,
            "ownerFollowTrackConfirmed": now - self.last_ocr_confirmed <= self.IDENTITY_GRACE_SECONDS if self.last_ocr_confirmed > 0 else False,
            "ownerFollowMouseYawAvailable": self.yaw.mouse_available if self.yaw is not None else None,
            "ownerFollowLastError": self.yaw.last_error if self.yaw is not None else None,
        }
        if observation is not None:
            updates["ownerFollowCaptureFresh"] = True
            plate = observation.nameplate
            updates["ownerFollowNameplateText"] = plate.text if plate else None
            updates["ownerFollowNameplateBox"] = list(plate.box) if plate else None
            updates["ownerFollowNameplateSource"] = plate.source if plate else None
            updates["ownerFollowIdentitySource"] = "ocr" if plate else ("body" if observation.body else None)
            updates["ownerFollowIdentityConfirmed"] = plate is not None
            body = observation.body
            updates["ownerFollowPoseConf"] = round(body.conf, 3) if body else None
            updates["ownerFollowPoseYaw"] = round(body.yaw_deg, 1) if body else None
            updates["ownerFollowFaceVisible"] = bool(body.face_visible) if body else None
            updates["ownerFollowPoseDistanceM"] = round(body.distance_m, 3) if body else None
            updates["ownerFollowPlateDistanceM"] = (
                round(observation.plate_distance_m, 3) if observation.plate_distance_m is not None else None
            )
            updates["ownerFollowNameplateHeightM"] = round(self.nameplate_height_m, 3)
            updates["ownerFollowPitchOffsetCounts"] = round(self.pitch_offset_counts, 1)
            updates["ownerFollowBodyBox"] = [round(v, 1) for v in body.bbox] if body else None
            if seen:
                updates["ownerFollowLastSeenAt"] = time.time()
                updates["ownerFollowFacingUs"] = bool(body is not None and abs(body.yaw_deg) > 120.0)
        if self.finder is not None:
            updates["ownerFollowOCRTexts"] = list(self.finder.last_texts[:12])
            updates["ownerFollowOCRCropSeconds"] = round(self.finder.last_crop_ocr_seconds, 3)
            updates["ownerFollowOCRFullSeconds"] = round(self.finder.last_full_ocr_seconds, 3)
        if phase == "searching":
            updates["ownerFollowSearchStep"] = self.search_steps
            updates["ownerFollowSearchDirection"] = "left" if self.search_direction < 0 else "right"
        if phase == "orbiting":
            updates["ownerFollowOrbitDelta"] = round(getattr(self, "last_orbit_delta", 0.0), 2)
        if phase == "waiting":
            updates["ownerFollowWaitingReason"] = self.waiting_reason
            updates["ownerFollowWaitingSeconds"] = round(now - self.waiting_since, 1)
        self._report(**updates)
