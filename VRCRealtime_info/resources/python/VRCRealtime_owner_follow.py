"""Local, screen-based owner formation control for the VRChat bridge.

The controller deliberately uses only local VRChat pixels and OSC input axes.
It never sends OCR text or frames to OpenAI.  It first approaches a recognised
owner nameplate, then orbits to the owner's left/rear before maintaining that
formation.  If the owner leaves the field of view, translation stops and the
avatar rotates in place to reacquire the nameplate.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from difflib import SequenceMatcher
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from vrcpilot import OscSender
from vrcpilot.ocr import OCRWord, RapidOCREngine

from VRCRealtime_person_detector import AvatarDetection, MediaPipePersonDetector


class _OCRLike(Protocol):
    def recognize(self, image: NDArray[np.uint8]) -> Sequence[OCRWord]: ...


class _ControllerLike(Protocol):
    def vertical(self, value: float) -> None: ...

    def horizontal(self, value: float) -> None: ...

    def look_horizontal(self, value: float) -> None: ...

    def look_left(self, active: bool = True) -> None: ...

    def look_right(self, active: bool = True) -> None: ...

    def chatbox(self, text: str, *, send: bool = True, sfx: bool = True) -> None: ...


class _PersonDetectorLike(Protocol):
    def detect(self, frame_rgb: NDArray[np.uint8]) -> Sequence[AvatarDetection]: ...


class OwnerFollowController:
    """Follow one visible VRChat nameplate using a bounded OSC controller."""

    SEARCH_STEP_SECONDS = 0.90
    SEARCH_TURN_SECONDS = 0.14
    SEARCH_OBSERVE_AFTER = 0.32
    SEARCH_STEPS_PER_SWEEP = 12
    SEARCH_RETURN_OBSERVE_SECONDS = 1.8
    PRIVATE_OWNER_NEAR_RATIO = 0.55
    PRIVATE_OWNER_CONFIRM_RATIO = 0.90
    PRIVATE_OWNER_CONFIRM_TTL_SECONDS = 30.0
    PRIVATE_OWNER_VISIBLE_GRACE_SECONDS = 3.5
    OWNER_REACQUIRE_GRACE_SECONDS = 5.0
    SPEECH_RELEASE_GRACE_SECONDS = 1.0

    def __init__(
        self,
        *,
        owner_account: str,
        initial_mode: str,
        frame_provider: Callable[..., NDArray[np.uint8] | None],
        tracking_fps: float,
        target_screen_x: float,
        target_nameplate_height: float,
        lost_seconds: float,
        formation_seconds: float,
        orbit_speed: float,
        search_turn_speed: float,
        assume_single_avatar: bool,
        ocr_min_side: int,
        ocr_box_threshold: float,
        ocr_text_score: float,
        osc_host: str,
        osc_port: int,
        state_callback: Callable[..., None] | None = None,
        ocr_factory: Callable[[], _OCRLike] | None = None,
        person_detector_factory: Callable[[], _PersonDetectorLike] | None = None,
        controller: _ControllerLike | None = None,
    ) -> None:
        self.owner_account = owner_account.strip()
        if not self.owner_account:
            raise ValueError("owner_account must not be empty")
        self.frame_provider = frame_provider
        self.tracking_fps = tracking_fps
        self.target_screen_x = target_screen_x
        self.target_nameplate_height = target_nameplate_height
        self.lost_seconds = lost_seconds
        self.formation_seconds = formation_seconds
        self.orbit_speed = orbit_speed
        self.search_turn_speed = search_turn_speed
        self.assume_single_avatar = bool(assume_single_avatar)
        self.ocr_min_side = ocr_min_side
        self.ocr_box_threshold = ocr_box_threshold
        self.ocr_text_score = ocr_text_score
        self.osc_host = osc_host
        self.osc_port = osc_port
        self.state_callback = state_callback
        self.ocr_factory = ocr_factory or (
            lambda: RapidOCREngine(
                params={
                    "Det.limit_side_len": self.ocr_min_side,
                    "Det.box_thresh": self.ocr_box_threshold,
                    "Global.text_score": self.ocr_text_score,
                    "Global.log_level": "warning",
                }
            )
        )
        self.ocr_engine: _OCRLike | None = None
        self.person_detector_factory = person_detector_factory or (
            lambda: MediaPipePersonDetector(
                MediaPipePersonDetector.default_model_path(),
                score_threshold=0.35,
            )
        )
        self.person_detector: _PersonDetectorLike | None = None
        self.person_detector_attempted = False
        self.controller = controller

        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.lock = threading.Lock()
        self.axis_lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.mode = self._canonical_mode(initial_mode)
        self.last_seen_monotonic = 0.0
        self.smoothed_x = 0.5
        self.smoothed_height = 0.0
        self.last_axes = (0.0, 0.0, 0.0)
        self.last_button_turn = 0
        self.motion_stopped = True
        self.phase = (
            "acquiring"
            if self.mode == "OwnerFollow"
            else ("presentation" if self.mode == "Presentation" else "standalone")
        )
        self.phase_before_search = "acquiring"
        self.formation_started = 0.0
        self.formation_saw_front_face = False
        self.formation_face_lost_at = 0.0
        self.owner_facing_us = False
        self.tracking_front_face_hits = 0
        self.search_started = 0.0
        self.search_direction = 1.0
        self.search_tile_index = 0
        self.search_best_candidate: dict[str, Any] | None = None
        self.search_return_started = 0.0
        self.search_return_steps = 0
        self.search_return_direction = 0.0
        self.search_last_chat_step = 0
        self.chat_last_key: str | None = None
        self.chat_last_at = 0.0
        self.chat_key_times: dict[str, float] = {}
        self.magnified_tracking_until = 0.0
        self.magnified_tracking_center = (0.5, 0.5)
        self.appearance_descriptor: NDArray[np.float32] | None = None
        self.appearance_tracker: Any | None = None
        self.appearance_tracker_scale = 1.0
        self.appearance_nameplate_offset = (0.0, 0.0)
        self.appearance_last_similarity = 0.0
        self.appearance_failures = 0
        self.appearance_only_started = 0.0
        self.last_ocr_confirmed = 0.0
        self.identity_source: str | None = None
        self.owner_track_confirmed = False
        self.owner_identity_latched = False
        self.owner_private_confirmed_at = 0.0
        self.avatar_candidate_height = 0.0
        self.avatar_candidate_started = 0.0
        self.manual_turn_direction = 0
        self.manual_turn_until = 0.0
        self.manual_turn_label: str | None = None
        self.attention_hold_until = 0.0
        self.conversation_hold = False
        self.speech_hold = False
        self.speech_hold_until = 0.0
        self.listener_hold = False
        self.listener_hold_until = 0.0
        self.previous_motion_gray: NDArray[np.uint8] | None = None
        self.last_visual_motion = 0.0
        self.visual_motion_sample = 0
        self.consumed_visual_motion_sample = 0
        self.translation_no_motion_seconds = 0.0
        self.last_control_at = time.monotonic()
        self.escape_started = 0.0
        self.escape_source_phase = "acquiring"
        self.escape_blocked_axes = (0.0, 0.0)
        self.escape_turn_direction = 1.0
        self.escape_attempt = 0

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("OpenCV frontal-face cascade could not be loaded")
        self._report(
            avatarMode=self.mode,
            ownerAccount=self.owner_account,
            ownerFollowTargetNameplateHeight=self.target_nameplate_height,
            ownerFollowOCRMinSide=self.ocr_min_side,
            ownerFollowOCRBoxThreshold=self.ocr_box_threshold,
            ownerFollowOCRTextScore=self.ocr_text_score,
            ownerFollowAssumeSingleAvatar=self.assume_single_avatar,
            ownerFollowSearchStepsPerSweep=self.SEARCH_STEPS_PER_SWEEP,
            ownerFollowSearchCandidateRemembered=False,
            ownerFollowChatboxText=None,
            ownerFollowChatboxAt=None,
            ownerFollowIdentityLatched=False,
            ownerFollowTrackConfirmed=False,
            ownerFollowPrivateConfirmationCurrent=False,
            ownerFollowPrivateConfirmationAgeSeconds=None,
            ownerFollowSpeechHold=False,
            ownerFollowListenerHold=False,
            ownerFollowState=self.phase,
            ownerFollowDetected=False,
            ownerFollowFacingUs=False,
        )

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
    def _normalise_name(text: str) -> str:
        normalised = unicodedata.normalize("NFKC", text).casefold()
        return "".join(ch for ch in normalised if ch.isalnum())

    def _report(self, **updates: Any) -> None:
        if self.state_callback is not None:
            self.state_callback(**updates)

    def announce_status(
        self,
        key: str,
        text: str,
        *,
        min_interval: float = 2.0,
    ) -> bool:
        """Publish a short, rate-limited status through VRChat's chatbox."""

        controller = self.controller
        if controller is None or not hasattr(controller, "chatbox"):
            return False
        now = time.monotonic()
        with self.chat_lock:
            previous_at = self.chat_key_times.get(key, 0.0)
            if previous_at > 0.0 and now - previous_at < max(1.2, min_interval):
                return False
            # VRChat currently accepts at most 144 characters for chatbox text.
            message = str(text).strip()[:144]
            if not message:
                return False
            try:
                controller.chatbox(message, send=True, sfx=False)
            except Exception as exc:
                self._report(
                    ownerFollowChatboxLastError=f"{type(exc).__name__}: {exc}"
                )
                return False
            self.chat_last_key = key
            self.chat_last_at = now
            self.chat_key_times[key] = now
            if len(self.chat_key_times) > 128:
                oldest = sorted(self.chat_key_times, key=self.chat_key_times.get)[:32]
                for old_key in oldest:
                    self.chat_key_times.pop(old_key, None)
        self._report(
            ownerFollowChatboxText=message,
            ownerFollowChatboxAt=time.time(),
            ownerFollowChatboxLastError=None,
        )
        return True

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="VRCRealtimeOwnerFollow",
            daemon=True,
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
            if canonical == "OwnerFollow":
                self._reset_follow_cycle_locked()
            else:
                self.phase = (
                    "presentation" if canonical == "Presentation" else "standalone"
                )
        if canonical != "OwnerFollow":
            self._stop_motion(force=True)
        self._report(
            avatarMode=canonical,
            ownerFollowState=(
                "acquiring"
                if canonical == "OwnerFollow"
                else ("presentation" if canonical == "Presentation" else "standalone")
            ),
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
            self.owner_account = value
            self._reset_follow_cycle_locked()
        self._stop_motion(force=True)
        self._report(ownerAccount=value, ownerFollowDetected=False)
        self.wake_event.set()

    def set_target_nameplate_height(self, value: float) -> float:
        """Adjust the apparent-size target used as the following distance."""

        target = float(value)
        if not 0.005 <= target <= 0.15:
            raise ValueError("target nameplate height must be between 0.005 and 0.15")
        with self.lock:
            self.target_nameplate_height = target
        self._report(ownerFollowTargetNameplateHeight=target)
        self.wake_event.set()
        return target

    def turn(self, direction: str) -> str:
        """Schedule a voice-directed 90 or 180 degree in-place turn."""

        compact = "".join(ch for ch in str(direction).casefold() if ch.isalnum())
        if compact in {"left", "left90", "左", "左旋回"}:
            label, sign, seconds = "left", -1, 0.55
        elif compact in {"right", "right90", "右", "右旋回"}:
            label, sign, seconds = "right", 1, 0.55
        elif compact in {
            "around",
            "turnaround",
            "back",
            "180",
            "反対",
            "反対向き",
        }:
            label, sign, seconds = "around", 1, 1.10
        else:
            raise ValueError("direction must be left, right, or around")
        with self.lock:
            self.manual_turn_direction = sign
            self.manual_turn_until = time.monotonic() + seconds
            self.manual_turn_label = label
        self._stop_motion(force=True)
        self._report(
            ownerFollowState="manual_turning",
            ownerFollowManualTurn=label,
            ownerFollowManualTurnActive=True,
        )
        self.wake_event.set()
        return label

    def private_access_status(self) -> dict[str, Any]:
        """Return the live owner-proximity decision for high-PL local speech.

        A close-range OCR observation establishes a 30-second identity lease.
        While the same follow track remains intact, appearance/person tracking
        may maintain it even when the nearby nameplate leaves the capture.
        Entering search clears the lease and requires close OCR confirmation.
        """

        now = time.monotonic()
        with self.lock:
            mode = self.mode
            phase = self.phase
            identity_source = self.identity_source
            identity_latched = self.owner_identity_latched
            private_confirmed_at = self.owner_private_confirmed_at
            last_seen = self.last_seen_monotonic
            height = self.smoothed_height
            target_height = self.target_nameplate_height
        visible_grace = max(
            self.PRIVATE_OWNER_VISIBLE_GRACE_SECONDS,
            self.lost_seconds * 2.0,
        )
        seen_age = now - last_seen if last_seen > 0.0 else float("inf")
        fresh = seen_age <= visible_grace
        confirmation_age = (
            now - private_confirmed_at
            if private_confirmed_at > 0.0
            else float("inf")
        )
        confirmation_current = (
            confirmation_age <= self.PRIVATE_OWNER_CONFIRM_TTL_SECONDS
        )
        identity_confirmed = bool(identity_latched and confirmation_current)
        distance_ratio = height / max(target_height, 1.0e-6)
        tracking_phase = phase in {"forming_left", "following_left"}
        allowed = bool(
            mode == "OwnerFollow"
            and fresh
            and identity_confirmed
            and tracking_phase
            and distance_ratio >= self.PRIVATE_OWNER_NEAR_RATIO
        )
        if mode != "OwnerFollow":
            reason = (
                "presentation_mode" if mode == "Presentation" else "standalone_mode"
            )
        elif not fresh:
            reason = "owner_not_visible"
        elif not identity_confirmed:
            reason = (
                "owner_close_confirmation_expired"
                if identity_latched and private_confirmed_at > 0.0
                else "owner_identity_unconfirmed"
            )
        elif not tracking_phase:
            reason = "owner_search_or_approach_incomplete"
        elif distance_ratio < self.PRIVATE_OWNER_NEAR_RATIO:
            reason = "owner_too_far"
        else:
            reason = "allowed"
        status = {
            "allowed": allowed,
            "reason": reason,
            "mode": mode,
            "phase": phase,
            "identityConfirmed": identity_confirmed,
            "identitySource": identity_source,
            "identityLatched": identity_latched,
            "closeConfirmationCurrent": confirmation_current,
            "closeConfirmationAgeSeconds": (
                round(confirmation_age, 3)
                if confirmation_age != float("inf")
                else None
            ),
            "closeConfirmationTTLSeconds": self.PRIVATE_OWNER_CONFIRM_TTL_SECONDS,
            "fresh": fresh,
            "seenAgeSeconds": (
                round(seen_age, 3) if seen_age != float("inf") else None
            ),
            "visibleGraceSeconds": round(visible_grace, 3),
            "distanceRatio": round(distance_ratio, 3),
            "nearRatioThreshold": self.PRIVATE_OWNER_NEAR_RATIO,
            "confirmRatioThreshold": self.PRIVATE_OWNER_CONFIRM_RATIO,
        }
        self._report(
            ownerFollowPrivateAccessAllowed=allowed,
            ownerFollowPrivateAccessReason=reason,
            ownerFollowIdentityLatched=identity_latched,
            ownerFollowPrivateConfirmationCurrent=confirmation_current,
            ownerFollowPrivateConfirmationAgeSeconds=(
                round(confirmation_age, 3)
                if confirmation_age != float("inf")
                else None
            ),
        )
        return status

    def hold_for_attention(self, seconds: float = 1.0) -> None:
        """Release all movement while the visual gate stabilises on a speaker."""

        until = time.monotonic() + max(0.25, float(seconds))
        with self.lock:
            self.attention_hold_until = max(self.attention_hold_until, until)
        self._stop_motion()
        self._report(
            ownerFollowState="attention_hold",
            ownerFollowAttentionHold=True,
            ownerFollowAxes=[0.0, 0.0, 0.0],
        )

    def set_conversation_active(self, active: bool) -> None:
        """Pause formation control for the complete latched voice session."""

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
            (
                "💬 会話中：発話中は停止・全周探索を抑制"
                if active
                else "✅ 会話終了：追尾を再開"
            ),
            min_interval=1.2,
        )
        self.wake_event.set()

    def set_speech_active(self, active: bool) -> None:
        """Pause navigation while either OpenAI or local audio is speaking."""

        now = time.monotonic()
        with self.lock:
            self.speech_hold = bool(active)
            if active:
                self.speech_hold_until = 0.0
            else:
                self.speech_hold_until = now + self.SPEECH_RELEASE_GRACE_SECONDS
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
        self.announce_status(
            "speech_hold" if active else "speech_release",
            (
                "🔊 発話中：追尾・探索を一時停止"
                if active
                else "⏸ 発話終了：1秒後に追尾を再開"
            ),
            min_interval=1.2,
        )
        self.wake_event.set()

    def set_listener_active(self, active: bool) -> None:
        """Pause movement only while the human is actively speaking."""

        now = time.monotonic()
        with self.lock:
            self.listener_hold = bool(active)
            self.listener_hold_until = (
                0.0 if active else now + 0.45
            )
        if active:
            self._stop_motion(force=True)
        self._report(
            ownerFollowListenerHold=bool(active),
            ownerFollowState=("listener_hold" if active else self.phase),
            ownerFollowAxes=[0.0, 0.0, 0.0] if active else list(self.last_axes),
        )
        self.wake_event.set()

    def _current_mode(self) -> str:
        with self.lock:
            return self.mode

    def _reset_follow_cycle_locked(self) -> None:
        """Reset the acquire/approach/formation cycle while holding ``lock``."""

        self.last_seen_monotonic = 0.0
        self.smoothed_x = 0.5
        self.smoothed_height = 0.0
        self.phase = "acquiring"
        self.phase_before_search = "acquiring"
        self.formation_started = 0.0
        self.formation_saw_front_face = False
        self.formation_face_lost_at = 0.0
        self.owner_facing_us = False
        self.tracking_front_face_hits = 0
        self.search_started = 0.0
        self.search_direction = 1.0
        self.search_tile_index = 0
        self.search_best_candidate = None
        self.search_return_started = 0.0
        self.search_return_steps = 0
        self.search_return_direction = 0.0
        self.search_last_chat_step = 0
        self.magnified_tracking_until = 0.0
        self.magnified_tracking_center = (0.5, 0.5)
        self.appearance_descriptor = None
        self.appearance_tracker = None
        self.appearance_tracker_scale = 1.0
        self.appearance_nameplate_offset = (0.0, 0.0)
        self.appearance_last_similarity = 0.0
        self.appearance_failures = 0
        self.appearance_only_started = 0.0
        self.last_ocr_confirmed = 0.0
        self.identity_source = None
        self.owner_track_confirmed = False
        self.owner_identity_latched = False
        self.owner_private_confirmed_at = 0.0
        self.avatar_candidate_height = 0.0
        self.avatar_candidate_started = 0.0
        self.attention_hold_until = 0.0
        self.conversation_hold = False
        self.speech_hold = False
        self.speech_hold_until = 0.0
        self.listener_hold = False
        self.listener_hold_until = 0.0
        self.previous_motion_gray = None
        self.last_visual_motion = 0.0
        self.visual_motion_sample = 0
        self.consumed_visual_motion_sample = 0
        self.translation_no_motion_seconds = 0.0
        self.last_control_at = time.monotonic()
        self.escape_started = 0.0
        self.escape_source_phase = "acquiring"
        self.escape_blocked_axes = (0.0, 0.0)

    def _ensure_runtime(self) -> None:
        if self.ocr_engine is None:
            self._report(ownerFollowState="loading_ocr", ownerFollowLastError=None)
            self.ocr_engine = self.ocr_factory()
        if not self.person_detector_attempted:
            self.person_detector_attempted = True
            try:
                self.person_detector = self.person_detector_factory()
                self._report(
                    ownerFollowAvatarDetectorReady=True,
                    ownerFollowAvatarDetectorError=None,
                )
            except Exception as exc:
                self.person_detector = None
                self._report(
                    ownerFollowAvatarDetectorReady=False,
                    ownerFollowAvatarDetectorError=f"{type(exc).__name__}: {exc}",
                )
        if self.controller is None:
            self.controller = OscSender(self.osc_host, self.osc_port).controller()
        self._report(ownerFollowOCRReady=True, ownerFollowLastError=None)

    def _run(self) -> None:
        ocr_interval = 1.0 / self.tracking_fps
        next_ocr = 0.0
        next_appearance = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                manual_turn_active = (
                    self.manual_turn_direction != 0
                    and now < self.manual_turn_until
                )
                manual_turn_pending_release = (
                    self.manual_turn_direction != 0 and not manual_turn_active
                )
            if manual_turn_active:
                try:
                    self._ensure_runtime()
                    self._apply_manual_turn(now)
                except Exception as exc:
                    self._stop_motion(force=True)
                    self._report(
                        ownerFollowState="error",
                        ownerFollowLastError=f"{type(exc).__name__}: {exc}",
                    )
                self.stop_event.wait(0.05)
                continue
            if manual_turn_pending_release:
                self._finish_manual_turn()
            if self._current_mode() != "OwnerFollow":
                self._stop_motion()
                self.wake_event.wait(0.25)
                self.wake_event.clear()
                next_ocr = 0.0
                next_appearance = 0.0
                continue
            try:
                self._ensure_runtime()
                now = time.monotonic()
                # Release digital turn input before capture. During search,
                # recognize only after the view has remained still long enough
                # for CaptureLoop to publish a sharp frame.
                self._apply_control(now)
                now = time.monotonic()
                if now >= next_ocr and self._ocr_observation_allowed(now):
                    observation_started = time.monotonic()
                    self._observe_once(now)
                    self._pause_search_clock(
                        time.monotonic() - observation_started
                    )
                    next_ocr = time.monotonic() + ocr_interval
                    next_appearance = time.monotonic() + 0.22
                elif (
                    now >= next_appearance
                    and self._appearance_observation_allowed(now)
                ):
                    observation_started = time.monotonic()
                    self._track_appearance_once(now)
                    self._pause_search_clock(
                        time.monotonic() - observation_started
                    )
                    next_appearance = time.monotonic() + 0.22
            except Exception as exc:
                self._stop_motion(force=True)
                self._report(
                    ownerFollowState="error",
                    ownerFollowDetected=False,
                    ownerFollowLastError=f"{type(exc).__name__}: {exc}",
                )
                self.stop_event.wait(1.0)
            self.stop_event.wait(0.10)
        self._stop_motion(force=True)

    def _apply_manual_turn(self, now: float) -> None:
        with self.lock:
            direction = self.manual_turn_direction
            label = self.manual_turn_label
            remaining = max(0.0, self.manual_turn_until - now)
        self._send_axes(0.0, 0.0, direction * 0.42, turn_mode="hold")
        self._report(
            ownerFollowState="manual_turning",
            ownerFollowManualTurn=label,
            ownerFollowManualTurnActive=True,
            ownerFollowManualTurnSeconds=round(remaining, 2),
            ownerFollowAxes=[0.0, 0.0, round(direction * 0.42, 3)],
        )

    def _finish_manual_turn(self) -> None:
        with self.lock:
            label = self.manual_turn_label
            self.manual_turn_direction = 0
            self.manual_turn_until = 0.0
            self.manual_turn_label = None
            self.last_seen_monotonic = 0.0
            if self.mode == "OwnerFollow":
                self.phase = "searching"
                self.search_started = 0.0
                next_state = "searching"
            else:
                next_state = (
                    "presentation" if self.mode == "Presentation" else "standalone"
                )
        self._stop_motion(force=True)
        self._report(
            ownerFollowState=next_state,
            ownerFollowManualTurn=label,
            ownerFollowManualTurnActive=False,
            ownerFollowAxes=[0.0, 0.0, 0.0],
        )

    def _ocr_observation_allowed(self, now: float) -> bool:
        with self.lock:
            if self.phase != "searching" or self.search_started <= 0.0:
                return True
            return_started = self.search_return_started
            return_steps = self.search_return_steps
            search_started = self.search_started
        if return_started > 0.0:
            elapsed = max(0.0, now - return_started)
            turn_duration = return_steps * self.SEARCH_STEP_SECONDS
            if elapsed >= turn_duration:
                return True
            step_time = elapsed % self.SEARCH_STEP_SECONDS
        else:
            step_time = max(0.0, now - search_started) % self.SEARCH_STEP_SECONDS
        return step_time >= self.SEARCH_OBSERVE_AFTER

    def _appearance_observation_allowed(self, now: float) -> bool:
        with self.lock:
            if self.appearance_tracker is None:
                return False
            if self.phase != "searching" or self.search_started <= 0.0:
                return True
        return self._ocr_observation_allowed(now)

    def _pause_search_clock(self, seconds: float) -> None:
        """Exclude local image-analysis time from the physical scan clock."""

        delay = max(0.0, float(seconds))
        if delay <= 0.0:
            return
        with self.lock:
            if self.phase == "searching" and self.search_started > 0.0:
                self.search_started += delay
                if self.search_return_started > 0.0:
                    self.search_return_started += delay

    def _observe_once(self, now: float) -> None:
        frame = self.frame_provider(max_age_seconds=max(1.0, self.lost_seconds))
        if frame is None:
            self._report(ownerFollowCaptureFresh=False)
            return
        with self.lock:
            searching = self.phase == "searching"
        detections: list[AvatarDetection] = []
        detector = self.person_detector
        if detector is not None:
            multiscale = getattr(detector, "detect_multiscale", None)
            if searching and callable(multiscale):
                detections = list(multiscale(frame))
            else:
                detections = list(detector.detect(frame))
        avatar_match = self._select_avatar_detection(frame, detections)
        if avatar_match is None and searching and detections:
            if self._search_return_ready(now):
                fallback = self._best_search_detection(frame, detections)
                if fallback is not None:
                    detection, similarity = fallback
                    avatar_match = (
                        detection,
                        "avatar_detector_scan_candidate",
                        similarity,
                    )
            else:
                self._remember_search_candidate(now, frame, detections)
        self._report(
            ownerFollowAvatarDetectorReady=detector is not None,
            ownerFollowAvatarCandidates=len(detections),
            ownerFollowAvatarCandidateBoxes=[
                [*detection.bbox, round(float(detection.score), 4)]
                for detection in detections[:8]
            ],
            ownerFollowAvatarCandidateSelected=avatar_match is not None,
            ownerFollowAvatarScanEmptyDirection=False if detections else None,
        )
        if avatar_match is not None:
            avatar_detection, avatar_source, avatar_similarity = avatar_match
            self._record_avatar_detection(
                now,
                frame,
                avatar_detection,
                avatar_source,
                avatar_similarity,
            )
        elif searching and detector is not None and not detections:
            if self._skip_ocr_for_empty_direction(now):
                self._report(
                    ownerFollowCaptureFresh=True,
                    ownerFollowOCRMode="skipped_empty_direction",
                    ownerFollowOCRWords=0,
                    ownerFollowOCRTexts=[],
                    ownerFollowOCRMatches=0,
                    ownerFollowAvatarScanEmptyDirection=True,
                )
                self._advance_empty_search_direction(now)
                return
        engine = self.ocr_engine
        if engine is None:
            return
        with self.lock:
            searching = self.phase == "searching"
            magnified_tracking = now < self.magnified_tracking_until
            magnified_center = self.magnified_tracking_center
        if searching:
            crop, crop_box, tile_number = self._next_search_tile(frame)
            local_words = engine.recognize(crop)
            crop_x, crop_y, _crop_width, _crop_height = crop_box
            words = [
                OCRWord(
                    text=word.text,
                    polygon=tuple(
                        (point_x + crop_x, point_y + crop_y)
                        for point_x, point_y in word.polygon
                    ),
                    confidence=word.confidence,
                )
                for word in local_words
            ]
            ocr_mode = "search_tile"
            ocr_updates = {
                "ownerFollowOCRSearchTile": tile_number,
                "ownerFollowOCRSearchTileBox": list(crop_box),
            }
        elif magnified_tracking:
            crop, crop_box = self._tracking_tile(frame, magnified_center)
            local_words = engine.recognize(crop)
            crop_x, crop_y, _crop_width, _crop_height = crop_box
            words = [
                OCRWord(
                    text=word.text,
                    polygon=tuple(
                        (point_x + crop_x, point_y + crop_y)
                        for point_x, point_y in word.polygon
                    ),
                    confidence=word.confidence,
                )
                for word in local_words
            ]
            ocr_mode = "target_tile"
            ocr_updates = {
                "ownerFollowOCRSearchTile": None,
                "ownerFollowOCRSearchTileBox": list(crop_box),
            }
        else:
            words = list(engine.recognize(frame))
            ocr_mode = "full_frame"
            ocr_updates = {
                "ownerFollowOCRSearchTile": None,
                "ownerFollowOCRSearchTileBox": None,
            }
        selected = self._select_owner_word(words)
        identity_source = "ocr"
        appearance_similarity = 1.0
        if selected is None:
            appearance_match = self._select_appearance_word(frame, words)
            if appearance_match is not None:
                selected, appearance_similarity = appearance_match
                identity_source = "appearance_nameplate"
        self._report(
            ownerFollowCaptureFresh=True,
            ownerFollowOCRWords=len(words),
            ownerFollowOCRTexts=[word.text for word in words[:12]],
            ownerFollowOCRMode=ocr_mode,
            **ocr_updates,
        )
        if selected is None:
            self._report(
                ownerFollowOCRMatches=0,
                ownerFollowAppearanceReady=self.appearance_descriptor is not None,
            )
            return

        x, y, width, height = selected.bbox
        image_height, image_width = frame.shape[:2]
        centre_x = (x + width * 0.5) / max(image_width, 1)
        centre_y = (y + height * 0.5) / max(image_height, 1)
        height_fraction = height / max(image_height, 1)
        avatar_box = self._avatar_box_from_nameplate(frame, selected)
        if identity_source == "ocr":
            appearance_similarity = self._learn_owner_appearance(
                frame,
                avatar_box,
            )
        if avatar_box is not None:
            self._initialise_appearance_tracker(frame, avatar_box, selected)
        facing_us = self._owner_face_is_frontal(frame, selected)
        visual_motion = self._camera_motion_fraction(frame)
        with self.lock:
            previous_last_seen = self.last_seen_monotonic
            previous_identity_source = self.identity_source
            was_stale = (
                previous_last_seen <= 0.0
                or now - previous_last_seen > self.lost_seconds
            )
            alpha = 1.0 if was_stale else 0.45
            self.smoothed_x = self.smoothed_x * (1.0 - alpha) + centre_x * alpha
            self.smoothed_height = (
                self.smoothed_height * (1.0 - alpha) + height_fraction * alpha
            )
            self.last_seen_monotonic = now
            self.identity_source = identity_source
            self.appearance_last_similarity = appearance_similarity
            if identity_source == "ocr":
                self.last_ocr_confirmed = now
                self.owner_track_confirmed = True
                if (
                    height_fraction
                    / max(self.target_nameplate_height, 1.0e-6)
                    >= self.PRIVATE_OWNER_CONFIRM_RATIO
                ):
                    self.owner_identity_latched = True
                    self.owner_private_confirmed_at = now
                self.appearance_only_started = 0.0
                self.avatar_candidate_started = 0.0
            elif self.appearance_only_started <= 0.0:
                self.appearance_only_started = now
            # A distant nameplate found by tiled search is usually too small
            # for the next full-frame OCR pass. Keep a high-resolution crop
            # centred on the match until repeated observations have brought
            # the owner substantially closer.
            magnified_height_limit = max(
                0.022,
                self.target_nameplate_height * 0.70,
            )
            if searching or height_fraction < magnified_height_limit:
                self.magnified_tracking_center = (centre_x, centre_y)
                self.magnified_tracking_until = now + max(
                    5.0,
                    self.lost_seconds * 4.0,
                )
            else:
                self.magnified_tracking_until = 0.0
            self.owner_facing_us = facing_us
            self.last_visual_motion = visual_motion
            self.visual_motion_sample += 1
            if self.phase == "forming_left":
                if facing_us:
                    self.formation_saw_front_face = True
                    self.formation_face_lost_at = 0.0
                elif (
                    self.formation_saw_front_face
                    and self.formation_face_lost_at <= 0.0
                ):
                    self.formation_face_lost_at = now
            if self.phase == "following_left":
                self.tracking_front_face_hits = (
                    self.tracking_front_face_hits + 1 if facing_us else 0
                )
            else:
                self.tracking_front_face_hits = 0
            smooth_x = self.smoothed_x
            smooth_height = self.smoothed_height
            if self.conversation_hold:
                report_phase = "conversation_hold"
            elif time.monotonic() < self.attention_hold_until:
                report_phase = "attention_hold"
            else:
                report_phase = self.phase
        self._report(
            ownerFollowDetected=True,
            ownerFollowState=report_phase,
            ownerFollowFacingUs=facing_us,
            ownerFollowOCRMatches=1,
            ownerFollowIdentitySource=identity_source,
            ownerFollowIdentityConfirmed=identity_source == "ocr",
            ownerFollowTrackConfirmed=self.owner_track_confirmed,
            ownerFollowIdentityLatched=self.owner_identity_latched,
            ownerFollowAppearanceReady=self.appearance_descriptor is not None,
            ownerFollowAppearanceTracked=identity_source != "ocr",
            ownerFollowAppearanceSimilarity=round(appearance_similarity, 4),
            ownerFollowAppearanceBox=(list(avatar_box) if avatar_box else None),
            ownerFollowNameplateText=selected.text,
            ownerFollowNameplateConfidence=round(float(selected.confidence), 4),
            ownerFollowNameplateBox=[x, y, width, height],
            ownerFollowScreenPosition=[round(smooth_x, 4), round(centre_y, 4)],
            ownerFollowNameplateHeight=round(smooth_height, 5),
            ownerFollowVisualMotion=round(visual_motion, 5),
            ownerFollowLastSeenAt=time.time(),
            ownerFollowLastError=None,
        )
        if identity_source == "ocr" and (
            was_stale or previous_identity_source != "ocr"
        ):
            self.announce_status(
                "owner_found",
                f"✅ オーナー発見：{selected.text}",
                min_interval=4.0,
            )

    def _select_avatar_detection(
        self,
        frame: NDArray[np.uint8],
        detections: Sequence[AvatarDetection],
    ) -> tuple[AvatarDetection, str, float] | None:
        """Select a person before attempting name recognition."""

        if not detections:
            return None
        with self.lock:
            reference = self.appearance_descriptor
            assume_single = self.assume_single_avatar
        if len(detections) == 1 and assume_single:
            detection = detections[0]
            descriptor = self._appearance_feature(frame, detection.bbox)
            similarity = self._appearance_similarity(reference, descriptor)
            return detection, "avatar_detector_single", similarity
        if reference is None:
            return None
        candidates: list[tuple[float, AvatarDetection]] = []
        for detection in detections:
            descriptor = self._appearance_feature(frame, detection.bbox)
            similarity = self._appearance_similarity(reference, descriptor)
            if similarity >= 0.68:
                candidates.append((similarity, detection))
        if not candidates:
            return None
        similarity, detection = max(candidates, key=lambda item: item[0])
        return detection, "avatar_detector_appearance", similarity

    def _best_search_detection(
        self,
        frame: NDArray[np.uint8],
        detections: Sequence[AvatarDetection],
    ) -> tuple[AvatarDetection, float] | None:
        """Rank a scanned person using clothing colour and detector confidence."""

        if not detections:
            return None
        with self.lock:
            reference = self.appearance_descriptor
        ranked: list[tuple[float, float, AvatarDetection]] = []
        for detection in detections:
            descriptor = self._appearance_feature(frame, detection.bbox)
            similarity = self._appearance_similarity(reference, descriptor)
            if reference is None:
                quality = float(detection.score)
            else:
                quality = similarity * 0.72 + float(detection.score) * 0.28
            ranked.append((quality, similarity, detection))
        _quality, similarity, detection = max(ranked, key=lambda item: item[0])
        return detection, similarity

    def _remember_search_candidate(
        self,
        now: float,
        frame: NDArray[np.uint8],
        detections: Sequence[AvatarDetection],
    ) -> None:
        """Remember the best direction in the current full-circle scan."""

        best = self._best_search_detection(frame, detections)
        if best is None:
            return
        detection, similarity = best
        with self.lock:
            if self.search_started <= 0.0:
                return
            elapsed = max(0.0, now - self.search_started)
            absolute_step = int(elapsed // self.SEARCH_STEP_SECONDS)
            cycle = absolute_step // self.SEARCH_STEPS_PER_SWEEP
            step = absolute_step % self.SEARCH_STEPS_PER_SWEEP + 1
            reference_ready = self.appearance_descriptor is not None
            quality = (
                similarity * 0.72 + float(detection.score) * 0.28
                if reference_ready
                else float(detection.score)
            )
            previous = self.search_best_candidate
            if (
                previous is None
                or int(previous.get("cycle", -1)) != cycle
                or quality > float(previous.get("quality", -1.0))
            ):
                self.search_best_candidate = {
                    "cycle": cycle,
                    "step": step,
                    "quality": quality,
                    "similarity": similarity,
                    "detectorScore": float(detection.score),
                    "bbox": list(detection.bbox),
                }
                remembered = True
            else:
                remembered = False
        if remembered:
            self._report(
                ownerFollowSearchCandidateRemembered=True,
                ownerFollowSearchCandidateStep=step,
                ownerFollowSearchCandidateQuality=round(quality, 4),
                ownerFollowSearchCandidateSimilarity=round(similarity, 4),
                ownerFollowSearchCandidateBox=list(detection.bbox),
            )
            self.announce_status(
                f"search_candidate_{cycle}_{step}",
                (
                    f"👤 人物候補を検出：スキャン {step}/"
                    f"{self.SEARCH_STEPS_PER_SWEEP}、外観一致 {similarity:.0%}"
                ),
                min_interval=0.0,
            )

    def _search_return_ready(self, now: float) -> bool:
        with self.lock:
            started = self.search_return_started
            steps = self.search_return_steps
        return bool(
            started > 0.0
            and now - started >= steps * self.SEARCH_STEP_SECONDS
        )

    def _record_avatar_detection(
        self,
        now: float,
        frame: NDArray[np.uint8],
        detection: AvatarDetection,
        identity_source: str,
        appearance_similarity: float,
    ) -> None:
        """Use a person candidate for navigation before OCR confirmation."""

        x, y, width, height = detection.bbox
        image_height, image_width = frame.shape[:2]
        centre_x = (x + width * 0.5) / max(image_width, 1)
        centre_y = (y + height * 0.5) / max(image_height, 1)
        predicted_name_y = max(0.0, y - height * 0.035) / max(image_height, 1)
        descriptor = self._appearance_feature(frame, detection.bbox)
        with self.lock:
            if self.appearance_descriptor is None and descriptor is not None:
                # In explicitly enabled single-avatar mode this provisional
                # descriptor lets MIL keep following until OCR confirms.
                self.appearance_descriptor = descriptor
            previous_last_seen = self.last_seen_monotonic
            stale = (
                previous_last_seen <= 0.0
                or now - previous_last_seen > self.lost_seconds
            )
            alpha = 1.0 if stale else 0.55
            self.smoothed_x = self.smoothed_x * (1.0 - alpha) + centre_x * alpha
            self.last_seen_monotonic = now
            previous_source = self.identity_source
            self.identity_source = identity_source
            self.avatar_candidate_height = height / max(image_height, 1)
            if (
                self.avatar_candidate_started <= 0.0
                or previous_source is None
                or not previous_source.startswith("avatar_detector")
                or stale
            ):
                self.avatar_candidate_started = now
            self.magnified_tracking_center = (centre_x, predicted_name_y)
            self.magnified_tracking_until = now + max(
                5.0,
                self.lost_seconds * 4.0,
            )
            smooth_x = self.smoothed_x
            report_phase = self.phase
            if self.conversation_hold:
                report_phase = "conversation_hold"
            elif time.monotonic() < self.attention_hold_until:
                report_phase = "attention_hold"

        name_height = max(6.0, height * 0.035)
        name_width = max(30.0, width * 0.36)
        name_centre_x = x + width * 0.5
        synthetic_nameplate = OCRWord(
            text="",
            polygon=(
                (name_centre_x - name_width * 0.5, max(0.0, y - name_height * 1.5)),
                (name_centre_x + name_width * 0.5, max(0.0, y - name_height * 1.5)),
                (name_centre_x + name_width * 0.5, max(0.0, y - name_height * 0.5)),
                (name_centre_x - name_width * 0.5, max(0.0, y - name_height * 0.5)),
            ),
            confidence=float(detection.score),
        )
        self._initialise_appearance_tracker(frame, detection.bbox, synthetic_nameplate)
        self._report(
            ownerFollowDetected=True,
            ownerFollowState=(
                "avatar_candidate" if report_phase == "searching" else report_phase
            ),
            ownerFollowIdentitySource=identity_source,
            ownerFollowIdentityConfirmed=False,
            ownerFollowAvatarCandidateSelected=True,
            ownerFollowAvatarScanEmptyDirection=False,
            ownerFollowAvatarCandidateBox=list(detection.bbox),
            ownerFollowAvatarCandidateConfidence=round(float(detection.score), 4),
            ownerFollowAvatarCandidateHeight=round(
                height / max(image_height, 1), 4
            ),
            ownerFollowAppearanceReady=self.appearance_descriptor is not None,
            ownerFollowAppearanceSimilarity=round(appearance_similarity, 4),
            ownerFollowScreenPosition=[round(smooth_x, 4), round(centre_y, 4)],
            ownerFollowLastSeenAt=time.time(),
            ownerFollowLastError=None,
        )
        self.announce_status(
            "avatar_candidate",
            (
                f"👤 人物を認識：外観一致 {appearance_similarity:.0%}。"
                "接近してネームプレートを確認します"
            ),
            min_interval=4.0,
        )

    def _skip_ocr_for_empty_direction(self, now: float) -> bool:
        """Keep occasional OCR fallback steps for detector false negatives."""

        with self.lock:
            if self.phase != "searching" or self.search_started <= 0.0:
                return False
            if self.search_return_started > 0.0:
                return False
            absolute_step = int(
                max(0.0, now - self.search_started) // self.SEARCH_STEP_SECONDS
            )
        scan_step = absolute_step % self.SEARCH_STEPS_PER_SWEEP + 1
        return scan_step % 4 != 0

    def _advance_empty_search_direction(self, now: float) -> None:
        """Mark an empty view without skipping an angle not physically scanned."""

        self._report(ownerFollowAvatarScanEmptyDirection=True)

    def _next_search_tile(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], tuple[int, int, int, int], int]:
        """Return one overlapping quadrant for high-magnification search OCR.

        RapidOCR scales the short side of its input to ``ocr_min_side``.  A
        roughly 58% by 58% crop therefore gives a 1280x720 capture about 2.6x
        OCR magnification while remaining under its 2000-pixel long-side
        limit.  The overlap keeps a nameplate near a quadrant boundary intact.
        """

        image_height, image_width = frame.shape[:2]
        tile_width = min(image_width, max(1, int(round(image_width * 0.58))))
        tile_height = min(image_height, max(1, int(round(image_height * 0.58))))
        right_x = max(0, image_width - tile_width)
        bottom_y = max(0, image_height - tile_height)
        positions = ((0, 0), (right_x, 0), (0, bottom_y), (right_x, bottom_y))
        with self.lock:
            index = self.search_tile_index % len(positions)
            self.search_tile_index = (index + 1) % len(positions)
        x, y = positions[index]
        crop = frame[y : y + tile_height, x : x + tile_width]
        return crop, (x, y, tile_width, tile_height), index + 1

    @staticmethod
    def _tracking_tile(
        frame: NDArray[np.uint8],
        centre: tuple[float, float],
    ) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
        """Return an overlapping-size crop centred on a reacquired owner."""

        image_height, image_width = frame.shape[:2]
        tile_width = min(image_width, max(1, int(round(image_width * 0.58))))
        tile_height = min(image_height, max(1, int(round(image_height * 0.58))))
        centre_x = float(max(0.0, min(1.0, centre[0]))) * image_width
        centre_y = float(max(0.0, min(1.0, centre[1]))) * image_height
        x = int(round(centre_x - tile_width * 0.5))
        y = int(round(centre_y - tile_height * 0.5))
        x = max(0, min(image_width - tile_width, x))
        y = max(0, min(image_height - tile_height, y))
        crop = frame[y : y + tile_height, x : x + tile_width]
        return crop, (x, y, tile_width, tile_height)

    @staticmethod
    def _avatar_box_from_nameplate(
        frame: NDArray[np.uint8],
        nameplate: OCRWord,
    ) -> tuple[int, int, int, int] | None:
        """Estimate the avatar body region immediately below a nameplate."""

        x, y, width, height = nameplate.bbox
        image_height, image_width = frame.shape[:2]
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            return None
        centre_x = x + width * 0.5
        body_width = int(round(max(width * 1.35, height * 10.0, 24.0)))
        body_height = int(round(max(height * 24.0, body_width * 1.9, 48.0)))
        body_width = min(body_width, max(1, int(round(image_width * 0.32))))
        body_height = min(body_height, max(1, int(round(image_height * 0.58))))
        left = int(round(centre_x - body_width * 0.5))
        top = int(round(y + height * 1.45))
        left = max(0, min(image_width - body_width, left))
        top = max(0, min(image_height - body_height, top))
        if body_width < 16 or body_height < 32:
            return None
        return left, top, body_width, body_height

    @staticmethod
    def _appearance_feature(
        frame: NDArray[np.uint8],
        box: tuple[int, int, int, int] | None,
    ) -> NDArray[np.float32] | None:
        """Describe clothing by spatial hue/saturation/value distributions."""

        if box is None:
            return None
        x, y, width, height = box
        image_height, image_width = frame.shape[:2]
        left = max(0, min(image_width, int(x)))
        top = max(0, min(image_height, int(y)))
        right = max(left, min(image_width, int(x + width)))
        bottom = max(top, min(image_height, int(y + height)))
        if right - left < 12 or bottom - top < 24:
            return None
        roi = frame[top:bottom, left:right]
        sample = cv2.resize(roi, (48, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
        base_mask = np.zeros((96, 48), dtype=np.uint8)
        cv2.ellipse(base_mask, (24, 48), (20, 46), 0, 0, 360, 255, -1)
        features: list[NDArray[np.float32]] = []
        for band_top, band_bottom in ((0, 38), (28, 68), (58, 96)):
            mask = base_mask.copy()
            mask[:band_top, :] = 0
            mask[band_bottom:, :] = 0
            hue_sat = cv2.calcHist(
                [hsv], [0, 1], mask, [12, 6], [0, 180, 0, 256]
            ).astype(np.float32)
            sat_value = cv2.calcHist(
                [hsv], [1, 2], mask, [6, 8], [0, 256, 0, 256]
            ).astype(np.float32)
            for histogram in (hue_sat, sat_value):
                total = float(histogram.sum())
                if total > 0.0:
                    histogram /= total
                features.append(histogram.reshape(-1))
        descriptor = np.concatenate(features).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1.0e-8:
            return None
        return descriptor / norm

    @staticmethod
    def _appearance_similarity(
        first: NDArray[np.float32] | None,
        second: NDArray[np.float32] | None,
    ) -> float:
        if first is None or second is None or first.shape != second.shape:
            return 0.0
        return float(max(0.0, min(1.0, np.dot(first, second))))

    def _learn_owner_appearance(
        self,
        frame: NDArray[np.uint8],
        avatar_box: tuple[int, int, int, int] | None,
    ) -> float:
        descriptor = self._appearance_feature(frame, avatar_box)
        if descriptor is None:
            return 0.0
        with self.lock:
            previous = self.appearance_descriptor
            similarity = self._appearance_similarity(previous, descriptor)
            if previous is None:
                self.appearance_descriptor = descriptor
                similarity = 1.0
            else:
                blended = previous * 0.78 + descriptor * 0.22
                norm = float(np.linalg.norm(blended))
                self.appearance_descriptor = (
                    blended / norm if norm > 1.0e-8 else descriptor
                ).astype(np.float32)
        return similarity

    def _select_appearance_word(
        self,
        frame: NDArray[np.uint8],
        words: Sequence[OCRWord],
    ) -> tuple[OCRWord, float] | None:
        """Use garbled nameplates as proposals and clothing as identity."""

        with self.lock:
            reference = self.appearance_descriptor
        if reference is None:
            return None
        candidates: list[tuple[float, OCRWord]] = []
        for word in words:
            if float(word.confidence) < 0.25:
                continue
            avatar_box = self._avatar_box_from_nameplate(frame, word)
            descriptor = self._appearance_feature(frame, avatar_box)
            similarity = self._appearance_similarity(reference, descriptor)
            if similarity >= 0.78:
                candidates.append((similarity, word))
        if not candidates:
            return None
        similarity, word = max(candidates, key=lambda item: item[0])
        return word, similarity

    def _initialise_appearance_tracker(
        self,
        frame: NDArray[np.uint8],
        avatar_box: tuple[int, int, int, int],
        nameplate: OCRWord,
    ) -> None:
        """Start a local MIL tracker after OCR or appearance confirmation."""

        image_height, image_width = frame.shape[:2]
        scale = min(1.0, 640.0 / max(image_width, 1))
        tracker_frame = (
            frame
            if scale >= 0.999
            else cv2.resize(
                frame,
                (
                    max(1, int(round(image_width * scale))),
                    max(1, int(round(image_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        )
        x, y, width, height = avatar_box
        scaled_box = (
            int(round(x * scale)),
            int(round(y * scale)),
            max(12, int(round(width * scale))),
            max(24, int(round(height * scale))),
        )
        tracker_height, tracker_width = tracker_frame.shape[:2]
        scaled_x = max(0, min(tracker_width - 1, scaled_box[0]))
        scaled_y = max(0, min(tracker_height - 1, scaled_box[1]))
        scaled_width = max(1, min(tracker_width - scaled_x, scaled_box[2]))
        scaled_height = max(1, min(tracker_height - scaled_y, scaled_box[3]))
        if scaled_width < 12 or scaled_height < 24:
            return
        scaled_box = (scaled_x, scaled_y, scaled_width, scaled_height)
        try:
            tracker = cv2.TrackerMIL_create()
            tracker.init(tracker_frame, scaled_box)
        except Exception:
            return
        name_x, name_y, name_width, name_height = nameplate.bbox
        name_centre = (name_x + name_width * 0.5, name_y + name_height * 0.5)
        body_centre = (x + width * 0.5, y + height * 0.5)
        with self.lock:
            self.appearance_tracker = tracker
            self.appearance_tracker_scale = scale
            self.appearance_nameplate_offset = (
                name_centre[0] - body_centre[0],
                name_centre[1] - body_centre[1],
            )
            self.appearance_failures = 0

    def _track_appearance_once(self, now: float) -> None:
        """Track a confirmed avatar between readable nameplate frames."""

        with self.lock:
            tracker = self.appearance_tracker
            scale = self.appearance_tracker_scale
            reference = self.appearance_descriptor
            nameplate_offset = self.appearance_nameplate_offset
        if tracker is None or reference is None:
            return
        frame = self.frame_provider(max_age_seconds=0.75)
        if frame is None:
            return
        image_height, image_width = frame.shape[:2]
        tracker_frame = (
            frame
            if scale >= 0.999
            else cv2.resize(
                frame,
                (
                    max(1, int(round(image_width * scale))),
                    max(1, int(round(image_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        )
        try:
            tracked, scaled_box = tracker.update(tracker_frame)
        except Exception:
            tracked = False
            scaled_box = (0.0, 0.0, 0.0, 0.0)
        if not tracked:
            self._appearance_tracking_failed(0.0)
            return
        x, y, width, height = (
            float(value) / max(scale, 1.0e-6) for value in scaled_box
        )
        box = (
            max(0, int(round(x))),
            max(0, int(round(y))),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )
        descriptor = self._appearance_feature(frame, box)
        similarity = self._appearance_similarity(reference, descriptor)
        if similarity < 0.68:
            self._appearance_tracking_failed(similarity)
            return

        body_centre_x = box[0] + box[2] * 0.5
        body_centre_y = box[1] + box[3] * 0.5
        predicted_name_x = body_centre_x + nameplate_offset[0]
        predicted_name_y = body_centre_y + nameplate_offset[1]
        centre_x = predicted_name_x / max(image_width, 1)
        centre_y = predicted_name_y / max(image_height, 1)
        with self.lock:
            alpha = 0.58
            self.smoothed_x = self.smoothed_x * (1.0 - alpha) + centre_x * alpha
            self.last_seen_monotonic = now
            self.identity_source = "appearance_tracker"
            self.appearance_last_similarity = similarity
            self.appearance_failures = 0
            if self.appearance_only_started <= 0.0:
                self.appearance_only_started = now
            self.magnified_tracking_center = (centre_x, centre_y)
            self.magnified_tracking_until = now + max(
                5.0,
                self.lost_seconds * 4.0,
            )
            smooth_x = self.smoothed_x
            if self.conversation_hold:
                report_phase = "conversation_hold"
            elif time.monotonic() < self.attention_hold_until:
                report_phase = "attention_hold"
            else:
                report_phase = self.phase
        self._report(
            ownerFollowDetected=True,
            ownerFollowState=report_phase,
            ownerFollowIdentitySource="appearance_tracker",
            ownerFollowIdentityConfirmed=False,
            ownerFollowAppearanceReady=True,
            ownerFollowAppearanceTracked=True,
            ownerFollowAppearanceSimilarity=round(similarity, 4),
            ownerFollowAppearanceBox=list(box),
            ownerFollowScreenPosition=[round(smooth_x, 4), round(centre_y, 4)],
            ownerFollowLastSeenAt=time.time(),
            ownerFollowLastError=None,
        )

    def _appearance_tracking_failed(self, similarity: float) -> None:
        with self.lock:
            self.appearance_failures += 1
            failures = self.appearance_failures
            self.appearance_last_similarity = similarity
            if failures >= 3:
                self.appearance_tracker = None
        self._report(
            ownerFollowAppearanceTracked=False,
            ownerFollowAppearanceSimilarity=round(similarity, 4),
            ownerFollowAppearanceFailures=failures,
        )

    def _select_owner_word(self, words: Sequence[OCRWord]) -> OCRWord | None:
        with self.lock:
            owner = self._normalise_name(self.owner_account)
        candidates: list[tuple[float, OCRWord]] = []
        for word in words:
            text = self._normalise_name(word.text)
            if not text or float(word.confidence) < 0.30:
                continue
            if owner == text:
                score = float(word.confidence) + 1.0
            elif (
                min(len(owner), len(text)) >= 3
                and SequenceMatcher(None, owner, text).ratio() >= 0.62
            ):
                score = float(word.confidence) + 0.5 * SequenceMatcher(
                    None, owner, text
                ).ratio()
            else:
                continue
            candidates.append((score, word))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _owner_face_is_frontal(
        self,
        frame: NDArray[np.uint8],
        nameplate: OCRWord,
    ) -> bool:
        """Detect a frontal face only in the region below the owner nameplate."""

        x, y, width, height = nameplate.bbox
        image_height, image_width = frame.shape[:2]
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            return False
        left = max(0, int(round(x - 1.5 * width)))
        right = min(image_width, int(round(x + 2.5 * width)))
        top = max(0, int(round(y + height)))
        bottom = min(image_height, int(round(y + height + 12.0 * height)))
        if right - left < 24 or bottom - top < 24:
            return False
        roi = frame[top:bottom, left:right]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        minimum = max(24, int(round(height * 0.9)))
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(minimum, minimum),
        )
        return len(faces) > 0

    def _camera_motion_fraction(self, frame: NDArray[np.uint8]) -> float:
        """Estimate ego-view motion from the median flow of background features."""

        height, width = frame.shape[:2]
        scale = min(1.0, 240.0 / max(width, 1))
        small = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous = self.previous_motion_gray
        self.previous_motion_gray = gray
        if previous is None or previous.shape != gray.shape:
            return 0.0
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=120,
            qualityLevel=0.02,
            minDistance=7,
            blockSize=7,
        )
        if points is None or len(points) < 8:
            return 0.0
        moved, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if moved is None or status is None:
            return 0.0
        valid = status.reshape(-1).astype(bool)
        if int(np.count_nonzero(valid)) < 8:
            return 0.0
        displacement = moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        lengths = np.linalg.norm(displacement, axis=1)
        # The median rejects an animated avatar or video occupying a minority
        # of the screen while retaining broad camera translation/rotation.
        return float(np.median(lengths) / max(gray.shape[1], 1))

    def _apply_control(self, now: float) -> None:
        with self.lock:
            last_seen = self.last_seen_monotonic
            x = self.smoothed_x
            height = self.smoothed_height
            phase = self.phase
            identity_source = self.identity_source
            appearance_only_started = self.appearance_only_started
            last_ocr_confirmed = self.last_ocr_confirmed
            avatar_candidate_height = self.avatar_candidate_height
            avatar_candidate_started = self.avatar_candidate_started
            track_confirmed = self.owner_track_confirmed
            conversation_hold = self.conversation_hold
            attention_hold = now < self.attention_hold_until
            speech_hold = self.speech_hold or now < self.speech_hold_until
            listener_hold = self.listener_hold or now < self.listener_hold_until
        if attention_hold or speech_hold or listener_hold:
            with self.lock:
                self.translation_no_motion_seconds = 0.0
            self._stop_motion()
            hold_state = (
                "speech_hold"
                if speech_hold
                else ("listener_hold" if listener_hold else "attention_hold")
            )
            self._report(
                ownerFollowState=hold_state,
                ownerFollowAttentionHold=attention_hold,
                ownerFollowConversationHold=conversation_hold,
                ownerFollowSpeechHold=speech_hold,
                ownerFollowListenerHold=listener_hold,
                ownerFollowAxes=[0.0, 0.0, 0.0],
            )
            return
        self._report(
            ownerFollowAttentionHold=False,
            ownerFollowConversationHold=conversation_hold,
            ownerFollowSpeechHold=False,
            ownerFollowListenerHold=False,
            ownerFollowConversationSearchSuppressed=False,
        )
        if phase == "escaping":
            self._apply_escape(now)
            return
        seen_age = now - last_seen if last_seen > 0.0 else float("inf")
        if seen_age > self.lost_seconds:
            if conversation_hold:
                self._apply_conversation_reacquire(now, x, seen_age)
                return
            if (
                track_confirmed
                and seen_age <= self.OWNER_REACQUIRE_GRACE_SECONDS
            ):
                self._apply_nearby_reacquire(now, x, seen_age)
                return
            self._search_for_owner(now, x)
            return

        size_ratio = height / max(self.target_nameplate_height, 1.0e-6)
        if phase == "searching":
            with self.lock:
                search_duration = max(0.0, now - self.search_started)
                resumed = self.phase_before_search
                if resumed == "forming_left" and self.formation_started > 0.0:
                    self.formation_started += search_duration
                    if self.formation_face_lost_at > 0.0:
                        self.formation_face_lost_at += search_duration
                self.phase = resumed
                self.search_started = 0.0
                self.search_best_candidate = None
                self.search_return_started = 0.0
                self.search_return_steps = 0
                self.search_return_direction = 0.0
                self.search_last_chat_step = 0
                phase = self.phase

        if (
            identity_source is not None
            and identity_source.startswith("avatar_detector")
        ):
            self._apply_avatar_candidate_approach(
                now,
                x,
                avatar_candidate_height,
                avatar_candidate_started,
            )
            return

        if (
            identity_source is not None
            and identity_source.startswith("appearance")
            and not track_confirmed
            and now - last_ocr_confirmed > max(0.9, 1.4 / self.tracking_fps)
        ):
            self._apply_appearance_approach(
                now,
                x,
                appearance_only_started,
            )
            return

        if phase in {"acquiring", "approaching"}:
            if size_ratio >= 0.90:
                self._begin_left_formation(now)
                phase = "forming_left"
            else:
                phase = "approaching"
                with self.lock:
                    self.phase = phase

        if phase == "forming_left":
            if self._left_formation_complete(now):
                with self.lock:
                    self.phase = "following_left"
                    self.tracking_front_face_hits = 0
                phase = "following_left"
            else:
                self._apply_left_formation(now, x, size_ratio)
                return

        with self.lock:
            front_face_hits = self.tracking_front_face_hits
        if phase == "following_left" and front_face_hits >= 2:
            # The owner has turned to face this avatar (or the formation has
            # drifted to the front).  Re-form on the left rather than staying
            # in the face-to-face position.
            self._begin_left_formation(now)
            self._apply_left_formation(now, x, size_ratio)
            return

        if phase == "approaching":
            horizontal_error = x - 0.50
            turn = self._clip(horizontal_error * 1.55, -0.34, 0.34)
            # Do not walk and turn at the same time. Digital yaw buttons are
            # much stronger than LookHorizontal axes; centre the owner first,
            # then advance in a straight line on later observations.
            if abs(horizontal_error) > 0.055:
                forward = 0.0
            else:
                turn = 0.0
                forward = self._distance_forward(size_ratio)
            strafe = 0.0
        else:
            # After formation the owner should remain to the screen's right.
            # A camera translation shifts the observed owner in the opposite
            # direction, so the strafe axis has the same sign as this error.
            horizontal_error = x - self.target_screen_x
            turn = self._clip(horizontal_error * 1.55, -0.34, 0.34)
            strafe = self._clip(horizontal_error * 0.90, -0.28, 0.28)
            if abs(horizontal_error) < 0.025:
                turn = 0.0
                strafe = 0.0
            forward = self._distance_forward(size_ratio)
            if abs(horizontal_error) > 0.30 and forward > 0.0:
                forward *= 0.25

        if self._dispatch_control(now, phase, forward, strafe, turn):
            return
        self._report(
            ownerFollowState=phase,
            ownerFollowAxes=[round(forward, 3), round(strafe, 3), round(turn, 3)],
            ownerFollowDistanceRatio=round(size_ratio, 3),
            ownerFollowSearchPhase=None,
            ownerFollowSearchSeconds=0.0,
        )

    def _apply_conversation_reacquire(
        self,
        now: float,
        last_x: float,
        seen_age: float,
    ) -> None:
        """Avoid full search rotation while a conversation remains latched."""

        horizontal_error = last_x - 0.50
        if seen_age < max(1.0, self.lost_seconds + 0.5):
            turn = 0.0
            state = "conversation_reacquire_observe"
        elif abs(horizontal_error) >= 0.08:
            turn = self._clip(horizontal_error * 0.55, -0.10, 0.10)
            state = "conversation_reacquire_small_turn"
        else:
            turn = 0.0
            state = "conversation_reacquire_observe"
        self._send_axes(0.0, 0.0, turn, turn_mode="pulse")
        self._report(
            ownerFollowDetected=False,
            ownerFollowState=state,
            ownerFollowAxes=[0.0, 0.0, round(turn, 3)],
            ownerFollowConversationHold=True,
            ownerFollowConversationSearchSuppressed=True,
            ownerFollowReacquireSeconds=round(seen_age, 2),
            ownerFollowSearchPhase=None,
        )

    def _apply_nearby_reacquire(
        self,
        now: float,
        last_x: float,
        seen_age: float,
    ) -> None:
        """Try a small last-known-direction correction before a full search."""

        horizontal_error = last_x - 0.50
        if seen_age < max(0.8, self.lost_seconds + 0.35):
            turn = 0.0
            state = "reacquire_observe"
        elif abs(horizontal_error) >= 0.06:
            turn = self._clip(horizontal_error * 0.75, -0.16, 0.16)
            state = "reacquire_small_turn"
        else:
            turn = 0.0
            state = "reacquire_observe"
        self._send_axes(
            0.0,
            0.0,
            turn,
            turn_mode=("pulse" if abs(turn) > 0.0 else "pulse"),
        )
        self._report(
            ownerFollowDetected=False,
            ownerFollowState=state,
            ownerFollowAxes=[0.0, 0.0, round(turn, 3)],
            ownerFollowReacquireGrace=True,
            ownerFollowReacquireSeconds=round(seen_age, 2),
            ownerFollowReacquireLimitSeconds=self.OWNER_REACQUIRE_GRACE_SECONDS,
            ownerFollowSearchPhase=None,
        )

    def _apply_avatar_candidate_approach(
        self,
        now: float,
        x: float,
        height_fraction: float,
        started: float,
    ) -> None:
        """Centre a detected person and alternate approach with OCR pauses."""

        if started <= 0.0:
            with self.lock:
                self.avatar_candidate_started = now
            started = now
        cycle_time = max(0.0, now - started) % 4.0
        horizontal_error = x - 0.50
        if abs(horizontal_error) > 0.055:
            state = "centering_avatar"
            forward = 0.0
            turn = self._clip(horizontal_error * 1.55, -0.32, 0.32)
        elif cycle_time < 2.5 and height_fraction < 0.68:
            state = "approaching_avatar"
            turn = 0.0
            forward = self._clip(
                (0.62 - height_fraction) * 0.72,
                0.16,
                0.34,
            )
        else:
            state = "verifying_avatar"
            forward = 0.0
            turn = 0.0
        if abs(forward) > 0.0 or abs(turn) > 0.0:
            if self._dispatch_control(now, state, forward, 0.0, turn):
                return
        else:
            self._stop_motion()
        self._report(
            ownerFollowState=state,
            ownerFollowAxes=[round(forward, 3), 0.0, round(turn, 3)],
            ownerFollowIdentityConfirmed=False,
            ownerFollowAvatarCandidateHeight=round(height_fraction, 4),
            ownerFollowAvatarApproachCycle=round(cycle_time, 2),
            ownerFollowSearchPhase=None,
            ownerFollowSearchSeconds=0.0,
        )

    def _apply_appearance_approach(
        self,
        now: float,
        x: float,
        started: float,
    ) -> None:
        """Approach a visual candidate briefly, then require OCR identity."""

        if started <= 0.0:
            with self.lock:
                self.appearance_only_started = now
            started = now
        elapsed = max(0.0, now - started)
        horizontal_error = x - 0.50
        if elapsed < 3.0:
            state = "approaching_appearance"
            turn = self._clip(horizontal_error * 1.55, -0.30, 0.30)
            if abs(horizontal_error) > 0.055:
                forward = 0.0
            else:
                turn = 0.0
                forward = 0.24
            if self._dispatch_control(now, state, forward, 0.0, turn):
                return
        elif elapsed < 6.0:
            state = "verifying_owner"
            forward = 0.0
            turn = 0.0
            self._stop_motion()
        else:
            # Never continue walking solely on an appearance hypothesis. Keep
            # the learned clothing descriptor, discard the drifting tracker,
            # and resume a tiled search for an OCR/appearance proposal.
            with self.lock:
                self.appearance_tracker = None
                self.appearance_only_started = 0.0
                self.identity_source = None
                self.last_seen_monotonic = 0.0
                self.phase = "searching"
                self.search_started = 0.0
            self._search_for_owner(now, x)
            self._report(
                ownerFollowIdentitySource=None,
                ownerFollowIdentityConfirmed=False,
                ownerFollowAppearanceTracked=False,
                ownerFollowVerificationExpired=True,
            )
            return
        self._report(
            ownerFollowState=state,
            ownerFollowAxes=[round(forward, 3), 0.0, round(turn, 3)],
            ownerFollowIdentityConfirmed=False,
            ownerFollowAppearanceApproachSeconds=round(elapsed, 2),
            ownerFollowVerificationExpired=False,
            ownerFollowSearchPhase=None,
            ownerFollowSearchSeconds=0.0,
        )

    def _distance_forward(self, size_ratio: float) -> float:
        if size_ratio < 0.90:
            return self._clip((1.02 - size_ratio) * 0.90, 0.12, 0.55)
        if size_ratio > 1.18:
            return -self._clip((size_ratio - 1.08) * 0.55, 0.12, 0.36)
        return 0.0

    def _begin_left_formation(self, now: float) -> None:
        with self.lock:
            self.phase = "forming_left"
            self.formation_started = now
            self.formation_saw_front_face = self.owner_facing_us
            self.formation_face_lost_at = 0.0
            self.tracking_front_face_hits = 0
        self._report(
            ownerFollowState="forming_left",
            ownerFollowFormationProgress=0.0,
        )

    def _left_formation_complete(self, now: float) -> bool:
        with self.lock:
            elapsed = max(0.0, now - self.formation_started)
            saw_face = self.formation_saw_front_face
            face_lost_at = self.formation_face_lost_at
        face_completed = (
            saw_face
            and face_lost_at > 0.0
            and elapsed >= 2.0
            and now - face_lost_at >= 1.2
        )
        return face_completed or elapsed >= self.formation_seconds

    def _apply_left_formation(
        self,
        now: float,
        x: float,
        size_ratio: float,
    ) -> None:
        with self.lock:
            elapsed = max(0.0, now - self.formation_started)
        centre_error = x - 0.50
        cycle_position = elapsed % 0.90
        if cycle_position < 0.55:
            # Separate translation and yaw.  Besides making the path less
            # likely to spiral into a wall, this gives collision detection a
            # view interval in which camera motion should be measurable.
            strafe = self.orbit_speed
            turn = 0.0
            forward = self._distance_forward(size_ratio)
            if size_ratio < 0.72:
                strafe *= 0.45
        else:
            strafe = 0.0
            forward = 0.0
            turn = self._clip(
                centre_error * 1.80 - self.orbit_speed * 0.65,
                -0.42,
                0.34,
            )
        if self._dispatch_control(now, "forming_left", forward, strafe, turn):
            return
        self._report(
            ownerFollowState="forming_left",
            ownerFollowAxes=[round(forward, 3), round(strafe, 3), round(turn, 3)],
            ownerFollowDistanceRatio=round(size_ratio, 3),
            ownerFollowFormationProgress=round(
                min(1.0, elapsed / max(self.formation_seconds, 1.0e-6)), 3
            ),
            ownerFollowSearchSeconds=0.0,
        )

    def _dispatch_control(
        self,
        now: float,
        phase: str,
        forward: float,
        strafe: float,
        turn: float,
    ) -> bool:
        """Send axes, or enter collision recovery when translation makes no progress."""

        translation = max(abs(forward), abs(strafe))
        with self.lock:
            dt = max(0.0, min(0.25, now - self.last_control_at))
            self.last_control_at = now
            has_new_motion = self.visual_motion_sample != self.consumed_visual_motion_sample
            if has_new_motion:
                self.consumed_visual_motion_sample = self.visual_motion_sample
                if self.last_visual_motion >= 0.0025:
                    self.translation_no_motion_seconds = 0.0
            if translation >= 0.16 and abs(turn) <= 0.07:
                self.translation_no_motion_seconds += dt
            else:
                self.translation_no_motion_seconds = max(
                    0.0,
                    self.translation_no_motion_seconds - 0.25 * dt,
                )
            stuck_seconds = self.translation_no_motion_seconds
        if stuck_seconds >= 2.4:
            self._begin_escape(now, phase, forward, strafe)
            self._apply_escape(now)
            return True
        self._send_axes(forward, strafe, turn, turn_mode="pulse")
        return False

    def _begin_escape(
        self,
        now: float,
        phase: str,
        forward: float,
        strafe: float,
    ) -> None:
        with self.lock:
            self.escape_attempt += 1
            self.escape_started = now
            self.escape_source_phase = phase
            self.escape_blocked_axes = (forward, strafe)
            self.escape_turn_direction = -1.0 if self.escape_attempt % 2 else 1.0
            self.translation_no_motion_seconds = 0.0
            self.phase = "escaping"
        self._report(
            ownerFollowState="escaping",
            ownerFollowCollisionRecovery=True,
            ownerFollowCollisionAttempts=self.escape_attempt,
        )

    def _apply_escape(self, now: float) -> None:
        with self.lock:
            elapsed = max(0.0, now - self.escape_started)
            blocked_forward, blocked_strafe = self.escape_blocked_axes
            turn_direction = self.escape_turn_direction
            source_phase = self.escape_source_phase
        if elapsed < 0.85:
            forward = -0.30 if blocked_forward > 0.08 else (
                0.30 if blocked_forward < -0.08 else 0.0
            )
            strafe = -0.28 if blocked_strafe > 0.08 else (
                0.28 if blocked_strafe < -0.08 else 0.0
            )
            turn = 0.0
        elif elapsed < 1.85:
            forward = 0.0
            strafe = 0.0
            turn = turn_direction * 0.32
        else:
            self._stop_motion(force=True)
            with self.lock:
                self.phase = source_phase
                # Collision recovery changes the camera abruptly. Preserve a
                # short local-reacquisition window instead of jumping straight
                # into an unrelated full-circle search.
                self.last_seen_monotonic = now - self.lost_seconds
                self.escape_started = 0.0
            self._report(
                ownerFollowState="reacquire_observe",
                ownerFollowCollisionRecovery=False,
                ownerFollowReacquireGrace=True,
                ownerFollowAxes=[0.0, 0.0, 0.0],
            )
            return
        self._send_axes(
            forward,
            strafe,
            turn,
            turn_mode=("hold" if abs(turn) > 0.0 else "pulse"),
        )
        self._report(
            ownerFollowState="escaping",
            ownerFollowCollisionRecovery=True,
            ownerFollowEscapeSeconds=round(elapsed, 2),
            ownerFollowAxes=[round(forward, 3), round(strafe, 3), round(turn, 3)],
        )

    def _search_for_owner(self, now: float, last_x: float) -> None:
        entered_search = False
        began_return = False
        completed_empty_cycle = False
        with self.lock:
            self.translation_no_motion_seconds = 0.0
            if self.phase != "searching" or self.search_started <= 0.0:
                if self.phase != "searching":
                    self.phase_before_search = self.phase
                self.phase = "searching"
                self.search_started = now
                self.owner_track_confirmed = False
                self.owner_identity_latched = False
                self.owner_private_confirmed_at = 0.0
                self.search_direction = -1.0 if last_x < 0.50 else 1.0
                self.search_best_candidate = None
                self.search_return_started = 0.0
                self.search_return_steps = 0
                self.search_return_direction = 0.0
                self.search_last_chat_step = 0
                entered_search = True
            base_direction = self.search_direction
            elapsed = max(0.0, now - self.search_started)
            absolute_step = int(elapsed // self.SEARCH_STEP_SECONDS)
            cycle_index = absolute_step // self.SEARCH_STEPS_PER_SWEEP
            candidate = self.search_best_candidate
            if (
                self.search_return_started <= 0.0
                and candidate is not None
                and cycle_index > int(candidate.get("cycle", cycle_index))
            ):
                target_step = int(candidate.get("step", 1))
                forward_steps = target_step % self.SEARCH_STEPS_PER_SWEEP
                reverse_steps = (
                    self.SEARCH_STEPS_PER_SWEEP - forward_steps
                ) % self.SEARCH_STEPS_PER_SWEEP
                if forward_steps <= reverse_steps:
                    self.search_return_steps = forward_steps
                    self.search_return_direction = base_direction
                else:
                    self.search_return_steps = reverse_steps
                    self.search_return_direction = -base_direction
                self.search_return_started = now
                began_return = True
            elif (
                self.search_return_started <= 0.0
                and candidate is None
                and cycle_index > 0
                and absolute_step % self.SEARCH_STEPS_PER_SWEEP == 0
            ):
                completed_empty_cycle = True
            return_started = self.search_return_started
            return_steps = self.search_return_steps
            return_direction = self.search_return_direction
            candidate = self.search_best_candidate

        if entered_search:
            self.announce_status(
                "owner_search_started",
                (
                    "🔎 オーナー探索中："
                    + ("左" if base_direction < 0.0 else "右")
                    + "へ全周スキャンを開始"
                ),
                min_interval=2.0,
            )

        if began_return:
            candidate_step = int(candidate.get("step", 0)) if candidate else 0
            similarity = float(candidate.get("similarity", 0.0)) if candidate else 0.0
            self.announce_status(
                "candidate_return",
                (
                    f"↩ 人物候補の方向へ復帰：スキャン {candidate_step}/"
                    f"{self.SEARCH_STEPS_PER_SWEEP}、外観一致 {similarity:.0%}"
                ),
                min_interval=0.0,
            )

        if return_started > 0.0:
            return_elapsed = max(0.0, now - return_started)
            return_turn_duration = return_steps * self.SEARCH_STEP_SECONDS
            if return_elapsed < return_turn_duration:
                return_step = min(
                    return_steps,
                    int(return_elapsed // self.SEARCH_STEP_SECONDS) + 1,
                )
                step_time = return_elapsed % self.SEARCH_STEP_SECONDS
                if step_time < self.SEARCH_TURN_SECONDS:
                    search_phase = "candidate_return_turn"
                    turn = return_direction * self.search_turn_speed
                else:
                    search_phase = "candidate_return_observe"
                    turn = 0.0
                self._send_axes(
                    0.0,
                    0.0,
                    turn,
                    turn_mode=("hold" if abs(turn) > 0.0 else "pulse"),
                )
                self._report(
                    ownerFollowDetected=False,
                    ownerFollowState="searching",
                    ownerFollowAxes=[0.0, 0.0, round(turn, 3)],
                    ownerFollowSearchDirection=(
                        "left" if return_direction < 0.0 else "right"
                    ),
                    ownerFollowSearchPhase=search_phase,
                    ownerFollowSearchReturnStep=return_step,
                    ownerFollowSearchReturnSteps=return_steps,
                    ownerFollowSearchSeconds=round(return_elapsed, 2),
                )
                return

            observe_elapsed = return_elapsed - return_turn_duration
            if observe_elapsed <= self.SEARCH_RETURN_OBSERVE_SECONDS:
                self._send_axes(0.0, 0.0, 0.0, turn_mode="pulse")
                self._report(
                    ownerFollowDetected=False,
                    ownerFollowState="searching",
                    ownerFollowAxes=[0.0, 0.0, 0.0],
                    ownerFollowSearchPhase="candidate_verify",
                    ownerFollowSearchReturnStep=return_steps,
                    ownerFollowSearchReturnSteps=return_steps,
                    ownerFollowSearchSeconds=round(return_elapsed, 2),
                )
                return

            with self.lock:
                self.search_started = now
                self.search_best_candidate = None
                self.search_return_started = 0.0
                self.search_return_steps = 0
                self.search_return_direction = 0.0
                self.search_last_chat_step = 0
            self.announce_status(
                "candidate_not_verified",
                "🔎 人物候補を再確認できません。新しい全周スキャンを開始します",
                min_interval=1.0,
            )
            self._stop_motion()
            self._report(
                ownerFollowState="searching",
                ownerFollowSearchPhase="scan_restart",
                ownerFollowSearchCandidateRemembered=False,
                ownerFollowAxes=[0.0, 0.0, 0.0],
            )
            return

        # One complete sweep always keeps turning in the same direction.
        # Empty detector frames never advance this clock, so every reported
        # angle corresponds to a turn that was physically sent to VRChat.
        absolute_step = int(elapsed // self.SEARCH_STEP_SECONDS)
        cycle_index = absolute_step // self.SEARCH_STEPS_PER_SWEEP
        scan_step = absolute_step % self.SEARCH_STEPS_PER_SWEEP + 1
        step_time = elapsed % self.SEARCH_STEP_SECONDS
        direction = base_direction
        if step_time < self.SEARCH_TURN_SECONDS:
            search_phase = "scan_turn"
            forward = 0.0
            strafe = 0.0
            turn = direction * self.search_turn_speed
        else:
            search_phase = "scan_observe"
            forward = 0.0
            strafe = 0.0
            turn = 0.0
        self._send_axes(
            forward,
            strafe,
            turn,
            turn_mode=("hold" if abs(turn) > 0.0 else "pulse"),
        )
        chat_progress = False
        progress_steps = {
            1,
            max(1, self.SEARCH_STEPS_PER_SWEEP // 4),
            max(1, self.SEARCH_STEPS_PER_SWEEP // 2),
            max(1, self.SEARCH_STEPS_PER_SWEEP * 3 // 4),
            self.SEARCH_STEPS_PER_SWEEP,
        }
        if search_phase == "scan_turn" and scan_step in progress_steps:
            with self.lock:
                chat_step = cycle_index * self.SEARCH_STEPS_PER_SWEEP + scan_step
                if chat_step != self.search_last_chat_step:
                    self.search_last_chat_step = chat_step
                    chat_progress = True
        if chat_progress:
            self.announce_status(
                f"scan_{cycle_index}_{scan_step}",
                (
                    f"🔄 探索旋回中：{'左' if direction < 0.0 else '右'} "
                    f"{scan_step}/{self.SEARCH_STEPS_PER_SWEEP}"
                ),
                min_interval=0.0,
            )
        if completed_empty_cycle:
            self.announce_status(
                f"empty_sweep_{cycle_index}",
                f"🔎 全周スキャン {cycle_index} 回完了：人物候補なし。探索を継続",
                min_interval=0.0,
            )
        self._report(
            ownerFollowDetected=False,
            ownerFollowState="searching",
            ownerFollowAxes=[round(forward, 3), round(strafe, 3), round(turn, 3)],
            ownerFollowSearchDirection=("left" if direction < 0.0 else "right"),
            ownerFollowSearchPhase=search_phase,
            ownerFollowSearchCycle=cycle_index + 1,
            ownerFollowSearchStep=scan_step,
            ownerFollowSearchStepsPerSweep=self.SEARCH_STEPS_PER_SWEEP,
            ownerFollowSearchCandidateRemembered=candidate is not None,
            ownerFollowIdentityLatched=False,
            ownerFollowPrivateConfirmationCurrent=False,
            ownerFollowPrivateConfirmationAgeSeconds=None,
            ownerFollowSearchSeconds=round(elapsed, 2),
        )

    @staticmethod
    def _clip(value: float, minimum: float, maximum: float) -> float:
        return float(max(minimum, min(maximum, value)))

    def _send_axes(
        self,
        forward: float,
        strafe: float,
        turn: float,
        *,
        turn_mode: str = "pulse",
    ) -> None:
        controller = self.controller
        if controller is None:
            return
        axes = (float(forward), float(strafe), float(turn))
        if abs(turn) < 0.06:
            button_turn = 0
        elif turn_mode == "hold":
            button_turn = -1 if turn < 0.0 else 1
        else:
            # Ordinary formation corrections use short digital pulses rather
            # than a full-speed held turn.  This VRChat environment receives
            # LookHorizontal OSC but does not apply it to the view.
            period = 0.85
            active_seconds = 0.08 + 0.08 * min(1.0, abs(turn) / 0.42)
            active = time.monotonic() % period < active_seconds
            button_turn = (-1 if turn < 0.0 else 1) if active else 0
        with self.axis_lock:
            axes_changed = any(
                abs(a - b) >= 0.01 for a, b in zip(axes, self.last_axes)
            )
            if axes_changed:
                controller.vertical(axes[0])
                controller.horizontal(axes[1])
                controller.look_horizontal(0.0)
            self._set_button_turn_locked(controller, button_turn)
            self.last_axes = axes
            self.motion_stopped = (
                all(abs(value) < 1.0e-6 for value in axes[:2])
                and button_turn == 0
            )
        self._report(
            ownerFollowTurnInput=(
                "LookLeft" if button_turn < 0
                else ("LookRight" if button_turn > 0 else None)
            )
        )

    def _set_button_turn_locked(
        self,
        controller: _ControllerLike,
        direction: int,
    ) -> None:
        """Change held LookLeft/LookRight state while holding axis_lock."""

        direction = -1 if direction < 0 else (1 if direction > 0 else 0)
        if direction == self.last_button_turn:
            return
        if self.last_button_turn < 0:
            controller.look_left(False)
        elif self.last_button_turn > 0:
            controller.look_right(False)
        if direction < 0:
            controller.look_left(True)
        elif direction > 0:
            controller.look_right(True)
        self.last_button_turn = direction

    def _stop_motion(self, *, force: bool = False) -> None:
        controller = self.controller
        if controller is None:
            return
        with self.axis_lock:
            if self.motion_stopped and not force:
                return
            controller.vertical(0.0)
            controller.horizontal(0.0)
            controller.look_horizontal(0.0)
            controller.look_left(False)
            controller.look_right(False)
            self.last_axes = (0.0, 0.0, 0.0)
            self.last_button_turn = 0
            self.motion_stopped = True
