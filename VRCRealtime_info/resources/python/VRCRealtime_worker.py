"""Low-latency VRChat <-> OpenAI Realtime audio bridge.

This process is intentionally separate from the Wolfram Language kernel.  It
owns the long-lived WebSocket and audio devices so notebook evaluation and the
front end never sit on an audio or network hot path.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from collections.abc import Callable
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import websocket
import cv2
from numpy.typing import NDArray
from vrcpilot import Mic
from vrcpilot.speaker import SpeakerLoop

from VRCRealtime_visual_gate import VisualAttentionGate
from VRCRealtime_owner_follow import OwnerFollowController
from VRCRealtime_owner_formation import FormationConfig, OwnerFormationController
from VRCRealtime_ownerjoin import OwnerLocator
from VRCRealtime_voice_commands import VoiceCommandSpotter, control_action, has_greeting
from VRCRealtime_sliden import (
    SlidenCancelled,
    SlidenConsoleController,
    SlidenError,
    prepare_slide_images,
)


OPENAI_INPUT_RATE = 24_000
VRCHAT_RATE = 48_000
PLAYBACK_SLICE_SAMPLES = 480  # 20 ms at 24 kHz.
PLAYBACK_PREBUFFER_CHUNKS = 3  # 60 ms jitter buffer before each response.

# Presentations live in SourceVault's registry (SourceVault_slidedeck.wl), which
# is written from the SlideWorkflow palette.  This table is only the last-resort
# fallback for when the registry cannot be reached at all: without it a stopped
# SourceVault service would mean no presentation can ever start, and the room
# would just see the avatar do nothing.  Which route was used is reported in
# presentationCatalogRoute rather than being hidden.
PRESENTATION_CATALOG: dict[str, dict[str, Any]] = {
    "計算と自然31": {
        "title": "計算と自然31",
        "url": "https://transreal.sakura.ne.jp/cn/31.mp4",
        "secondsPerSlide": 25.0,
        "startSlide": 1,
        "endSlide": None,
        "narrationInstructions": "専門外の聴衆にも分かるように説明してください。",
        "talk": None,
    }
}


def _normalise_presentation_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _presentation_catalog_entry(title: Any) -> dict[str, Any]:
    requested = _normalise_presentation_title(title)
    for catalogue_title, entry in PRESENTATION_CATALOG.items():
        key = _normalise_presentation_title(catalogue_title)
        if requested == key:
            return dict(entry)
        # "計算と自然31のプレゼン" is the same talk; "計算と自然312" is not.
        # The registry refuses candidates whose trailing number differs and
        # this table has to refuse them too -- starting the wrong deck is
        # worse than starting none.
        if requested.startswith(key) and not requested[len(key) : len(key) + 1].isdigit():
            return dict(entry)
    raise ValueError(f"presentation title is not configured: {title}")


def _normalise_talk(value: Any) -> dict[str, Any] | None:
    """Keep only what the narration loop can act on: numbered slides that
    actually carry script text.  A talk with no usable slide is None, so the
    caller falls back to describing the slide image instead of pretending a
    script exists."""

    if not isinstance(value, dict):
        return None
    raw_slides = value.get("slides")
    if isinstance(raw_slides, dict):
        # Already normalised once (the voice-command path never round-trips
        # through JSON), so accept that shape too and stay idempotent.
        raw_slides = list(raw_slides.values())
    if not isinstance(raw_slides, list):
        return None
    slides: dict[int, dict[str, Any]] = {}
    for item in raw_slides:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("slide", 0))
        except (TypeError, ValueError):
            continue
        text = item.get("text")
        if number < 1 or not isinstance(text, str) or not text.strip():
            continue
        seconds = item.get("seconds")
        slides[number] = {
            "slide": number,
            "text": text.strip(),
            "seconds": float(seconds)
            if isinstance(seconds, (int, float)) and float(seconds) > 0.0
            else None,
        }
    if not slides:
        return None
    opening = value.get("opening")
    closing = value.get("closing")
    return {
        "opening": opening.strip() if isinstance(opening, str) else "",
        "closing": closing.strip() if isinstance(closing, str) else "",
        "slides": slides,
    }


def _presentation_entry_from_deck(deck: dict[str, Any]) -> dict[str, Any]:
    """Turn a SourceVault registry payload into the fields _run_presentation
    consumes.  Raises when the registry says the title is unknown or withheld,
    because starting the wrong deck is worse than not starting one."""

    if not deck.get("found"):
        available = deck.get("available")
        raise ValueError(
            "presentation is not registered"
            + (
                f" (registered: {', '.join(str(a) for a in available)})"
                if isinstance(available, list) and available
                else ""
            )
        )
    if deck.get("released") is False:
        raise ValueError("presentation is withheld by the privacy gate")
    url = deck.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("registered presentation has no usable slide URL")
    seconds = deck.get("secondsPerSlide")
    start = deck.get("startSlide")
    end = deck.get("endSlide")
    instructions = deck.get("narrationInstructions")
    return {
        "title": str(deck.get("title", "")) or None,
        "url": url,
        "secondsPerSlide": float(seconds)
        if isinstance(seconds, (int, float)) and float(seconds) > 0.0
        else 25.0,
        "startSlide": int(start) if isinstance(start, int) and start >= 1 else 1,
        "endSlide": int(end) if isinstance(end, int) and end >= 1 else None,
        "narrationInstructions": instructions if isinstance(instructions, str) else "",
        "talk": None if deck.get("talkWithheld") else _normalise_talk(deck.get("talk")),
    }


#: How questions are taken during a talk (set_question_mode / --question-mode).
#:   question_time -- only in the window at each slide boundary (the room is
#:                    not heard while a slide is narrated);
#:   keywords      -- also mid-slide, but only when a listener says an allowed
#:                    word ("質問", "スライド", "プレゼン", "発表" -- the last
#:                    three so "発表を中断して" is heard), spotted locally by Vosk;
#:   detect        -- also mid-slide whenever a listener speaks; whether it was
#:                    a question is then the model's call.
#: VRChat's room audio never contains the avatar's own voice (it goes out
#: through the virtual cable), so listening during narration needs no echo
#: cancellation here.
QUESTION_MODES = ("question_time", "keywords", "detect")
QUESTION_MODE_LABELS = {
    "question_time": "質問タイム",
    "keywords": "特定ワード",
    "detect": "二段構え",
}
DEFAULT_INTERRUPT_WORDS = ("質問", "スライド", "プレゼン", "発表")


class _TalkInterrupted(Exception):
    """A listener interrupted the slide being narrated (keywords / detect)."""

    def __init__(self, kind: str, heard: str) -> None:
        super().__init__(kind)
        self.kind = kind
        self.heard = heard


def normalise_question_mode(value: Any) -> str | None:
    """Accept the canonical names, the palette's names and the Japanese labels."""

    text = str(value or "").strip()
    key = text.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "question_time": "question_time",
        "questiontime": "question_time",
        "質問タイム": "question_time",
        "keywords": "keywords",
        "keyword": "keywords",
        "words": "keywords",
        "特定ワード": "keywords",
        "detect": "detect",
        "二段構え": "detect",
    }
    return aliases.get(key) or aliases.get(text)


def resume_text(script: str, spoken: str) -> str:
    """The script from the sentence the talk was stopped in.

    Measured by length on the spoken transcript (punctuation and spaces
    ignored): the sentence the transcript reaches into is said again whole.
    """

    sentences = [part for part in _SENTENCE_BOUNDARY.split(script) if part.strip()]
    if not sentences:
        return script
    strip = lambda text: len(_NON_SPOKEN.sub("", text))  # noqa: E731
    said = strip(spoken)
    done = 0
    for index, sentence in enumerate(sentences):
        length = strip(sentence)
        if done + length > said * 1.02:
            return "".join(sentences[index:])
        done += length
    return ""


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?\n])")
_NON_SPOKEN = re.compile(r"[\s、。，．,.！？!?「」『』（）()・…ー-]")


_KANJI_DIGIT = {
    "〇": 0, "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_KANJI_NUMBER = re.compile(r"[〇零一二三四五六七八九十百]+")


def _kanji_number(token: str) -> int | None:
    """三十三 -> 33, 百二 -> 102, 三三 -> 33 (digit by digit)."""

    if not token:
        return None
    if "十" not in token and "百" not in token:
        digits = [_KANJI_DIGIT.get(ch) for ch in token]
        if any(d is None for d in digits):
            return None
        return int("".join(str(d) for d in digits))
    total = 0
    for unit, scale in (("百", 100), ("十", 10)):
        if unit in token:
            head, token = token.split(unit, 1)
            total += (_KANJI_DIGIT.get(head, 0) if head else 1) * scale
    if token:
        value = _KANJI_DIGIT.get(token)
        if value is None:
            return None
        total += value
    return total


def spoken_numbers(text: str) -> list[int]:
    """Numbers written in arabic digits (any width) or kanji numerals.

    Single kanji 一 is skipped: it is 「一つ」「一度」 far more often than a
    talk number."""

    normal = unicodedata.normalize("NFKC", str(text))
    found = [int(match) for match in re.findall(r"\d+", normal)]
    for match in _KANJI_NUMBER.findall(normal):
        if match == "一":
            continue
        value = _kanji_number(match)
        if value is not None:
            found.append(value)
    return found


_TRAILING_KANJI_NUMBER = re.compile(r"([〇零一二三四五六七八九十百]+)(回)?$")


def normalise_title_numbers(title: str) -> str:
    """「計算と自然三十三」 -> 「計算と自然33」 (a trailing kanji numeral only).

    The registry tells talks apart by their trailing number, and does not
    read kanji: 「計算と自然三十三」 tied with 31, 32 and 33 and the first
    one, 31, was started (live 2026-09-19)."""

    text = unicodedata.normalize("NFKC", str(title)).strip()
    match = _TRAILING_KANJI_NUMBER.search(text)
    if not match or match.group(1) == "一":
        return text
    value = _kanji_number(match.group(1))
    if value is None:
        return text
    return text[: match.start()] + str(value) + (match.group(2) or "")


def title_stem(title: str) -> str:
    """The words of a talk title without its number: 「計算と自然33」 -> 「計算と自然」."""

    text = unicodedata.normalize("NFKC", str(title))
    text = re.sub(r"\d+|[〇零一二三四五六七八九十百]+(回)?$", "", text)
    return re.sub(r"[\s、。,.・「」『』()（）]", "", text)


_STOP_INTENT = re.compile(r"終了|中止|打ち切|おしまいに|終わりにし|終わらせ")
_PAUSE_INTENT = re.compile(r"一時停止|停止|中断|止めて|止めてください|止まって|ちょっと待って")
_RESUME_INTENT = re.compile(r"再開|続けて|続きを")
_TALK_NAMED = re.compile(r"発表|プレゼン|スライド|説明")


def presentation_control_intent(text: str) -> str | None:
    """"stop" / "pause" / "resume" when an utterance asks that of the talk.

    Used on GPT-Live's input transcript: its model answered 「発表を終了
    します」 to 「発表を終了してください」 and never called the tool, three
    times (live 2026-09-19).  Questions (「何時に終了しますか」) are not
    requests.  「やめて」 counts only with the talk named."""

    t = re.sub(r"\s", "", unicodedata.normalize("NFKC", str(text)))
    if not t or re.search(r"[?？]$|か[。.!！]*$", t):
        return None
    if _STOP_INTENT.search(t) or (_TALK_NAMED.search(t) and re.search(r"やめて|やめよう", t)):
        return "stop"
    if _PAUSE_INTENT.search(t):
        return "pause"
    if _RESUME_INTENT.search(t):
        return "resume"
    return None


def _clear_queue(q: queue.Queue[Any]) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


class PrivateTTSBrokerClient:
    """Synchronous JSONL client for the credential-free private TTS broker."""

    def __init__(self, args: argparse.Namespace, state_callback: Any) -> None:
        self.args = args
        self.state_callback = state_callback
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.ready = False

    def start(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.ready:
                return
            env = dict(os.environ)
            for key in list(env):
                upper = key.upper()
                if (
                    upper.startswith(
                        ("OPENAI_", "ANTHROPIC_", "AWS_", "AZURE_", "GOOGLE_")
                    )
                    or upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
                ):
                    env.pop(key, None)
            command = [
                sys.executable,
                self.args.private_tts_broker,
                "--sourcevault-service-dir",
                self.args.sourcevault_service_dir,
                "--sourcevault-kinds",
                self.args.sourcevault_kinds,
                "--sourcevault-limit",
                str(self.args.sourcevault_limit),
                "--timeout-seconds",
                str(self.args.sourcevault_wait_seconds),
                "--local-speech-max-seconds",
                str(self.args.local_speech_max_seconds),
                "--piper-executable",
                self.args.piper_executable,
                "--piper-model",
                self.args.piper_model,
                "--audio-dir",
                self.args.private_audio_dir,
            ]
            # Config and language belong to the voice, not to this bridge:
            # whoever launched us resolved them from the voice that is
            # actually installed.  Empty means "let piper decide".
            if self.args.piper_config:
                command.extend(["--piper-config", self.args.piper_config])
            if getattr(self.args, "piper_language", ""):
                command.extend(["--piper-language", self.args.piper_language])
            if getattr(self.args, "piper_length_scale", 0.0):
                command.extend(
                    ["--piper-length-scale", str(self.args.piper_length_scale)]
                )
            if getattr(self.args, "piper_noise_scale", 0.0):
                command.extend(
                    ["--piper-noise-scale", str(self.args.piper_noise_scale)]
                )
            if getattr(self.args, "tts_engine", "piper") == "aivis":
                command.extend(
                    [
                        "--tts-engine",
                        "aivis",
                        "--aivis-endpoint",
                        self.args.aivis_endpoint,
                        "--aivis-style-id",
                        str(self.args.aivis_style_id),
                        "--aivis-speed-scale",
                        str(self.args.aivis_speed_scale),
                    ]
                )
            if self.args.sourcevault_kb:
                command.extend(["--sourcevault-kb", self.args.sourcevault_kb])
            if self.args.sourcevault_group:
                command.extend(["--sourcevault-group", self.args.sourcevault_group])
            if self.args.sourcevault_release_context:
                command.extend(
                    ["--sourcevault-release-context", self.args.sourcevault_release_context]
                )
            startup = subprocess.STARTUPINFO() if os.name == "nt" else None
            if startup is not None:
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            line = self.process.stdout.readline() if self.process.stdout is not None else ""
            try:
                message = json.loads(line)
            except Exception as exc:
                raise RuntimeError("private broker did not become ready") from exc
            if message.get("type") != "ready":
                raise RuntimeError("private broker failed to start")
            self.ready = True
            self.state_callback(privateBrokerReady=True, privateBrokerPID=message.get("pid"))

    def request(self, query: str, *, scope: str = "") -> dict[str, Any]:
        return self._exchange({"query": query, "scope": scope})

    def request_presentation(self, title: str) -> dict[str, Any]:
        """Resolve a presentation title through SourceVault's registry.

        Routed through the broker rather than queued from here directly so
        SourceVault keeps one ingress and the registry's release gate -- not
        this OpenAI-connected process -- decides what text crosses over."""

        return self._exchange({"kind": "presentation", "title": title})

    def request_private_readout(self, token: str, action: str) -> dict[str, Any]:
        """Accept or decline the private remainder the listener was offered."""

        return self._exchange(
            {"kind": "private_readout", "token": token, "action": action}
        )

    def request_web_search(self, query: str) -> dict[str, Any]:
        """Local SearXNG search, used only after the fast local index misses."""

        return self._exchange({"kind": "web", "query": query})

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.start()
        with self.lock:
            process = self.process
            if (
                process is None
                or process.poll() is not None
                or process.stdin is None
                or process.stdout is None
            ):
                raise RuntimeError("private broker is not running")
            request_id = uuid.uuid4().hex
            process.stdin.write(
                json.dumps(
                    {"id": request_id} | payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            process.stdin.flush()
            line = process.stdout.readline()
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != request_id:
                raise RuntimeError("invalid private broker response")
            return response

    def close(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            self.ready = False
            if process is None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.close()
                process.wait(timeout=4)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


class _SourceVaultHandled(Exception):
    """Raised inside the SourceVault tool handler when the reply was already sent."""


class RealtimeBridge:
    #: Which OpenAI API this bridge speaks ("live" in VRCRealtime_live).
    API = "realtime"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from the child environment")

        self.stop_event = threading.Event()
        self.ws_ready = threading.Event()
        self.output_reset = threading.Event()
        self.output_active = threading.Event()
        self.output_done = threading.Event()
        self.muted = threading.Event()
        #: Last time each chatbox key was sent, for the fallback path used
        #: when no owner-follow controller is around to rate-limit them.
        self._chat_times: dict[str, float] = {}
        self._chatbox: Any | None = None
        self.privacy_muted = threading.Event()
        self.sourcevault_waiting = threading.Event()
        self.local_handoff_active = threading.Event()
        self.private_playback = threading.Event()
        self.response_in_progress = threading.Event()
        self.response_create_pending = threading.Event()
        self.response_create_inflight = threading.Event()
        self.presentation_stop_event = threading.Event()
        self.ws_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state_file_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.tool_lock = threading.Lock()
        self.handoff_lock = threading.Lock()
        self.response_request_lock = threading.Lock()
        self.output_purpose_lock = threading.Lock()
        self.visual_session_lock = threading.Lock()
        self.vision_lock = threading.Lock()
        #: Focus-free window capture used for look_at_vrchat when the visual
        #: attention gate (and its frame loop) is not running.
        self._vision_capture: Any | None = None
        self._vision_capture_factory: Callable[[], Any] | None = None
        self.presentation_lock = threading.Lock()
        self.ws: websocket.WebSocketApp | None = None
        self.speaker: SpeakerLoop | None = None
        self.private_broker: PrivateTTSBrokerClient | None = None
        self.visual_gate: VisualAttentionGate | None = None
        self.owner_follower: OwnerFollowController | None = None
        self.pending_private_wav: str | None = None
        self.response_create_requests: deque[dict[str, Any]] = deque()
        self.inflight_response_request: dict[str, Any] | None = None
        self.current_output_purpose = "normal"
        self.presentation_thread: threading.Thread | None = None
        #: The most recent avatar tool-call batch, kept so shutdown and tests
        #: can wait for it instead of racing it.
        self.avatar_command_thread: threading.Thread | None = None
        self.presentation_audio_done: queue.Queue[str] = queue.Queue()
        #: Question window between slides.  The mic is opened only while this
        #: is set, so a stray remark during narration cannot become a turn.
        #: Wall-clock of the last input audio the server VAD committed as a
        #: conversation item.  With automatic responses off during a talk,
        #: this is what says the question is there to be answered.
        self.input_committed_at = 0.0
        self.presentation_qa_open = threading.Event()
        self.presentation_qa_speech = threading.Event()
        self.presentation_qa_done: queue.Queue[str] = queue.Queue()
        #: pause / resume / stop asked for by voice during a question window.
        self.presentation_control_lock = threading.Lock()
        self.presentation_control_request: str | None = None
        #: Token for private material offered to the listener but not yet
        #: accepted.  Only the token crosses into this process; the text stays
        #: in the credential-free broker until the listener asks for it.
        self.private_offer_lock = threading.Lock()
        self.private_offer_token: str | None = None
        self.private_offer_level: Any = 1.0
        #: Slide the room asked to see during questions.  Console I/O all
        #: happens on the presentation thread, so this is a request it picks
        #: up rather than something the websocket thread drives directly.
        self.slide_request_lock = threading.Lock()
        self.slide_request: int | None = None
        #: Where the console itself is, which is not where narration got to
        #: once questions start moving it around.
        self.console_slide = 0
        self.presentation_controller: Any = None
        self.presentation_talk: dict[str, Any] | None = None
        #: How questions are taken during a talk (QUESTION_MODES).  Changed by
        #: voice (set_question_mode), by the notebook ("question-mode") or by a
        #: presentation-start command that names one.
        self.question_mode = (
            normalise_question_mode(getattr(args, "question_mode", "question_time"))
            or "question_time"
        )
        raw_words = getattr(args, "interrupt_words", "") or ""
        words = raw_words.split(",") if isinstance(raw_words, str) else list(raw_words)
        self.interrupt_words = [
            " ".join(str(word).split()) for word in words if str(word).strip()
        ] or list(DEFAULT_INTERRUPT_WORDS)
        #: A listener interrupted the slide being narrated: {"kind", "heard"}.
        self.talk_interrupt_lock = threading.Lock()
        self.talk_interrupt: dict[str, Any] | None = None
        self.talk_interrupts = 0
        #: Only while a slide or a segment is being narrated do keywords /
        #: detect listen for interruptions.
        self.presentation_narrating = threading.Event()
        #: detect: how long the room has been loud (seconds of audio), and
        #: the quiet that ends a stretch.
        self._detect_loud_seconds = 0.0
        self._detect_quiet_seconds = 0.0
        #: No new interruption before this (monotonic): a room that is still
        #: chatting after its question must not stop the talk every sentence.
        self.talk_interrupt_quiet_until = 0.0
        #: The last few seconds of room audio heard during narration.  The
        #: utterance that interrupted ("スライドを戻して", a question) is in
        #: it, and the model has to hear that, not only what follows.
        self.talk_preroll: deque[NDArray[np.float32]] = deque(
            maxlen=max(1, int(6000 / max(1, int(getattr(args, "chunk_ms", 20)))))
        )
        #: Output samples received per item: with played_samples this says
        #: how much of a narration the room actually heard before a cut.
        self.received_samples: dict[str, int] = {}
        #: Did the model say anything in the last question window?
        self.question_window_spoke = False

        # Input must never accumulate stale audio.  At 20 ms per event this
        # queue contains at most about 200 ms before dropping the oldest chunk.
        self.input_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10)
        self.output_audio: queue.Queue[tuple[str, int, NDArray[np.float32]]] = (
            queue.Queue(maxsize=3000)
        )
        self.visual_audio_preroll: deque[NDArray[np.float32]] = deque(
            maxlen=max(
                1,
                (args.visual_gate_preroll_ms + args.chunk_ms - 1)
                // args.chunk_ms,
            )
        )
        self.visual_gate_audio_open = False
        self.visual_session_active = threading.Event()
        self.visual_session_started_monotonic = 0.0
        self.visual_session_deadline_monotonic = 0.0

        self.played_samples: dict[str, int] = {}
        self.handled_tool_calls: set[str] = set()
        self.last_item_id = ""
        self.last_content_index = 0
        self.input_carry: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.input_blocked_until = 0.0
        self.instructions = args.instructions
        self.session_started_at = 0.0
        #: HTTP status of a refused WebSocket handshake (set by _on_error).
        self._handshake_status: int | None = None
        #: Slide images of the running talk (GPT-Live prepares scripts for
        #: unscripted slides ahead of narrating them).
        self.presentation_slides: list[Path] = []
        self.state: dict[str, Any] = {
            "status": "starting",
            "connected": False,
            "pid": os.getpid(),
            "api": self.API,
            "questionMode": self.question_mode,
            "questionModeLabel": QUESTION_MODE_LABELS[self.question_mode],
            "interruptWords": list(self.interrupt_words),
            "talkInterrupts": 0,
            "lastTalkInterrupt": None,
            "model": args.model,
            "voice": args.voice,
            "outputDevice": args.output_device,
            "vrchatPID": args.vrchat_pid,
            "inputSource": args.input_source,
            "inputDevice": args.input_device or None,
            "allowBargeIn": args.allow_barge_in,
            "visualGateEnabled": args.visual_attention_gate,
            "visualGateCaptureRunning": False,
            "visualGateFaceDetected": False,
            "visualGateFaceBox": None,
            "visualGateNameplateBox": None,
            "visualGateNameplateBlueRatio": 0.0,
            "visualGateNameplateSpeaking": False,
            "visualGateMouthMotion": 0.0,
            "visualGateAudioRMS": 0.0,
            "visualGateStereoBalance": 0.0,
            "visualGateOpen": False,
            "visualGateAudioOpen": False,
            "visualGateSessionMode": (
                "start_only"
                if args.visual_attention_gate
                and args.visual_gate_conversation_latch
                else ("per_turn" if args.visual_attention_gate else "disabled")
            ),
            "visualGateSessionActive": False,
            "visualGateProtected": bool(
                args.visual_attention_gate
                and args.visual_gate_conversation_latch
            ),
            "visualGateSessionIdleSeconds": (
                args.visual_gate_conversation_idle_seconds
            ),
            "visualGateSessionStartedAt": None,
            "visualGateSessionLastActivityAt": None,
            "visualGateSessionExpiresAt": None,
            "visualGateSessionLastActivityReason": None,
            "visualGateSessionEndReason": None,
            "visualGateLastError": None,
            "visualGateAttentionCandidate": False,
            "avatarMode": (
                "OwnerFollow"
                if args.avatar_mode == "owner-follow"
                else (
                    "Presentation"
                    if args.avatar_mode == "presentation"
                    else "Standalone"
                )
            ),
            "ownerAccount": args.owner_account,
            "ownerFollowEnabled": bool(
                args.visual_attention_gate and args.owner_account
            ),
            "ownerFollowState": (
                "starting"
                if args.avatar_mode == "owner-follow"
                else (
                    "presentation"
                    if args.avatar_mode == "presentation"
                    else "standalone"
                )
            ),
            "ownerFollowOCRReady": False,
            "ownerFollowDetected": False,
            "ownerFollowIdentitySource": None,
            "ownerFollowIdentityConfirmed": False,
            "ownerFollowIdentityLatched": False,
            "ownerFollowTrackConfirmed": False,
            "ownerFollowPrivateConfirmationCurrent": False,
            "ownerFollowPrivateConfirmationAgeSeconds": None,
            "ownerFollowSpeechHold": False,
            "ownerFollowListenerHold": False,
            "ownerFollowReacquireGrace": False,
            "ownerFollowAppearanceReady": False,
            "ownerFollowAppearanceTracked": False,
            "ownerFollowAppearanceSimilarity": 0.0,
            "ownerFollowAppearanceBox": None,
            "ownerFollowAssumeSingleAvatar": bool(
                args.owner_follow_assume_single_avatar
            ),
            "ownerFollowAvatarDetectorReady": False,
            "ownerFollowAvatarCandidates": 0,
            "ownerFollowAvatarCandidateBoxes": [],
            "ownerFollowAvatarCandidateSelected": False,
            "ownerFollowAvatarCandidateBox": None,
            "ownerFollowAvatarCandidateConfidence": 0.0,
            "ownerFollowAvatarScanEmptyDirection": False,
            "ownerFollowSearchStepsPerSweep": (
                OwnerFollowController.SEARCH_STEPS_PER_SWEEP
            ),
            "ownerFollowSearchCandidateRemembered": False,
            "ownerFollowChatboxText": None,
            "ownerFollowChatboxAt": None,
            "ownerFollowCaptureFresh": False,
            "ownerFollowNameplateText": None,
            "ownerFollowNameplateBox": None,
            "ownerFollowTargetNameplateHeight": (
                args.owner_follow_target_nameplate_height
            ),
            "ownerFollowFacingUs": False,
            "ownerFollowFormationProgress": 0.0,
            "ownerFollowSearchDirection": None,
            "ownerFollowSearchPhase": None,
            "ownerFollowSearchCycle": 0,
            "ownerFollowSearchStep": 0,
            "ownerFollowTurnInput": None,
            "ownerFollowSearchSeconds": 0.0,
            "ownerFollowAttentionHold": False,
            "ownerFollowConversationHold": False,
            "ownerFollowConversationSearchSuppressed": False,
            "ownerFollowVisualMotion": 0.0,
            "ownerFollowCollisionRecovery": False,
            "ownerFollowCollisionAttempts": 0,
            "ownerFollowAxes": [0.0, 0.0, 0.0],
            "ownerFollowLastSeenAt": None,
            "ownerFollowLastError": None,
            "visionCaptureEnabled": bool(args.enable_vision_capture),
            "visionCaptureInProgress": False,
            "visionCaptures": 0,
            "lastVisionCaptureAt": None,
            "lastVisionImageSize": None,
            "lastVisionError": None,
            "presentationStatus": "idle",
            "presentationPhase": "idle",
            "presentationTitle": None,
            "presentationURL": None,
            "presentationSlideCount": 0,
            "presentationCurrentSlide": 0,
            "presentationStartSlide": None,
            "presentationEndSlide": None,
            "presentationDownloadedBytes": 0,
            "presentationExtractionProgress": 0.0,
            "presentationConsoleConfidence": 0.0,
            "presentationConsoleModal": False,
            "presentationConsoleInputMode": None,
            "presentationConsoleAimError": None,
            "presentationConsoleAimErrorX": None,
            "presentationConsoleAimErrorY": None,
            "presentationConsoleClickAttempt": None,
            "presentationConsoleClearDetected": None,
            "presentationConsoleClearMethod": None,
            "presentationConsoleTarget": None,
            "presentationConsoleReticle": None,
            "presentationNavigationEnabled": False,
            "presentationNavigationStatus": "idle",
            "presentationNavigationScanStep": 0,
            "presentationNavigationSweep": 0,
            "presentationNavigationPanelScanStep": 0,
            "presentationNavigationForwardSeconds": 0.0,
            "presentationNavigationCandidateRejected": False,
            "presentationSlidenScreenConfidence": 0.0,
            "presentationSlidenScreenBox": None,
            "presentationSlidenScreenWidthFraction": 0.0,
            "presentationSlidenScreenCenterErrorX": None,
            "presentationSlidenScreenCenterErrorY": None,
            "presentationSlidenLandmarkReady": False,
            "presentationNavigationRepositioned": False,
            "presentationNavigationPitchAdjustment": 0.0,
            "presentationConsoleWidthFraction": 0.0,
            "presentationConsoleCenterErrorX": None,
            "presentationConsoleDistanceReady": False,
            "presentationLastPreviewDifference": None,
            "presentationReferenceImageSize": None,
            "presentationLastTranscript": "",
            "presentationLastError": None,
            "presentationCatalogRoute": None,
            "presentationCatalogError": None,
            "presentationTalkAvailable": False,
            "presentationTalkSlides": [],
            "presentationTalkOpening": False,
            "presentationTalkClosing": False,
            "presentationScriptedSlides": [],
            "presentationAdvanceDifferences": [],
            "presentationUnconfirmedAdvances": 0,
            "presentationConsoleNavFit": None,
            "presentationConsoleStepBacks": 0,
            "presentationConsoleSlide": 0,
            "presentationResumeNarrating": None,
            "presentationResumeDisplayed": None,
            "presentationResumeChoice": None,
            "presentationQAOpen": False,
            "presentationQuestionGate": None,
            "presentationLastQuestionHeard": None,
            "presentationQuestionUncommitted": False,
            "presentationQuestionsAnswered": 0,
            "presentationPaused": False,
            "presentationLastControl": None,
            "presentationStoppedBy": None,
            "presentationStartedAt": None,
            "presentationFinishedAt": None,
            "sourceVaultEnabled": args.enable_sourcevault,
            "privateBrokerReady": False,
            "privacyMuted": False,
            "sourceVaultWaiting": False,
            "localHandoffActive": False,
            "localHandoffStage": "idle",
            "sourceVaultWaitSeconds": args.sourcevault_wait_seconds,
            "sourceVaultDeferred": False,
            "lastSourceVaultRoute": None,
            "lastSourceVaultSource": None,
            "privateOfferPending": False,
            "privateOfferCount": 0,
            "privateReadoutTruncated": False,
            "lastSourceVaultKBStatus": None,
            "lastSourceVaultScope": None,
            "lastSourceVaultPrivacyLevel": None,
            "ownerPresencePrivacyThreshold": (
                args.owner_presence_privacy_threshold
            ),
            "ownerPresencePrivateAccessAllowed": False,
            "ownerPresencePrivateAccessReason": "not_checked",
            # None until VRChat reports it.  The bot has no other way to know
            # whether the voice it is generating reaches the world at all.
            "vrchatMicMuted": None,
            "vrchatMicMutedAt": None,
            "vrchatMicUnmuteAttempts": 0,
            "chatboxLast": None,
            "chatboxLastAt": None,
            "chatboxLastError": None,
            "vrchatOSCListen": "starting",
            "vrchatOSCPackets": 0,
            "inputChunks": 0,
            "outputSamples": 0,
            "droppedInputChunks": 0,
            "droppedOutputChunks": 0,
            "reconnects": 0,
            "lastEvent": None,
            "lastTranscript": "",
            "voiceCommandReady": False,
            "voiceCommandModelStatus": None,
            "voiceCommandLast": None,
            "voiceKeywordLeft": False,
            "voiceCommandLastText": None,
            "voiceCommandLastAt": None,
            "lastError": None,
            "startedAt": time.time(),
            "updatedAt": time.time(),
        }
        #: Room-audio capture health.  vrcpilot's SpeakerLoop ends its thread
        #: on the first exception -- from the capture or from our callback --
        #: and says so only when stopped: the room went silent for good and
        #: 「会話に参加して」 was never heard (live 2026-09-19).
        self.speaker_chunks = 0
        self.speaker_last_chunk_at = 0.0
        self.speaker_callback_errors = 0
        self.speaker_restarts = 0
        self._speaker_error_logged_at = 0.0
        self._write_state()
        if args.visual_attention_gate:
            self.visual_gate = VisualAttentionGate(
                fps=args.visual_gate_fps,
                pid=args.vrchat_pid if args.vrchat_pid > 0 else None,
                center_tolerance=args.visual_gate_center_tolerance,
                min_face_fraction=args.visual_gate_min_face_fraction,
                motion_threshold=args.visual_gate_motion_threshold,
                audio_rms_threshold=args.visual_gate_audio_rms_threshold,
                max_stereo_imbalance=args.visual_gate_max_stereo_imbalance,
                hangover_ms=args.visual_gate_hangover_ms,
                require_nameplate_pulse=args.visual_gate_require_nameplate_pulse,
                nameplate_blue_threshold=args.visual_gate_nameplate_blue_threshold,
                state_callback=self._set_state,
            )
            if args.owner_account and args.owner_follow_engine == "formation":
                # Geometric formation controller: mouse yaw odometry, body
                # landmark range and heading, owner at the right edge of a
                # level view.  See VRCRealtime_owner_formation.py.
                self.owner_follower = OwnerFormationController(
                    config=FormationConfig(
                        owner_account=args.owner_account,
                        follow_distance_m=args.owner_follow_distance_m,
                        rear_angle_deg=args.owner_follow_rear_angle,
                        owner_bearing_deg=args.owner_follow_owner_bearing,
                        tracking_fps=max(4.0, args.owner_follow_tracking_fps),
                        lost_seconds=args.owner_follow_lost_seconds,
                        search_patrol=bool(getattr(args, "owner_follow_search_patrol", True)),
                        idle_after_seconds=float(getattr(args, "owner_follow_idle_after_seconds", 360.0)),
                        idle_recheck_seconds=float(getattr(args, "owner_follow_idle_recheck_seconds", 300.0)),
                        ocr_box_threshold=args.owner_follow_ocr_box_threshold,
                        ocr_text_score=args.owner_follow_ocr_text_score,
                    ),
                    initial_mode=args.avatar_mode,
                    frame_provider=self.visual_gate.snapshot,
                    osc_host=args.osc_host,
                    osc_port=args.osc_port,
                    vrchat_pid=args.vrchat_pid if args.vrchat_pid > 0 else None,
                    state_callback=self._set_state,
                    locator_factory=(
                        self._make_owner_locator
                        if getattr(args, "owner_follow_menu_locate", True)
                        else None
                    ),
                )
            elif args.owner_account:
                self.owner_follower = OwnerFollowController(
                    owner_account=args.owner_account,
                    initial_mode=args.avatar_mode,
                    frame_provider=self.visual_gate.snapshot,
                    tracking_fps=args.owner_follow_tracking_fps,
                    target_screen_x=args.owner_follow_target_screen_x,
                    target_nameplate_height=args.owner_follow_target_nameplate_height,
                    lost_seconds=args.owner_follow_lost_seconds,
                    formation_seconds=args.owner_follow_formation_seconds,
                    orbit_speed=args.owner_follow_orbit_speed,
                    search_turn_speed=args.owner_follow_search_turn_speed,
                    assume_single_avatar=args.owner_follow_assume_single_avatar,
                    ocr_min_side=args.owner_follow_ocr_min_side,
                    ocr_box_threshold=args.owner_follow_ocr_box_threshold,
                    ocr_text_score=args.owner_follow_ocr_text_score,
                    osc_host=args.osc_host,
                    osc_port=args.osc_port,
                    state_callback=self._set_state,
                )
        if args.enable_sourcevault:
            self.private_broker = PrivateTTSBrokerClient(args, self._set_state)
        #: Offline "会話に参加 / 会話から離脱" spotter on the room audio.  It
        #: runs regardless of the visual gate and of avatar motion, so the
        #: phrases work from the side, from behind and while following.
        self.voice_commands: VoiceCommandSpotter | None = None
        #: Set by an explicit spoken leave ("会話から離脱", or the model's
        #: conversation_control leave).  While it is set no room audio reaches
        #: OpenAI at all: the visual criterion alone may NOT re-open the
        #: conversation, because the person who just left is usually still
        #: standing in front of the avatar and still talking.  Only a spoken
        #: "会話に参加", or a greeting spoken while facing the avatar
        #: ("正面からこんにちは"), lifts it -- there is no timeout.
        self.voice_keyword_left = threading.Event()
        self.voice_keyword_left_at = 0.0
        if getattr(args, "voice_commands", False) and args.input_source == "vrchat":
            self.voice_commands = VoiceCommandSpotter(
                on_command=self._on_voice_keyword,
                model_dir=getattr(args, "voice_command_model_dir", "") or None,
                auto_download=not getattr(args, "no_voice_command_download", False),
                min_confidence=float(getattr(args, "voice_command_min_confidence", 0.45)),
                state_callback=self._set_state,
                question_words=tuple(self.interrupt_words),
            )

    #: Publishing the state file competes with the notebook's Dynamic monitor,
    #: which imports the same path several times a second.  A Windows reader
    #: holding the destination makes os.replace fail with WinError 5, and that
    #: exception used to propagate into whichever thread happened to be
    #: reporting progress -- a presentation died 16 ms after starting because
    #: the monitor was reading at that moment.
    _STATE_WRITE_ATTEMPTS = 6
    _STATE_WRITE_BACKOFF_SECONDS = 0.02

    def _write_state(self, **updates: Any) -> None:
        with self.state_lock:
            self.state.update(updates)
            self.state["updatedAt"] = time.time()
            payload = dict(self.state)
        with self.state_file_lock:
            path = Path(self.args.state_file)
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
            except OSError:
                return
            for attempt in range(self._STATE_WRITE_ATTEMPTS):
                try:
                    os.replace(temporary, path)
                    return
                except OSError:
                    time.sleep(self._STATE_WRITE_BACKOFF_SECONDS * (attempt + 1))
            # Best effort by design: every update writes the whole payload, so
            # a dropped snapshot costs nothing but the next poll, whereas
            # raising costs the caller its thread.
            try:
                temporary.unlink()
            except OSError:
                pass

    def _set_state(self, **updates: Any) -> None:
        with self.state_lock:
            self.state.update(updates)

    def _increment_state(self, field: str, amount: int = 1) -> None:
        with self.state_lock:
            self.state[field] = int(self.state.get(field, 0)) + amount

    def _status_worker(self) -> None:
        while not self.stop_event.wait(0.25):
            self._expire_visual_session_if_idle()
            self._write_state()

    @staticmethod
    def _osc_bool(packet: bytes, address: str) -> bool | None:
        """Read a boolean OSC argument out of *packet*, or ``None``.

        A whole OSC library for one parameter is not worth a dependency, but
        the packet does have to be parsed properly rather than sniffed: an
        OSC sender is free to wrap messages in a ``#bundle``, and a reader
        that only recognizes a bare message then sees a steady stream of
        traffic and reports nothing at all -- which is indistinguishable from
        VRChat not sending, and sends the search off in the wrong direction.
        """

        if packet.startswith(b"#bundle\x00"):
            # Header is the tag plus an eight byte time tag; then each
            # element is a length-prefixed packet of its own.
            cursor = 16
            while cursor + 4 <= len(packet):
                size = int.from_bytes(packet[cursor : cursor + 4], "big")
                cursor += 4
                element = packet[cursor : cursor + size]
                cursor += size
                found = RealtimeBridge._osc_bool(element, address)
                if found is not None:
                    return found
            return None
        encoded = address.encode("ascii")
        if not packet.startswith(encoded + b"\x00"):
            return None
        cursor = (len(encoded) + 4) & ~3
        end = packet.find(b"\x00", cursor)
        tags = packet[cursor:end] if end >= 0 else packet[cursor:]
        if tags == b",T":
            return True
        if tags == b",F":
            return False
        # VRChat declares MuteSelf as a Bool and sends T/F, but a sender that
        # spells the same flag as a number is not wrong, and treating that as
        # "no reading" would strand the caller on "unknown" for ever.
        payload = packet[((end + 4) & ~3) if end >= 0 else len(packet) :]
        if tags == b",i" and len(payload) >= 4:
            return int.from_bytes(payload[:4], "big", signed=True) != 0
        if tags == b",f" and len(payload) >= 4:
            import struct

            return abs(struct.unpack(">f", payload[:4])[0]) > 0.5
        return None

    def _vrchat_state_worker(self) -> None:
        """Listen for what VRChat says about itself, notably the mic.

        The client publishes ``MuteSelf`` on its OSC output port.  Without it
        the bot cannot tell a presentation nobody heard from one that was
        never spoken: the audio path looks identical from this side either
        way, because it is -- the samples are generated, upsampled and pushed
        into the virtual cable whether or not VRChat is passing them on.
        """

        import socket

        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Deliberately no SO_REUSEADDR: on Windows it lets this bind take
            # a port another OSC application is already using, and the two
            # would then fight over VRChat's output.  Losing the race cleanly
            # is the correct outcome.
            listener.bind(("127.0.0.1", self.args.osc_listen_port))
            listener.settimeout(0.25)
        except OSError as exc:
            # Another OSC application already owns the port, which is a
            # perfectly ordinary setup.  Everything else still works; the
            # mute state simply stays unknown.
            self._write_state(
                vrchatMicMuted=None,
                vrchatOSCListen=f"port {self.args.osc_listen_port} unavailable",
                lastError=f"osc listen: {type(exc).__name__}: {exc}",
            )
            return
        self._write_state(vrchatOSCListen=f"listening on {self.args.osc_listen_port}")
        packets = 0
        try:
            while not self.stop_event.is_set():
                try:
                    packet, _sender = listener.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    return
                packets += 1
                # Counting traffic is what separates the two reasons the mute
                # state can be unknown -- nothing is arriving, or plenty is
                # arriving and none of it is the parameter we want.  A run
                # cannot be watched live, because VRChat holds the foreground
                # throughout, so the state has to answer that afterwards.
                if packets % 50 == 1:
                    self._write_state(vrchatOSCPackets=packets)
                muted = self._osc_bool(packet, "/avatar/parameters/MuteSelf")
                if muted is None:
                    continue
                with self.state_lock:
                    unchanged = self.state.get("vrchatMicMuted") is muted
                if unchanged:
                    continue
                self._write_state(
                    vrchatMicMuted=muted,
                    vrchatMicMutedAt=time.time(),
                    vrchatOSCPackets=packets,
                )
        finally:
            listener.close()

    #: What each presentation phase looks like from the room.  The console
    #: work is a long silent stretch of an avatar staring at a panel, so the
    #: chatbox is the only place anyone can see whether it is working or
    #: stuck -- and the notebook cannot be watched during a run, because
    #: VRChat holds the foreground throughout.
    _PRESENTATION_CHAT = {
        "preparing": "準備中",
        "downloading": "スライド取得中",
        "extracting": "スライド展開中",
        "navigating_to_console": "操作パネルへ移動中",
        "recognizing_console": "操作パネルを確認中",
        "clearing_console_url": "URL欄を消去中",
        "loading_console_url": "スライド読み込み中",
        "seeking_start_slide": "開始位置へ移動中",
        "showing_slide": "スライドを表示中",
        "opening": "発表を始めます",
        "narrating": "発表中",
        "question_window": "質問があればどうぞ",
        "paused": "一時停止中(「再開して」で再開)",
        "final_questions": "質問があればどうぞ",
        "closing": "まとめ",
        "advancing_console": "次のスライドへ",
        "at_sliden_console": "操作パネルに到着",
        "completed": "発表を終了しました",
        "stopped": "発表を停止しました",
    }

    def _announce(self, key: str, text: str, *, min_interval: float = 2.5) -> bool:
        """Put a short line in VRChat's chatbox.

        Sent straight down OSC rather than through the owner-follow
        controller.  That controller is built lazily inside the follower's
        own loop, and in presentation mode the loop is parked -- so every
        announcement was quietly answered "no channel" and the room saw
        nothing at all while the bot worked in silence.
        """

        line = text.strip()[:144]
        if not line:
            return False
        now = time.monotonic()
        with self.state_lock:
            previous = self._chat_times.get(key, 0.0)
            if previous and now - previous < max(1.2, min_interval):
                return False
            self._chat_times[key] = now
        try:
            if self._chatbox is None:
                from vrcpilot import OscSender

                self._chatbox = OscSender(
                    self.args.osc_host, self.args.osc_port
                ).controller()
            self._chatbox.chatbox(line, send=True, sfx=False)
        except Exception as exc:
            self._chatbox = None
            # Recorded rather than raised: a status line failing to send is
            # not a reason to stop presenting, but it is a reason the room
            # saw nothing, and that has to be findable afterwards.
            self._write_state(chatboxLastError=f"{type(exc).__name__}: {exc}")
            return False
        self._write_state(chatboxLast=line, chatboxLastAt=time.time())
        return True

    def _presentation_state(self, **updates: Any) -> None:
        """Write presentation state and narrate the change to the room."""

        self._write_state(**updates)
        phase = updates.get("presentationPhase")
        if not isinstance(phase, str):
            return
        label = self._PRESENTATION_CHAT.get(phase)
        if label is None:
            return
        with self.state_lock:
            title = str(self.state.get("presentationTitle") or "")
            current = self.state.get("presentationCurrentSlide") or 0
            total = self.state.get("presentationSlideCount") or 0
            end = self.state.get("presentationEndSlide") or total
        where = ""
        if phase in ("narrating", "advancing_console", "seeking_start_slide"):
            if current and end:
                where = f" {current}/{end}"
        elif phase == "extracting" and total:
            where = f" {total}枚"
        heading = f"[{title}] " if title and phase != "narrating" else ""
        self._announce(f"presentation:{phase}:{current}", f"{heading}{label}{where}")

    def _ensure_vrchat_microphone_live(self) -> bool:
        """Unmute VRChat if it has told us it is muted.

        Only ever acts on a reported state.  ``/input/Voice`` is a toggle, so
        acting on a guess would mute a bot that was working perfectly.
        """

        with self.state_lock:
            muted = self.state.get("vrchatMicMuted")
            attempts = int(self.state.get("vrchatMicUnmuteAttempts") or 0)
        if muted is not True:
            return muted is not None
        try:
            from vrcpilot import OscSender

            OscSender(self.args.osc_host, self.args.osc_port).controller().voice()
        except Exception as exc:
            self._write_state(
                lastError=f"unmute: {type(exc).__name__}: {exc}",
            )
            return False
        self._write_state(vrchatMicUnmuteAttempts=attempts + 1)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self.state_lock:
                if self.state.get("vrchatMicMuted") is False:
                    return True
            time.sleep(0.05)
        return False

    def _visual_session_enabled(self) -> bool:
        return bool(
            self.visual_gate is not None
            and self.args.visual_gate_conversation_latch
        )

    def _activate_visual_session(self, reason: str = "visual_gate") -> bool:
        """Latch the visual gate open after one deliberately addressed turn."""

        if not self._visual_session_enabled():
            return False
        now_monotonic = time.monotonic()
        now_epoch = time.time()
        with self.visual_session_lock:
            newly_active = not self.visual_session_active.is_set()
            self.visual_session_active.set()
            if newly_active:
                self.visual_session_started_monotonic = now_monotonic
            self.visual_session_deadline_monotonic = (
                now_monotonic + self.args.visual_gate_conversation_idle_seconds
            )
        updates: dict[str, Any] = {
            "visualGateSessionActive": True,
            "visualGateProtected": False,
            "visualGateSessionLastActivityAt": now_epoch,
            "visualGateSessionExpiresAt": (
                now_epoch + self.args.visual_gate_conversation_idle_seconds
            ),
            "visualGateSessionLastActivityReason": reason,
            "visualGateSessionEndReason": None,
        }
        if newly_active:
            updates["visualGateSessionStartedAt"] = now_epoch
        if self.owner_follower is not None:
            # A conversation should feel face-to-face. Full navigation and
            # search rotation remain paused until the latched session ends.
            self.owner_follower.hold_for_attention(1.0)
            self.owner_follower.set_conversation_active(True)
        self._set_state(**updates)
        return newly_active

    def _touch_visual_session(self, reason: str) -> None:
        """Extend an already-open conversation without opening it by itself."""

        if not self._visual_session_enabled() or not self.visual_session_active.is_set():
            return
        now_monotonic = time.monotonic()
        now_epoch = time.time()
        with self.visual_session_lock:
            if not self.visual_session_active.is_set():
                return
            self.visual_session_deadline_monotonic = (
                now_monotonic + self.args.visual_gate_conversation_idle_seconds
            )
        self._set_state(
            visualGateSessionLastActivityAt=now_epoch,
            visualGateSessionExpiresAt=(
                now_epoch + self.args.visual_gate_conversation_idle_seconds
            ),
            visualGateSessionLastActivityReason=reason,
        )

    def _end_visual_session(self, reason: str) -> bool:
        """Re-arm the visual gate and discard any audio buffered before it."""

        if not self._visual_session_enabled():
            return False
        with self.visual_session_lock:
            was_active = self.visual_session_active.is_set()
            self.visual_session_active.clear()
            self.visual_session_started_monotonic = 0.0
            self.visual_session_deadline_monotonic = 0.0
        if self.owner_follower is not None:
            self.owner_follower.set_conversation_active(False)
        self._reset_input_gate()
        self._write_state(
            visualGateSessionActive=False,
            visualGateProtected=True,
            visualGateSessionStartedAt=None,
            visualGateSessionExpiresAt=None,
            visualGateSessionEndReason=reason,
        )
        return was_active

    def _expire_visual_session_if_idle(self) -> bool:
        if not self._visual_session_enabled() or not self.visual_session_active.is_set():
            return False
        with self.visual_session_lock:
            deadline = self.visual_session_deadline_monotonic
        if deadline <= 0.0 or time.monotonic() < deadline:
            return False
        if any(
            event.is_set()
            for event in (
                self.output_active,
                self.private_playback,
                self.response_in_progress,
                self.response_create_pending,
                self.response_create_inflight,
                self.sourcevault_waiting,
                self.local_handoff_active,
            )
        ):
            return False
        return self._end_visual_session("idle_timeout")

    def _send_now(self, event: dict[str, Any]) -> bool:
        with self.ws_lock:
            ws = self.ws
        if ws is None or not self.ws_ready.is_set():
            return False
        try:
            ws.send(json.dumps(event, separators=(",", ":")))
            return True
        except Exception as exc:  # connection failures are handled by run_forever
            self._set_state(lastError=f"send: {type(exc).__name__}: {exc}")
            return False

    def _put_latest_input(self, event: dict[str, Any]) -> None:
        try:
            self.input_events.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self.input_events.get_nowait()
        except queue.Empty:
            pass
        try:
            self.input_events.put_nowait(event)
        except queue.Full:
            pass
        self._increment_state("droppedInputChunks")

    def _speaker_callback_safe(self, chunk: NDArray[np.float32]) -> None:
        """What the SpeakerLoop calls: one bad chunk must not end the loop."""

        self.speaker_chunks += 1
        self.speaker_last_chunk_at = time.monotonic()
        try:
            self._speaker_callback(chunk)
        except Exception as exc:  # keep hearing the room; say what broke
            self.speaker_callback_errors += 1
            now = time.monotonic()
            if now - self._speaker_error_logged_at >= 5.0:
                self._speaker_error_logged_at = now
                import traceback

                where = traceback.extract_tb(exc.__traceback__)[-1]
                self._write_state(
                    speakerCallbackErrors=self.speaker_callback_errors,
                    lastSpeakerCallbackError=(
                        f"{type(exc).__name__}: {exc} "
                        f"({where.name}:{where.lineno})"
                    )[:300],
                )

    def _start_speaker(self) -> None:
        pid = self.args.vrchat_pid if self.args.vrchat_pid > 0 else None
        self.speaker = SpeakerLoop(
            self._speaker_callback_safe,
            chunk_seconds=self.args.chunk_ms / 1000.0,
            pid=pid,
        )
        self.speaker.start()
        self.speaker_last_chunk_at = time.monotonic()

    def _speaker_watchdog(self) -> None:
        """Bring the room audio back when its capture thread has died or hung."""

        while not self.stop_event.wait(2.0):
            self._check_speaker()

    def _check_speaker(self) -> bool:
        """One watchdog look; True when the capture was restarted."""

        speaker = self.speaker
        if speaker is None:
            return False
        stale = time.monotonic() - self.speaker_last_chunk_at
        alive = bool(getattr(speaker, "is_running", True))
        self._set_state(
            speakerAlive=alive,
            speakerChunks=self.speaker_chunks,
            speakerLastChunkAgeSeconds=round(stale, 1),
        )
        # The loop forwards empty chunks as silence ticks, so a quiet
        # room still ticks: 10 s without one is a dead or hung capture.
        if alive and stale < 10.0:
            return False
        reason = "thread_died" if not alive else f"no_audio_{stale:.0f}s"
        error = getattr(speaker, "_exception", None)
        # close() joins the capture thread with no timeout; a capture hung
        # in read() would take the watchdog down with it.  Let it go on
        # its own thread.
        threading.Thread(
            target=self._close_quietly, args=(speaker,), daemon=True
        ).start()
        self.speaker_restarts += 1
        try:
            self._start_speaker()
            restarted = True
        except Exception as exc:
            restarted = False
            error = error or exc
        self._write_state(
            speakerRestarts=self.speaker_restarts,
            lastSpeakerRestart={
                "reason": reason,
                "error": (f"{type(error).__name__}: {error}"[:300] if error else None),
                "restarted": restarted,
                "at": time.time(),
            },
        )
        return restarted

    @staticmethod
    def _close_quietly(speaker: Any) -> None:
        try:
            speaker.close()
        except BaseException:
            pass  # the error is already recorded; close() re-raises it

    def _speaker_callback(self, chunk: NDArray[np.float32]) -> None:
        if self._control_listening() and chunk.size:
            # A talk in keywords / detect mode listens to the room while it
            # narrates.  Local only: the mic stays muted towards OpenAI, and
            # the room audio has no echo of the avatar (its voice goes out
            # through the virtual cable, VRChat does not play it back).
            self._observe_talk_audio(chunk)
            return
        if (
            self.stop_event.is_set()
            or self.muted.is_set()
            or self.privacy_muted.is_set()
            or self.sourcevault_waiting.is_set()
            or self.local_handoff_active.is_set()
            or not self.ws_ready.is_set()
            or (
                not self._full_duplex()
                and (
                    (self.output_active.is_set() and not self.args.allow_barge_in)
                    or time.monotonic() < self.input_blocked_until
                )
            )
            or chunk.size == 0
        ):
            return

        spotter = self.voice_commands
        if (
            spotter is not None
            and (not self.output_active.is_set() or self._full_duplex())
            and not self.private_playback.is_set()
        ):
            # Local only.  The spotter never sees audio the bridge would not
            # consider at all (muted, echo guard) nor the avatar's own speech.
            spotter.feed(chunk)
        left = self.voice_keyword_left.is_set()
        visual_gate = self.visual_gate
        if visual_gate is not None:
            # The gate keeps watching even while the conversation is closed:
            # its frontal-face evidence is what makes a spoken greeting count
            # as "addressed from the front".  Observation is local only.
            visual_gate.observe_audio(chunk)
            attention_candidate = bool(
                getattr(visual_gate, "attention_candidate", lambda: False)()
            )
            self._set_state(visualGateAttentionCandidate=attention_candidate)
            if attention_candidate and self.owner_follower is not None and not left:
                # Stop navigation before the strict nameplate/mouth gate opens;
                # otherwise capture motion can keep destabilising the very
                # evidence required to enter voice mode.
                self.owner_follower.hold_for_attention(1.0)
            if left:
                # Explicitly left: hold everything back until a join keyword
                # or a frontal greeting (_on_voice_keyword clears the latch).
                self.visual_audio_preroll.clear()
                if self.visual_gate_audio_open:
                    self.visual_gate_audio_open = False
                    self.input_carry = np.empty(0, dtype=np.float32)
                    _clear_queue(self.input_events)
                    self._set_state(visualGateAudioOpen=False)
                return
            session_open = (
                self.args.visual_gate_conversation_latch
                and self.visual_session_active.is_set()
            )
            if not session_open and not visual_gate.allows_audio():
                self.visual_audio_preroll.append(
                    np.asarray(chunk, dtype=np.float32).copy()
                )
                if self.visual_gate_audio_open:
                    self.visual_gate_audio_open = False
                    self.input_carry = np.empty(0, dtype=np.float32)
                    _clear_queue(self.input_events)
                    self._set_state(visualGateAudioOpen=False)
                return
            if not self.visual_gate_audio_open:
                if not session_open:
                    self._activate_visual_session("visual_gate")
                self.visual_gate_audio_open = True
                buffered = list(self.visual_audio_preroll)
                self.visual_audio_preroll.clear()
                self._set_state(visualGateAudioOpen=True)
                for buffered_chunk in buffered:
                    self._enqueue_input_audio(buffered_chunk)
        elif left:
            # No visual gate configured: the latch is the whole barrier.
            return

        self._enqueue_input_audio(chunk)

    def _full_duplex(self) -> bool:
        """Does the room keep being heard while the avatar speaks?

        Not with Realtime: its server VAD turns any sound in the room into
        a barge-in, so the room is closed while the avatar speaks and for a
        moment after.  The GPT-Live bridge answers differently."""

        return False

    def _enqueue_input_audio(self, chunk: NDArray[np.float32]) -> None:
        """Downmix and enqueue one visual-gate-approved audio chunk."""

        # VRC Pilot returns 48 kHz stereo float32.  Speech is downmixed and
        # decimated by two for OpenAI's 24 kHz PCM input.  Pair averaging is a
        # deliberately tiny low-pass stage that keeps the callback inexpensive.
        mono = np.mean(chunk, axis=1, dtype=np.float32)
        if self.input_carry.size:
            mono = np.concatenate((self.input_carry, mono))
        if mono.size < 2:
            self.input_carry = mono.copy()
            return
        if mono.size % 2:
            self.input_carry = mono[-1:].copy()
            mono = mono[:-1]
        else:
            self.input_carry = np.empty(0, dtype=np.float32)
        downsampled = mono.reshape(-1, 2).mean(axis=1, dtype=np.float32)
        pcm16 = np.clip(downsampled, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2", copy=False)
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16.tobytes()).decode("ascii"),
        }
        self._put_latest_input(event)
        self._increment_state("inputChunks")

    def _input_sender(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.input_events.get(timeout=0.1)
            except queue.Empty:
                continue
            if (
                self.muted.is_set()
                or self.privacy_muted.is_set()
                or self.sourcevault_waiting.is_set()
                or self.local_handoff_active.is_set()
            ):
                continue
            self._send_now(event)

    @staticmethod
    def _upsample_for_vrchat(
        samples: NDArray[np.float32], previous: float
    ) -> tuple[NDArray[np.float32], float]:
        """Linear 24 -> 48 kHz interpolation with no heavyweight DSP dependency."""
        if samples.size == 0:
            return samples, previous
        out = np.empty(samples.size * 2, dtype=np.float32)
        out[1::2] = samples
        out[0] = (np.float32(previous) + samples[0]) * np.float32(0.5)
        if samples.size > 1:
            out[2::2] = (samples[:-1] + samples[1:]) * np.float32(0.5)
        return out, float(samples[-1])

    def _open_mic(self) -> Mic:
        return Mic(self.args.output_device, sample_rate=VRCHAT_RATE, channels=1)

    def _microphone_worker(self) -> None:
        import soundcard as sc  # VRC Pilot already depends on soundcard.

        try:
            device = (
                sc.get_microphone(self.args.input_device, include_loopback=False)
                if self.args.input_device
                else sc.default_microphone()
            )
            self._set_state(resolvedInputDevice=str(device.name))
            frames = max(1, int(VRCHAT_RATE * self.args.chunk_ms / 1000))
            with device.recorder(
                samplerate=VRCHAT_RATE,
                channels=1,
                blocksize=frames,
            ) as recorder:
                while not self.stop_event.is_set():
                    chunk = recorder.record(numframes=frames)
                    self._speaker_callback(np.asarray(chunk, dtype=np.float32))
        except Exception as exc:
            self._write_state(
                status="error",
                lastError=f"input microphone: {type(exc).__name__}: {exc}",
            )
            self.stop_event.set()
            with self.ws_lock:
                ws = self.ws
            if ws is not None:
                ws.close()

    def _send_vad_mode(self, enabled: bool) -> None:
        self._send_now(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": self.args.vad_threshold,
                                "prefix_padding_ms": self.args.prefix_padding_ms,
                                "silence_duration_ms": self.args.silence_duration_ms,
                                "create_response": enabled,
                                "interrupt_response": (
                                    self.args.allow_barge_in if enabled else False
                                ),
                            }
                        }
                    },
                },
            }
        )

    def _reset_input_gate(self) -> None:
        _clear_queue(self.input_events)
        self.visual_audio_preroll.clear()
        self.visual_gate_audio_open = False
        self.input_carry = np.empty(0, dtype=np.float32)
        self._send_now({"type": "input_audio_buffer.clear"})
        self._set_state(visualGateAudioOpen=False)

    def _set_sourcevault_waiting(self, enabled: bool) -> None:
        """Pause conversational VAD while a synchronous local lookup runs."""
        if enabled:
            self.sourcevault_waiting.set()
            self._send_vad_mode(False)
            self._reset_input_gate()
            self._write_state(sourceVaultWaiting=True)
            return
        self.sourcevault_waiting.clear()
        self._reset_input_gate()
        if (
            not self.privacy_muted.is_set()
            and not self.local_handoff_active.is_set()
        ):
            self._send_vad_mode(True)
        self.input_blocked_until = time.monotonic() + 0.3
        self._write_state(sourceVaultWaiting=False)

    def _set_privacy_mute(self, enabled: bool) -> None:
        """Hard gate all audio toward OpenAI while private speech is audible."""
        if enabled:
            self.privacy_muted.set()
            # No private playback starts until automatic VAD responses are off
            # and all already-ordered input audio has been discarded.
            self._send_vad_mode(False)
            self._reset_input_gate()
            self._write_state(privacyMuted=True)
            return

        self._reset_input_gate()
        self.privacy_muted.clear()
        if (
            not self.sourcevault_waiting.is_set()
            and not self.local_handoff_active.is_set()
        ):
            self._send_vad_mode(True)
        self.input_blocked_until = (
            time.monotonic() + self.args.output_cooldown_ms / 1000.0
        )
        self._write_state(privacyMuted=False)

    @staticmethod
    def _read_private_wav(path: Path) -> NDArray[np.float32]:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if width != 2 or rate <= 0:
            raise ValueError("private TTS WAV must be PCM16")
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1, dtype=np.float32)
        samples = pcm / np.float32(32768.0)
        if rate != OPENAI_INPUT_RATE and samples.size > 1:
            output_size = max(1, int(round(samples.size * OPENAI_INPUT_RATE / rate)))
            source_x = np.arange(samples.size, dtype=np.float64)
            target_x = np.linspace(0.0, samples.size - 1.0, output_size)
            samples = np.interp(target_x, source_x, samples).astype(np.float32)
        return samples

    def _queue_private_wav(self, wav_path: str) -> None:
        path = Path(wav_path).resolve()
        self._set_privacy_mute(True)
        if self.owner_follower is not None:
            self.owner_follower.set_speech_active(True)
        try:
            samples = self._read_private_wav(path)
            _clear_queue(self.output_audio)
            self.output_reset.set()
            self.output_active.set()
            self.output_done.clear()
            self.private_playback.set()
            item_id = "private-tts-" + uuid.uuid4().hex
            for start in range(0, samples.size, PLAYBACK_SLICE_SAMPLES):
                part = samples[start : start + PLAYBACK_SLICE_SAMPLES].copy()
                self.output_audio.put((item_id, 0, part), timeout=1.0)
            self.output_done.set()
            self._write_state(privatePlayback=True)
        except Exception:
            self.private_playback.clear()
            self._set_privacy_mute(False)
            raise
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    def _begin_local_handoff(
        self, call_id: str, wav_path: str, privacy_level: Any
    ) -> None:
        """Let OpenAI announce the privacy handoff before local speech starts."""
        path = Path(wav_path).resolve()
        if not path.is_file():
            raise RuntimeError("private broker returned no audio")
        with self.handoff_lock:
            if self.local_handoff_active.is_set() or self.pending_private_wav:
                raise RuntimeError("another local handoff is already active")
            self.pending_private_wav = str(path)
            self.local_handoff_active.set()
        self._write_state(
            localHandoffActive=True,
            localHandoffStage="openai_announcement",
        )
        self._send_function_output(
            call_id,
            {
                "status": "local_handoff_required",
                "privacyLevel": privacy_level,
                "message": (
                    "結果はプライバシーレベル0.5以上のためモデルへ渡されません。"
                    "これからローカル音声が本人へ回答します。"
                    "内容を推測・復唱しないでください。"
                ),
            },
        )
        self._request_response(
            purpose="sourcevault_local_announcement",
            instructions=(
                "SourceVaultの結果はプライバシーレベル0.5以上であり、"
                "本文はあなたには提供されません。これからローカル音声モデルが"
                "本人へ回答します。次の一文だけを日本語で自然に発話してください。"
                "『この情報はプライバシーレベルが高いため、ここからローカル音声に"
                "切り替えて回答します。』"
            ),
        )

    def _start_pending_private_playback(self) -> None:
        with self.handoff_lock:
            wav_path = self.pending_private_wav
            self.pending_private_wav = None
        if not wav_path:
            self._abort_local_handoff("local handoff audio is missing")
            return
        self._write_state(localHandoffStage="local_playback")
        follower = self.owner_follower
        if follower is not None:
            follower.announce_status(
                "local_speech_started",
                "🔊 ローカル音声で回答中",
                min_interval=0.0,
            )
        try:
            self._queue_private_wav(wav_path)
        except Exception as exc:
            self._abort_local_handoff(
                f"local handoff playback: {type(exc).__name__}: {exc}"
            )

    def _notify_local_playback_complete(self) -> None:
        """Add only safe completion metadata, then let OpenAI resume the dialog."""
        follower = self.owner_follower
        if follower is not None:
            follower.announce_status(
                "local_speech_finished",
                "✅ ローカル音声の回答が終了",
                min_interval=0.0,
            )
        self._write_state(
            privatePlayback=False,
            localHandoffStage="openai_completion",
        )
        self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "[ローカル実行通知] プライバシーレベル0.5以上の"
                                "SourceVault結果について、ローカル音声モデルによる"
                                "本人への回答が終了しました。非公開本文はあなたには"
                                "提供されていません。内容を推測・復唱せず、通常の会話を"
                                "再開してください。"
                            ),
                        }
                    ],
                },
            }
        )
        self._request_response(
            purpose="sourcevault_local_completion",
            instructions=(
                "ローカル音声モデルの回答が終了したことを把握してください。"
                "非公開の内容は推測も復唱もしないでください。"
                "次の一文だけを日本語で自然に発話してください。"
                "『ローカル音声での回答が終わりました。会話を続けましょう。』"
            ),
        )

    def _complete_local_handoff(self) -> None:
        with self.handoff_lock:
            self.pending_private_wav = None
            self.local_handoff_active.clear()
        self._reset_input_gate()
        if not self.privacy_muted.is_set() and not self.sourcevault_waiting.is_set():
            self._send_vad_mode(True)
        self.input_blocked_until = (
            time.monotonic() + self.args.output_cooldown_ms / 1000.0
        )
        self._write_state(
            localHandoffActive=False,
            localHandoffStage="idle",
        )

    def _abort_local_handoff(self, message: str) -> None:
        with self.handoff_lock:
            wav_path = self.pending_private_wav
            self.pending_private_wav = None
            self.local_handoff_active.clear()
        if wav_path:
            try:
                Path(wav_path).unlink()
            except OSError:
                pass
        self.private_playback.clear()
        self._set_privacy_mute(False)
        if self.owner_follower is not None:
            self.owner_follower.set_speech_active(False)
        self._write_state(
            localHandoffActive=False,
            localHandoffStage="error",
            lastError=message,
        )

    def _finish_output_period(self) -> None:
        # Refresh before clearing output_active so the idle watchdog cannot
        # re-lock in the small gap between playback completion and bookkeeping.
        self._touch_visual_session("output_finishing")
        was_private = self.private_playback.is_set()
        with self.output_purpose_lock:
            output_purpose = self.current_output_purpose
            self.current_output_purpose = "normal"
        self.private_playback.clear()
        self.output_done.clear()
        self.output_active.clear()
        self.input_blocked_until = (
            time.monotonic() + self.args.output_cooldown_ms / 1000.0
        )
        if not self._full_duplex():
            _clear_queue(self.input_events)
        # Discard any echo tail already waiting on the server.
        self._send_now({"type": "input_audio_buffer.clear"})
        if was_private:
            self._set_privacy_mute(False)
            if self.local_handoff_active.is_set():
                self._notify_local_playback_complete()
            else:
                self._write_state(privatePlayback=False)
        elif output_purpose == "sourcevault_local_announcement":
            threading.Thread(
                target=self._start_pending_private_playback, daemon=True
            ).start()
        elif output_purpose == "sourcevault_local_completion":
            self._complete_local_handoff()
        elif output_purpose.startswith("presentation_"):
            try:
                self.presentation_audio_done.put_nowait(output_purpose)
            except queue.Full:
                pass
        elif self.presentation_qa_open.is_set():
            # An answer to a question asked inside the window finished playing.
            # (detect lets the model stay silent when what it heard was not
            # addressed to it; whether it spoke decides how the talk resumes.)
            with self.state_lock:
                if str(self.state.get("lastTranscript", "")).strip():
                    self.question_window_spoke = True
            try:
                self.presentation_qa_done.put_nowait(output_purpose)
            except queue.Full:
                pass
        keep_speech_hold = bool(
            output_purpose == "sourcevault_local_announcement"
            or (was_private and self.local_handoff_active.is_set())
        )
        if not keep_speech_hold and self.owner_follower is not None:
            self.owner_follower.set_speech_active(False)
        self._touch_visual_session("output_finished")

    def _playback_worker(self) -> None:
        mic: Mic | None = None
        previous_sample = 0.0
        playback_started = False
        try:
            mic = self._open_mic()
            while not self.stop_event.is_set():
                if self.output_reset.is_set():
                    self.output_reset.clear()
                    mic.close()
                    mic = self._open_mic()
                    previous_sample = 0.0
                    playback_started = False
                if (
                    self.output_active.is_set()
                    and not playback_started
                    and self.output_audio.qsize() < PLAYBACK_PREBUFFER_CHUNKS
                    and not self.output_done.is_set()
                ):
                    time.sleep(0.005)
                    continue
                try:
                    item_id, content_index, samples = self.output_audio.get(timeout=0.02)
                except queue.Empty:
                    if self.output_done.is_set():
                        self._finish_output_period()
                        playback_started = False
                    continue
                if self.output_reset.is_set():
                    continue
                playback_started = True
                vrchat_samples, previous_sample = self._upsample_for_vrchat(
                    samples, previous_sample
                )
                mic.play(vrchat_samples)
                with self.playback_lock:
                    self.last_item_id = item_id
                    self.last_content_index = content_index
                    self.played_samples[item_id] = (
                        self.played_samples.get(item_id, 0) + int(samples.size)
                    )
                self._increment_state("outputSamples", int(samples.size))
                if self.output_done.is_set() and self.output_audio.empty():
                    self._finish_output_period()
                    playback_started = False
        except Exception as exc:
            self._write_state(
                status="error",
                lastError=f"playback: {type(exc).__name__}: {exc}",
            )
            self.stop_event.set()
            with self.ws_lock:
                ws = self.ws
            if ws is not None:
                ws.close()
        finally:
            if mic is not None:
                mic.close()

    def _queue_output_delta(self, event: dict[str, Any]) -> None:
        if self.privacy_muted.is_set():
            return
        try:
            raw = base64.b64decode(event["delta"])
            pcm = np.frombuffer(raw, dtype="<i2")
        except Exception as exc:
            self._set_state(lastError=f"audio decode: {type(exc).__name__}: {exc}")
            return
        samples = pcm.astype(np.float32) / np.float32(32768.0)
        item_id = str(event.get("item_id", ""))
        content_index = int(event.get("content_index", 0))
        if not self.output_active.is_set():
            self.output_active.set()
            self.output_done.clear()
            if self.owner_follower is not None:
                self.owner_follower.set_speech_active(True)
            if not self.args.allow_barge_in and not self._full_duplex():
                _clear_queue(self.input_events)
        with self.playback_lock:
            self.received_samples[item_id] = (
                self.received_samples.get(item_id, 0) + int(samples.size)
            )
        for start in range(0, samples.size, PLAYBACK_SLICE_SAMPLES):
            part = samples[start : start + PLAYBACK_SLICE_SAMPLES].copy()
            try:
                self.output_audio.put_nowait((item_id, content_index, part))
            except queue.Full:
                self._increment_state("droppedOutputChunks")
                break

    def _interrupt_playback(self) -> None:
        _clear_queue(self.output_audio)
        self.output_reset.set()
        self.output_done.set()
        with self.playback_lock:
            item_id = self.last_item_id
            content_index = self.last_content_index
            played = self.played_samples.get(item_id, 0)
        if item_id and played > 0:
            self._send_now(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": int(played * 1000 / OPENAI_INPUT_RATE),
                }
            )

    def _effective_instructions(self) -> str:
        instructions = self.instructions
        if self.args.enable_sourcevault:
            instructions += (
                "\nSourceVault、保存資料、過去の作業やメールについて質問された場合は、"
                "必ず sourcevault_lookup を使ってから答えてください。"
                "結果は公開資料が先に返ります。tool が privateRemaining を返したら、"
                "まず公開分で答えたうえで『非公開の資料も読み上げますか』と一言たずね、"
                "希望されたら private_readout を action=play、不要と言われたら "
                "action=discard で呼んでください。勝手に読み上げないでください。"
                "tool が local_handoff_required を返した場合、非公開内容は推測も復唱もせず、"
                "指定された切替案内だけを発話してください。ローカル実行通知を受け取ったら、"
                "ローカル音声が終了した事実だけを把握して通常会話へ戻ってください。"
                "tool が preparing を返した場合は、検索をバックグラウンドで準備しているので"
                "後からもう一度確認してほしい、と短く伝えてください。"
                "『今日』だけでなく『昨日』『一昨日』『3日前』『先週』『先月』や"
                "具体的な日付を含むSourceVaultの依頼でも必ずtoolを呼び、利用者の"
                "日付表現をqueryへそのまま含めてください。"
                "sourcevault_lookup は事前構築の高速索引を最初に引くので、"
                "調べ物は常にこれを最初の手段にしてください。"
            )
            if self.args.enable_voice_web_search:
                instructions += (
                    "\nsourcevault_lookup が not_found を返したら、そこで打ち切らずに"
                    "『ウェブで調べましょうか』と一言たずねてください。"
                    "了承されたら web_search を呼び、返ってきた題名と抜粋だけを根拠に"
                    "日本語で簡潔にまとめ、ウェブで調べた結果である旨を添えてください。"
                    "本文は取得していないので、抜粋に無いことを補わないでください。"
                    "断られたらそこで終え、追加の説明や再検索はしないでください。"
                    "sourcevault_lookup を飛ばして web_search を呼ばないでください。"
                )
        if self.owner_follower is not None:
            instructions += (
                "\n利用者がスタンドアロン／フォロー／プレゼンモードへの切替を命じたら、"
                "口頭で答えるだけでなく必ず set_avatar_mode を呼んでください。"
                "『〇〇集会のプレゼンを開始して』のように発表名を挙げた依頼には、"
                "start_presentation を一度だけ呼び、title には利用者が言った発表名を"
                "そのまま渡してください (set_avatar_mode を先に呼ぶ必要はありません。"
                "title 付きの set_avatar_mode で始めた場合も start_presentation を"
                "重ねて呼ばないでください)。登録簿の照合は tool 側が行うので、"
                "タイトルを自分で言い換えたり番号を推測したりしないでください。"
                "ツールは登録簿からスライドURLと発表原稿を取得し、操作パネルへの投入と"
                "発表開始までをバックグラウンドで実行します。"
                "tool が not registered を返した場合は、その発表は登録されていないと"
                "短く伝え、別の発表を勝手に始めないでください。"
                "\n質疑応答で『一つ前のスライド』『最後のスライド』『3枚目を出して』"
                "と言われたら show_slide を呼んでスライドを出し直してください。"
                "『〜について質問が』のように話題を挙げられた場合も show_slide を"
                "topic=その話題 で呼び、該当スライドを出してから答えてください"
                "(見つからないと返ったら、無理に別のスライドを出さずそのまま答えてください)。"
                "\n発表中はスライドの切り替え時に短い質疑応答の時間があります。"
                "そこで『一時停止して』『打ち切って』『発表を終了』『再開して』と"
                "言われたら、必ず presentation_control を呼んでください。"
                "『スタンドアロンモードに移行』『フォローモードに移行』のような"
                "モード変更を言われた場合は set_avatar_mode を呼んでください"
                "(発表は自動的に終了します)。"
                "右旋回、左旋回、反対を向く指示には必ず turn_avatar を呼んでください。"
                "会話への参加または離脱を明示されたら必ず conversation_control を"
                "呼んでください。ツール実行前に完了したとは言わないでください。"
            )
        if self._visual_session_enabled():
            instructions += (
                "\n利用者が『じゃあね』『会話を終わる』など明確に会話終了を告げた場合だけ、"
                "短い別れの挨拶をしたうえで end_conversation を呼んでください。"
                "単なる沈黙や話題の区切りでは呼ばないでください。"
            )
        if self.args.enable_vision_capture:
            instructions += (
                "\nあなたは VRChat 内のアバターで、look_at_vrchat を呼べば今見ている画面の"
                "静止画が得られます。利用者が『目の前に何がある』『今何が見える』"
                "『これは何』『あれは何』『前/周りに何がある』『画面を見て』『見て答えて』"
                "『どんな場所』など、現在の視界・目の前の物・景色・文字について尋ねたら、"
                "映像がないとは答えず、画像を送ってほしいとも言わず、必ず先に look_at_vrchat を"
                "呼んでください。ツール後に現在画面の静止画が会話へ追加されるので、"
                "その画像だけに基づいて簡潔に答えてください。"
            )
        return instructions

    def _session_update(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "type": "realtime",
            "model": self.args.model,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_INPUT_RATE},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.args.vad_threshold,
                        "prefix_padding_ms": self.args.prefix_padding_ms,
                        "silence_duration_ms": self.args.silence_duration_ms,
                        "create_response": True,
                        "interrupt_response": self.args.allow_barge_in,
                    },
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": OPENAI_INPUT_RATE,
                    },
                    "voice": self.args.voice,
                },
            },
            "instructions": self._effective_instructions(),
        }
        tools: list[dict[str, Any]] = []
        if self.args.enable_sourcevault:
            tools.append(
                {
                    "type": "function",
                    "name": "sourcevault_lookup",
                    "description": (
                        "SourceVaultのローカル情報を検索する。SourceVault、過去の作業、"
                        "保存資料、メール、ノート、パッケージ文書について尋ねられたら使う。"
                        "「計算と自然」など発表スライドの内容、スライド中の図やグラフ、"
                        "学生便覧の規程についての質問は事前構築の索引から即座に返るので、"
                        "推測で答えずまずこのtoolを呼ぶ。"
                        "公開可能な結果だけがモデルへ返り、非公開結果はローカル音声で処理される。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SourceVaultで検索する短い日本語クエリ。",
                            }
                        },
                        "required": ["query"],
                    },
                }
            )
        if self.args.enable_sourcevault:
            tools.append(
                {
                    "type": "function",
                    "name": "private_readout",
                    "description": (
                        "sourcevault_lookup が privateRemaining を返したときだけ使う。"
                        "利用者が続きを聞きたいと言えば action=play で保留中の非公開分を"
                        "ローカル音声で読み上げ、不要と言われたら action=discard で破棄する。"
                        "本文はモデルには渡らないので、推測も復唱もしないこと。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["play", "discard"]}
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }
            )
        if self.args.enable_sourcevault and self.args.enable_voice_web_search:
            tools.append(
                {
                    "type": "function",
                    "name": "web_search",
                    "description": (
                        "ローカルのメタ検索エンジンでウェブを検索し、"
                        "候補の題名・URL・抜粋だけを返す(本文は取得しない)。"
                        "sourcevault_lookup が該当なしを返した場合にだけ使う。"
                        "先に必ず sourcevault_lookup を試すこと。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "検索クエリ。",
                            }
                        },
                        "required": ["query"],
                    },
                }
            )
        if self.owner_follower is not None:
            tools.extend(
                [
                    {
                        "type": "function",
                        "name": "set_avatar_mode",
                        "description": (
                            "音声指示に従ってスタンドアロン、オーナーフォロー、"
                            "またはプレゼンモードへ実際に切り替える。"
                            "mode=presentation はモードを変えるだけで発表は始まらない。"
                            "利用者が発表名を挙げて開始を求めたときは title に"
                            "その発表名をそのまま入れること (入れれば発表も始まるので、"
                            "start_presentation を重ねて呼ばないこと)。"
                            "title が分からないときは start_presentation を使うこと。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": [
                                        "standalone",
                                        "owner_follow",
                                        "presentation",
                                    ],
                                },
                                "title": {
                                    "type": "string",
                                    "description": (
                                        "mode=presentation のときだけ。利用者が言った"
                                        "発表名を番号も含めてそのまま (漢数字は算用数字に"
                                        "してよい)。番号を推測・補完しないこと。"
                                    ),
                                },
                            },
                            "required": ["mode"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "start_presentation",
                        "description": (
                            "指定タイトルの発表を開始する。SourceVaultの発表登録簿から"
                            "スライド(Sliden用mp4)のURLと発表原稿を取得し、プレゼンモードへ"
                            "切り替え、操作パネルへURLを投入して自動発表を始める。"
                            "タイトルの表記ゆれは登録簿側で吸収するので、利用者が言った"
                            "発表名をそのまま渡すこと。未登録なら開始せずエラーを返す。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": (
                                        "利用者が言った発表名を番号も含めてそのまま"
                                        "(漢数字は算用数字にしてよい)。番号を推測・"
                                        "補完しないこと。"
                                    ),
                                }
                            },
                            "required": ["title"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "presentation_control",
                        "description": (
                            "実行中の発表を一時停止・再開・停止する。"
                            "『一時停止して』『ちょっと待って』『中断して』"
                            "『停止して』『止めて』は pause、"
                            "『再開して』『続けて』は resume、"
                            "『発表を終了して』『打ち切って』『もうやめて』は stop。"
                            "終了は取り消せないので、迷うときは pause にする。"
                            "質疑応答中にスライドを動かすと、発表側から"
                            "『表示中のスライドから続けますか、元の位置へ戻りますか』"
                            "と尋ねる。その返事に答えるときだけ、"
                            "表示中から続けるなら resume_here、"
                            "発表が進んでいた位置へ戻るなら resume_back を使う。"
                            "口頭で答えるだけにせず必ずこのツールを呼ぶこと。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "pause",
                                        "resume",
                                        "stop",
                                        "resume_here",
                                        "resume_back",
                                    ],
                                }
                            },
                            "required": ["action"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "show_slide",
                        "description": (
                            "発表したスライドを操作パネルに表示し直す。質疑応答で"
                            "「一つ前のスライドを見せて」「最後のスライド」"
                            "「3枚目を出して」と言われたとき、および"
                            "「〜について質問が」と話題を挙げられたときに使う。"
                            "slide=枚数指定、relative=previous/next/first/last、"
                            "topic=話題 のいずれか一つを渡す。"
                            "表示できたら、その内容を踏まえて質問に答えること。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "slide": {
                                    "type": "integer",
                                    "description": "1 から始まるスライド番号。",
                                },
                                "relative": {
                                    "type": "string",
                                    "enum": ["previous", "next", "first", "last"],
                                },
                                "topic": {
                                    "type": "string",
                                    "description": "利用者が挙げた話題。原稿と照合して該当スライドを選ぶ。",
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "turn_avatar",
                        "description": (
                            "VRChatアバターをその場で左90度、右90度、"
                            "または反対向きへ実際に旋回させる。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "direction": {
                                    "type": "string",
                                    "enum": ["left", "right", "around"],
                                }
                            },
                            "required": ["direction"],
                            "additionalProperties": False,
                        },
                    },
                    {
                        "type": "function",
                        "name": "conversation_control",
                        "description": (
                            "利用者の明示的な指示に従い、音声会話へ参加するか"
                            "会話から離脱して開始ゲートを再有効化する。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["join", "leave"],
                                }
                            },
                            "required": ["action"],
                            "additionalProperties": False,
                        },
                    },
                ]
            )
        tools.append(self._question_mode_tool())
        if self._visual_session_enabled():
            tools.append(
                {
                    "type": "function",
                    "name": "end_conversation",
                    "description": (
                        "利用者が明確に別れや会話終了を告げたとき、"
                        "視覚的な開始ゲートを再び有効にする。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            )
        if self.args.enable_vision_capture:
            tools.append(
                {
                    "type": "function",
                    "name": "look_at_vrchat",
                    "description": (
                        "今このアバターが見ているVRChat画面を1枚取得して会話に追加する。"
                        "『目の前に何がある』『何が見える』『これは何』『周りの様子は』など、"
                        "現在の視界・目の前の物・景色・文字について尋ねられたら、画像の提供を"
                        "求めずにまずこれを呼ぶ。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "画像を見て答える利用者の質問。",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if tools:
            session["tools"] = tools
            session["tool_choice"] = "auto"
        return {
            "type": "session.update",
            "session": session,
        }

    def _question_mode_tool(self) -> dict[str, Any]:
        words = "」「".join(self.interrupt_words)
        return {
            "type": "function",
            "name": "set_question_mode",
            "description": (
                "発表中の質問の受け方を切り替える。利用者が『質問モードを切り替えたい』"
                "『質問の受け方を変えて』などと言ったら、まず次の3つを日本語でごく短く"
                "説明し、どれにするか尋ねる。選ばれたらこのツールを呼ぶ。"
                "question_time=質問タイム(スライドの区切りでだけ質問を受ける。既定)、"
                f"keywords=特定ワード(発表中でも「{words}」を含む発話があれば中断して"
                "質問を受け、終われば中断した所から再開)、"
                "detect=二段構え(発表中に声が聞こえたら一旦止め、自分への質問か判断して"
                "応じ、違えば続ける)。利用者がモード名を最初から言っていれば、"
                "尋ね直さずに呼んでよい。現在のモードを尋ねられたら mode を省いて呼ぶ。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": list(QUESTION_MODES),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        }

    def _handle_question_mode_call(self, item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id", ""))
        try:
            arguments = json.loads(str(item.get("arguments", "{}")) or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        requested = arguments.get("mode")
        previous = self.question_mode
        if requested in (None, ""):
            payload = {
                "status": "current",
                "mode": previous,
                "label": QUESTION_MODE_LABELS[previous],
            }
            instructions = (
                f"現在の質問モードは『{QUESTION_MODE_LABELS[previous]}』です。"
                "日本語でごく短く伝えてください。"
            )
        else:
            mode = normalise_question_mode(requested)
            ready, why = self._question_mode_ready(mode or "")
            if mode is None:
                payload = {"status": "error", "error": f"unknown mode: {requested}"}
                instructions = (
                    "指定された質問モードは無効でした。質問タイム・特定ワード・"
                    "二段構えのどれにするか、日本語で短く尋ね直してください。"
                )
            elif not ready:
                payload = {
                    "status": "unavailable",
                    "mode": mode,
                    "reason": why,
                    "current": previous,
                }
                instructions = (
                    f"『{QUESTION_MODE_LABELS[mode]}』は、音声のキーワード認識が"
                    "準備できていないため今は使えません。現在の"
                    f"『{QUESTION_MODE_LABELS[previous]}』のままであることを"
                    "日本語でごく短く伝えてください。"
                )
            else:
                self._set_question_mode(mode, source="voice")
                payload = {
                    "status": "changed" if mode != previous else "unchanged",
                    "mode": mode,
                    "label": QUESTION_MODE_LABELS[mode],
                    "previous": previous,
                    "presentationRunning": self._presentation_running(),
                }
                instructions = (
                    f"質問モードを『{QUESTION_MODE_LABELS[mode]}』に"
                    + ("しました。" if mode != previous else "したままです。")
                    + "そのことを日本語でごく短く伝えてください。"
                )
                if mode == "keywords":
                    instructions += (
                        f"発表中に質問したいときは「{self.interrupt_words[0]}」と"
                        "言ってもらえばよい、と一言添えてください。"
                    )
        self._send_function_output(call_id, payload)
        self._write_state(lastVoiceCommand="set_question_mode", lastVoiceCommandValue=str(requested))
        self._request_response(purpose="question_mode", instructions=instructions)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        with self.ws_lock:
            self.ws = ws
        try:
            ws.send(json.dumps(self._session_update(), separators=(",", ":")))
        except Exception as exc:
            self._write_state(lastError=f"session update: {type(exc).__name__}: {exc}")
            ws.close()
            return
        self.ws_ready.set()
        self.session_started_at = time.monotonic()
        self._write_state(status="connected", connected=True, lastError=None)

    def _send_function_output(self, call_id: str, payload: dict[str, Any]) -> None:
        self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )

    def _vision_enabled(self) -> bool:
        return bool(self.args.enable_vision_capture)

    def _direct_visual_frame(self) -> NDArray[np.uint8] | None:
        """One focus-free grab of the VRChat window (no visual gate running)."""

        if self._vision_capture is None:
            if self._vision_capture_factory is not None:
                self._vision_capture = self._vision_capture_factory()
            else:
                from vrcpilot.capture import Capture

                pid = self.args.vrchat_pid if self.args.vrchat_pid > 0 else None
                self._vision_capture = Capture(frame_timeout=3.0, pid=pid)
        try:
            frame = self._vision_capture.read()
        except Exception:
            try:
                self._vision_capture.close()
            except Exception:
                pass
            self._vision_capture = None
            raise
        if frame is None:
            return None
        return np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))

    def _latest_visual_frame(self, wait_seconds: float = 1.25) -> NDArray[np.uint8] | None:
        gate = self.visual_gate
        if gate is None:
            return self._direct_visual_frame()
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while not self.stop_event.is_set():
            frame = gate.snapshot(max_age_seconds=2.0)
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                return None
            self.stop_event.wait(0.05)
        return None

    def _encode_visual_frame(self, frame: NDArray[np.uint8]) -> tuple[str, int, int]:
        height, width = frame.shape[:2]
        max_width = self.args.vision_image_max_width
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        output_height, output_width = frame.shape[:2]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.vision_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode the VRChat frame")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
        return data_uri, output_width, output_height

    def _handle_vision_capture(
        self,
        call_id: str | None = None,
        question: str = "",
    ) -> None:
        if not self._vision_enabled():
            if call_id:
                self._send_function_output(
                    call_id,
                    {"status": "error", "message": "VRChat画像取得は無効です。"},
                )
                self._request_response()
            self._write_state(lastVisionError="vision capture is not enabled")
            return
        if not self.vision_lock.acquire(blocking=False):
            if call_id:
                self._send_function_output(
                    call_id,
                    {"status": "busy", "message": "画像取得はすでに実行中です。"},
                )
                self._request_response()
            return
        self._write_state(visionCaptureInProgress=True, lastVisionError=None)
        try:
            frame = self._latest_visual_frame()
            if frame is None:
                raise RuntimeError("a recent VRChat frame is not available")
            data_uri, width, height = self._encode_visual_frame(frame)
            if call_id:
                self._send_function_output(
                    call_id,
                    {"status": "captured", "width": width, "height": height},
                )
            prompt = question.strip() or "現在のVRChat画面には何が見えますか。"
            sent = self._send_now(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "現在のVRChat画面の静止画です。質問: " + prompt
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": data_uri,
                                "detail": "auto",
                            },
                        ],
                    },
                }
            )
            if not sent:
                raise RuntimeError("the VRChat image could not be sent")
            with self.state_lock:
                capture_count = int(self.state.get("visionCaptures", 0)) + 1
            self._write_state(
                visionCaptures=capture_count,
                lastVisionCaptureAt=time.time(),
                lastVisionImageSize=[width, height],
                lastVisionError=None,
            )
            self._request_response(
                purpose="vision_capture",
                instructions=(
                    "直前に追加されたVRChat画面の静止画を観察し、利用者の質問に"
                    "日本語で簡潔に答えてください。見えない情報は推測しないでください。"
                ),
            )
        except Exception as exc:
            message = f"vision capture: {type(exc).__name__}: {exc}"
            if call_id:
                self._send_function_output(
                    call_id,
                    {"status": "error", "message": "現在画面を取得できませんでした。"},
                )
                self._request_response()
            self._write_state(lastVisionError=message)
        finally:
            self._write_state(visionCaptureInProgress=False)
            self.vision_lock.release()

    #: Preview change that confirms the console actually turned the page.
    #: Calibrated on live runs (2026-08-15): a click that hit the lock button
    #: instead of Next measured 0.00159, while real page turns measured
    #: 0.00867, 0.03532 and 0.13907.  The old floor of 0.0015 sat inside the
    #: noise and passed the lock click as an advance; 0.01 sat above the
    #: quietest real turn.  Both figures come from measurements, not taste.
    _ADVANCE_CONFIRM_DIFFERENCE = 0.004
    #: Consecutive unconfirmed advances tolerated before stopping.  Two
    #: near-identical slides in a row are believable; three are not, and
    #: narrating a deck nobody can follow is worse than stopping.
    _ADVANCE_UNCONFIRMED_LIMIT = 3

    def _resolve_presentation(self, title: Any) -> tuple[dict[str, Any], str]:
        """Resolve a spoken title to a deck.  Registry first; the built-in table
        only when the registry cannot answer at all.  Returns the route so a
        presentation given from stale built-in data is visible afterwards."""

        broker = self.private_broker
        registry_error: str | None = None
        if broker is not None:
            try:
                response = broker.request_presentation(str(title))
            except Exception as exc:
                registry_error = f"{type(exc).__name__}"
            else:
                route = str(response.get("route", ""))
                if route == "presentation":
                    deck = response.get("deck")
                    if isinstance(deck, dict):
                        # A registry that answers "not registered" is an answer,
                        # not an outage: do not paper over it with the fallback.
                        return _presentation_entry_from_deck(deck), "registry"
                    registry_error = "malformed"
                else:
                    registry_error = str(response.get("error", route or "unavailable"))
        self._write_state(presentationCatalogError=registry_error)
        try:
            entry = _presentation_catalog_entry(title)
        except ValueError:
            # The built-in table holds one talk.  Everything else lives in the
            # registry, so a registry that could not answer is the reason this
            # talk cannot start -- report that, in the language it will be
            # spoken in, instead of "not configured" (live 2026-09-04: the
            # service kernel was inside a 300 s maintenance tick, the lookup
            # timed out, and the room was told a presentation was on its way).
            if registry_error:
                self._write_state(
                    presentationLastError=f"registry unavailable: {registry_error}"
                )
                raise ValueError(
                    f"発表登録簿を引けなかったため『{title}』を開始できません "
                    f"({registry_error})。SourceVault サービスの応答を待って"
                    "もう一度声をかけてもらってください。"
                ) from None
            self._write_state(presentationLastError=f"unknown title: {title}")
            raise ValueError(
                f"『{title}』という発表は登録簿にありません。"
            ) from None
        entry.setdefault("talk", None)
        return entry, "builtin"

    _PRESENTATION_CONTROL_ACTIONS = {
        "pause",
        "resume",
        "stop",
        # Answers to "carry on from the slide on screen, or from the one the
        # talk had reached?", asked whenever questions moved the deck.
        "resume_here",
        "resume_back",
    }
    #: How long that question waits for an answer before carrying on from
    #: where the talk was.  Long enough to think about, short enough that a
    #: room that has stopped listening does not strand the talk.
    _RESUME_CHOICE_SECONDS = 25.0

    def _sliden_text_locator(self, frame: NDArray[np.uint8]) -> list[tuple[str, Any]]:
        """OCR every text on *frame*, for the navigation's wall-screen search.

        The wall screen's template detector needs a head-on, mid-distance
        view; from beside or close to the screen only its OCR'd wordmark
        remains.  The engine is created on first use (a few seconds) and
        kept; a missing OCR install just leaves navigation without it.
        """

        engine = getattr(self, "_sliden_ocr_engine", None)
        if engine is None:
            try:
                from vrcpilot.ocr import RapidOCREngine

                engine = RapidOCREngine(
                    params={
                        "Det.limit_side_len": 1080,
                        "Global.text_score": 0.45,
                        "Global.log_level": "warning",
                    }
                )
            except Exception as exc:
                self._write_state(presentationNavigationOCR=f"unavailable: {type(exc).__name__}")
                engine = False
            self._sliden_ocr_engine = engine
        if engine is False:
            return []
        from VRCRealtime_sliden import Rect

        boxes: list[tuple[str, Any]] = []
        for word in engine.recognize(frame):
            x, y, w, h = word.bbox
            boxes.append((str(word.text), Rect(int(x), int(y), max(1, int(w)), max(1, int(h)))))
        return boxes

    def _presentation_running(self) -> bool:
        thread = self.presentation_thread
        return thread is not None and thread.is_alive()

    def _request_presentation_control(self, action: str) -> str:
        """Record a pause/resume/stop asked for by voice.

        Kept as a request the presentation thread picks up rather than acted on
        here: stopping mid-narration from the websocket thread would leave the
        console controller and the mute state half-restored."""

        if action not in self._PRESENTATION_CONTROL_ACTIONS:
            raise ValueError(
                "action must be pause, resume, stop, resume_here or resume_back"
            )
        with self.presentation_control_lock:
            self.presentation_control_request = action
        self._write_state(presentationLastControl=action)
        if action == "stop":
            # Also trip the hard stop so a presentation that is not currently
            # inside a question window still ends.
            self._stop_presentation()
        return action

    def _take_presentation_control(self) -> str | None:
        with self.presentation_control_lock:
            action = self.presentation_control_request
            self.presentation_control_request = None
        return action

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        stripped = "".join(ch for ch in text if not ch.isspace())
        return {stripped[i : i + 2] for i in range(max(0, len(stripped) - 1))}

    def _slide_for_topic(self, topic: str) -> int | None:
        """The slide whose script best matches *topic*.

        Matched here rather than asked of the model: the scenario text and the
        slide numbers are both already in this process, and character bigram
        overlap is enough to tell "キノコの記憶" from "量子ゲート" without a
        round trip or a chance of an invented slide number.
        """

        talk = self.presentation_talk
        if not talk or not topic.strip():
            return None
        wanted = self._bigrams(topic)
        if not wanted:
            return None
        best: tuple[float, int] | None = None
        for number, entry in talk["slides"].items():
            haystack = self._bigrams(
                str(entry.get("title", "")) + str(entry.get("text", ""))
            )
            if not haystack:
                continue
            score = len(wanted & haystack) / len(wanted)
            if best is None or score > best[0]:
                best = (score, number)
        if best is None or best[0] < 0.25:
            return None
        return best[1]

    def _resolve_slide_request(self, arguments: dict[str, Any]) -> int | None:
        """Turn "3枚目" / "一つ前" / "最後" / a topic into a slide number."""

        count = int(self.state.get("presentationSlideCount", 0) or 0)
        end = int(self.state.get("presentationEndSlide", 0) or 0) or count
        current = self.console_slide or int(
            self.state.get("presentationCurrentSlide", 1) or 1
        )
        slide = arguments.get("slide")
        if isinstance(slide, (int, float)) and int(slide) >= 1:
            target = int(slide)
        else:
            relative = str(arguments.get("relative", "")).strip()
            topic = str(arguments.get("topic", "")).strip()
            if relative == "previous":
                target = current - 1
            elif relative == "next":
                target = current + 1
            elif relative == "first":
                target = 1
            elif relative == "last":
                target = end or current
            elif topic:
                found = self._slide_for_topic(topic)
                if found is None:
                    return None
                target = found
            else:
                return None
        if count > 0:
            target = max(1, min(target, count))
        return max(1, target)

    def _serve_slide_request(self, controller: Any) -> None:
        """Move the console to a requested slide.  Runs on the presentation
        thread, which is the only thread allowed to touch the controller."""

        with self.slide_request_lock:
            target = self.slide_request
            self.slide_request = None
        if target is None or controller is None:
            return
        if self.console_slide <= 0:
            self.console_slide = 1
        if target == self.console_slide:
            self._announce("slide_jump", f"スライド {target}", min_interval=0.0)
            return
        self._presentation_state(presentationPhase="showing_slide")
        self._announce("slide_jump", f"スライド {target} へ", min_interval=0.0)
        self._move_console_to(controller, target)

    def _move_console_to(self, controller: Any, target: int) -> None:
        """Step the console to *target*.  Presentation thread only."""

        if controller is None or target < 1:
            return
        if self.console_slide <= 0:
            self.console_slide = 1
        steps = target - self.console_slide
        if steps == 0:
            return
        move = controller.next_slide if steps > 0 else controller.previous_slide
        for _ in range(abs(steps)):
            if self._presentation_cancelled():
                return
            try:
                move()
            except (SlidenCancelled, SlidenError):
                break
            self.console_slide += 1 if steps > 0 else -1
            self._write_state(presentationConsoleSlide=self.console_slide)
        self._write_state(presentationConsoleSlide=self.console_slide)

    def _resume_after_slide_detour(
        self, controller: Any, narrating: int, run_id: str
    ) -> int | None:
        """Put the talk and the screen back in agreement after questions.

        Questions are taken at a slide boundary and the room can move the deck
        while they are open ("go back one, what was that?").  The narration
        then carried straight on with the next slide's script over whatever
        was left on screen, and the two never met again for the rest of the
        run.  Which way to reconcile them is not this process's call: the room
        may have moved back to re-read a slide it wants covered again, or
        merely to glance at it.  So ask.

        Returns the slide to narrate next when the room chose to carry on from
        where the deck now stands, or None when the console was put back to
        the slide the talk had reached -- in which case the caller advances as
        it always did.
        """

        displayed = self.console_slide
        if displayed <= 0 or displayed == narrating:
            return None
        choice = self._ask_presentation_resume_point(narrating, displayed, run_id)
        # The deck may have moved again while the room was deciding; "here"
        # means wherever it stands now, not where it stood when asked.
        displayed = self.console_slide
        if choice == "here" and displayed != narrating:
            self._presentation_state(
                presentationResumeChoice="displayed",
                presentationCurrentSlide=displayed,
            )
            return displayed
        self._presentation_state(presentationResumeChoice="narration")
        self._move_console_to(controller, narrating)
        return None

    def _ask_presentation_resume_point(
        self, narrating: int, displayed: int, run_id: str
    ) -> str:
        """Ask where to carry on from.  Returns "here" or "back"."""

        self._presentation_state(
            presentationPhase="awaiting_resume_choice",
            presentationResumeNarrating=narrating,
            presentationResumeDisplayed=displayed,
            presentationResumeChoice=None,
        )
        self._announce(
            f"presentation:resume:{run_id}",
            f"{displayed}枚目から続けますか、{narrating}枚目へ戻りますか",
            min_interval=0.0,
        )
        # The question is spoken from fixed text rather than left to the model:
        # it has to name the right two slide numbers every time, and this
        # process is the only thing that knows them.
        self._speak_presentation_line(
            f"presentation_resume_choice:{run_id}",
            f"いま{displayed}枚目を表示しています。"
            f"このまま{displayed}枚目から続けますか、"
            f"それとも{narrating}枚目に戻って続けますか",
        )
        self._activate_visual_session("presentation_resume_choice")
        # Same rule as the slide-boundary window: room noise must not be able
        # to make the avatar talk.  The answer is a tool call, so the model
        # only needs a turn once the room has actually said something.
        gated = self._local_speech_gate_ready()
        if gated:
            self._send_vad_mode(False)
            self.input_committed_at = 0.0
        _clear_queue(self.presentation_qa_done)
        self.presentation_qa_speech.clear()
        self.presentation_qa_open.set()
        self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "いま発表の再開位置を尋ねたところです。"
                                "返事を聞いたら必ず presentation_control を呼んでください。\n"
                                f"1. 『このまま』『ここから』『{displayed}枚目から』など、"
                                "表示中のスライドから続ける意味なら action=resume_here。\n"
                                f"2. 『戻って』『元の場所から』『{narrating}枚目から』など、"
                                "発表が進んでいた場所へ戻る意味なら action=resume_back。\n"
                                "3. 『再開して』『続けて』だけのときは、利用者はいま画面に"
                                "出ているスライドを見て言っているので resume_here。\n"
                                "4. 『やめて』『終了』なら action=stop。\n"
                                "余計な説明はせず、短く復唱してからツールを呼んでください。"
                            ),
                        }
                    ],
                },
            }
        )
        self.muted.clear()
        self._write_state(muted=False, presentationQAOpen=True)
        try:
            heard_since = time.time()
            deadline = time.monotonic() + self._RESUME_CHOICE_SECONDS
            while time.monotonic() < deadline:
                if self._presentation_cancelled():
                    return "back"
                action = self._take_presentation_control()
                if action == "resume_here":
                    return "here"
                if action == "resume_back":
                    return "back"
                if action == "stop":
                    self._write_state(presentationStoppedBy="voice")
                    raise SlidenCancelled("presentation was stopped by voice")
                if action == "resume":
                    # The model was told to disambiguate and did not.  Someone
                    # looking at a slide who says "carry on" means this one.
                    return "here"
                if action == "pause":
                    # Not an answer -- give the room more time rather than
                    # deciding for it.
                    deadline = time.monotonic() + self._RESUME_CHOICE_SECONDS
                if gated and self._room_utterance_after(heard_since) is not None:
                    # Somebody answered.  Give the model the one turn it needs
                    # to hear them and call presentation_control.
                    heard_since = time.time()
                    self._answer_room_question()
                self._serve_slide_request(self.presentation_controller)
                if self.console_slide != displayed and self.console_slide > 0:
                    # They moved the deck again; the question now stands about
                    # the new slide, so do not run the clock out on the old one.
                    displayed = self.console_slide
                    self._presentation_state(presentationResumeDisplayed=displayed)
                    deadline = time.monotonic() + self._RESUME_CHOICE_SECONDS
                time.sleep(0.05)
            # Nobody answered.  Carrying on from where the talk was is the
            # choice that keeps the script and the screen together, which is
            # the whole point of asking.
            return "back"
        finally:
            self.presentation_qa_open.clear()
            self.presentation_qa_speech.clear()
            self.muted.set()
            if gated and not (
                self.privacy_muted.is_set()
                or self.sourcevault_waiting.is_set()
                or self.local_handoff_active.is_set()
            ):
                self._send_vad_mode(True)
            self._write_state(muted=True, presentationQAOpen=False)

    # ------------------------------------------------------------------
    # Question modes: question_time / keywords / detect
    # ------------------------------------------------------------------

    def _set_question_mode(self, value: Any, *, source: str) -> str | None:
        """Switch how the talk takes questions; None for an unknown mode."""

        mode = normalise_question_mode(value)
        if mode is None:
            return None
        self.question_mode = mode
        self._write_state(
            questionMode=mode,
            questionModeLabel=QUESTION_MODE_LABELS[mode],
            questionModeSource=source,
            questionModeChangedAt=time.time(),
        )
        return mode

    def _question_mode_ready(self, mode: str) -> tuple[bool, str]:
        """Can *mode* actually work here?  (ok, reason)"""

        if mode == "keywords" and not self._keyword_spotter_ready():
            return False, "keyword_recognizer_not_ready"
        return True, ""

    def _effective_question_mode(self) -> str:
        """The mode in force: one that cannot work falls back to question_time
        rather than leaving a talk that nobody can interrupt *and* that never
        pauses for questions."""

        mode = self.question_mode
        return mode if self._question_mode_ready(mode)[0] else "question_time"

    def _talk_listening(self) -> bool:
        """Is the room being listened to for interruptions right now?"""

        return bool(
            self.presentation_narrating.is_set()
            and self._effective_question_mode() != "question_time"
            and not self.presentation_qa_open.is_set()
            and not self.stop_event.is_set()
        )

    def _control_listening(self) -> bool:
        """Is the room listened to (locally) for 「発表を止めて」 right now?

        In every question mode: stopping a talk is not a question, and in
        question-time mode nobody could stop one mid-slide."""

        return bool(
            self.presentation_narrating.is_set()
            and not self.presentation_qa_open.is_set()
            and not self.stop_event.is_set()
        )

    def _keyword_interrupt_armed(self) -> bool:
        return self.question_mode == "keywords" and self._talk_listening()

    def _arm_talk_listening(self) -> None:
        with self.talk_interrupt_lock:
            self.talk_interrupt = None
        self.talk_preroll.clear()
        self._detect_loud_seconds = 0.0
        self._detect_quiet_seconds = 0.0
        self.presentation_narrating.set()

    def _disarm_talk_listening(self) -> None:
        self.presentation_narrating.clear()
        with self.talk_interrupt_lock:
            # An interruption that landed as the slide ended belongs to no
            # narration; the next slide must not start by stopping.
            self.talk_interrupt = None

    def _observe_talk_audio(self, chunk: NDArray[np.float32]) -> None:
        """Listen to the room during narration.  Nothing is sent anywhere."""

        self.talk_preroll.append(np.asarray(chunk, dtype=np.float32).copy())
        # The recogniser always listens (it also hears 「発表を止めて」);
        # what it may interrupt for depends on the mode (_on_voice_keyword).
        spotter = self.voice_commands
        if spotter is not None:
            spotter.feed(chunk)
        if self._effective_question_mode() != "detect":
            return
        data = np.asarray(chunk, dtype=np.float32)
        frames = int(data.shape[0]) if data.ndim else 0
        if frames <= 0:
            return
        seconds = frames / float(VRCHAT_RATE)
        level = float(np.sqrt(np.mean(np.square(data))))
        if level >= float(getattr(self.args, "interrupt_rms", 0.02)):
            self._detect_loud_seconds += seconds
            self._detect_quiet_seconds = 0.0
        else:
            self._detect_quiet_seconds += seconds
            # A breath between words is not the end of speaking.
            if self._detect_quiet_seconds >= 0.3:
                self._detect_loud_seconds = 0.0
        needed = float(getattr(self.args, "interrupt_ms", 450)) / 1000.0
        if self._detect_loud_seconds >= needed:
            self._detect_loud_seconds = 0.0
            self._request_talk_interrupt("detect", "")

    def _request_talk_interrupt(self, kind: str, heard: str, *, force: bool = False) -> bool:
        """Ask the narration in progress to stop for questions (or, with
        *force*, for a pause asked by voice -- in any question mode)."""

        if not (self._control_listening() if force else self._talk_listening()):
            return False
        if not force and time.monotonic() < self.talk_interrupt_quiet_until:
            self._write_state(talkInterruptIgnored=f"{kind}:cooldown")
            return False
        with self.talk_interrupt_lock:
            if self.talk_interrupt is not None:
                return False
            self.talk_interrupt = {"kind": kind, "heard": heard[:120], "at": time.time()}
        self.talk_interrupts += 1
        self._write_state(
            talkInterrupts=self.talk_interrupts,
            lastTalkInterrupt={"kind": kind, "heard": heard[:120], "at": time.time()},
        )
        return True

    def _cut_presentation_audio(self) -> None:
        """Stop the narration the room cut into and wait until it is quiet.

        Waiting matters: a cancelled response still ends with response.done,
        and the bookkeeping that follows clears the input buffer -- which
        would throw away the utterance about to be replayed to the model."""

        with self.response_request_lock:
            self.response_create_requests = deque(
                request
                for request in self.response_create_requests
                if not str(
                    request.get("response", {})
                    .get("metadata", {})
                    .get("vrc_realtime_purpose", "")
                ).startswith("presentation_")
            )
            if not self.response_create_requests:
                self.response_create_pending.clear()
        deadline = time.monotonic() + 4.0
        last_cancel = 0.0
        while time.monotonic() < deadline:
            with self.output_purpose_lock:
                purpose = self.current_output_purpose
            in_progress = self.response_in_progress.is_set()
            speaking = in_progress or self.output_active.is_set()
            if (
                speaking
                and purpose.startswith("presentation_")
                and time.monotonic() - last_cancel >= 0.5
            ):
                if in_progress:
                    self._send_now({"type": "response.cancel"})
                if self.output_active.is_set():
                    self._interrupt_playback()
                last_cancel = time.monotonic()
            if not (
                speaking
                or self.response_create_inflight.is_set()
                or self.output_done.is_set()
            ):
                break
            time.sleep(0.05)
        self._write_state(presentationPhase="interrupted")

    def _spoken_so_far(self) -> str:
        """Roughly what the room heard of the narration that was cut.

        The transcript runs ahead of playback (it arrives with the audio,
        faster than real time), so it is cut to the share of the received
        audio that was actually played."""

        with self.state_lock:
            transcript = str(self.state.get("lastTranscript", ""))
        with self.playback_lock:
            item = self.last_item_id
            played = self.played_samples.get(item, 0)
            received = self.received_samples.get(item, 0)
        if not transcript or received <= 0:
            return transcript
        share = max(0.0, min(1.0, played / received))
        return transcript[: int(round(len(transcript) * share))]

    def _send_talk_preroll(self) -> None:
        """Replay the last seconds of room audio (the interrupting utterance)."""

        chunks = list(self.talk_preroll)
        self.talk_preroll.clear()
        if not chunks:
            return
        frames = [
            np.asarray(chunk, dtype=np.float32).reshape(len(chunk), -1)
            for chunk in chunks
            if len(chunk)
        ]
        if not frames:
            return
        channels = min(frame.shape[1] for frame in frames)
        mono = np.concatenate([frame[:, :channels] for frame in frames]).mean(
            axis=1, dtype=np.float32
        )
        if mono.size % 2:
            mono = mono[:-1]
        if mono.size == 0:
            return
        down = mono.reshape(-1, 2).mean(axis=1, dtype=np.float32)
        pcm = (np.clip(down, -1.0, 1.0) * 32767.0).astype("<i2")
        step = OPENAI_INPUT_RATE // 10
        for start in range(0, pcm.size, step):
            self._send_now(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm[start : start + step].tobytes()
                    ).decode("ascii"),
                }
            )
        self._write_state(talkPrerollSeconds=round(pcm.size / OPENAI_INPUT_RATE, 2))

    def _interrupt_rule(self, interrupt: _TalkInterrupted | None) -> str:
        """The first rule of a question window opened by an interruption."""

        if interrupt is None:
            return ""
        if interrupt.kind == "control":
            return (
                "0. 聴衆の依頼で発表を一時停止しました。最初に届く音声がその依頼です。"
                "『了解です』とだけ言ってください。再開を求められたら presentation_control"
                " を action=resume で、終了を求められたら action=stop で呼んでください。\n"
            )
        # The room was not being asked for questions: whatever interrupted
        # the talk has already been said, and the model hears it first.
        # 「はい、どうぞ」 then answers nothing -- it invites the listener to
        # say again what they just said (live 2026-09-19: 「発表を停止して」
        # got 「はい、どうぞ」 and, straight after, 「発表を終了します」).  The
        # reply has to follow on from what was received: 「了解です」 when it
        # is understood, 「もう一度」 when it is not.
        replies = (
            "内容はもう受け取っているので、『はい、どうぞ』と促し直さず、"
            "受け取った内容に続けて次のとおり応じてください。\n"
            "- 発表の一時停止・再開・終了やスライド移動などの依頼で、内容が分かったとき:"
            "『了解です』とだけ言ってから presentation_control や show_slide を呼ぶ"
            "(結果を伝えるときに『了解です』を繰り返さない)。\n"
            "- 質問のとき: 前置きを付けずにそのまま答える。\n"
            "- 質問してよいかを尋ねただけで中身がまだ無いとき(『質問いいですか』など)に"
            "限り、『はい、どうぞ』とだけ言う。\n"
        )
        if interrupt.kind == "detect":
            return (
                "0. 発表の読み上げ中に部屋から声が聞こえたので、読み上げを止めました。"
                "最初に届く音声がその発話です。独り言・雑談・物音など、あなた(発表者)"
                "に向けたものでなければ、何も言わずに黙ってください。"
                "あなたへの質問・依頼なら、" + replies
                + "- あなたに向けた発話だが聞き取れない、または何を求めているか判断"
                "できないとき:『すみません、もう一度お願いします』とだけ言う。\n"
            )
        words = "」「".join(self.interrupt_words)
        return (
            f"0. 発表の読み上げ中に聴衆が「{words}」を含む発話をしたので、"
            "読み上げを止めました。最初に届く音声がその発話です。" + replies
            + "- 聞き取れない、または何を求めているか判断できないとき:"
            "『すみません、もう一度お願いします』とだけ言う。\n"
        )

    def _segment_speaker(
        self, kind: str, script: str, run_id: str, seconds_hint: float
    ) -> Callable[[str | None, str | None, str], None]:
        """The speak callback for an opening / closing segment."""

        def speak(text: str | None, phrase: str | None, _said: str) -> None:
            if phrase is None:
                self._narrate_presentation_segment(
                    kind, text or script, run_id, seconds_hint
                )
            else:
                self._narrate_presentation_segment(
                    kind, text or script, run_id, seconds_hint, resume_phrase=phrase
                )

        return speak

    def _slide_speaker(
        self,
        image: Path,
        number: int,
        count: int,
        seconds: float,
        narration_instructions: str,
        run_id: str,
    ) -> Callable[[str | None, str | None, str], None]:
        """The speak callback for one slide."""

        def speak(text: str | None, phrase: str | None, said: str) -> None:
            if phrase is None:
                self._narrate_presentation_slide(
                    image, number, count, seconds, narration_instructions, run_id,
                    script=text,
                )
            else:
                self._narrate_presentation_slide(
                    image, number, count, seconds, narration_instructions, run_id,
                    script=text, resume_phrase=phrase, already_said=said,
                )

        return speak

    _RESUME_PHRASE = "それでは、説明に戻ります"
    _RESUME_PHRASE_SPOKEN = re.compile(r"^.{0,24}?戻ります[。、!！]?")

    def _narrate_with_interruptions(
        self,
        run_id: str,
        label: str,
        speak: Callable[[str | None, str | None, str], None],
        *,
        script: str | None,
        slide_number: int | None = None,
    ) -> int | None:
        """Narrate, taking questions whenever the room cuts in.

        ``speak(text, resume_phrase, already_said)`` narrates *text* (the
        script, or what is left of it after an interruption; None without a
        script).  In question_time mode nothing listens, so this is one call.
        After questions the talk carries on from the sentence it was stopped
        in.  Returns the slide to carry on from when the questions moved the
        deck and the room chose to stay there, else None.
        """

        remaining = script
        already_said = ""
        resume_phrase: str | None = None
        rounds = 0
        while True:
            self._arm_talk_listening()
            try:
                speak(remaining, resume_phrase, already_said)
                return None
            except _TalkInterrupted as interrupted:
                caught = interrupted
                heard = self._spoken_so_far()
            finally:
                self._disarm_talk_listening()
            rounds += 1
            if resume_phrase:
                heard = self._RESUME_PHRASE_SPOKEN.sub("", heard, count=1)
            if remaining:
                remaining = resume_text(remaining, heard)
            else:
                already_said = (already_said + heard)[-1200:]
            console_before = self.console_slide
            self._write_state(
                presentationInterruptKind=caught.kind,
                presentationInterruptHeard=caught.heard or None,
                presentationInterruptSpoken=heard[-200:],
                presentationInterruptRemainingChars=(
                    len(remaining) if remaining is not None else None
                ),
            )
            outcome = self._presentation_question_window(
                run_id,
                f"{label}:interrupt{rounds}",
                float(getattr(self.args, "interrupt_resume_seconds", 4.0)),
                interrupt=caught,
            )
            self.talk_interrupt_quiet_until = time.monotonic() + float(
                getattr(self.args, "interrupt_cooldown_seconds", 8.0)
            )
            if outcome == "stop":
                raise SlidenCancelled("presentation was stopped by voice")
            controller = self.presentation_controller
            if slide_number is not None:
                moved = self._resume_after_slide_detour(
                    controller, slide_number, run_id
                )
                if moved is not None:
                    return moved
            elif console_before > 0 and self.console_slide != console_before:
                self._move_console_to(controller, console_before)
            if script and not (remaining or "").strip():
                # Cut in the last breath of the slide: nothing is left to say.
                return None
            resume_phrase = self._RESUME_PHRASE if self.question_window_spoke else ""
            self._presentation_state(presentationPhase="narrating")

    def _presentation_cancelled(self) -> bool:
        return self.stop_event.is_set() or self.presentation_stop_event.is_set()

    def _speak_presentation_line(self, purpose: str, text: str) -> None:
        """Say one fixed short line (e.g. "それでは再開します") and wait for it.

        Failures here are swallowed: losing the transition line is not a reason
        to abandon a presentation that is otherwise fine."""

        _clear_queue(self.presentation_audio_done)
        self._request_response(
            purpose=purpose,
            instructions=(
                f"日本語で「{text}」とだけ発話してください。"
                "他のことは何も言わず、質問もしないでください。"
            ),
        )
        try:
            self._await_presentation_audio(purpose, 30.0)
        except SlidenCancelled:
            raise
        except SlidenError:
            pass

    def _presentation_question_window(
        self,
        run_id: str,
        label: str,
        seconds: float,
        *,
        final: bool = False,
        hard_cap_seconds: float | None = None,
        interrupt: _TalkInterrupted | None = None,
    ) -> str:
        """Open the mic for questions.  Returns "continue" or "stop".

        The window is the only point in a presentation where the room can speak
        to the bot, so it is also where pause/stop/resume are heard.  It closes
        the mic again on every exit path -- an open mic during narration makes
        the bot answer its own voice.

        *interrupt* is set when the room cut into a slide (question modes
        keywords / detect): the utterance that did it is replayed to the model
        first, and no "それでは再開します" is said on the way out -- the
        resumed narration says it."""

        if seconds <= 0.0:
            return "continue"
        phase = (
            "final_questions" if final
            else "interrupt_questions" if interrupt is not None
            else "question_window"
        )
        answered = 0
        paused = False
        self.question_window_spoke = False
        self._presentation_state(presentationPhase=phase)
        self._announce(
            f"presentation:qa:{label}",
            "質問をどうぞ" if interrupt is not None else "質問があればどうぞ",
            min_interval=0.0,
        )
        # The strict nameplate/mouth gate is meant for starting a conversation
        # from across a room; during a talk the audience is not framed that way.
        self._activate_visual_session("presentation_question_window")
        _clear_queue(self.presentation_qa_done)
        self.presentation_qa_speech.clear()
        self.presentation_qa_open.set()
        self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                self._interrupt_rule(interrupt)
                                + "ここから短い質疑応答です。次の規則を守ってください。\n"
                                "1. 質問には簡潔に、日本語で答える。長広舌にしない。\n"
                                "2. 発表資料や保存資料に関する事実を問われたら、"
                                "推測で答えず必ず sourcevault_lookup を先に呼ぶ"
                                "(高速な事前索引が最初に引かれる)。"
                                + (
                                    "そこで見つからないときに限り web_search を使う。"
                                    if self.args.enable_voice_web_search
                                    else "そこで見つからなければ、分からないと正直に言う。"
                                )
                                + "\n"
                                "3. 『一時停止して』『中断して』『停止して』『止めて』は "
                                "presentation_control を action=pause で、"
                                "『発表を終了して』『打ち切って』『もうやめて』は "
                                "action=stop で、『再開して』は action=resume で呼ぶ。"
                                "終了か一時停止か迷うときは pause にする。\n"
                                "3b. スライドを動かしたあとで再開位置を尋ねられたら、"
                                "表示中のスライドから続けるなら action=resume_here、"
                                "発表が進んでいた位置へ戻るなら action=resume_back。\n"
                                "4. この質疑応答の時間に誰も何も言わなければ、"
                                "何も話さずに待ってください。"
                                "『質問はありますか』と自分から尋ねてはいけません。"
                                "この決まりは質疑応答の時間が終われば解除されます。"
                            ),
                        }
                    ],
                },
            }
        )
        # Nothing the avatar says at a slide boundary was asked for.  The
        # server VAD was creating responses on its own, so any energy in the
        # room -- world ambience, a footstep, a breath -- became a turn, and
        # the model, primed by the rules just sent, filled that turn with
        # 「何か質問はありますか」 and then 「それでは再開します」.  Live
        # 2026-09-05: the talk interrupted and restarted itself repeatedly
        # while nobody had said a word.  A threshold cannot fix that -- even a
        # perfectly tuned one lets a cough become a question -- because the
        # decision was being made on energy alone.  So automatic responses go
        # off for the whole window, and the one thing that can create a turn
        # is the local recogniser putting words to what it heard.
        # The gate needs a recogniser that is actually running.  Without one
        # there is no local way to tell a question from a noise, so the old
        # VAD behaviour stays rather than the talk going deaf -- and which of
        # the two is in force is reported, not left to be guessed at.
        gated = self._local_speech_gate_ready()
        if gated:
            self._send_vad_mode(False)
            self.input_committed_at = 0.0
        self._write_state(presentationQuestionGate="local_speech" if gated else "vad")
        if interrupt is not None:
            # The utterance that stopped the talk goes first: "スライドを
            # 戻して" has to be acted on, not asked for again.
            self.input_committed_at = 0.0
            self._send_talk_preroll()
        self.muted.clear()
        self._write_state(muted=False, presentationQAOpen=True)
        try:
            if interrupt is not None and gated:
                # Automatic responses are off; the replayed utterance is the
                # turn this window was opened for.  detect waits longer: the
                # speaker may still be mid-sentence.
                if self._answer_room_question(
                    settle_seconds=10.0 if interrupt.kind == "detect" else 4.0
                ):
                    answered += 1
                    self._write_state(presentationQuestionsAnswered=answered)
            # Only utterances from now on are this window's business.
            heard_since = time.time()
            deadline = time.monotonic() + seconds
            # A pause nobody lifts, or an endless run of questions, must not
            # hold a presentation open forever; the room may simply have left.
            hard_deadline = time.monotonic() + (
                hard_cap_seconds
                if hard_cap_seconds is not None
                else max(seconds * 20.0, 1800.0)
            )
            while True:
                if self._presentation_cancelled():
                    return "stop"
                if time.monotonic() >= hard_deadline:
                    self._write_state(presentationStoppedBy="question_window_timeout")
                    return "stop"
                action = self._take_presentation_control()
                if action == "stop":
                    self._write_state(presentationStoppedBy="voice")
                    return "stop"
                if action == "pause":
                    paused = True
                    self._presentation_state(presentationPhase="paused")
                    self._write_state(presentationPaused=True)
                elif action == "resume" and paused:
                    paused = False
                    self._presentation_state(presentationPhase=phase)
                    self._write_state(presentationPaused=False)
                    deadline = time.monotonic() + seconds
                self._serve_slide_request(self.presentation_controller)
                if gated:
                    heard = self._room_utterance_after(heard_since)
                    # A VAD onset on its own says only that the room made a
                    # sound.  Clear it either way: unanswered, it must not
                    # leak into the next window.  And fall through to the
                    # deadline below when nothing was said -- a window that
                    # only ever "continue"s never expires and never sleeps.
                    self.presentation_qa_speech.clear()
                    if heard is not None:
                        heard_since = time.time()
                        self._write_state(presentationLastQuestionHeard=heard[:120])
                        if self._answer_room_question():
                            answered += 1
                            self._write_state(presentationQuestionsAnswered=answered)
                            deadline = time.monotonic() + seconds
                        continue
                elif self.presentation_qa_speech.is_set():
                    self.presentation_qa_speech.clear()
                    # A tool-assisted answer can legitimately take a while; the
                    # window must not close under the answer it is waiting for.
                    self._wait_for_answer(180.0)
                    answered += 1
                    self._write_state(presentationQuestionsAnswered=answered)
                    deadline = time.monotonic() + seconds
                    continue
                if not paused and time.monotonic() >= deadline:
                    return "continue"
                time.sleep(0.05)
        finally:
            self.presentation_qa_open.clear()
            self.presentation_qa_speech.clear()
            self.muted.set()
            if gated and not (
                self.privacy_muted.is_set()
                or self.sourcevault_waiting.is_set()
                or self.local_handoff_active.is_set()
            ):
                self._send_vad_mode(True)
            self._write_state(
                muted=True, presentationQAOpen=False, presentationPaused=False
            )
            if not final:
                self._end_question_context()
            if (
                answered > 0
                and not self._presentation_cancelled()
                and not final
                and interrupt is None
            ):
                # Never let the transition line change the window's outcome:
                # raising out of a finally block would swallow the return value.
                try:
                    self._speak_presentation_line(
                        f"presentation_resume:{run_id}:{label}", "それでは再開します"
                    )
                except (SlidenCancelled, SlidenError):
                    pass

    #: Characters of recognised speech, ignoring out-of-vocabulary tokens,
    #: below which an utterance is treated as noise rather than a question.
    #: One stray syllable off a footstep is not somebody asking something.
    _QUESTION_MINIMUM_CHARACTERS = 2

    def _end_question_context(self) -> None:
        """Close the question context explicitly, without a word to the room.

        The rules item of a window ends with "say nothing if there is no
        question", and left as the last thing said to the model it silences
        the next slide: slide 2 was skipped in silence and the talk ran on to
        the closing (observed 2026-08-15).  A Realtime conversation item is
        never spoken by itself; the GPT-Live bridge overrides this, because
        there the note was answered aloud (「質問時間は終わりました」 after
        every slide, live 2026-09-19).
        """

        self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "質疑応答はここまでです。直前の質疑応答の規則は"
                                "もう適用されません。ここから発表の続きに戻ります。"
                                "次に渡されるスライドと原稿は必ず発話してください。"
                                "この文には返事をせず、何も話さないでください。"
                            ),
                        }
                    ],
                },
            }
        )

    def _keyword_spotter_ready(self) -> bool:
        """Is the local (Vosk) recogniser running?"""

        listener = self.voice_commands
        if listener is None:
            return False
        ready = getattr(listener, "ready", None)
        return bool(ready is not None and ready.is_set())

    def _local_speech_gate_ready(self) -> bool:
        """Can this process tell speech from noise without asking OpenAI?

        (GPT-Live answers only what it judges to be addressed to it and has
        no automatic-response switch, so its bridge does not gate.)"""

        return self._keyword_spotter_ready()

    def _room_utterance_after(self, since: float) -> str | None:
        """What the local recogniser heard since *since*, if it was words.

        The server VAD fires on energy; it cannot tell a question from a door
        closing.  The Vosk recogniser already running for the join/leave
        keywords transcribes the same audio locally, and an utterance it can
        put words to is the evidence that somebody actually said something.
        """

        listener = self.voice_commands
        if listener is None:
            return None
        heard_at = float(getattr(listener, "last_text_at", 0.0) or 0.0)
        if heard_at <= since:
            return None
        text = str(getattr(listener, "last_text", "") or "")
        # "[unk]" is Vosk saying it heard *something* it has no word for --
        # exactly what noise produces, so it cannot count towards the length.
        words = "".join(
            part for part in text.replace("[unk]", " ").split() if part
        )
        if len(words) < self._QUESTION_MINIMUM_CHARACTERS:
            return None
        return text.strip()

    def _answer_room_question(self, settle_seconds: float = 2.5) -> bool:
        """Turn the utterance the room just made into an answered turn.

        Automatic responses are off for the whole window, so nothing has been
        said back yet; the audio itself is already on its way to OpenAI, which
        commits it as an input item.  Wait for that commit, then ask for the
        one response this question is owed.  False when the audio never
        arrived -- there is then nothing to answer and nothing to say.
        """

        deadline = time.monotonic() + settle_seconds
        while self.input_committed_at <= 0.0:
            if self._presentation_cancelled():
                return False
            if time.monotonic() >= deadline:
                self._write_state(presentationQuestionUncommitted=True)
                return False
            time.sleep(0.05)
        self.input_committed_at = 0.0
        _clear_queue(self.presentation_qa_done)
        self._request_response(purpose="question_answer")
        self._wait_for_answer(180.0)
        return True

    def _wait_for_answer(self, budget_seconds: float) -> None:
        """Wait until the answer to a question has finished playing."""

        deadline = time.monotonic() + budget_seconds
        while time.monotonic() < deadline:
            if self._presentation_cancelled():
                return
            with self.presentation_control_lock:
                if self.presentation_control_request is not None:
                    return
            try:
                self.presentation_qa_done.get(timeout=0.2)
            except queue.Empty:
                continue
            # Give the model a moment to start a follow-up tool round; if it
            # does, another answer marker arrives and we wait again.
            if self.response_in_progress.is_set() or self.sourcevault_waiting.is_set():
                continue
            return

    @staticmethod
    def _presentation_image_data_uri(path: Path) -> tuple[str, int, int]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SlidenError(f"could not read extracted slide image: {path}")
        height, width = bgr.shape[:2]
        if width > 1920:
            scale = 1920.0 / width
            bgr = cv2.resize(
                bgr,
                (1920, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        output_height, output_width = bgr.shape[:2]
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            raise SlidenError("OpenCV could not encode an extracted slide")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
        return data_uri, output_width, output_height

    def _await_presentation_audio(
        self, purpose: str, budget_seconds: float, *, interruptible: bool = False
    ) -> None:
        deadline = time.monotonic() + budget_seconds
        while time.monotonic() < deadline:
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            if interruptible:
                with self.talk_interrupt_lock:
                    interrupt = self.talk_interrupt
                    self.talk_interrupt = None
                if interrupt is not None:
                    self._cut_presentation_audio()
                    raise _TalkInterrupted(
                        str(interrupt.get("kind", "")), str(interrupt.get("heard", ""))
                    )
            try:
                completed = self.presentation_audio_done.get(timeout=0.2)
            except queue.Empty:
                continue
            if completed == purpose:
                with self.state_lock:
                    transcript = str(self.state.get("lastTranscript", ""))
                self._write_state(presentationLastTranscript=transcript)
                return
        raise SlidenError(f"timed out waiting for narration ({purpose})")

    def _narrate_presentation_segment(
        self,
        kind: str,
        text: str,
        run_id: str,
        seconds_hint: float,
        *,
        resume_phrase: str | None = None,
    ) -> None:
        """Speak the opening or the closing -- the parts of a talk that belong
        to no single slide.  Without these the deck is a sequence of slide
        descriptions rather than a presentation, which is the whole point of
        carrying a scenario.

        *resume_phrase* is not None when a question interrupted the segment:
        *text* is then what is left of it, and the phrase (possibly empty) is
        said first."""

        self._ensure_vrchat_microphone_live()
        purpose = f"presentation_{kind}:{run_id}"
        resumed = resume_phrase is not None
        _clear_queue(self.presentation_audio_done)
        ready_deadline = time.monotonic() + 30.0
        while not self.ws_ready.is_set():
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            if time.monotonic() >= ready_deadline:
                raise SlidenError("Realtime connection was not ready for narration")
            time.sleep(0.1)
        label = "導入" if kind == "opening" else "締め"
        sent = self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                (
                                    f"発表の{label}のうち、質疑で中断した箇所から"
                                    "後の原稿です。\n---\n"
                                    if resumed
                                    else f"これから行う発表の{label}の原稿です。\n---\n"
                                )
                                + f"{text}\n---"
                            ),
                        }
                    ],
                },
            }
        )
        if not sent:
            raise SlidenError(f"the {kind} script could not be sent to Realtime")
        self._request_response(
            purpose=purpose,
            instructions=(
                self._resume_lead(resume_phrase)
                + f"上の{label}の原稿の内容だけを、自然な発表口調の日本語で話してください。"
                "原稿に無い事実を足さず、読み上げ調にせず、記号やURLは読まないでください。"
                "話し終えたら、次へ進む合図や質問をせずに停止してください。"
            ),
        )
        self._await_presentation_audio(
            purpose, max(90.0, seconds_hint * 5.0), interruptible=True
        )

    @staticmethod
    def _resume_lead(resume_phrase: str | None) -> str:
        """The instruction prefix for narration resumed after questions."""

        if resume_phrase is None:
            return ""
        if resume_phrase:
            return (
                f"質疑で中断していた説明を再開します。最初に「{resume_phrase}」と"
                "一言だけ言い、すでに話した部分は繰り返さずに続けてください。"
            )
        return "中断していた説明を、すでに話した部分を繰り返さずに再開してください。"

    def _narrate_presentation_slide(
        self,
        path: Path,
        slide_number: int,
        slide_count: int,
        seconds_per_slide: float,
        narration_instructions: str,
        run_id: str,
        script: str | None = None,
        *,
        resume_phrase: str | None = None,
        already_said: str = "",
    ) -> None:
        # A narrated slide nobody can hear is the same failure as a slide that
        # was never narrated, and from inside this process the two look
        # identical.  Check before speaking, and unmute if VRChat says it is
        # muted -- but never on a guess, because the control is a toggle.
        self._ensure_vrchat_microphone_live()
        purpose = f"presentation_slide:{run_id}:{slide_number}"
        _clear_queue(self.presentation_audio_done)
        ready_deadline = time.monotonic() + 30.0
        while not self.ws_ready.is_set():
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            if time.monotonic() >= ready_deadline:
                raise SlidenError("Realtime connection was not ready for narration")
            time.sleep(0.1)
        if resume_phrase is not None:
            # Questions interrupted this slide.  The image is already in the
            # conversation; what is needed is where the talk had got to.
            if script:
                body = (
                    f"第{slide_number}枚のスライドの発表原稿のうち、質疑で中断した"
                    "箇所から後の部分です。\n---\n"
                    f"{script}\n---"
                )
                instructions = (
                    self._resume_lead(resume_phrase)
                    + "上の原稿の残りだけを、自然な発表口調の日本語で話してください。"
                    "原稿に書かれていない事実・数値・引用を足さないでください。"
                    "箇条書き記号・記号列・URLは読み上げないでください。"
                    "話し終えたら次へ進む合図や質問をせず停止してください。"
                )
            else:
                body = (
                    f"第{slide_number}枚のスライド(先ほど添付した画像)の説明を"
                    "質疑で中断しました。中断までに話した内容は次のとおりです。\n---\n"
                    f"{already_said.strip() or '(ほとんど話していません)'}\n---"
                )
                instructions = (
                    self._resume_lead(resume_phrase)
                    + "このスライドの説明の残りだけを、日本語で短く話してください。"
                    "画像にない事実を作らず、話し終えたら次へ進む合図や質問をせず"
                    "停止してください。"
                )
            if narration_instructions.strip():
                instructions += "\n追加指示: " + narration_instructions.strip()
            if not self._send_now(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": body}],
                    },
                }
            ):
                raise SlidenError("the resumed script could not be sent to Realtime")
            self._request_response(purpose=purpose, instructions=instructions)
            self._await_presentation_audio(
                purpose, max(60.0, seconds_per_slide * 5.0), interruptible=True
            )
            return
        data_uri, width, height = self._presentation_image_data_uri(path)
        self._write_state(presentationReferenceImageSize=[width, height])
        sent = self._send_now(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Sliden用MP4から元解像度で抽出した発表スライドです。"
                                f"全{slide_count}枚中の第{slide_number}枚です。"
                                "VRChat内コンソールの縮小プレビューではなく、"
                                "添付した画像だけを発表内容の根拠にしてください。"
                            )
                            + (
                                # The image still goes with the script: it is
                                # what keeps the spoken part and the slide on
                                # screen in sync, and it grounds the wording.
                                "\nこのスライドの発表原稿は次のとおりです。"
                                "この原稿に沿って話してください。\n---\n"
                                f"{script}\n---"
                                if script
                                else ""
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": data_uri,
                            "detail": "high",
                        },
                    ],
                },
            }
        )
        if not sent:
            raise SlidenError("the extracted slide could not be sent to Realtime")
        if script:
            instructions = (
                f"上の発表原稿の内容だけを、自然な発表口調の日本語で"
                f"{seconds_per_slide:g}秒前後を目安に話してください。"
                "原稿に書かれていない事実・数値・引用を足さないでください。"
                "一字一句の読み上げにはせず、話し言葉として自然に整えてください。"
                "箇条書き記号・記号列・URLは読み上げないでください。"
                "この1枚の分だけを話し、話し終えたら次へ進む合図や質問をせず停止してください。"
            )
        else:
            instructions = (
                f"添付スライドを日本語で{seconds_per_slide:g}秒前後を目安に発表してください。"
                "画像に実際に書かれている内容を正確に読み、単なる全文読み上げではなく、"
                "聴衆に分かる自然な説明にしてください。画像にない事実を作らないでください。"
                "この1枚の説明だけを行い、説明を終えたら次へ進む合図や質問をせず停止してください。"
            )
        if narration_instructions.strip():
            instructions += "\n追加指示: " + narration_instructions.strip()
        self._request_response(purpose=purpose, instructions=instructions)
        self._await_presentation_audio(
            purpose, max(90.0, seconds_per_slide * 5.0), interruptible=True
        )

    def _run_presentation(self, command: dict[str, Any]) -> None:
        url = str(command.get("url", "")).strip()
        presentation_title = str(command.get("presentationTitle", "")).strip() or None
        seconds_per_slide = float(command.get("secondsPerSlide", 30.0))
        console_timeout = float(command.get("consoleTimeoutSeconds", 30.0))
        # "auto" (default): look for the panel from where the avatar stands and
        # only walk to it when it is not already usable.  True always walks,
        # False never does (and fails if the panel is not already in front).
        navigate_mode = _navigation_mode(command.get("navigateToConsole", "auto"))
        navigation_timeout = float(command.get("navigationTimeoutSeconds", 120.0))
        download_timeout = float(command.get("downloadTimeoutSeconds", 120.0))
        maximum_megabytes = float(command.get("maxDownloadMegabytes", 256.0))
        narration_instructions = str(command.get("narrationInstructions", ""))
        talk = _normalise_talk(command.get("talk"))
        catalog_route = str(command.get("catalogRoute", "direct"))
        question_seconds = float(command.get("questionSeconds", 5.0))
        final_question_seconds = float(command.get("finalQuestionSeconds", 120.0))
        start_slide = int(command.get("startSlide", 1))
        requested_end = command.get("endSlide")
        end_slide = None if requested_end is None else int(requested_end)
        if command.get("questionMode") not in (None, "", "automatic", "Automatic"):
            self._set_question_mode(command.get("questionMode"), source="presentation")
        run_id = uuid.uuid4().hex[:12]
        was_muted = self.muted.is_set()
        follower = self.owner_follower
        follower_was_held = bool(
            follower is not None and getattr(follower, "conversation_hold", False)
        )
        workspace = Path(tempfile.mkdtemp(prefix="vrc-sliden-"))
        controller: SlidenConsoleController | None = None
        try:
            if not url.startswith(("http://", "https://")):
                raise SlidenError("slide URL must use HTTP or HTTPS")
            if not 3.0 <= seconds_per_slide <= 300.0:
                raise SlidenError("seconds per slide must be between 3 and 300")
            if not 3.0 <= console_timeout <= 180.0:
                raise SlidenError("console timeout must be between 3 and 180 seconds")
            if navigate_mode is None:
                raise SlidenError(
                    'navigateToConsole must be "auto", true or false'
                )
            if navigate_mode != "never" and not 10.0 <= navigation_timeout <= 300.0:
                raise SlidenError(
                    "navigation timeout must be between 10 and 300 seconds"
                )
            if not 5.0 <= download_timeout <= 600.0:
                raise SlidenError("download timeout must be between 5 and 600 seconds")
            if not 1.0 <= maximum_megabytes <= 1024.0:
                raise SlidenError("maximum download size must be between 1 and 1024 MB")
            if not 0.0 <= question_seconds <= 60.0:
                raise SlidenError("question window must be between 0 and 60 seconds")
            if not 0.0 <= final_question_seconds <= 1800.0:
                raise SlidenError(
                    "final question window must be between 0 and 1800 seconds"
                )
            self._take_presentation_control()
            self.muted.set()
            self._write_state(
                muted=True,
                presentationStatus="running",
                presentationPhase="preparing",
                presentationTitle=presentation_title,
                presentationURL=url,
                presentationCurrentSlide=0,
                presentationSlideCount=0,
                presentationStartSlide=start_slide,
                presentationEndSlide=end_slide,
                presentationDownloadedBytes=0,
                presentationExtractionProgress=0.0,
                presentationConsoleInputMode=None,
                presentationConsoleAimError=None,
                presentationConsoleAimErrorX=None,
                presentationConsoleAimErrorY=None,
                presentationConsoleAimGainX=None,
                presentationConsoleAimGainY=None,
                presentationConsoleAimIterations=0,
                presentationConsoleAimLost=0,
                presentationConsoleAimProbe=None,
                presentationConsoleAimQuantized=False,
                presentationConsoleAimStalled=False,
                presentationConsoleSearchPitch=0.0,
                presentationConsoleClickAttempt=None,
                presentationConsoleClearDetected=None,
                presentationConsoleClearMethod=None,
                presentationConsoleTarget=None,
                presentationConsoleReticle=None,
                presentationNavigationEnabled=(navigate_mode != "never"),
                presentationNavigationMode=navigate_mode,
                presentationNavigationStatus=(
                    "pending" if navigate_mode != "never" else "disabled"
                ),
                presentationNavigationScanStep=0,
                presentationNavigationSweep=0,
                presentationNavigationPanelScanStep=0,
                presentationNavigationForwardSeconds=0.0,
                presentationNavigationCandidateRejected=False,
                presentationSlidenScreenConfidence=0.0,
                presentationSlidenScreenBox=None,
                presentationSlidenScreenWidthFraction=0.0,
                presentationSlidenScreenCenterErrorX=None,
                presentationSlidenScreenCenterErrorY=None,
                presentationSlidenLandmarkReady=False,
                presentationNavigationRepositioned=False,
                presentationNavigationPitchAdjustment=0.0,
                presentationConsoleWidthFraction=0.0,
                presentationConsoleCenterErrorX=None,
                presentationConsoleDistanceReady=False,
                presentationLastPreviewDifference=None,
                presentationLastTranscript="",
                presentationLastError=None,
                # Without these, a flowless presentation is indistinguishable
                # from one whose scenario was never registered or was withheld.
                presentationCatalogRoute=catalog_route,
                presentationTalkAvailable=talk is not None,
                presentationTalkSlides=sorted(talk["slides"]) if talk else [],
                presentationTalkOpening=bool(talk and talk["opening"]),
                presentationTalkClosing=bool(talk and talk["closing"]),
                presentationScriptedSlides=[],
                presentationAdvanceDifferences=[],
                presentationUnconfirmedAdvances=0,
                presentationQAOpen=False,
                presentationQuestionsAnswered=0,
                presentationPaused=False,
                presentationLastControl=None,
                presentationStoppedBy=None,
                presentationQuestionSeconds=question_seconds,
                presentationFinalQuestionSeconds=final_question_seconds,
                presentationStartedAt=time.time(),
                presentationFinishedAt=None,
            )
            if follower is not None:
                follower.set_conversation_active(True)
            # Say so up front rather than at the end of a silent deck.  A
            # muted client is not a reason to refuse: the run is still worth
            # making, and the state carries the reason the room heard nothing.
            self._ensure_vrchat_microphone_live()
            with self.state_lock:
                microphone = self.state.get("vrchatMicMuted")
            self._write_state(
                presentationMicrophoneConfirmed=microphone is False,
                presentationMicrophoneState=(
                    "unknown" if microphone is None else
                    "muted" if microphone else "live"
                ),
            )
            slides = prepare_slide_images(
                url,
                workspace,
                timeout_seconds=download_timeout,
                max_bytes=int(maximum_megabytes * 1024 * 1024),
                cancelled=self._presentation_cancelled,
                state_callback=self._presentation_state,
            )
            slide_count = len(slides)
            self.presentation_slides = list(slides)
            if not 1 <= start_slide <= slide_count:
                raise SlidenError(
                    f"start slide {start_slide} is outside 1..{slide_count}"
                )
            actual_end = slide_count if end_slide is None else end_slide
            if not start_slide <= actual_end <= slide_count:
                raise SlidenError(
                    f"end slide {actual_end} is outside {start_slide}..{slide_count}"
                )
            self._write_state(
                presentationSlideCount=slide_count,
                presentationEndSlide=actual_end,
            )
            # Kept outside the workspace, which is deleted with the run: the
            # frames around a failed aim are exactly what is needed after one.
            import shutil

            diagnostics = Path(tempfile.gettempdir()) / "vrc-sliden-aim"
            shutil.rmtree(diagnostics, ignore_errors=True)
            controller = SlidenConsoleController(
                pid=self.args.vrchat_pid if self.args.vrchat_pid > 0 else None,
                timeout_seconds=console_timeout,
                # The aim and confirm loops sleep this between frames; the
                # camera answers a correction within a frame, so a quarter
                # second per iteration was mostly waiting.
                poll_seconds=0.12,
                cancelled=self._presentation_cancelled,
                state_callback=self._presentation_state,
                osc_host=self.args.osc_host,
                osc_port=self.args.osc_port,
                diagnostic_directory=diagnostics,
                text_locator=self._sliden_text_locator,
            )
            self._write_state(presentationDiagnosticDirectory=str(diagnostics))
            self.presentation_controller = controller
            self.presentation_talk = talk
            if navigate_mode != "never":
                in_view = None
                if navigate_mode == "auto":
                    # Asking first costs a few seconds and saves the whole walk
                    # when the avatar is already parked at the panel -- which is
                    # the common case, and used to be the *only* supported case.
                    self._presentation_state(
                        presentationPhase="checking_console",
                        presentationNavigationStatus="checking",
                    )
                    probe = getattr(controller, "console_in_view", None)
                    if probe is not None:
                        in_view = probe(
                            timeout_seconds=min(8.0, max(1.0, console_timeout))
                        )
                if in_view is not None:
                    self._presentation_state(
                        presentationNavigationStatus="already_in_front",
                    )
                else:
                    # Not respawning first: in this world the spawn point is in a
                    # different room from the presentation area, so it is a worse
                    # starting place than wherever the avatar already stands.  The
                    # respawn helper stays available for unsticking by hand.
                    self._presentation_state(
                        presentationNavigationStatus="navigating",
                    )
                    controller.locate_and_approach(
                        timeout_seconds=navigation_timeout,
                    )
            controller.configure_url(url)
            self.console_slide = 1
            self._write_state(presentationConsoleSlide=1)
            for skipped in range(1, start_slide):
                self._presentation_state(
                    presentationPhase="seeking_start_slide",
                    presentationCurrentSlide=skipped,
                )
                difference = controller.next_slide()
                self.console_slide += 1
                self._write_state(
                    presentationLastPreviewDifference=difference,
                    presentationConsoleSlide=self.console_slide,
                )
            ready, why = self._question_mode_ready(self.question_mode)
            self._write_state(
                presentationQuestionMode=self.question_mode,
                presentationQuestionModeEffective=(
                    self.question_mode if ready else "question_time"
                ),
                presentationQuestionModeReason=why or None,
            )
            if talk and talk["opening"]:
                self._presentation_state(presentationPhase="opening")
                self._narrate_with_interruptions(
                    run_id,
                    "opening",
                    self._segment_speaker(
                        "opening", talk["opening"], run_id, seconds_per_slide
                    ),
                    script=talk["opening"],
                )
            scripted: list[int] = []
            advance_differences: list[float] = []
            unconfirmed = 0
            # A while loop rather than a range: after questions the room may
            # choose to carry on from a slide it moved to, and the talk has to
            # be able to go there.
            slide_number = start_slide
            while slide_number <= actual_end:
                if self._presentation_cancelled():
                    raise SlidenCancelled("presentation was cancelled")
                entry = talk["slides"].get(slide_number) if talk else None
                script = entry["text"] if entry else None
                slide_seconds = (
                    entry["seconds"]
                    if entry and entry["seconds"] is not None
                    else seconds_per_slide
                )
                self._presentation_state(
                    presentationPhase="narrating",
                    presentationCurrentSlide=slide_number,
                )
                moved = self._narrate_with_interruptions(
                    run_id,
                    f"slide{slide_number}",
                    self._slide_speaker(
                        slides[slide_number - 1],
                        slide_number,
                        slide_count,
                        slide_seconds,
                        narration_instructions,
                        run_id,
                    ),
                    script=script,
                    slide_number=slide_number,
                )
                if script and slide_number not in scripted:
                    scripted.append(slide_number)
                    self._write_state(presentationScriptedSlides=list(scripted))
                if moved is not None:
                    # Questions in the middle of the slide moved the deck and
                    # the room chose to carry on from there.
                    slide_number = moved
                    continue
                if slide_number >= actual_end:
                    break
                # Questions are taken at the slide boundary, before the
                # console advances: answering about a slide the room can no
                # longer see is worse than a slightly longer pause.  Every
                # question mode takes it: cutting in mid-slide (keywords /
                # detect) is for the question that cannot wait, and without
                # the pause a question about what was just said had nowhere
                # to go -- the next slide began the moment the last one ended
                # (live, GPT-Live in keywords mode, 2026-09-19).
                if self._presentation_question_window(
                    run_id, f"slide{slide_number}", question_seconds
                ) == "stop":
                    raise SlidenCancelled("presentation was stopped by voice")
                # Those questions can move the deck.  Narrating slide N+1 over
                # slide N-3 is how the script and the screen came apart for the
                # rest of a run, so reconcile them here -- by asking, because
                # only the room knows whether it went back to re-hear a slide
                # or merely to glance at it.
                resumed = self._resume_after_slide_detour(
                    controller, slide_number, run_id
                )
                if resumed is not None:
                    slide_number = resumed
                    continue
                self._presentation_state(presentationPhase="advancing_console")
                difference = controller.next_slide()
                self.console_slide += 1
                self._write_state(presentationConsoleSlide=self.console_slide)
                advance_differences.append(round(float(difference), 5))
                if difference >= self._ADVANCE_CONFIRM_DIFFERENCE:
                    unconfirmed = 0
                else:
                    unconfirmed += 1
                self._write_state(
                    presentationLastPreviewDifference=difference,
                    presentationAdvanceDifferences=advance_differences[-12:],
                    presentationUnconfirmedAdvances=unconfirmed,
                )
                if unconfirmed >= self._ADVANCE_UNCONFIRMED_LIMIT:
                    raise SlidenError(
                        "the console did not visibly change slide on "
                        f"{unconfirmed} consecutive advances "
                        f"(last difference {difference:.5f}); the room is "
                        "not seeing the slide being narrated"
                    )
                slide_number += 1
            if talk and talk["closing"]:
                self._presentation_state(presentationPhase="closing")
                self._narrate_with_interruptions(
                    run_id,
                    "closing",
                    self._segment_speaker(
                        "closing", talk["closing"], run_id, seconds_per_slide
                    ),
                    script=talk["closing"],
                )
            if final_question_seconds > 0.0:
                self._speak_presentation_line(
                    f"presentation_final_qa:{run_id}",
                    "この発表に関して質問があればどうぞ",
                )
                self._presentation_question_window(
                    run_id, "final", final_question_seconds, final=True
                )
            self._presentation_state(
                presentationStatus="completed",
                presentationPhase="completed",
                presentationFinishedAt=time.time(),
            )
        except SlidenCancelled:
            self._presentation_state(
                presentationStatus="stopped",
                presentationPhase="stopped",
                presentationFinishedAt=time.time(),
            )
        except Exception as exc:
            self._write_state(
                presentationStatus="error",
                presentationPhase="error",
                presentationLastError=f"{type(exc).__name__}: {exc}",
                presentationFinishedAt=time.time(),
            )
            # Whoever is in the room watched the avatar stop doing anything.
            # Tell them why rather than leaving them to guess.
            with self.state_lock:
                where = str(self.state.get("presentationPhase") or "")
            self._announce(
                "presentation:error",
                f"発表を中断しました ({self._PRESENTATION_CHAT.get(where, where) or type(exc).__name__})",
                min_interval=0.0,
            )
        finally:
            self.presentation_controller = None
            self.presentation_slides = []
            with self.slide_request_lock:
                self.slide_request = None
            if controller is not None:
                controller.close()
            if follower is not None and not follower_was_held:
                follower.set_conversation_active(False)
            if not was_muted:
                self.muted.clear()
                self._write_state(muted=False)
            try:
                import shutil

                shutil.rmtree(workspace, ignore_errors=True)
            except Exception:
                pass
            try:
                self._presentation_finished()
            except Exception as exc:
                self._write_state(lastError=f"after presentation: {type(exc).__name__}: {exc}")

    def _presentation_finished(self) -> None:
        """Called once a talk has ended, however it ended.  (Realtime: its
        per-response instructions leave nothing behind to undo.)"""

    def _start_presentation(self, command: dict[str, Any]) -> bool:
        with self.presentation_lock:
            if self.presentation_thread is not None and self.presentation_thread.is_alive():
                self._write_state(
                    presentationLastError="a presentation is already running"
                )
                return False
            activate_mode = bool(
                command.get(
                    "activatePresentationMode",
                    command.get("navigateToConsole", False),
                )
            )
            if activate_mode:
                if self.owner_follower is not None:
                    self.owner_follower.set_mode("Presentation")
                else:
                    self._write_state(avatarMode="Presentation")
            self.presentation_stop_event.clear()
            # Named before the thread even starts: a second request for the
            # same talk can arrive within the same model response, and it
            # has to read as a repeat rather than "another presentation".
            self._write_state(
                presentationTitle=str(command.get("presentationTitle") or ""),
                presentationURL=str(command.get("url") or ""),
                presentationStatus="starting",
                presentationPhase="starting",
            )
            self.presentation_thread = threading.Thread(
                target=self._run_presentation,
                args=(dict(command),),
                daemon=True,
                name="VRCRealtimeSlidenPresentation",
            )
            self.presentation_thread.start()
            return True

    def _stop_presentation(self) -> None:
        with self.presentation_lock:
            thread = self.presentation_thread
            if thread is None or not thread.is_alive():
                self._write_state(
                    presentationStatus="idle",
                    presentationPhase="idle",
                )
                return
            self.presentation_stop_event.set()
            self._write_state(
                presentationStatus="stopping",
                presentationPhase="stopping",
            )
        with self.response_request_lock:
            self.response_create_requests = deque(
                request
                for request in self.response_create_requests
                if not str(
                    request.get("response", {})
                    .get("metadata", {})
                    .get("vrc_realtime_purpose", "")
                ).startswith("presentation_")
            )
            if not self.response_create_requests:
                self.response_create_pending.clear()
        with self.output_purpose_lock:
            presentation_audio_active = self.current_output_purpose.startswith(
                "presentation_"
            )
        if presentation_audio_active:
            self._send_now({"type": "response.cancel"})
            self._interrupt_playback()

    def _request_response(
        self, purpose: str = "normal", instructions: str | None = None
    ) -> None:
        """Queue a purpose-tagged response.create after the active response."""
        response: dict[str, Any] = {
            "metadata": {"vrc_realtime_purpose": purpose},
        }
        if instructions:
            response.update(
                {
                    "output_modalities": ["audio"],
                    "instructions": instructions,
                    "tool_choice": "none",
                }
            )
        request = {"type": "response.create", "response": response}
        with self.response_request_lock:
            self.response_create_requests.append(request)
            self.response_create_pending.set()

    def _response_create_worker(self) -> None:
        while not self.stop_event.wait(0.05):
            if not self.response_create_pending.is_set():
                continue
            if self.response_in_progress.is_set() or self.sourcevault_waiting.is_set():
                continue
            if self.response_create_inflight.is_set():
                continue
            with self.response_request_lock:
                if not self.response_create_requests:
                    self.response_create_pending.clear()
                    continue
                request = self.response_create_requests.popleft()
                self.inflight_response_request = request
                self.response_create_inflight.set()
                if not self.response_create_requests:
                    self.response_create_pending.clear()
            if not self._send_now(request):
                with self.response_request_lock:
                    self.response_create_requests.appendleft(request)
                    self.inflight_response_request = None
                    self.response_create_inflight.clear()
                    self.response_create_pending.set()

    def _presentation_start_instruction(
        self, entry: dict[str, Any], catalog_route: str
    ) -> str:
        """What to say once a presentation has been started.

        A registry that could not answer is worth a sentence: the deck still
        loads from the built-in table, but the *script* is gone, so the talk
        silently becomes slide-by-slide image description.  Degrading without
        saying so is how a stopped MCP service looked like a working one
        (2026-08-21).
        """

        opening = (
            f"{entry['title']}のプレゼンモードへ切り替え、"
            "スライドの準備を開始しました。完了したとは言わず、"
            "開始した事実だけを日本語で短く伝えてください。"
        )
        if catalog_route == "registry":
            return opening
        return opening + (
            "ただし発表登録簿を引けなかったため内蔵の一覧から開始しており、"
            "発表原稿は使えません。原稿なしで進める旨も一言添えてください。"
        )

    def _spoken_request_text(self) -> str:
        """What the room just said, when this bridge has it as text.

        The Realtime model calls tools on what it heard itself; nothing to
        cross-check here.  (The GPT-Live bridge returns the transcript.)"""

        return ""

    def _reconcile_title_number(self, title: Any, heard: str) -> Any:
        """Give the talk the number the room actually said.

        A tool call made by a model that did not hear the request itself
        (GPT-Live delegates to a backend) came with 「計算と自然31」 for a
        spoken 「計算と自然、三十三」 (live 2026-09-19), and the room got the
        wrong deck.  When the utterance names exactly one number and the
        title carries a different one (or none), the spoken one wins.
        """

        if not isinstance(title, str) or not heard:
            return title
        # Only a request that names the talk counts: 「十三のプレゼンをして
        # ください」, the tail of a split transcript, would otherwise turn 33
        # into 13 (live 2026-09-19).
        stem = title_stem(title)
        if not stem or stem not in title_stem(heard):
            return title
        spoken = sorted(set(spoken_numbers(heard)))
        if len(spoken) != 1:
            return title
        said = spoken[0]
        written = spoken_numbers(title)
        if written and written[-1] == said:
            return title
        text = unicodedata.normalize("NFKC", title).strip()
        corrected = (
            re.sub(r"\d+(?!.*\d)", str(said), text) if written else f"{text}{said}"
        )
        self._write_state(
            presentationTitleCorrected={"given": title, "heard": heard[-120:], "used": corrected}
        )
        return corrected

    def _presentation_start_command(
        self, title: Any
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Resolve a spoken title and build the presentation-start command.

        Shared by ``start_presentation`` and by ``set_avatar_mode`` when it is
        given a title, so whichever tool the model reaches for, the room gets
        the same behaviour.
        """

        if isinstance(title, str):
            title = normalise_title_numbers(title)
        title = self._reconcile_title_number(title, self._spoken_request_text())
        entry, catalog_route = self._resolve_presentation(title)
        command = {
            "command": "presentation-start",
            "presentationTitle": entry["title"],
            "url": entry["url"],
            "secondsPerSlide": entry["secondsPerSlide"],
            "startSlide": entry["startSlide"],
            "endSlide": entry["endSlide"],
            "narrationInstructions": entry["narrationInstructions"],
            "talk": entry.get("talk"),
            "catalogRoute": catalog_route,
            # Asked by voice, nothing guarantees the avatar is standing at the
            # panel.  "auto" checks first and only walks when it has to, so
            # "start the talk" works from anywhere in the room without a
            # pointless walk when it is already parked.
            "navigateToConsole": "auto",
            # The recipe (frame the wall screen, find the bar, walk at it,
            # about-face, back off) plus one retry on the other side needs
            # more than the two minutes that cut a live run off mid-retry
            # (2026-09-03); the world-join budget is 210 for the same walk.
            "navigationTimeoutSeconds": 180.0,
            "activatePresentationMode": True,
        }
        return entry, catalog_route, command

    def _presentation_duplicate_result(
        self, entry: dict[str, Any]
    ) -> tuple[dict[str, Any], str] | None:
        """The tool result for a talk that is already the running one.

        Asked to start a named talk, the model called set_avatar_mode with the
        title *and* start_presentation (live 2026-09-03).  The second call
        found the first one's thread alive, raised, and the room heard
        "already running, so it could not be started" -- about the talk it
        had just started.  A repeat of the running talk is a repeat, not a
        failure: the result says so and says nothing of failing.  ``None``
        when nothing is running or a different talk is.
        """

        if not self._presentation_running():
            return None
        with self.state_lock:
            running_title = str(self.state.get("presentationTitle") or "")
            running_url = str(self.state.get("presentationURL") or "")
            phase = str(self.state.get("presentationPhase") or "")
            navigation = str(self.state.get("presentationNavigationStatus") or "")
        same = bool(
            (running_title and running_title == str(entry.get("title", "")))
            or (running_url and running_url == str(entry.get("url", "")))
        )
        if not same:
            return None
        payload = {
            "status": "already_started",
            "mode": "Presentation",
            "title": entry["title"],
            "url": entry["url"],
            "phase": phase,
            "navigation": navigation,
        }
        instruction = (
            f"{entry['title']}の発表は既に開始済みで、いま操作パネルへの移動と"
            "準備を進めています。『開始できなかった』『既に実行中のため』とは"
            "決して言わず、『はい、準備を進めています』程度に日本語でごく短く"
            "答えてください。直前に同じことを伝えていれば何も言わなくて構いません。"
        )
        return payload, instruction

    def _start_presentation_from_tool(
        self,
        call_id: str,
        name: str,
        entry: dict[str, Any],
        catalog_route: str,
        command: dict[str, Any],
        extra: dict[str, Any],
    ) -> str:
        """Start *entry* for a tool call; returns what to say about it.

        Shared by ``start_presentation`` and ``set_avatar_mode`` with a title
        so both answer a repeat the same way.
        """

        if self._start_presentation(command):
            payload = {
                "status": "started",
                "mode": "Presentation",
                "title": entry["title"],
                "url": entry["url"],
                "navigation": "auto",
                "scripted": entry.get("talk") is not None,
                "catalog": catalog_route,
                "catalogError": self.state.get("presentationCatalogError"),
            }
            payload.update(extra)
            self._send_function_output(call_id, payload)
            self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=entry["title"])
            return self._presentation_start_instruction(entry, catalog_route)
        duplicate = self._presentation_duplicate_result(entry)
        if duplicate is None and self._presentation_running():
            # Another talk is running: asked for this one, the room wants to
            # switch.  Refusing left the wrong talk on and the room waiting
            # after 「了解です」 (live 2026-09-19).
            with self.state_lock:
                previous = str(self.state.get("presentationTitle") or "")
            self._write_state(presentationSwitching={"from": previous, "to": entry["title"]})
            self._request_presentation_control("stop")
            thread = self.presentation_thread
            if thread is not None:
                thread.join(timeout=20.0)
            if self._start_presentation(command):
                payload = {
                    "status": "switched",
                    "mode": "Presentation",
                    "stopped": previous,
                    "title": entry["title"],
                    "url": entry["url"],
                    "scripted": entry.get("talk") is not None,
                    "catalog": catalog_route,
                }
                payload.update(extra)
                self._send_function_output(call_id, payload)
                self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=entry["title"])
                return (
                    f"『{previous}』を終了し、{entry['title']}のプレゼンを開始しました。"
                    "完了したとは言わず、切り替えて準備を始めたことだけを日本語で短く"
                    "伝えてください。"
                )
        if duplicate is None:
            with self.state_lock:
                running = str(self.state.get("presentationTitle") or "")
            raise ValueError(
                f"別の発表『{running or '不明'}』が実行中です。新しい発表を始めるには、"
                "先に presentation_control の stop で終了してください。"
            )
        payload, instruction = duplicate
        payload = dict(payload)
        payload.update(extra)
        self._send_function_output(call_id, payload)
        self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=entry["title"])
        return instruction

    def _await_avatar_commands(self, timeout: float = 90.0) -> None:
        """Wait for the running avatar tool-call batch, if any."""

        thread = self.avatar_command_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _handle_avatar_command_calls(self, items: list[dict[str, Any]]) -> None:
        """Run avatar tool calls in the order the model asked for them."""

        for item in items:
            try:
                self._handle_avatar_command_call(item)
            except Exception as exc:  # never let one call kill the thread
                self._write_state(
                    lastError=f"avatar command: {type(exc).__name__}: {exc}"
                )

    def _handle_avatar_command_call(self, item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id", ""))
        name = str(item.get("name", ""))
        try:
            arguments = json.loads(str(item.get("arguments", "{}")))
        except json.JSONDecodeError:
            arguments = {}
        if not call_id:
            return
        follower = self.owner_follower
        if follower is None:
            self._send_function_output(
                call_id,
                {"status": "error", "message": "アバター制御は利用できません。"},
            )
            self._request_response()
            return
        try:
            if name == "set_avatar_mode":
                requested = str(arguments.get("mode", ""))
                mode_map = {
                    "standalone": "Standalone",
                    "owner_follow": "OwnerFollow",
                    "presentation": "Presentation",
                }
                if requested not in mode_map:
                    raise ValueError(
                        "mode must be standalone, owner_follow, or presentation"
                    )
                # Leaving presentation mode ends the presentation.  Continuing
                # to narrate while following someone around the room would be a
                # mode change in name only.
                if (
                    mode_map[requested] != "Presentation"
                    and self._presentation_running()
                ):
                    self._request_presentation_control("stop")
                    self._write_state(presentationStoppedBy="mode_change")
                canonical = follower.set_mode(mode_map[requested])
                # "Switch to presentation mode" and "start the talk" are the
                # same request to a person, and the model does conflate them:
                # asked for a named talk it called this tool alone and then
                # reported success, leaving the room staring at nothing
                # (2026-08-21).  Prompt text already told it not to; the fix
                # has to be here.  With a title, do the whole job.  Without
                # one, say plainly in the tool result that nothing has started,
                # so the next turn corrects itself instead of stalling.
                spoken_title = str(arguments.get("title", "") or "").strip()
                if canonical == "Presentation" and spoken_title:
                    entry, catalog_route, command = self._presentation_start_command(
                        spoken_title
                    )
                    response_instruction = self._start_presentation_from_tool(
                        call_id, name, entry, catalog_route, command,
                        {"mode": canonical},
                    )
                elif canonical == "Presentation" and not self._presentation_running():
                    self._write_state(presentationStatus="awaiting_title")
                    self._send_function_output(
                        call_id,
                        {
                            "status": "ok",
                            "mode": canonical,
                            "presentationRunning": False,
                            "next": (
                                "モードを変えただけで発表は始まっていません。"
                                "利用者が発表名を挙げていたなら、その名前を title に"
                                "そのまま入れて start_presentation を呼んでください。"
                            ),
                        },
                    )
                    self._write_state(
                        lastVoiceCommand=name, lastVoiceCommandValue=canonical
                    )
                    response_instruction = (
                        "プレゼンモードへ切り替えました。発表はまだ始まっていません。"
                        "利用者が発表名を挙げていたなら、いま start_presentation を"
                        "その名前で呼んでください。挙げていなければ、どの発表を"
                        "始めるか日本語で短く尋ねてください。"
                    )
                else:
                    self._send_function_output(
                        call_id,
                        {"status": "ok", "mode": canonical},
                    )
                    self._write_state(
                        lastVoiceCommand=name, lastVoiceCommandValue=canonical
                    )
                    response_instruction = (
                        f"アバターモードを{canonical}へ実際に切り替えました。"
                        "その事実だけを日本語で短く確認してください。"
                    )
            elif name == "start_presentation":
                entry, catalog_route, command = self._presentation_start_command(
                    arguments.get("title", "")
                )
                response_instruction = self._start_presentation_from_tool(
                    call_id, name, entry, catalog_route, command, {}
                )
            elif name == "presentation_control":
                action = self._request_presentation_control(
                    str(arguments.get("action", ""))
                )
                running = self._presentation_running()
                self._send_function_output(
                    call_id,
                    {"status": "ok", "action": action, "presentationRunning": running},
                )
                self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=action)
                response_instruction = {
                    "pause": (
                        "発表を一時停止しました。『再開して』と言われるまで待つこと、"
                        "終わりにするなら『発表を終了して』と言ってもらえばよいことを、"
                        "日本語で短く伝えてください。直前に『了解です』と言っていれば繰り返さないでください。"
                    ),
                    "resume": "発表を再開します。その事実だけを日本語でごく短く伝えてください。",
                    "stop": (
                        "発表を終了しました。その事実だけを日本語でごく短く伝えてください。"
                        "直前に『了解です』と言っていれば繰り返さないでください。"
                    ),
                    "resume_here": "いま表示しているスライドから発表を続けます。"
                    "その事実だけを日本語でごく短く伝えてください。",
                    "resume_back": "発表が進んでいた場所へ戻って続けます。"
                    "その事実だけを日本語でごく短く伝えてください。",
                }[action]
            elif name == "show_slide":
                if not self._presentation_running():
                    raise ValueError(
                        "スライドを操作できるのは発表の実行中だけです"
                    )
                target = self._resolve_slide_request(
                    arguments if isinstance(arguments, dict) else {}
                )
                if target is None:
                    raise ValueError(
                        "その話題に対応するスライドが見つかりませんでした"
                    )
                with self.slide_request_lock:
                    self.slide_request = target
                self._send_function_output(
                    call_id,
                    {"status": "ok", "slide": target, "from": self.console_slide},
                )
                self._write_state(
                    lastVoiceCommand=name, lastVoiceCommandValue=target
                )
                response_instruction = (
                    f"{target}枚目のスライドを表示しています。"
                    "表示した旨を日本語で短く伝え、そのスライドの内容を踏まえて"
                    "質問に答えてください。"
                )
            elif name == "turn_avatar":
                direction = follower.turn(str(arguments.get("direction", "")))
                self._send_function_output(
                    call_id,
                    {"status": "ok", "direction": direction},
                )
                self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=direction)
                response_instruction = (
                    "指定された旋回を実際に開始しました。"
                    "その事実だけを日本語で短く確認してください。"
                )
            elif name == "conversation_control":
                action = str(arguments.get("action", ""))
                if action == "join":
                    self.voice_keyword_left.clear()
                    self._set_state(voiceKeywordLeft=False)
                    self._activate_visual_session("voice_join_command")
                    self._send_function_output(
                        call_id,
                        {"status": "joined", "conversationActive": True},
                    )
                    response_instruction = (
                        "会話への参加を開始しました。日本語で短く確認してください。"
                    )
                elif action == "leave":
                    already_left = self.voice_keyword_left.is_set()
                    self._send_function_output(
                        call_id,
                        {"status": "left", "conversationActive": False},
                    )
                    # Latches exactly like the spoken keyword: only a join or a
                    # frontal greeting resumes.
                    self._leave_conversation("voice_leave_command", announce=False)
                    if already_left:
                        # The local keyword spotter already said goodbye for
                        # this very utterance; do not answer twice.
                        self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=action)
                        return
                    response_instruction = (
                        "会話から離脱しました。ごく短い別れの挨拶だけをしてください。"
                    )
                else:
                    raise ValueError("action must be join or leave")
                self._write_state(lastVoiceCommand=name, lastVoiceCommandValue=action)
            else:
                raise ValueError("unknown avatar command")
            self._request_response(
                purpose="avatar_voice_command",
                instructions=response_instruction,
            )
        except ValueError as exc:
            self._send_function_output(
                call_id,
                {"status": "error", "message": str(exc)},
            )
            # An unsteered response.create let the model answer a failed
            # start_presentation with "では準備しますね" and then do nothing,
            # which is indistinguishable from a working start until the room
            # has waited a few minutes (live 2026-09-04).  Say what failed and
            # forbid the reassuring version.
            self._request_response(
                purpose="avatar_voice_command_error",
                instructions=(
                    "直前の操作は失敗しました。理由: " + str(exc) + " "
                    "『始めます』『準備します』のように成功したかのようには決して"
                    "言わず、できなかった事実とこの理由を日本語でごく短く伝えて"
                    "ください。同じ操作をもう一度呼び直さないでください。"
                ),
            )

    def _addressed_frontally(self, grace_seconds: float = 3.0) -> bool:
        """Is somebody facing the avatar and speaking to it right now?

        Used to decide whether a spoken greeting counts as "正面から話しかけ
        られた".  Without a visual gate there is no frontal evidence to weigh,
        so the greeting itself is taken as the address.
        """

        gate = self.visual_gate
        if gate is None:
            return True
        try:
            if gate.allows_audio():
                return True
            if bool(getattr(gate, "attention_candidate", lambda: False)()):
                return True
            last_face = float(getattr(gate, "last_face_seen", 0.0) or 0.0)
        except Exception:
            return False
        return last_face > 0.0 and time.monotonic() - last_face <= max(1.0, grace_seconds)

    def _resume_after_leave(self, reason: str, instructions: str) -> None:
        """Lift the leave latch and open the conversation again."""

        self.voice_keyword_left.clear()
        self._set_state(voiceKeywordLeft=False)
        if self._visual_session_enabled():
            self._activate_visual_session(reason)
            # Anything said before the consent stays local.
            self.visual_audio_preroll.clear()
            self.visual_gate_audio_open = True
            self._set_state(visualGateAudioOpen=True)
        else:
            self._reset_input_gate()
        self._request_response(purpose=reason, instructions=instructions)

    def _apply_voice_control(self, action: str, heard: str, *, source: str) -> bool:
        """Pause / resume / stop the running talk because the room asked.

        Acted on here, not left to a model's tool call.  A pause during
        narration also cuts the narration at once; the question window it
        opens holds the talk until 「再開して」."""

        if action not in ("stop", "pause", "resume") or not self._presentation_running():
            return False
        if action == "resume":
            with self.state_lock:
                paused = bool(self.state.get("presentationPaused"))
            if not paused:
                return False
        self._write_state(
            lastVoiceControl={"action": action, "heard": heard[:120], "source": source,
                              "at": time.time()}
        )
        self._request_presentation_control(action)
        if action == "pause" and self.presentation_narrating.is_set():
            self._request_talk_interrupt("control", heard, force=True)
        if action == "stop" and source == "keyword":
            # Heard only by the local recogniser: the model knows nothing of
            # it, so say what happened.
            self._request_response(
                purpose="presentation_stopped",
                instructions="日本語で「発表を終了しました」とだけ言ってください。",
            )
        return True

    def _on_voice_keyword(self, action: str, text: str, confidence: float) -> None:
        """A locally spotted "会話に参加" / "会話から離脱" / greeting."""

        self._write_state(
            lastVoiceCommand=f"keyword_{action}",
            lastVoiceCommandValue=text,
            voiceCommandLastConfidence=round(float(confidence), 3),
        )
        if action == "control":
            if self._apply_voice_control(control_action(text) or "", text, source="keyword"):
                return
            action = "question"
        if action == "question":
            if self._keyword_interrupt_armed():
                self._request_talk_interrupt("keywords", text)
                return
            # Outside a talk listening for keywords, "すみません、質問…" is
            # still the greeting it contains.
            if not has_greeting(text):
                return
            action = "greeting"
        if action == "greeting":
            # Only meaningful while the conversation was explicitly left; the
            # ordinary visual gate handles greetings the rest of the time.
            if not self.voice_keyword_left.is_set():
                return
            if not self._addressed_frontally():
                self._write_state(voiceCommandRejected="greeting:not_frontal")
                return
            self._resume_after_leave(
                "voice_greeting_resume",
                (
                    "正面から挨拶されました。会話を再開します。"
                    "日本語でごく短く挨拶を返し、質問があれば答える姿勢を示してください。"
                ),
            )
            return
        if action == "join":
            was_left = self.voice_keyword_left.is_set()
            already_open = self.visual_session_active.is_set()
            self.voice_keyword_left.clear()
            self._set_state(voiceKeywordLeft=False)
            if not was_left and already_open and self._visual_session_enabled():
                # Already talking: refresh the session without a second hello.
                self._activate_visual_session("voice_join_keyword")
                return
            self._resume_after_leave(
                "voice_join_keyword",
                (
                    "会話に参加するよう求められました。『はい、参加します』のように"
                    "日本語でごく短く応じ、質問があれば答える姿勢を示してください。"
                ),
            )
            return
        if action == "leave":
            self._leave_conversation("voice_leave_keyword")

    def _make_owner_locator(self, owner_account: str) -> OwnerLocator:
        """Menu automation used by the follower when the room search fails."""

        diagnostics = Path(tempfile.gettempdir()) / "vrc-owner-locate"
        return OwnerLocator(
            owner_account,
            pid=self.args.vrchat_pid if self.args.vrchat_pid > 0 else None,
            cancelled=lambda: self.stop_event.is_set()
            or (self.owner_follower is not None and self.owner_follower.mode != "OwnerFollow"),
            state_callback=self._set_state,
            diagnostic_directory=diagnostics,
            osc_host=self.args.osc_host,
            osc_port=self.args.osc_port,
        )

    def _owner_private_access_status(self) -> dict[str, Any]:
        follower = self.owner_follower
        if follower is None:
            status = {
                "allowed": False,
                "reason": "owner_follow_unavailable",
            }
        else:
            status = follower.private_access_status()
        self._write_state(
            ownerPresencePrivateAccessAllowed=bool(status.get("allowed", False)),
            ownerPresencePrivateAccessReason=str(status.get("reason", "unknown")),
            ownerPresencePrivateAccessDetails=status,
        )
        return status

    def _high_privacy_owner_required(self, privacy_level: Any) -> bool:
        return bool(
            isinstance(privacy_level, (int, float))
            and float(privacy_level) >= self.args.owner_presence_privacy_threshold
        )

    def _leave_conversation(self, reason: str, *, announce: bool = True) -> bool:
        """Explicit "会話から離脱": close and latch until a join or greeting."""

        if self.voice_keyword_left.is_set():
            return False
        if self._presentation_running():
            self._request_presentation_control("stop")
            self._write_state(presentationStoppedBy=reason)
        self._end_visual_session(reason)
        self._reset_input_gate()
        self.voice_keyword_left.set()
        self.voice_keyword_left_at = time.monotonic()
        # Whatever the model was about to say -- including its own reply to
        # the very utterance that asked to leave -- is dropped, and anything
        # already playing is cut: 離脱 means stop talking now.
        with self.response_request_lock:
            self.response_create_requests.clear()
            self.response_create_pending.clear()
        if self.response_in_progress.is_set() or self.output_active.is_set():
            self._send_now({"type": "response.cancel"})
            self._interrupt_playback()
        self._set_state(voiceKeywordLeft=True, visualGateSessionEndReason=reason)
        if announce:
            self._request_response(
                purpose=reason,
                instructions=(
                    "会話から離脱するよう求められました。『はい、離脱します』のように"
                    "日本語でごく短く応じ、それ以上は話さないでください。"
                ),
            )
        return True

    def _owner_present_for(self, privacy_level: Any) -> tuple[bool, dict[str, Any]]:
        """(allowed, status) for answering material at ``privacy_level``.

        Anything at or above ``--owner-presence-privacy-threshold`` (default
        0.25 on SourceVault's 0..1 scale) is answered only while the
        owner-follow engine has a fresh, OCR-identified owner in formation
        range -- whatever the route: text to OpenAI, a local readout, or the
        public half of a mixed answer.  Below the threshold nothing is
        required.
        """

        if not self._high_privacy_owner_required(privacy_level):
            return True, {"allowed": True, "reason": "below_threshold"}
        status = self._owner_private_access_status()
        return bool(status.get("allowed", False)), status

    def _deny_private_audio_without_owner(
        self,
        call_id: str,
        wav_path: str | None,
        privacy_level: Any,
        access_status: dict[str, Any],
    ) -> None:
        if wav_path:
            try:
                Path(wav_path).resolve().unlink()
            except OSError:
                pass
        self._send_function_output(
            call_id,
            {
                "status": "owner_presence_required",
                "privacyLevel": privacy_level,
                "message": (
                    "このプライバシーレベルの回答には、フォローモードで"
                    "本人確認済みのオーナーが近くにいる必要があります。"
                ),
                "reason": access_status.get("reason"),
            },
        )
        self._write_state(
            lastSourceVaultRoute="owner_presence_required",
            ownerPresencePrivateAccessAllowed=False,
            ownerPresencePrivateAccessReason=access_status.get("reason"),
        )
        self._request_response(
            purpose="owner_presence_required",
            instructions=(
                "非公開の検索結果はあなたには提供されていません。"
                "次の内容だけを日本語で簡潔に伝えてください。"
                "『この情報は、フォローモードでオーナーが近くにいるときだけ"
                "回答できます。オーナーの追跡完了後にもう一度尋ねてください。』"
            ),
        )

    def _handle_sourcevault_call(self, item: dict[str, Any]) -> None:
        call_id = str(item.get("call_id", ""))
        try:
            arguments = json.loads(str(item.get("arguments", "{}")))
        except json.JSONDecodeError:
            arguments = {}
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not call_id or not isinstance(query, str) or not query.strip():
            if call_id:
                self._send_function_output(
                    call_id, {"status": "error", "message": "検索語がありません。"}
                )
                self._request_response()
            return
        broker = self.private_broker
        if broker is None:
            self._send_function_output(
                call_id, {"status": "error", "message": "SourceVaultは利用できません。"}
            )
            self._request_response()
            return
        with self.tool_lock:
            self._set_sourcevault_waiting(True)
            self._write_state(sourceVaultToolActive=True)
            try:
                # A question asked during a talk is about the talk.  Scoping
                # it keeps the answer out of the owner's mail and coding logs:
                # unscoped, "what was the 9th meeting about" was answered from a
                # VRCPilot work log (2026-08-15).
                scope = "presentation" if self._presentation_running() else ""
                result = broker.request(query.strip(), scope=scope)
                route = str(result.get("route", "error"))
                level = result.get("privacyLevel")
                count = int(result.get("count", 0) or 0)
                self._write_state(
                    lastSourceVaultRoute=route,
                    lastSourceVaultPrivacyLevel=level,
                    lastSourceVaultCount=count,
                    lastSourceVaultScope=scope or "general",
                    # Which layer answered, and why the fast index did not:
                    # an unbuilt KB otherwise degrades invisibly.
                    lastSourceVaultSource=result.get("source"),
                    lastSourceVaultKBStatus=result.get("kbStatus"),
                    sourceVaultDeferred=(route == "deferred"),
                )
                if route == "openai":
                    # Defense in depth: result-bearing text crosses into the
                    # OpenAI process only when PL is explicitly below 0.5.
                    safe_level = count == 0 or (
                        isinstance(level, (int, float)) and float(level) < 0.5
                    )
                    text = result.get("text", "")
                    owner_ok, access_status = (
                        self._owner_present_for(level) if count > 0 else (True, {})
                    )
                    if not owner_ok:
                        # PL 0.25..0.5 material: only with the owner nearby.
                        self._deny_private_audio_without_owner(
                            call_id, None, level, access_status
                        )
                        self._write_state(lastSourceVaultRoute="owner_presence_required")
                        continue_response = False
                    elif not safe_level or not isinstance(text, str):
                        continue_response = True
                        self._send_function_output(
                            call_id,
                            {
                                "status": "withheld",
                                "message": "プライバシー判定により結果をモデルへ渡せません。",
                            },
                        )
                    else:
                        continue_response = True
                        self._send_function_output(
                            call_id,
                            {
                                "status": "ok",
                                "privacyLevel": level,
                                "results": text,
                            },
                        )
                    if continue_response:
                        self._request_response()
                elif route == "no_results":
                    # Finding nothing is not the end of the question.  Offer the
                    # web, ask first, and stop cleanly if the answer is no --
                    # a query leaving this machine is the listener's call.
                    self._send_function_output(
                        call_id,
                        {
                            "status": "not_found",
                            "message": (
                                "SourceVaultに該当する情報はありませんでした。"
                                "ウェブで調べてよいか利用者に一言たずねてください。"
                                "了承されたら web_search をその質問で呼び、"
                                "断られたらそこで話を終えて追加の説明はしないでください。"
                                if self.args.enable_voice_web_search
                                else "SourceVaultに該当する情報はありませんでした。"
                                "分からないと正直に伝え、推測で答えないでください。"
                            ),
                            "webSearchAvailable": bool(
                                self.args.enable_voice_web_search
                            ),
                        },
                    )
                    self._request_response()
                elif route == "mixed":
                    # Public material answers now; the non-public remainder is
                    # only offered.  Reading it aloud unasked spends a minute of
                    # the room's time on something nobody requested.
                    text = result.get("text", "")
                    token = result.get("privateToken")
                    private_count = int(result.get("privateCount", 0) or 0)
                    safe_level = count == 0 or (
                        isinstance(level, (int, float)) and float(level) < 0.5
                    )
                    if not safe_level or not isinstance(text, str):
                        text = ""
                    owner_ok, access_status = self._owner_present_for(level)
                    if not owner_ok:
                        # Neither the PL>=threshold public half nor the private
                        # remainder may be answered without the owner; the
                        # remainder is dropped (the broker expires it).
                        with self.private_offer_lock:
                            self.private_offer_token = None
                        self._write_state(privateOfferPending=False, privateOfferCount=0)
                        self._deny_private_audio_without_owner(
                            call_id, None, level, access_status
                        )
                        self._write_state(lastSourceVaultRoute="owner_presence_required")
                        raise _SourceVaultHandled()
                    with self.private_offer_lock:
                        self.private_offer_token = (
                            token if isinstance(token, str) else None
                        )
                        self.private_offer_level = result.get("privateMaxPrivacyLevel", 1.0)
                    self._write_state(
                        privateOfferPending=isinstance(token, str),
                        privateOfferCount=private_count,
                    )
                    self._send_function_output(
                        call_id,
                        {
                            "status": "ok",
                            "privacyLevel": level,
                            "results": text,
                            "privateRemaining": private_count,
                            "message": (
                                f"公開資料の結果を返しました。ほかに非公開の資料が"
                                f"{private_count}件あります。本文はモデルへ渡せませんが、"
                                "利用者が希望すればローカル音声で読み上げられます。"
                                "まず公開分で答え、そのうえで続きを聞くか一言たずねてください。"
                                "希望されたら private_readout を action=play で、"
                                "不要と言われたら action=discard で呼んでください。"
                            ),
                        },
                    )
                    self._request_response()
                elif route == "local_tts":
                    wav_path = result.get("wavPath")
                    if not isinstance(wav_path, str):
                        raise RuntimeError("private broker returned no audio")
                    if self._high_privacy_owner_required(level):
                        access_status = self._owner_private_access_status()
                        if not bool(access_status.get("allowed", False)):
                            self._deny_private_audio_without_owner(
                                call_id,
                                wav_path,
                                level,
                                access_status,
                            )
                        else:
                            self._begin_local_handoff(call_id, wav_path, level)
                    else:
                        self._begin_local_handoff(call_id, wav_path, level)
                elif route == "deferred":
                    self._send_function_output(
                        call_id,
                        {
                            "status": "preparing",
                            "jobId": result.get("jobId"),
                            "message": (
                                "検索は音声会話の待機上限を超えました。"
                                "ローカルで準備を続けているため、"
                                "後から結果を確認してください。"
                            ),
                        },
                    )
                    self._request_response()
                else:
                    self._send_function_output(
                        call_id,
                        {"status": "error", "message": "SourceVault検索に失敗しました。"},
                    )
                    self._request_response()
            except _SourceVaultHandled:
                pass
            except Exception as exc:
                self._set_state(lastError=f"SourceVault tool: {type(exc).__name__}")
                self._send_function_output(
                    call_id,
                    {"status": "error", "message": "SourceVault検索に失敗しました。"},
                )
                self._request_response()
            finally:
                self._set_sourcevault_waiting(False)
                self._write_state(sourceVaultToolActive=False)
                self._touch_visual_session("sourcevault_finished")

    def _handle_private_readout_call(self, item: dict[str, Any]) -> None:
        """Play or discard the private remainder, on the listener's word.

        The text itself never enters this process: the broker holds it and
        returns only a WAV path if the listener said yes."""

        call_id = str(item.get("call_id", ""))
        try:
            arguments = json.loads(str(item.get("arguments", "{}")))
        except json.JSONDecodeError:
            arguments = {}
        action = str(arguments.get("action", "discard")) if isinstance(
            arguments, dict
        ) else "discard"
        if not call_id:
            return
        with self.private_offer_lock:
            token = self.private_offer_token
            self.private_offer_token = None
            offer_level = getattr(self, "private_offer_level", 1.0)
        self._write_state(privateOfferPending=False)
        broker = self.private_broker
        if broker is None or not isinstance(token, str):
            self._send_function_output(
                call_id,
                {"status": "error", "message": "保留中の非公開結果はありません。"},
            )
            self._request_response()
            return
        if action == "play":
            owner_ok, access_status = self._owner_present_for(offer_level)
            if not owner_ok:
                try:
                    broker.request_private_readout(token, "discard")
                except Exception:
                    pass
                self._write_state(lastPrivateReadout="owner_presence_required")
                self._deny_private_audio_without_owner(call_id, None, offer_level, access_status)
                return
        try:
            result = broker.request_private_readout(token, action)
        except Exception as exc:
            self._set_state(lastError=f"private readout: {type(exc).__name__}")
            self._send_function_output(
                call_id, {"status": "error", "message": "読み上げに失敗しました。"}
            )
            self._request_response()
            return
        route = str(result.get("route", "error"))
        if route == "discarded":
            self._write_state(lastPrivateReadout="discarded")
            self._send_function_output(
                call_id,
                {"status": "discarded", "message": "非公開の結果は破棄しました。"},
            )
            self._request_response()
            return
        if route != "local_tts" or not isinstance(result.get("wavPath"), str):
            self._send_function_output(
                call_id, {"status": "error", "message": "読み上げに失敗しました。"}
            )
            self._request_response()
            return
        self._write_state(
            lastPrivateReadout="played",
            privateReadoutTruncated=bool(result.get("truncated", False)),
        )
        with self.tool_lock:
            try:
                self._begin_local_handoff(call_id, result["wavPath"], None)
            except Exception:
                self._send_function_output(
                    call_id, {"status": "error", "message": "読み上げに失敗しました。"}
                )
                self._request_response()

    def _handle_web_search_call(self, item: dict[str, Any]) -> None:
        """Local metasearch fallback for questions the local index cannot answer.

        Only titles/URLs/snippets come back -- no page bodies -- and the query
        itself leaves this machine, which is why the tool is opt-in
        (--enable-voice-web-search) rather than always available."""

        call_id = str(item.get("call_id", ""))
        try:
            arguments = json.loads(str(item.get("arguments", "{}")))
        except json.JSONDecodeError:
            arguments = {}
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not call_id:
            return
        if not isinstance(query, str) or not query.strip():
            self._send_function_output(
                call_id, {"status": "error", "message": "検索語がありません。"}
            )
            self._request_response()
            return
        broker = self.private_broker
        if broker is None or not self.args.enable_voice_web_search:
            self._send_function_output(
                call_id, {"status": "error", "message": "ウェブ検索は利用できません。"}
            )
            self._request_response()
            return
        with self.tool_lock:
            self._set_sourcevault_waiting(True)
            self._write_state(webSearchToolActive=True, lastWebSearchQuery=query.strip())
            # Nothing is spoken while this runs, and a silent bot reads as a
            # broken one from inside the room.  Say what it is doing.
            self._announce("web_search", "Web検索中", min_interval=0.0)
            try:
                result = broker.request_web_search(query.strip())
                route = str(result.get("route", "error"))
                if route == "web":
                    self._announce(
                        "web_search_done",
                        "Web検索: " + str(result.get("count", 0)) + "件",
                        min_interval=0.0,
                    )
                    self._send_function_output(
                        call_id,
                        {
                            "status": "ok",
                            "count": result.get("count", 0),
                            "results": result.get("text", ""),
                        },
                    )
                else:
                    self._announce("web_search_done", "Web検索に失敗", min_interval=0.0)
                    self._send_function_output(
                        call_id,
                        {
                            "status": "error",
                            "message": "ウェブ検索に失敗しました。",
                        },
                    )
                self._write_state(lastWebSearchRoute=route)
            except Exception as exc:
                self._set_state(lastError=f"web search tool: {type(exc).__name__}")
                self._send_function_output(
                    call_id, {"status": "error", "message": "ウェブ検索に失敗しました。"}
                )
            finally:
                self._set_sourcevault_waiting(False)
                self._write_state(webSearchToolActive=False)
            self._request_response()

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response", {})
        output = response.get("output", []) if isinstance(response, dict) else []
        calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "sourcevault_lookup"
        ]
        for item in calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            threading.Thread(
                target=self._handle_sourcevault_call, args=(item,), daemon=True
            ).start()
        readout_calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "private_readout"
        ]
        for item in readout_calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            threading.Thread(
                target=self._handle_private_readout_call, args=(item,), daemon=True
            ).start()
        web_calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "web_search"
        ]
        for item in web_calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            threading.Thread(
                target=self._handle_web_search_call, args=(item,), daemon=True
            ).start()
        end_calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "end_conversation"
        ]
        for item in end_calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            self._send_function_output(call_id, {"status": "ended"})
            self._end_visual_session("model_request")
            self._request_response(
                purpose="conversation_end",
                instructions=(
                    "利用者との会話をここで終了します。ツールは使わず、"
                    "日本語でごく短い別れの挨拶だけを発話してください。"
                ),
            )
        avatar_command_calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name")
            in {
                "set_avatar_mode",
                "start_presentation",
                "presentation_control",
                "show_slide",
                "turn_avatar",
                "conversation_control",
            }
        ]
        fresh_avatar_calls = []
        for item in avatar_command_calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            fresh_avatar_calls.append(item)
        if fresh_avatar_calls:
            # Off the websocket thread.  start_presentation resolves the title
            # through the SourceVault registry, and that request waits on the
            # service kernel: when a maintenance tick had the service busy the
            # lookup took its full 60 s timeout *inside this callback*, so no
            # audio delta and no event was read for a minute and the session
            # reconnected (live 2026-09-04).  Every other blocking tool already
            # runs on its own thread; these have to as well.  One thread for
            # the whole list, not one each: a response that calls
            # set_avatar_mode and start_presentation together depends on being
            # handled in the order the model asked for.
            thread = threading.Thread(
                target=self._handle_avatar_command_calls,
                args=(fresh_avatar_calls,),
                daemon=True,
                name="VRCRealtimeAvatarCommands",
            )
            self.avatar_command_thread = thread
            thread.start()
        for item in output:
            if not (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and item.get("name") == "set_question_mode"
            ):
                continue
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            self._handle_question_mode_call(item)
        vision_calls = [
            item
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "function_call"
            and item.get("name") == "look_at_vrchat"
        ]
        for item in vision_calls:
            call_id = str(item.get("call_id", ""))
            if not call_id or call_id in self.handled_tool_calls:
                continue
            self.handled_tool_calls.add(call_id)
            try:
                arguments = json.loads(str(item.get("arguments", "{}")))
            except json.JSONDecodeError:
                arguments = {}
            question = (
                str(arguments.get("question", ""))
                if isinstance(arguments, dict)
                else ""
            )
            threading.Thread(
                target=self._handle_vision_capture,
                args=(call_id, question),
                daemon=True,
            ).start()

    def _on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            self._write_state(lastError="received malformed JSON")
            return
        event_type = str(event.get("type", ""))
        self._set_state(lastEvent=event_type)
        if event_type == "response.output_audio.delta":
            self._queue_output_delta(event)
        elif event_type == "input_audio_buffer.speech_started":
            self._touch_visual_session("speech_started")
            if self.presentation_qa_open.is_set():
                # Someone actually spoke, so the window must now wait for the
                # answer instead of expiring on its 5 second timer.
                self.presentation_qa_speech.set()
            if self.owner_follower is not None:
                self.owner_follower.set_listener_active(True)
            # With VAD the server cancels generation.  WebSocket clients still
            # own playback and must discard/truncate audio already received.
            if self.args.allow_barge_in and self.output_active.is_set():
                self._interrupt_playback()
        elif event_type == "input_audio_buffer.committed":
            self.input_committed_at = time.time()
        elif event_type == "input_audio_buffer.speech_stopped":
            self._touch_visual_session("speech_stopped")
            if self.owner_follower is not None:
                self.owner_follower.set_listener_active(False)
        elif event_type == "response.output_audio.done":
            self.output_done.set()
        elif event_type == "response.output_audio_transcript.delta":
            delta = str(event.get("delta", ""))
            with self.state_lock:
                transcript = str(self.state.get("lastTranscript", "")) + delta
            self._set_state(lastTranscript=transcript[-4000:])
        elif event_type == "response.created":
            self._touch_visual_session("response_created")
            self.response_in_progress.set()
            response = event.get("response", {})
            metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
            is_requested_response = (
                isinstance(metadata, dict)
                and "vrc_realtime_purpose" in metadata
            )
            purpose = (
                str(metadata.get("vrc_realtime_purpose", "normal"))
                if isinstance(metadata, dict)
                else "normal"
            )
            if is_requested_response:
                with self.response_request_lock:
                    self.inflight_response_request = None
                    self.response_create_inflight.clear()
            with self.output_purpose_lock:
                self.current_output_purpose = purpose
            if self.owner_follower is not None:
                self.owner_follower.set_speech_active(True)
            self.output_active.set()
            self.output_done.clear()
            _clear_queue(self.input_events)
            self._send_now({"type": "input_audio_buffer.clear"})
            self._set_state(lastTranscript="")
        elif event_type == "response.done":
            self._touch_visual_session("response_done")
            self.response_in_progress.clear()
            response = event.get("response", {})
            metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
            purpose = (
                str(metadata.get("vrc_realtime_purpose", "normal"))
                if isinstance(metadata, dict)
                else "normal"
            )
            if purpose != "normal":
                with self.output_purpose_lock:
                    self.current_output_purpose = purpose
            self.output_done.set()
            self._handle_response_done(event)
        elif event_type == "session.updated":
            self._set_state(lastError=None)
        elif event_type == "error":
            detail = event.get("error", event)
            if (
                isinstance(detail, dict)
                and detail.get("code") == "conversation_already_has_active_response"
            ):
                # A VAD response won a narrow race. Retry the queued tool
                # continuation after response.done instead of losing it.
                self.response_in_progress.set()
                with self.response_request_lock:
                    request = self.inflight_response_request
                    if request is not None:
                        self.response_create_requests.appendleft(request)
                    self.inflight_response_request = None
                    self.response_create_inflight.clear()
                    self.response_create_pending.set()
            self._write_state(lastError=json.dumps(detail, ensure_ascii=False))

    def _on_error(self, _ws: websocket.WebSocketApp, error: Any) -> None:
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            self._handshake_status = status
        if not self.stop_event.is_set():
            self._write_state(lastError=f"websocket: {error}")

    # ---- connection hooks (VRCRealtime_live.LiveVRCBridge overrides these)

    def _ws_url(self) -> str:
        return "wss://api.openai.com/v1/realtime?model=" + quote(self.args.model, safe="")

    def _ws_headers(self) -> list[str]:
        headers = [f"Authorization: Bearer {self.api_key}"]
        if self.args.safety_identifier:
            headers.append(f"OpenAI-Safety-Identifier: {self.args.safety_identifier}")
        return headers

    def _run_forever_options(self) -> dict[str, Any]:
        return {}

    def _extra_threads(self) -> list[threading.Thread]:
        return []

    def _handshake_message(self, status: int) -> str:
        if status == 401:
            return "OpenAI に認証されませんでした (401)。API キーを確認してください。"
        if status == 403:
            return (
                f"OpenAI への接続が拒否されました (403)。この API キーで {self.args.model} "
                "が使えるか (権限・モデル制限) を確認してください。"
            )
        return f"接続先が見つかりません (404)。モデル名 {self.args.model} を確認してください。"

    def _on_close(
        self,
        _ws: websocket.WebSocketApp,
        status_code: int | None,
        message: str | None,
    ) -> None:
        self.ws_ready.clear()
        self.response_in_progress.clear()
        _clear_queue(self.input_events)
        self._end_visual_session("connection_closed")
        if not self.stop_event.is_set():
            self._write_state(
                status="reconnecting",
                connected=False,
                lastError=f"closed: {status_code} {message or ''}".strip(),
            )

    def _control_worker(self) -> None:
        for line in sys.stdin:
            if self.stop_event.is_set():
                return
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(command, dict) and self._handle_control_command(command):
                return

    def _handle_control_command(self, command: dict[str, Any]) -> bool:
        """Apply one control command; return True when shutdown was requested."""
        name = str(command.get("command", "")).lower()
        if name == "stop":
            self.stop()
            return True
        if name == "mute":
            self.muted.set()
            self._write_state(muted=True)
        elif name == "unmute":
            self.muted.clear()
            self._write_state(muted=False)
        elif name == "end-conversation":
            self._end_visual_session("explicit_command")
        elif name == "look":
            question = str(
                command.get("question", "現在のVRChat画面には何が見えますか。")
            )
            threading.Thread(
                target=self._handle_vision_capture,
                args=(None, question),
                daemon=True,
            ).start()
        elif name == "presentation-start":
            self._start_presentation(command)
        elif name == "presentation-stop":
            self._stop_presentation()
        elif name == "presentation-control":
            try:
                self._request_presentation_control(str(command.get("value", "")))
            except ValueError as exc:
                self._write_state(presentationLastError=str(exc))
        elif name == "question-mode":
            if self._set_question_mode(command.get("value"), source="notebook") is None:
                self._write_state(
                    lastError=f"unknown question mode: {command.get('value')!r}"
                )
        elif name == "mode":
            follower = self.owner_follower
            value = command.get("value")
            # A mode change ends a running presentation for the same reason a
            # spoken one does, and that must not depend on whether the
            # owner-follow controller happens to exist.
            if (
                isinstance(value, str)
                and value != "Presentation"
                and self._presentation_running()
            ):
                self._request_presentation_control("stop")
                self._write_state(presentationStoppedBy="mode_change")
            if follower is None:
                self._write_state(ownerFollowLastError="owner follow is unavailable")
            elif isinstance(value, str):
                try:
                    follower.set_mode(value)
                except ValueError as exc:
                    self._write_state(ownerFollowLastError=str(exc))
        elif name == "owner-account":
            follower = self.owner_follower
            value = command.get("value")
            if follower is None:
                self._write_state(ownerFollowLastError="owner follow is unavailable")
            elif isinstance(value, str):
                try:
                    follower.set_owner_account(value)
                except ValueError as exc:
                    self._write_state(ownerFollowLastError=str(exc))
        elif name == "follow-distance":
            follower = self.owner_follower
            value = command.get("targetNameplateHeight")
            metres = command.get("distanceMeters")
            if follower is None:
                self._write_state(ownerFollowLastError="owner follow is unavailable")
            elif isinstance(metres, (int, float)) and hasattr(follower, "set_follow_distance"):
                try:
                    follower.set_follow_distance(float(metres))
                    self._write_state(ownerFollowLastError=None)
                except ValueError as exc:
                    self._write_state(ownerFollowLastError=str(exc))
            elif isinstance(value, (int, float)):
                try:
                    follower.set_target_nameplate_height(float(value))
                    self._write_state(ownerFollowLastError=None)
                except ValueError as exc:
                    self._write_state(ownerFollowLastError=str(exc))
        elif name == "locate-owner":
            follower = self.owner_follower
            if follower is None or not hasattr(follower, "request_locate"):
                self._write_state(ownerFollowLastError="owner locate needs the formation engine")
            elif not follower.request_locate():
                self._write_state(ownerFollowLastError="owner locate is disabled")
            else:
                self._write_state(ownerFollowLastError=None)
        elif name == "formation":
            follower = self.owner_follower
            if follower is None or not hasattr(follower, "set_formation"):
                self._write_state(
                    ownerFollowLastError="formation settings need the formation engine"
                )
            else:
                try:
                    rear = command.get("rearAngleDeg")
                    bearing = command.get("ownerBearingDeg")
                    follower.set_formation(
                        rear_angle_deg=float(rear) if isinstance(rear, (int, float)) else None,
                        owner_bearing_deg=(
                            float(bearing) if isinstance(bearing, (int, float)) else None
                        ),
                    )
                    self._write_state(ownerFollowLastError=None)
                except ValueError as exc:
                    self._write_state(ownerFollowLastError=str(exc))
        elif name == "instructions":
            value = command.get("value")
            if isinstance(value, str):
                self.instructions = value
                self._send_now(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": self._effective_instructions(),
                        },
                    }
                )
        return False

    def _control_file_worker(self) -> None:
        """Poll an atomic JSON command file, independent of uv stdin forwarding."""
        if not self.args.control_file:
            return
        path = Path(self.args.control_file)
        while not self.stop_event.wait(0.05):
            if not path.is_file():
                continue
            try:
                command = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._set_state(lastError=f"control file: {type(exc).__name__}")
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            try:
                path.unlink()
            except OSError:
                pass
            if isinstance(command, dict) and self._handle_control_command(command):
                return

    def _session_watchdog(self) -> None:
        # Realtime sessions are capped at 60 minutes.  Reconnect a little early
        # so the bridge recovers predictably instead of waiting for a hard close.
        while not self.stop_event.wait(1.0):
            if (
                self.ws_ready.is_set()
                and self.session_started_at > 0
                and time.monotonic() - self.session_started_at
                >= self.args.reconnect_seconds
            ):
                with self.ws_lock:
                    ws = self.ws
                if ws is not None:
                    ws.close()

    def _private_broker_worker(self) -> None:
        broker = self.private_broker
        if broker is None:
            return
        try:
            broker.start()
            self._write_state(privateBrokerReady=True)
        except Exception as exc:
            self._write_state(
                privateBrokerReady=False,
                lastError=f"private broker: {type(exc).__name__}: {exc}",
            )

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.presentation_stop_event.set()
        self.stop_event.set()
        self.ws_ready.clear()
        if self.owner_follower is not None:
            self.owner_follower.close()
        with self.ws_lock:
            ws = self.ws
        if ws is not None:
            ws.close()
        if self.private_broker is not None:
            self.private_broker.close()

    def run(self) -> int:
        if self.visual_gate is not None:
            self.visual_gate.start()
        if self.owner_follower is not None:
            self.owner_follower.start()
        if self.voice_commands is not None:
            self.voice_commands.start()
        if self.args.input_source == "vrchat":
            self._start_speaker()

        threads = [
            threading.Thread(target=self._input_sender, daemon=True),
            threading.Thread(target=self._playback_worker, daemon=True),
            threading.Thread(target=self._control_worker, daemon=True),
            threading.Thread(target=self._control_file_worker, daemon=True),
            threading.Thread(target=self._response_create_worker, daemon=True),
            threading.Thread(target=self._session_watchdog, daemon=True),
            threading.Thread(target=self._speaker_watchdog, daemon=True),
            threading.Thread(target=self._status_worker, daemon=True),
            threading.Thread(target=self._vrchat_state_worker, daemon=True),
        ]
        if self.private_broker is not None:
            threads.append(threading.Thread(target=self._private_broker_worker, daemon=True))
        if self.args.input_source == "microphone":
            threads.append(threading.Thread(target=self._microphone_worker, daemon=True))
        threads.extend(self._extra_threads())
        for thread in threads:
            thread.start()

        reconnect_count = 0
        try:
            while not self.stop_event.is_set():
                self._handshake_status = None
                app = websocket.WebSocketApp(
                    self._ws_url(),
                    header=self._ws_headers(),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self.ws_lock:
                    self.ws = app
                app.run_forever(
                    ping_interval=20, ping_timeout=10, **self._run_forever_options()
                )
                with self.ws_lock:
                    self.ws = None
                if self._handshake_status in (401, 403, 404):
                    # Refused, not dropped: retrying every second only hides
                    # the reason behind a reconnect counter.
                    self._write_state(
                        status="error",
                        handshakeStatus=self._handshake_status,
                        lastError=self._handshake_message(self._handshake_status),
                    )
                    break
                if self.stop_event.wait(1.0):
                    break
                reconnect_count += 1
                self._write_state(reconnects=reconnect_count)
        finally:
            self.stop_event.set()
            self.ws_ready.clear()
            if self.speaker is not None:
                # close() re-raises whatever ended the capture thread; that
                # must not skip the rest of the shutdown.
                self._close_quietly(self.speaker)
            if self.owner_follower is not None:
                self.owner_follower.close()
            if self.voice_commands is not None:
                self.voice_commands.close()
            if self._vision_capture is not None:
                try:
                    self._vision_capture.close()
                except Exception:
                    pass
                self._vision_capture = None
            if self.visual_gate is not None:
                self.visual_gate.close()
            if self.private_broker is not None:
                self.private_broker.close()
            if self.args.control_file:
                try:
                    Path(self.args.control_file).unlink()
                except OSError:
                    pass
            self._write_state(status="stopped", connected=False)
        return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--control-file", default="")
    parser.add_argument("--model", default="gpt-realtime-2.1")
    # auto: gpt-live-* speaks the GPT-Live protocol (VRCRealtime_live.py),
    # everything else the Realtime one.
    parser.add_argument("--api", choices=("auto", "realtime", "live"), default="auto")
    # GPT-Live delegates tool work to a Responses model (tools, vision).
    parser.add_argument("--live-backend-model", default="gpt-5.6-luna")
    parser.add_argument(
        "--live-backend-reasoning",
        choices=("none", "minimal", "low", "medium", "high"),
        default="low",
    )
    # GPT-Live: an utterance ends when no voiced audio has arrived for this
    # long (a script read aloud gets the longer one, with the coverage check).
    parser.add_argument("--live-idle-ms", type=int, default=900)
    parser.add_argument("--live-narration-idle-ms", type=int, default=1400)
    parser.add_argument("--live-narration-stall-ms", type=int, default=6000)
    parser.add_argument("--live-ack-timeout-ms", type=int, default=8000)
    # GPT-Live listens while it speaks.  Outside a talk the room is heard
    # all the time; --no-live-full-duplex restores the Realtime half duplex.
    parser.add_argument(
        "--live-full-duplex", action=argparse.BooleanOptionalAction, default=True
    )
    # GPT-Live exchange log (JSONL, no audio).  Empty = %TEMP%/vrc-live-events.jsonl.
    parser.add_argument("--live-event-log", default="")
    parser.add_argument("--voice", default="marin")
    parser.add_argument(
        "--instructions",
        default=os.environ.get(
            "VRC_REALTIME_INSTRUCTIONS",
            "日本語で自然かつ簡潔に会話してください。",
        ),
    )
    parser.add_argument("--output-device", default="CABLE-A Input")
    parser.add_argument(
        "--input-source", choices=("vrchat", "microphone"), default="vrchat"
    )
    parser.add_argument("--input-device", default="")
    parser.add_argument("--vrchat-pid", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--prefix-padding-ms", type=int, default=200)
    parser.add_argument("--silence-duration-ms", type=int, default=350)
    parser.add_argument("--reconnect-seconds", type=int, default=3300)
    parser.add_argument("--output-cooldown-ms", type=int, default=1200)
    parser.add_argument("--safety-identifier", default="")
    parser.add_argument("--allow-barge-in", action="store_true")
    parser.add_argument("--visual-attention-gate", action="store_true")
    parser.add_argument("--visual-gate-fps", type=float, default=8.0)
    parser.add_argument("--visual-gate-center-tolerance", type=float, default=0.18)
    parser.add_argument("--visual-gate-min-face-fraction", type=float, default=0.07)
    parser.add_argument("--visual-gate-motion-threshold", type=float, default=0.025)
    parser.add_argument("--visual-gate-audio-rms-threshold", type=float, default=0.008)
    parser.add_argument(
        "--visual-gate-max-stereo-imbalance", type=float, default=0.35
    )
    parser.add_argument("--visual-gate-hangover-ms", type=int, default=650)
    parser.add_argument("--visual-gate-preroll-ms", type=int, default=600)
    parser.add_argument(
        "--visual-gate-require-nameplate-pulse", action="store_true"
    )
    parser.add_argument(
        "--visual-gate-nameplate-blue-threshold", type=float, default=0.002
    )
    parser.add_argument("--visual-gate-conversation-latch", action="store_true")
    parser.add_argument(
        "--visual-gate-conversation-idle-seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--avatar-mode",
        choices=("standalone", "owner-follow", "presentation"),
        default="standalone",
    )
    parser.add_argument("--owner-account", default="nconc")
    parser.add_argument(
        "--owner-follow-engine", choices=("formation", "legacy"), default="formation"
    )
    parser.add_argument("--owner-follow-distance-m", type=float, default=1.8)
    parser.add_argument("--owner-follow-rear-angle", type=float, default=35.0)
    parser.add_argument("--owner-follow-owner-bearing", type=float, default=33.0)
    parser.add_argument(
        "--owner-follow-menu-locate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--owner-follow-search-patrol", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--owner-follow-idle-after-seconds", type=float, default=360.0)
    parser.add_argument("--owner-follow-idle-recheck-seconds", type=float, default=300.0)
    parser.add_argument("--owner-follow-tracking-fps", type=float, default=1.5)
    parser.add_argument("--owner-follow-target-screen-x", type=float, default=0.66)
    parser.add_argument(
        "--owner-follow-target-nameplate-height", type=float, default=0.055
    )
    parser.add_argument("--owner-follow-lost-seconds", type=float, default=1.25)
    parser.add_argument("--owner-follow-formation-seconds", type=float, default=4.5)
    parser.add_argument("--owner-follow-orbit-speed", type=float, default=0.32)
    parser.add_argument("--owner-follow-search-turn-speed", type=float, default=0.42)
    parser.add_argument(
        "--owner-follow-assume-single-avatar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--owner-follow-ocr-min-side", type=int, default=1080)
    parser.add_argument("--owner-follow-ocr-box-threshold", type=float, default=0.35)
    parser.add_argument("--owner-follow-ocr-text-score", type=float, default=0.35)
    parser.add_argument(
        "--voice-commands", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--voice-command-model-dir", default="")
    parser.add_argument("--no-voice-command-download", action="store_true")
    parser.add_argument("--voice-command-min-confidence", type=float, default=0.45)
    # How a talk takes questions: question_time (only at slide boundaries,
    # the historical behaviour), keywords (a spoken 「質問」「スライド」「プレゼン」「発表」
    # interrupts the slide being narrated) or detect (sustained voice in the
    # room pauses the talk and the model judges whether it was a question).
    parser.add_argument(
        "--question-mode",
        choices=QUESTION_MODES,
        default="question_time",
    )
    parser.add_argument("--interrupt-words", default=",".join(DEFAULT_INTERRUPT_WORDS))
    # detect: room-audio RMS and how long it has to stay above it.
    parser.add_argument("--interrupt-rms", type=float, default=0.02)
    parser.add_argument("--interrupt-ms", type=int, default=450)
    # After an interruption: quiet that ends the questions and resumes the
    # talk, and how long a fresh interruption is ignored afterwards.
    parser.add_argument("--interrupt-resume-seconds", type=float, default=4.0)
    parser.add_argument("--interrupt-cooldown-seconds", type=float, default=8.0)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--osc-listen-port", type=int, default=9001)
    parser.add_argument("--enable-vision-capture", action="store_true")
    parser.add_argument("--vision-image-max-width", type=int, default=960)
    parser.add_argument("--vision-jpeg-quality", type=int, default=78)
    parser.add_argument("--enable-sourcevault", action="store_true")
    # Opt-in: a question asked by anyone in the room would otherwise become a
    # query leaving this machine.  Off by default, on request only.
    parser.add_argument("--enable-voice-web-search", action="store_true")
    parser.add_argument("--private-tts-broker", default="")
    parser.add_argument("--sourcevault-service-dir", default="")
    parser.add_argument(
        "--sourcevault-kinds",
        default="search,packageapi,llmlog,mail,eagle,snapshot",
    )
    parser.add_argument("--sourcevault-group", default="")
    parser.add_argument("--sourcevault-release-context", default="")
    # Graph-RAG knowledge base id.  When set, the broker asks SourceVault to try
    # that prebuilt index first, which answers in tens of milliseconds instead of
    # the seconds-to-minutes an adapter-wide search can take.
    parser.add_argument("--sourcevault-kb", default="")
    parser.add_argument("--sourcevault-limit", type=int, default=3)
    parser.add_argument("--sourcevault-wait-seconds", type=float, default=60.0)
    # Ceiling on one local (Piper) readout.  A high-privacy hit can be
    # thousands of characters, and nobody in the room can skip it.
    parser.add_argument("--local-speech-max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--owner-presence-privacy-threshold", type=float, default=0.25
    )
    # Local TTS engine for the private broker.  "aivis" keeps the piper
    # arguments as an in-session fallback; all values are pass-through.
    parser.add_argument("--tts-engine", choices=["piper", "aivis"], default="piper")
    parser.add_argument("--aivis-endpoint", default="http://127.0.0.1:10101")
    parser.add_argument("--aivis-style-id", type=int, default=0)
    parser.add_argument("--aivis-speed-scale", type=float, default=1.0)
    parser.add_argument("--piper-executable", default="")
    parser.add_argument("--piper-model", default="")
    parser.add_argument("--piper-config", default="")
    # Resolved from the installed voice's config by the caller.  Empty =
    # unknown/not applicable; nothing here assumes a particular voice.
    parser.add_argument("--piper-language", default="")
    parser.add_argument("--piper-length-scale", type=float, default=0.0)
    parser.add_argument("--piper-noise-scale", type=float, default=0.0)
    parser.add_argument("--private-audio-dir", default="")
    args = parser.parse_args()
    args.local_speech_max_seconds = max(
        5.0, min(600.0, args.local_speech_max_seconds)
    )
    args.sourcevault_wait_seconds = max(
        1.0, min(120.0, args.sourcevault_wait_seconds)
    )
    if not 1.0 <= args.visual_gate_fps <= 20.0:
        parser.error("--visual-gate-fps must be between 1 and 20")
    if not 0.05 <= args.visual_gate_center_tolerance <= 0.5:
        parser.error("--visual-gate-center-tolerance must be between 0.05 and 0.5")
    if not 0.03 <= args.visual_gate_min_face_fraction <= 0.4:
        parser.error("--visual-gate-min-face-fraction must be between 0.03 and 0.4")
    if not 0.0 <= args.visual_gate_motion_threshold <= 0.2:
        parser.error("--visual-gate-motion-threshold must be between 0 and 0.2")
    if not 0.0 <= args.visual_gate_audio_rms_threshold <= 0.5:
        parser.error("--visual-gate-audio-rms-threshold must be between 0 and 0.5")
    if not 0.0 <= args.visual_gate_max_stereo_imbalance <= 1.0:
        parser.error("--visual-gate-max-stereo-imbalance must be between 0 and 1")
    if not 100 <= args.visual_gate_hangover_ms <= 3000:
        parser.error("--visual-gate-hangover-ms must be between 100 and 3000")
    if not 0 <= args.visual_gate_preroll_ms <= 2000:
        parser.error("--visual-gate-preroll-ms must be between 0 and 2000")
    if not 0.0001 <= args.visual_gate_nameplate_blue_threshold <= 0.1:
        parser.error(
            "--visual-gate-nameplate-blue-threshold must be between 0.0001 and 0.1"
        )
    if not 5.0 <= args.visual_gate_conversation_idle_seconds <= 600.0:
        parser.error(
            "--visual-gate-conversation-idle-seconds must be between 5 and 600"
        )
    if not 0.25 <= args.owner_follow_tracking_fps <= 5.0:
        parser.error("--owner-follow-tracking-fps must be between 0.25 and 5")
    if not 0.8 <= args.owner_follow_distance_m <= 6.0:
        parser.error("--owner-follow-distance-m must be between 0.8 and 6")
    if not 0.0 <= args.owner_follow_rear_angle <= 90.0:
        parser.error("--owner-follow-rear-angle must be between 0 and 90")
    if not 10.0 <= args.owner_follow_owner_bearing <= 42.0:
        parser.error("--owner-follow-owner-bearing must be between 10 and 42")
    if not 0.0 <= args.owner_follow_idle_after_seconds <= 3600.0:
        parser.error("--owner-follow-idle-after-seconds must be between 0 and 3600")
    if not 30.0 <= args.owner_follow_idle_recheck_seconds <= 3600.0:
        parser.error("--owner-follow-idle-recheck-seconds must be between 30 and 3600")
    if not 0.50 <= args.owner_follow_target_screen_x <= 0.80:
        parser.error("--owner-follow-target-screen-x must be between 0.50 and 0.80")
    if not 0.005 <= args.owner_follow_target_nameplate_height <= 0.15:
        parser.error(
            "--owner-follow-target-nameplate-height must be between 0.005 and 0.15"
        )
    if not 0.5 <= args.owner_follow_lost_seconds <= 5.0:
        parser.error("--owner-follow-lost-seconds must be between 0.5 and 5")
    if not 1.5 <= args.owner_follow_formation_seconds <= 10.0:
        parser.error("--owner-follow-formation-seconds must be between 1.5 and 10")
    if not 0.10 <= args.owner_follow_orbit_speed <= 0.60:
        parser.error("--owner-follow-orbit-speed must be between 0.10 and 0.60")
    if not 0.08 <= args.owner_follow_search_turn_speed <= 0.50:
        parser.error("--owner-follow-search-turn-speed must be between 0.08 and 0.50")
    if not 736 <= args.owner_follow_ocr_min_side <= 1440:
        parser.error("--owner-follow-ocr-min-side must be between 736 and 1440")
    if not 0.20 <= args.owner_follow_ocr_box_threshold <= 0.70:
        parser.error("--owner-follow-ocr-box-threshold must be between 0.20 and 0.70")
    if not 0.20 <= args.owner_follow_ocr_text_score <= 0.70:
        parser.error("--owner-follow-ocr-text-score must be between 0.20 and 0.70")
    if not args.owner_account.strip():
        parser.error("--owner-account must not be empty")
    if not 1 <= args.osc_port <= 65535:
        parser.error("--osc-port must be between 1 and 65535")
    if not 0.0 <= args.voice_command_min_confidence <= 1.0:
        parser.error("--voice-command-min-confidence must be between 0 and 1")
    if not 0.001 <= args.interrupt_rms <= 0.5:
        parser.error("--interrupt-rms must be between 0.001 and 0.5")
    if not 100 <= args.interrupt_ms <= 5000:
        parser.error("--interrupt-ms must be between 100 and 5000")
    if not 1.0 <= args.interrupt_resume_seconds <= 60.0:
        parser.error("--interrupt-resume-seconds must be between 1 and 60")
    if not 0.0 <= args.interrupt_cooldown_seconds <= 120.0:
        parser.error("--interrupt-cooldown-seconds must be between 0 and 120")
    if not 1 <= args.osc_listen_port <= 65535:
        parser.error("--osc-listen-port must be between 1 and 65535")
    if not 320 <= args.vision_image_max_width <= 1920:
        parser.error("--vision-image-max-width must be between 320 and 1920")
    if not 40 <= args.vision_jpeg_quality <= 95:
        parser.error("--vision-jpeg-quality must be between 40 and 95")
    if not 0.0 <= args.owner_presence_privacy_threshold <= 10.0:
        parser.error("--owner-presence-privacy-threshold must be between 0 and 10")
    if args.visual_attention_gate and args.input_source != "vrchat":
        parser.error("--visual-attention-gate requires --input-source vrchat")
    if (
        args.avatar_mode in {"owner-follow", "presentation"}
        and not args.visual_attention_gate
    ):
        parser.error(
            "--avatar-mode owner-follow/presentation requires --visual-attention-gate"
        )
    return args


def resolve_api(model: str, requested: str = "auto") -> str:
    """"live" for gpt-live-* models, "realtime" otherwise (or as requested)."""

    requested = str(requested or "auto").strip().casefold()
    if requested in ("live", "realtime"):
        return requested
    return "live" if str(model).strip().casefold().startswith("gpt-live") else "realtime"


def _navigation_mode(value: Any) -> str | None:
    """Normalize the navigateToConsole request to auto / always / never.

    Accepts the historical booleans so an older caller keeps working, and
    returns None for anything else so the caller can reject it loudly rather
    than silently picking a behaviour nobody asked for.
    """

    if isinstance(value, bool):
        return "always" if value else "never"
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "automatic"}:
            return "auto"
        if text in {"always", "true", "yes"}:
            return "always"
        if text in {"never", "false", "no", "none", "disabled"}:
            return "never"
    return None


def missing_sourcevault_paths(args: Any) -> list[str]:
    """Names of the local-TTS inputs that must exist on disk but do not.

    Only paths belong here.  The voice's language code is a *property of the
    voice*, not a file: putting it in this check made every start fail with
    "configuration is incomplete: Piper language" because a language code is
    never an existing path.  The config is optional too -- a voice that ships
    its settings as <model>.onnx.json needs no --piper-config at all, and the
    caller passes an empty string to mean "let piper decide".
    """

    required = {
        "private broker": args.private_tts_broker,
        "SourceVault service": args.sourcevault_service_dir,
    }
    # With the AivisSpeech engine the voice is an HTTP endpoint, not a file;
    # piper paths are then an optional fallback and must not gate the start.
    if getattr(args, "tts_engine", "piper") != "aivis":
        required["Piper executable"] = args.piper_executable
        required["Piper model"] = args.piper_model
        if getattr(args, "piper_config", ""):
            required["Piper config"] = args.piper_config
    missing = [name for name, path in required.items() if not path or not Path(path).exists()]
    if not args.private_audio_dir:
        missing.append("private audio directory")
    return missing


WORKER_LOCK_NAME = "vrc-realtime-worker.lock"


def worker_lock_path() -> Path:
    """The one well-known file that says a bridge is running.

    The state and control files are named with a fresh UUID per run, so a
    Wolfram kernel that reloaded the package (which resets its private
    $iProcess) had no way to reach the worker it had started: it silently
    started a second one, and two bridges then answered the same room and
    billed the same account (2026-08-21).  A fixed path is what lets a fresh
    kernel find, observe and stop a bridge it did not launch.
    """

    return Path(tempfile.gettempdir()) / WORKER_LOCK_NAME


def write_worker_lock(args: Any) -> Path | None:
    """Record this run at the well-known path.  Never fatal."""

    path = worker_lock_path()
    payload = {
        "worker": "VRCRealtime",
        "pid": os.getpid(),
        "stateFile": str(args.state_file),
        "controlFile": str(getattr(args, "control_file", "") or ""),
        "startedAt": time.time(),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return None
    return path


def clear_worker_lock() -> None:
    """Drop the lock if it is still ours.

    Checking the pid matters: a worker that is being replaced must not delete
    the incoming worker's lock on its way out.
    """

    path = worker_lock_path()
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(recorded, dict) and recorded.get("pid") == os.getpid():
        try:
            path.unlink()
        except OSError:
            pass


def main() -> int:
    args = _parse_args()
    bridge: RealtimeBridge | None = None
    try:
        if args.enable_sourcevault:
            missing = missing_sourcevault_paths(args)
            if missing:
                raise RuntimeError(
                    "SourceVault local TTS configuration is incomplete: "
                    + ", ".join(missing)
                )
        if resolve_api(args.model, args.api) == "live":
            from VRCRealtime_live import LiveVRCBridge

            bridge = LiveVRCBridge(args)
        else:
            bridge = RealtimeBridge(args)
        write_worker_lock(args)

        def request_stop(_signum: int, _frame: Any) -> None:
            if bridge is not None:
                bridge.stop()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        try:
            return bridge.run()
        finally:
            clear_worker_lock()
    except Exception as exc:
        payload = {
            "status": "error",
            "connected": False,
            "pid": os.getpid(),
            "lastError": f"{type(exc).__name__}: {exc}",
            "updatedAt": time.time(),
        }
        try:
            Path(args.state_file).write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
