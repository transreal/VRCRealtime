"""Local spoken-command spotter for the VRChat bridge ("会話に参加" / "会話から離脱").

The visual attention gate only lets room audio reach OpenAI once somebody is
facing the avatar and speaking to it.  Before that, nothing is transcribed
anywhere, so a spoken request to join a conversation could never be heard.
This module listens to the local VRChat room audio with an offline Vosk
recogniser restricted to a tiny grammar, and reports only two events:

* ``join``     -- "会話に参加" / "会話に参加して" / "…してください"
* ``leave``    -- "会話から離脱" / "会話から離脱して" / "…してください"
* ``greeting`` -- "こんにちは" / "こんばんは" / "おはよう(ございます)" /
  "はじめまして" / "もしもし" / "ハロー" / "すみません".  A greeting only
  matters after a spoken leave: the bridge resumes when one is heard *and*
  the visual gate has a frontal face, i.e. somebody addressed the avatar.

Audio never leaves the machine here; the grammar-restricted recogniser cannot
produce any other words (everything else decodes to ``[unk]``), so no
bystander speech is transcribed.  Measured on synthetic Piper utterances: all
five phrasings recognised, negatives ("参加者は三人です", "会話を終わります")
never yield the full phrase, decoding ~0.05-0.2 s per utterance.

The model (``vosk-model-small-ja-0.22``, ~50 MB zip) is a SourceVault_voice
asset: it is looked for through ``VRCRealtime_assets.speech_model_search_path``
(``SourceVault_voice/asr`` first, the old ``%LOCALAPPDATA%/VRCRealtime/models``
still readable) and downloaded into the first of those on first use unless
``auto_download`` is disabled.  ``python VRCRealtime_voice_commands.py
--install`` fetches it ahead of time; ``--test file.wav`` decodes a file.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import VRCRealtime_assets as assets

MODEL_NAME = "vosk-model-small-ja-0.22"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
SAMPLE_RATE = 16000

#: Grammar handed to the recogniser: every accepted phrasing, tokenised the way
#: the model's vocabulary is (``words.txt`` has 会話/に/参加/から/離脱/し/て/
#: して/ください).  ``[unk]`` absorbs everything else.
#: Wake greetings.  Measured against Piper-synthesized negatives: "はじめまして"
#: had to be dropped -- the recogniser reads "(発表を)始めましょう" as it, and a
#: talk in this world says that often.  The rest never fired on 12 negatives.
GREETING_GRAMMAR = [
    "こんにちは",
    "こんばんは",
    "おはよう",
    "おはよう ございます",
    "もしもし",
    "ハロー",
    "すみません",
]
GRAMMAR = [
    "会話 に 参加",
    "会話 に 参加 し て",
    "会話 に 参加 して",
    "会話 に 参加 し て ください",
    "会話 に 参加 して ください",
    "会話 から 離脱",
    "会話 から 離脱 し て",
    "会話 から 離脱 して",
    "会話 から 離脱 し て ください",
    "会話 から 離脱 して ください",
    *GREETING_GRAMMAR,
    "[unk]",
]
#: Words that interrupt a talk in the "特定ワード" question mode
#: ("質問いいですか", "スライドを戻して", "発表を中断して", "プレゼンを止めて").
#: All are single tokens in the small-ja vocabulary (graph/words.txt); the
#: rest of such an utterance decodes to [unk].
DEFAULT_QUESTION_WORDS = ("質問", "スライド", "プレゼン", "発表")
#: Stopping a talk must not depend on a model deciding to call a tool (live
#: 2026-09-19: 「発表を終了してください」 three times, 「止めます」 said, nothing
#: stopped) nor on the question mode: these are heard during narration in
#: every mode.  All single tokens in small-ja (graph/words.txt).
TALK_WORDS = ("発表", "プレゼン")
STOP_WORDS = ("終了", "中止", "打ち切り", "終わり", "やめ")
PAUSE_WORDS = ("停止", "中断", "止め")
JOIN_PHRASES = ("会話に参加",)
LEAVE_PHRASES = ("会話から離脱",)
#: Compact (whitespace-free) forms of the greetings above, plus the spellings
#: the recogniser produces for them.
GREETING_PHRASES = (
    "こんにちは",
    "こんにちわ",
    "こんばんは",
    "こんばんわ",
    "おはよう",
    "もしもし",
    "ハロー",
    "すみません",
)


def default_model_directory() -> Path:
    """Where the Japanese small model is, or where it should be installed.

    The model is a SourceVault_voice asset now, so an already-installed copy
    wins wherever it sits -- including the pre-SourceVault_voice location
    (%LOCALAPPDATA%/VRCRealtime/models), which stays readable so nobody has to
    re-download 50 MB after the move.  Only when nothing is installed do we
    fall through to the new install target.
    """

    found = assets.find_speech_model(MODEL_NAME)
    if found is not None:
        return found
    return assets.default_speech_model_directory() / MODEL_NAME


def model_present(directory: Path) -> bool:
    return (directory / "am" / "final.mdl").is_file() and (directory / "graph").is_dir()


def install_model(
    directory: Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    timeout: float = 600.0,
) -> Path:
    """Download and unpack the Japanese small model into ``directory``."""

    target = Path(directory) if directory is not None else default_model_directory()
    if model_present(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / f"{MODEL_NAME}.zip.part"
    if progress:
        progress(f"downloading {MODEL_URL}")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "VRCRealtime/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, archive.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 18)
            if not block:
                break
            out.write(block)
            done += len(block)
            if progress and total:
                progress(f"downloading {done * 100 // total}%")
    if progress:
        progress("extracting")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        root = names[0].split("/")[0] if names else MODEL_NAME
        bundle.extractall(target.parent)
    archive.unlink(missing_ok=True)
    extracted = target.parent / root
    if extracted != target and extracted.is_dir() and not target.exists():
        extracted.rename(target)
    if not model_present(target):
        raise RuntimeError(f"model archive did not contain {MODEL_NAME}")
    if progress:
        progress("ready")
    return target


def grammar_for(question_words: Any = DEFAULT_QUESTION_WORDS) -> list[str]:
    """The recogniser grammar: the fixed phrases plus the interrupt words."""

    extra = [
        " ".join(str(word).split())
        for word in (question_words or ())
        if str(word).strip()
    ]
    body = [phrase for phrase in GRAMMAR if phrase != "[unk]"]
    extra += [w for w in TALK_WORDS + STOP_WORDS + PAUSE_WORDS if w not in extra]
    return body + [word for word in extra if word not in body] + ["[unk]"]


def control_action(text: str) -> str | None:
    """"stop" / "pause" when a grammar-restricted transcript asks the talk to
    stop: the talk itself named (発表 / プレゼン) together with a stop or pause
    word.  A pause word alone is not enough -- 「止め」 turns up in chatter."""

    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if not any(word in compact for word in TALK_WORDS):
        return None
    if any(word in compact for word in STOP_WORDS):
        return "stop"
    if any(word in compact for word in PAUSE_WORDS):
        return "pause"
    return None


def classify_text(
    text: str, question_words: Any = DEFAULT_QUESTION_WORDS
) -> str | None:
    """Map a grammar-restricted transcript onto an action.

    ``leave`` / ``join`` first, then ``question`` (an interrupt word), then
    ``greeting``: "すみません、質問いいですか" is a question during a talk.
    The bridge treats a question that also carries a greeting as the greeting
    whenever no talk is listening for interruptions.
    """

    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if any(phrase in compact for phrase in LEAVE_PHRASES):
        return "leave"
    if any(phrase in compact for phrase in JOIN_PHRASES):
        return "join"
    if control_action(compact) is not None:
        return "control"
    words = [
        "".join(str(word).split()) for word in (question_words or ()) if str(word).strip()
    ]
    if any(word in compact for word in words):
        return "question"
    if any(phrase in compact for phrase in GREETING_PHRASES):
        return "greeting"
    return None


def has_greeting(text: str) -> bool:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    return any(phrase in compact for phrase in GREETING_PHRASES)


def downmix_to_16k(chunk: NDArray[np.float32], input_rate: int) -> NDArray[np.int16]:
    """Stereo/mono float32 at ``input_rate`` -> mono int16 at 16 kHz."""

    data = np.asarray(chunk, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1, dtype=np.float32)
    if input_rate != SAMPLE_RATE:
        factor = input_rate / SAMPLE_RATE
        if abs(factor - round(factor)) < 1e-6 and factor >= 1:
            step = int(round(factor))
            usable = (data.size // step) * step
            data = data[:usable].reshape(-1, step).mean(axis=1, dtype=np.float32)
        else:
            count = int(round(data.size / factor))
            positions = np.linspace(0.0, max(data.size - 1, 0), max(count, 1))
            data = np.interp(positions, np.arange(data.size), data).astype(np.float32)
    return (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)


class VoiceCommandSpotter:
    """Feed room audio in, get ``on_command(action, text, confidence)`` out."""

    def __init__(
        self,
        *,
        on_command: Callable[[str, str, float], None],
        model_dir: str | Path | None = None,
        auto_download: bool = True,
        input_rate: int = 48000,
        min_confidence: float = 0.45,
        cooldown_seconds: float = 2.5,
        silence_rms: float = 0.006,
        flush_silence_seconds: float = 0.7,
        state_callback: Callable[..., None] | None = None,
        recognizer_factory: Callable[[], Any] | None = None,
        question_words: Any = DEFAULT_QUESTION_WORDS,
    ) -> None:
        self.on_command = on_command
        self.question_words = tuple(
            " ".join(str(word).split())
            for word in (question_words or ())
            if str(word).strip()
        )
        self.model_dir = Path(model_dir) if model_dir else default_model_directory()
        self.auto_download = auto_download
        self.input_rate = int(input_rate)
        self.min_confidence = float(min_confidence)
        self.cooldown_seconds = float(cooldown_seconds)
        #: Vosk's own endpointer does not fire reliably on every utterance --
        #: measured live, two of four spoken commands were merged into the
        #: following one and never reported.  Room audio that has been quiet
        #: for ``flush_silence_seconds`` therefore forces a FinalResult.
        self.silence_rms = float(silence_rms)
        self.flush_silence_seconds = float(flush_silence_seconds)
        self.last_voice_at = 0.0
        self.pending_audio = False
        self.state_callback = state_callback
        self._recognizer_factory = recognizer_factory
        self.audio_queue: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=200)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.suppressed_until = 0.0
        self.last_command_at = 0.0
        self.last_greeting_at = 0.0
        self.last_question_at = 0.0
        self.last_text = ""
        self.last_text_at = 0.0
        self.dropped_chunks = 0
        self.status = "idle"
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def _report(self, **updates: Any) -> None:
        if self.state_callback is not None:
            self.state_callback(**updates)

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name="VRCRealtimeVoiceCommands", daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self.thread = None

    def suppress(self, seconds: float) -> None:
        """Ignore audio for a while (the avatar itself is speaking)."""

        until = time.monotonic() + max(0.0, float(seconds))
        with self.lock:
            self.suppressed_until = max(self.suppressed_until, until)

    def feed(self, chunk: NDArray[np.float32]) -> None:
        """Audio callback entry: cheap downmix, then hand off to the thread."""

        if not self.ready.is_set() or self.stop_event.is_set():
            return
        if time.monotonic() < self.suppressed_until:
            return
        try:
            self.audio_queue.put_nowait(downmix_to_16k(chunk, self.input_rate))
        except queue.Full:
            self.dropped_chunks += 1

    # ------------------------------------------------------------------
    def _build_recognizer(self) -> Any:
        if self._recognizer_factory is not None:
            return self._recognizer_factory()
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        model = Model(str(self.model_dir))
        recognizer = KaldiRecognizer(
            model,
            SAMPLE_RATE,
            json.dumps(grammar_for(self.question_words), ensure_ascii=False),
        )
        recognizer.SetWords(True)
        return recognizer

    def _prepare_model(self) -> bool:
        if model_present(self.model_dir):
            return True
        if not self.auto_download:
            self.status = "missing"
            self._report(
                voiceCommandModelStatus="missing",
                voiceCommandModelDirectory=str(self.model_dir),
                voiceCommandReady=False,
            )
            return False
        try:
            self.status = "downloading"
            self._report(voiceCommandModelStatus="downloading", voiceCommandReady=False)
            install_model(
                self.model_dir,
                progress=lambda text: self._report(voiceCommandModelStatus=text),
            )
            return True
        except Exception as exc:
            self.status = f"error: {type(exc).__name__}: {exc}"
            self._report(voiceCommandModelStatus=self.status, voiceCommandReady=False)
            return False

    def _run(self) -> None:
        if not self._prepare_model():
            return
        try:
            recognizer = self._build_recognizer()
        except Exception as exc:
            self.status = f"error: {type(exc).__name__}: {exc}"
            self._report(voiceCommandModelStatus=self.status, voiceCommandReady=False)
            return
        self.status = "ready"
        self.ready.set()
        self._report(
            voiceCommandReady=True,
            voiceCommandModelStatus="ready",
            voiceCommandModelDirectory=str(self.model_dir),
        )
        while not self.stop_event.is_set():
            try:
                pcm = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                # No audio at all (muted, or the avatar is speaking): whatever
                # was half-decoded should still be judged.
                self._flush_pending(recognizer)
                continue
            try:
                level = float(
                    np.sqrt(np.mean(np.square(pcm.astype(np.float32) / 32768.0)))
                )
                now = time.monotonic()
                if level >= self.silence_rms:
                    self.last_voice_at = now
                    self.pending_audio = True
                if recognizer.AcceptWaveform(pcm.tobytes()):
                    self.pending_audio = False
                    self._handle_result(recognizer.Result())
                elif (
                    self.pending_audio
                    and level < self.silence_rms
                    and now - self.last_voice_at >= self.flush_silence_seconds
                ):
                    self._flush_pending(recognizer)
            except Exception as exc:  # keep listening after a decoder hiccup
                self._report(voiceCommandLastError=f"{type(exc).__name__}: {exc}")

    def _flush_pending(self, recognizer: Any) -> None:
        """End the current utterance and judge it (resets the recogniser)."""

        if not self.pending_audio:
            return
        self.pending_audio = False
        try:
            self._handle_result(recognizer.FinalResult())
        except Exception as exc:
            self._report(voiceCommandLastError=f"{type(exc).__name__}: {exc}")

    def _handle_result(self, payload: str) -> None:
        try:
            result = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return
        text = str(result.get("text", "")).strip()
        if not text:
            return
        words = result.get("result") or []
        confidences = [
            float(item.get("conf", 1.0))
            for item in words
            if isinstance(item, dict) and item.get("word") != "[unk]"
        ]
        confidence = min(confidences) if confidences else 1.0
        self.last_text = text
        #: Wall-clock of the last recognised utterance.  Callers use it to ask
        #: "has the room said anything since ...?" -- which is a different
        #: question from "did the VAD fire?", and the only one that separates
        #: a spoken question from a noise in the room.
        self.last_text_at = time.time()
        self._report(voiceCommandLastText=text, voiceCommandLastAt=self.last_text_at)
        action = classify_text(text, self.question_words)
        if action is None:
            return
        now = time.monotonic()
        if confidence < self.min_confidence:
            self._report(voiceCommandRejected=f"{action}:{confidence:.2f}")
            return
        with self.lock:
            if now - self.last_command_at < self.cooldown_seconds:
                return
            if action in ("question", "control"):
                # Like a greeting, a word the listener may repeat; it must not
                # hold the join/leave cooldown.
                if now - self.last_question_at < self.cooldown_seconds:
                    return
                self.last_question_at = now
            elif action != "greeting":
                # A greeting is a wake word the listener may repeat; only the
                # join/leave commands hold the shared cooldown.
                self.last_command_at = now
            elif now - self.last_greeting_at < self.cooldown_seconds:
                return
            else:
                self.last_greeting_at = now
        self._report(
            voiceCommandLast=action,
            voiceCommandLastConfidence=round(confidence, 3),
            voiceCommandCount=None,
        )
        try:
            self.on_command(action, text, confidence)
        except Exception as exc:
            self._report(voiceCommandLastError=f"{type(exc).__name__}: {exc}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRCRealtime voice command spotter")
    parser.add_argument("--install", action="store_true", help="download the model")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--test", default="", help="decode a WAV file and print the result")
    args = parser.parse_args(argv)
    directory = Path(args.model_dir) if args.model_dir else default_model_directory()
    if args.install:
        path = install_model(directory, progress=lambda text: print(text, flush=True))
        print(json.dumps({"status": "ready", "modelDirectory": str(path)}, ensure_ascii=False))
        return 0
    if args.test:
        import wave

        if not model_present(directory):
            print(json.dumps({"status": "missing", "modelDirectory": str(directory)}))
            return 1
        with wave.open(args.test, "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        if channels > 1:
            frames = frames.reshape(-1, channels).mean(axis=1)
        samples = frames.astype(np.float32) / 32767.0
        events: list[tuple[str, str, float]] = []
        spotter = VoiceCommandSpotter(
            on_command=lambda action, text, conf: events.append((action, text, conf)),
            model_dir=directory,
            auto_download=False,
            input_rate=rate,
        )
        recognizer = spotter._build_recognizer()
        pcm = downmix_to_16k(samples, rate)
        for start in range(0, pcm.size, 4000):
            if recognizer.AcceptWaveform(pcm[start : start + 4000].tobytes()):
                spotter._handle_result(recognizer.Result())
        spotter._handle_result(recognizer.FinalResult())
        print(json.dumps({"text": spotter.last_text, "events": events}, ensure_ascii=False))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
