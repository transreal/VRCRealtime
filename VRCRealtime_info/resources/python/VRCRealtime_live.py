"""GPT-Live (gpt-live-*) for the VRChat bridge.

``LiveVRCBridge`` speaks the GPT-Live protocol (``wss://api.openai.com/v1/
live/sessions``) and translates it to and from the Realtime-shaped events the
rest of ``VRCRealtime_worker.RealtimeBridge`` was written against, so the
presentation, SourceVault, owner-follow and vision code runs unchanged.

What differs, and how it is bridged:

* Tools.  GPT-Live itself has none; it delegates.  This bridge uses
  **Responses delegation**: the tool list (the same one the Realtime session
  gets) goes to a Responses backend model (``--live-backend-model``), whose
  finished ``function_call`` items arrive nested in ``response.event``.  They
  are handed to the unchanged ``_handle_response_done``; the results go back
  with ``response.item.create`` followed by ``response.create`` (continue).
  When a handler asks for a reply with specific wording
  (``_request_response(purpose, instructions)``), the wording travels inside
  the tool result as ``reply_instructions``: the backend composes the reply
  and GPT-Live speaks it.
* Speech.  There is no response.create / response.done for speech: GPT-Live
  decides when to talk.  An utterance is a stretch of voiced output audio; it
  starts a period (the purpose a handler asked for, if any is waiting) and is
  over once no voiced audio has arrived for a while.  A requested line is
  ``session.instructions.append``.  A script is asked for with an
  instruction that says "now" and carries the words (the form the Live
  guide gives for exact wording), sent once the model has fallen quiet.
  Nothing goes in as private context (``session.thinking.append``):
  GPT-Live spoke what was put there -- labels, notes, rules -- and then
  ignored the request to read.  Standing rules are in the session
  instructions.  A script counts as heard only when most of it is in the
  output transcript.
* Input.  The Live timeline advances with input frames, so a frame is sent
  every chunk: the room when the Realtime bridge would have sent it, silence
  otherwise.  Speech start / end are derived from the input transcript.
* Images.  GPT-Live hears; the backend sees.  A slide without a script gets
  one written from its image by the backend model (Responses HTTP), ahead of
  time for the next slide.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import websocket

from VRCRealtime_worker import (
    OPENAI_INPUT_RATE,
    RealtimeBridge,
    presentation_control_intent,
    SlidenCancelled,
    SlidenError,
    _clear_queue,
    resume_text,
)

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
RESPONSES_URL = "https://api.openai.com/v1/responses"
#: session.instructions/thinking/commentary.append carry at most 500 tokens.
LIVE_APPEND_MAX_TOKENS = 460
#: Script carried by one narration piece.
LIVE_NARRATION_CHUNK_CHARS = 260
#: Output chunks whose peak is below this count as silence, not speech.
LIVE_VOICED_PEAK = 0.01
#: A script has been read when this share of it is in the output transcript.
LIVE_NARRATION_COVERAGE = 0.8
#: Purposes whose speech is a script (coverage decides when it is over).
NARRATION_PURPOSES = ("presentation_slide:", "presentation_opening:", "presentation_closing:")

#: Where the exchange with GPT-Live is recorded (no audio): what was sent,
#: acknowledged, said and heard, for working out afterwards why a slide was
#: or was not read.
#: Realtime instructions belong to one response; GPT-Live keeps every
#: instruction for the rest of the session.  「…とだけ言ってください」 and
#: 「読み終えたら黙って待ってください」, sent that way, left it answering
#: nobody after a talk (live 2026-09-19).  Every one-off request says so.
ONE_TIME_SCOPE = (
    "(この指示はこの一回の発話だけのものです。終わったら、話しかけられたときは"
    "普段どおり応じてください。)"
)
#: Once a talk is over: the reading rules are lifted.
TALK_OVER = (
    "発表は終わりました。原稿の読み上げや発表中の一回限りの発話の指示は、もう"
    "当てはまりません。ここからは普段どおりの会話です。話しかけられたら普段どおり"
    "応じてください。今は何も言わなくて構いません。"
)
LIVE_EVENT_LOG_NAME = "vrc-live-events.jsonl"
LIVE_EVENT_LOG_MAX_BYTES = 5 * 1024 * 1024
#: Too frequent to log one by one (speech begin/end and turns are logged).
_UNLOGGED_EVENTS = frozenset(
    {
        "session.output_audio.delta",
        "session.output_transcript.delta",
        "session.input_transcript.delta",
        "session.usage.updated",
    }
)

_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")
_NON_SPOKEN = re.compile(r"[\s、。，．,.！？!?「」『』（）()・…ー-]")
_RESUME_SPOKEN = re.compile(r"^.{0,24}?戻ります[。、!！]?")


def estimate_tokens(text: str) -> int:
    """Rough token count: CJK about one token a character, other text a quarter."""

    wide = sum(1 for ch in text if ord(ch) > 0x2E7F)
    return wide + (len(text) - wide + 3) // 4


def split_sentences(text: str) -> list[str]:
    return [piece for piece in _SENTENCE_END.split(str(text)) if piece.strip()]


def append_chunks(text: str, limit: int = LIVE_APPEND_MAX_TOKENS) -> list[str]:
    """Split context text into appends under the token cap (nothing dropped)."""

    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(str(text).strip()):
        while estimate_tokens(sentence) > limit:
            cut = max(1, int(len(sentence) * limit / max(1, estimate_tokens(sentence))) - 2)
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut])
            sentence = sentence[cut:]
        if current and estimate_tokens(current + sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def narration_chunks(text: str, limit: int = LIVE_NARRATION_CHUNK_CHARS) -> list[str]:
    """A script in pieces of at most ``limit`` characters, cut between
    sentences (or at a comma inside a very long one)."""

    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(str(text).strip()):
        while len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            cut = sentence.rfind("、", 0, limit)
            cut = cut + 1 if cut > limit // 3 else limit
            chunks.append(sentence[:cut])
            sentence = sentence[cut:]
        if current and len(current) + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def plain_length(text: str) -> int:
    return len(_NON_SPOKEN.sub("", str(text)))


def response_output_text(payload: Any) -> str:
    """The text of a Responses API result."""

    if not isinstance(payload, dict):
        return ""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in ("output_text", "text"):
                parts.append(str(content.get("text", "")))
    return "".join(parts).strip()


class LiveVRCBridge(RealtimeBridge):
    """The VRChat bridge over GPT-Live with Responses delegation."""

    API = "live"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._event_counter = 0
        self.session_id: str | None = None
        self.ack_lock = threading.Lock()
        self.acks: dict[str, threading.Event] = {}
        #: Held while a frame (or a replayed utterance) goes out, so the
        #: clocked sender never interleaves silence into a replay.
        self.send_lock = threading.Lock()
        self.live_lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.event_log_path = Path(
            self._opt("live_event_log", "")
            or Path(tempfile.gettempdir()) / LIVE_EVENT_LOG_NAME
        )
        #: Speech a handler asked for and that has not started yet.
        self.expected: deque[dict[str, Any]] = deque()
        self.period_seq = 0
        self.current_item_id = ""
        self.period_narration = False
        self._done_since = 0.0
        self.last_voiced_output_at = 0.0
        self.discard_until_gap = False
        self.discard_last_voiced = 0.0
        self.discard_deadline = 0.0
        #: The script piece being read, and what earlier pieces of the same
        #: narration already said (for resuming after an interruption).
        self.narration_target = ""
        self.narration_prefix = ""
        #: Responses backend: items waiting to be sent, calls waiting for
        #: their results, and whether a backend response is running.
        self.backend_items: list[dict[str, Any]] = []
        self.backend_flush_at = 0.0
        self.backend_calls_pending = 0
        self.backend_active = False
        #: When the backend last started writing a reply (monotonic).
        self.backend_text_at = 0.0
        self.delegations = 0
        #: The room's speech, from the input transcript.
        self.user_turn_open = False
        self.user_text = ""
        self.last_user_delta_at = 0.0
        self.user_turn_ended_at = 0.0
        self.awaiting_reply_since = 0.0
        self.history: deque[dict[str, str]] = deque(maxlen=24)
        #: The room's recent turns with their time: a request split over
        #: two transcript turns (「計算と自然三」 / 「十三のプレゼンを…」)
        #: is read whole.
        self.recent_user_turns: deque[tuple[float, str]] = deque(maxlen=10)
        #: The script piece a talk interruption cut, and when: GPT-Live may
        #: still read it from an instruction sent just before the cut.
        self.cut_script = ""
        self.cut_at = 0.0
        #: Scripts written from slide images: path -> text.
        self.slide_scripts: dict[str, str] = {}
        self.slide_script_threads: dict[str, threading.Thread] = {}
        self._write_state(
            api=self.API,
            delegation="responses",
            backendModel=self._opt("live_backend_model", "gpt-5.6-luna"),
            sessionId=None,
            usageSeconds=0,
            contextUsage=None,
            delegations=0,
            lastDelegation=None,
            backendActive=False,
            closeReason=None,
            liveEventLog=str(self.event_log_path),
            fullDuplex=bool(self._opt("live_full_duplex", True)),
        )

    # ------------------------------------------------------------ plumbing

    def _opt(self, name: str, default: Any) -> Any:
        value = getattr(self.args, name, None)
        return default if value is None else value

    def _ws_url(self) -> str:
        return LIVE_URL

    def _run_forever_options(self) -> dict[str, Any]:
        # websocket-client sends "Origin: https://api.openai.com" unless told
        # not to, and GPT-Live answers a request with an Origin (a browser) by
        # a bare 403 even when the key is fine (measured 2026-09-19).
        return {"suppress_origin": True}

    def _extra_threads(self) -> list[threading.Thread]:
        return [
            threading.Thread(
                target=self._live_housekeeping,
                daemon=True,
                name="VRCRealtimeLiveHousekeeping",
            )
        ]

    def _handshake_message(self, status: int) -> str:
        if status == 403:
            return (
                "GPT-Live への接続が拒否されました (403)。この API キー / プロジェクトで "
                f"{self.args.model} が使えるか (権限・モデル制限) を確認してください。"
            )
        return super()._handshake_message(status)

    def _next_event_id(self, prefix: str) -> str:
        with self.live_lock:
            self._event_counter += 1
            return f"vrc_{prefix}_{self._event_counter}"

    def _send_raw(self, event: dict[str, Any]) -> bool:
        return RealtimeBridge._send_now(self, event)

    def _log(self, event: str, **fields: Any) -> None:
        """One line of the event log.  Never fatal."""

        record = {"t": round(time.time(), 3), "event": event}
        record.update(fields)
        try:
            with self.log_lock:
                path = self.event_log_path
                if path.exists() and path.stat().st_size > LIVE_EVENT_LOG_MAX_BYTES:
                    os.replace(path, path.with_suffix(".jsonl.1"))
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _append(self, kind: str, delegation_id: str | None, content: str) -> str | None:
        """session.<kind>.append; the event id (see _wait_ack) or None."""

        text = str(content).strip()
        if not text:
            return None
        if estimate_tokens(text) > LIVE_APPEND_MAX_TOKENS:
            text = append_chunks(text)[0]
        event_id = self._next_event_id(kind)
        with self.ack_lock:
            if len(self.acks) > 200:
                self.acks.clear()
            self.acks[event_id] = threading.Event()
        sent = self._send_raw(
            {
                "type": f"session.{kind}.append",
                "event_id": event_id,
                "delegation_id": delegation_id,
                "content": text,
            }
        )
        self._log("send", kind=kind, id=event_id, sent=sent, content=text[:160])
        return event_id if sent else None

    def _ack(self, event_id: Any) -> None:
        with self.ack_lock:
            waiter = self.acks.get(str(event_id)) if event_id else None
        if waiter is not None:
            waiter.set()

    def _wait_ack(self, event_id: str | None, timeout: float) -> bool:
        if not event_id:
            return False
        with self.ack_lock:
            waiter = self.acks.get(event_id)
        if waiter is None:
            return False
        done = waiter.wait(timeout)
        with self.ack_lock:
            self.acks.pop(event_id, None)
        return done

    # ------------------------------------------------------------ session

    def _tools(self) -> list[dict[str, Any]]:
        tools = []
        for tool in self._session_update()["session"].get("tools", []):
            entry = dict(tool)
            # Responses defaults function tools to strict schemas, which the
            # optional arguments here (show_slide, look_at_vrchat) are not.
            entry["strict"] = False
            tools.append(entry)
        return tools

    def _front_instructions(self) -> str:
        lines = [
            str(self.instructions or "").strip(),
            "あなたはVRChatの中にいる音声アバターです。",
            "調べもの (保存資料・発表の内容・ウェブ)、発表の開始・一時停止・再開・停止、"
            "スライドの表示、アバターのモード切替や旋回、目の前の画面を見ること、"
            "会話への参加・離脱、質問モードの切り替えは、自分で『やりました』と言わず、"
            "必ずバックエンドに委譲してください。結果が返ったら、その内容を短く自然に"
            "伝えてください。『質問モードを切り替えたい』と言われたら、"
            "質問タイム・特定ワード・二段構えの3つを短く説明してどれにするか尋ね、"
            "答えを聞いたらバックエンドに委譲してください。",
            "原稿を読むよう指示されたときは、いつでも必ず、その原稿だけを書かれた"
            "とおりに声に出して読み、前置きや感想を足さず、読み終えたらその話を"
            "終えてください (その後も話しかけられたら普段どおり応じる)。",
            "聞き手の独り言・相槌・雑談・咳や物音には反応しないでください。"
            "あなたへの質問や呼びかけのときだけ答えてください。",
            # What the Realtime bridge says per window as conversation
            # items.  GPT-Live read injected notes aloud (「参考情報です」,
            # 「ただし、この規則は…」, live 2026-09-19), so they stand here,
            # where nothing is ever spoken.
            "発表中の決まり: スライドの区切りで『質問はありますか』と自分から"
            "尋ねることはしません。質問には日本語で簡潔に答え、発表資料や保存資料の事実は"
            "推測せずバックエンドに委譲して確かめます。『一時停止して』『中断して』"
            "『停止して』『止めて』は一時停止、『発表を終了して』『打ち切って』"
            "『もうやめて』は終了、『再開して』は再開として、バックエンドに委譲します"
            "(迷うときは一時停止)。再開位置を尋ねた後の返事 (『このまま』『戻って』"
            "など) もバックエンドに委譲します。",
            "原稿の読み上げを途中で打ち切られた直後に聞こえた発話は、すでに内容を"
            "受け取っています。『はい、どうぞ』と促し直さず、依頼で内容が分かれば"
            "『了解です』とだけ言ってから委譲し (結果を伝えるときに繰り返さない)、"
            "質問なら前置きなしにそのまま答え、『質問いいですか』だけなら"
            "『はい、どうぞ』と言い、聞き取れない・判断できないときは"
            "『すみません、もう一度お願いします』と言います。",
        ]
        return "\n".join(line for line in lines if line)

    def _backend_instructions(self) -> str:
        return (
            self._effective_instructions()
            + "\nあなたは音声会話のバックエンドです。返答は音声対話モデルが話すので、"
            "日本語の短い話し言葉の平文で返してください (箇条書き・記号・URL・"
            "マークダウンは使わない)。ツールの結果 (JSON) に reply_instructions が"
            "あれば、それに従って返答を作ってください。"
        )

    def _session_start_event(self) -> dict[str, Any]:
        backend: dict[str, Any] = {
            "model": self._opt("live_backend_model", "gpt-5.6-luna"),
            "instructions": self._backend_instructions(),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        tools = self._tools()
        if tools:
            backend["tools"] = tools
        effort = str(self._opt("live_backend_reasoning", "low") or "")
        if effort:
            backend["reasoning"] = {"effort": effort}
        session: dict[str, Any] = {
            "model": self.args.model,
            "instructions": self._front_instructions(),
            "audio": {
                "format": {"type": "audio/pcm", "rate": OPENAI_INPUT_RATE},
                "output": {"voice": self.args.voice},
            },
            "delegation": {"type": "responses", "responses": backend},
        }
        seed = self._seed_history()
        if seed:
            session["input"] = seed
        return {
            "type": "session.start",
            "event_id": self._next_event_id("start"),
            "session": session,
        }

    def _seed_history(self) -> list[dict[str, Any]]:
        """Recent text for a replacement session (empty on the first one)."""

        items: list[dict[str, Any]] = []
        budget = 4000
        for entry in reversed(list(self.history)):
            text = entry["text"][:400]
            cost = estimate_tokens(text)
            if cost > budget:
                break
            budget -= cost
            part = "input_text" if entry["role"] == "user" else "output_text"
            items.append(
                {
                    "type": "message",
                    "role": entry["role"],
                    "content": [{"type": part, "text": text}],
                }
            )
        items.reverse()
        return items

    def _remember(self, role: str, text: str) -> None:
        text = " ".join(str(text).split())
        if text:
            self.history.append({"role": role, "text": text})

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        # Audio waits for session.started: ws_ready is set there, not here.
        with self.ws_lock:
            self.ws = ws
        try:
            ws.send(
                json.dumps(
                    self._session_start_event(),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            self._write_state(lastError=f"session.start: {type(exc).__name__}: {exc}")
            ws.close()

    def _on_close(self, ws: Any, status_code: int | None, message: str | None) -> None:
        with self.live_lock:
            self.backend_items = []
            self.backend_calls_pending = 0
            self.backend_active = False
        super()._on_close(ws, status_code, message)

    # ------------------------------------------------------------ outbound

    def _send_now(self, event: dict[str, Any]) -> bool:
        """Realtime-shaped event in, GPT-Live event(s) out."""

        kind = str(event.get("type", ""))
        if kind == "input_audio_buffer.append":
            return self._send_raw(
                {"type": "session.input_audio.append", "audio": event.get("audio", "")}
            )
        if kind in (
            "input_audio_buffer.clear",
            "input_audio_buffer.commit",
            "conversation.item.truncate",
            "session.update",
        ):
            # No buffer to clear or commit (the model takes turns itself) and
            # no truncation; the session is configured once, at session.start.
            return True
        if kind == "response.cancel":
            self.response_in_progress.clear()
            return True
        if kind == "conversation.item.create":
            return self._live_item(event.get("item") or {})
        if kind == "response.create":
            response = event.get("response") or {}
            metadata = response.get("metadata") or {}
            self._request_response(
                purpose=str(metadata.get("vrc_realtime_purpose", "normal")),
                instructions=response.get("instructions"),
            )
            return True
        return self._send_raw(event)

    def _live_item(self, item: dict[str, Any]) -> bool:
        kind = item.get("type")
        if kind == "function_call_output":
            self._queue_backend_item(item)
            return True
        if kind != "message":
            return True
        content = [part for part in item.get("content") or [] if isinstance(part, dict)]
        if any(part.get("type") == "input_image" for part in content):
            # Only the backend sees.
            self._queue_backend_item(item)
            return True
        # A text-only note (a question window's rules, the resume-point
        # rules, "local readout finished") is not passed on.  GPT-Live read
        # injected notes aloud -- 「質問時間は終わりました」, 「ただし、この規則
        # は…」, 「参考情報です」 (live 2026-09-19) -- whatever they said about
        # not being read.  What the notes carry stands in the session
        # instructions (_front_instructions), and anything to be said is
        # asked for by the instruction that follows each note.
        with self.state_lock:
            count = int(self.state.get("liveContextNotesWithheld") or 0) + 1
        self._set_state(liveContextNotesWithheld=count)
        return True

    def _queue_backend_item(self, item: dict[str, Any]) -> None:
        with self.live_lock:
            self.backend_items.append(dict(item))
            # Sent with the reply request that normally follows at once; on
            # its own after a moment (the backend must not wait forever).
            self.backend_flush_at = time.monotonic() + 1.0

    def _flush_backend(self, purpose: str = "normal", instructions: str | None = None) -> None:
        with self.live_lock:
            items = self.backend_items
            self.backend_items = []
        if not items:
            return
        if instructions:
            for index in range(len(items) - 1, -1, -1):
                item = items[index]
                if item.get("type") != "function_call_output":
                    continue
                try:
                    payload = json.loads(str(item.get("output") or "{}"))
                except json.JSONDecodeError:
                    payload = {"result": item.get("output")}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                payload["reply_instructions"] = instructions
                items[index] = dict(
                    item,
                    output=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                )
                break
            else:
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "返答の指示: " + instructions}
                        ],
                    }
                )
        outputs = sum(1 for item in items if item.get("type") == "function_call_output")
        for item in items:
            self._send_raw(
                {
                    "type": "response.item.create",
                    "event_id": self._next_event_id("item"),
                    "item": item,
                }
            )
        self._send_raw({"type": "response.create", "event_id": self._next_event_id("continue")})
        with self.live_lock:
            self.backend_calls_pending = max(0, self.backend_calls_pending - outputs)
            self.backend_active = True
        self._write_state(backendActive=True)
        if purpose != "normal" or instructions:
            self._expect(purpose, backend=True)

    def _request_response(
        self, purpose: str = "normal", instructions: str | None = None
    ) -> None:
        """What a Realtime response.create asked for, done the Live way (at
        once: nothing here waits for a previous response to end)."""

        with self.live_lock:
            has_backend = bool(self.backend_items)
        if has_backend:
            self._flush_backend(purpose, instructions)
            return
        if instructions:
            scoped = instructions + ONE_TIME_SCOPE
            self._expect(
                purpose,
                timeout=30.0,
                narration=purpose.startswith(NARRATION_PURPOSES),
                resend=scoped,
            )
            self._append("instructions", None, scoped)
            return
        if purpose == "question_answer":
            self._expect_reply(purpose)
        # Otherwise: a continuation after a tool result already flushed, or a
        # reply the model gives by itself.  Nothing to send.

    # --------------------------------------------------------- speech periods

    def _expect(
        self,
        purpose: str,
        *,
        backend: bool = False,
        timeout: float | None = None,
        narration: bool = False,
        resend: str | None = None,
    ) -> None:
        now = time.monotonic()
        with self.live_lock:
            self.expected.append(
                {
                    "purpose": purpose,
                    "backend": backend,
                    "narration": narration,
                    "created": now,
                    "deadline": None if backend else now + float(timeout or 20.0),
                    "resend": resend,
                    "resent": False,
                }
            )

    def _expect_reply(self, purpose: str) -> None:
        """A turn the model should answer by itself (question windows)."""

        if self.output_active.is_set():
            with self.output_purpose_lock:
                if self.current_output_purpose == "normal":
                    self.current_output_purpose = purpose
            return
        if self.last_voiced_output_at > self.user_turn_ended_at > 0.0:
            # Already answered before anybody asked for it.
            self._emulate_empty(purpose)
            return
        self._expect(purpose, timeout=8.0)

    def _backend_idle(self) -> bool:
        with self.live_lock:
            return (
                not self.backend_active
                and not self.backend_items
                and self.backend_calls_pending == 0
            )

    def _entry_ready(self, entry: dict[str, Any]) -> bool:
        """May the next utterance be the speech *entry* asked for?

        A backend reply is ready once the backend is done -- or once it has
        started writing its answer: GPT-Live speaks the text as it streams in,
        before the nested response.completed, and treating that speech as
        unrelated left the reply's purpose (and a script queued after it)
        waiting for an utterance that had already happened."""

        if not entry["backend"]:
            return True
        return self._backend_idle() or self.backend_text_at >= entry["created"]

    def _take_expected(self) -> tuple[str, bool]:
        with self.live_lock:
            for entry in list(self.expected):
                if not self._entry_ready(entry):
                    continue
                self.expected.remove(entry)
                return str(entry["purpose"]), bool(entry["narration"])
        return "normal", False

    def _begin_period(self) -> None:
        self.period_seq += 1
        self.current_item_id = f"live_{self.period_seq}"
        purpose, narration = self._take_expected()
        with self.output_purpose_lock:
            self.current_output_purpose = purpose
        self.period_narration = narration or purpose.startswith(NARRATION_PURPOSES)
        self.response_in_progress.set()
        self.awaiting_reply_since = 0.0
        self._touch_visual_session("response_created")
        self._set_state(lastTranscript="", speaking=True)
        self._log("speech_begin", purpose=purpose, item=self.current_item_id)

    def _end_period(self) -> None:
        self.response_in_progress.clear()
        with self.state_lock:
            said = str(self.state.get("lastTranscript", ""))
        self._remember("assistant", said)
        self._touch_visual_session("response_done")
        self._set_state(speaking=False)
        with self.output_purpose_lock:
            purpose = self.current_output_purpose
        self._log("speech_end", purpose=purpose, said=said[-200:])
        self.output_done.set()

    def _emulate_empty(self, purpose: str) -> None:
        """A requested utterance that never came: close it as if it had."""

        with self.output_purpose_lock:
            self.current_output_purpose = purpose
        self._set_state(lastTranscript="")
        self._log("speech_missing", purpose=purpose)
        self.response_in_progress.clear()
        self.output_active.set()
        self.output_done.set()

    def _on_output_audio(self, audio: str) -> None:
        try:
            raw = base64.b64decode(audio)
        except Exception as exc:
            self._set_state(lastError=f"audio decode: {type(exc).__name__}: {exc}")
            return
        pcm = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2")
        peak = float(np.max(np.abs(pcm.astype(np.int32)))) / 32768.0 if pcm.size else 0.0
        voiced = peak >= LIVE_VOICED_PEAK
        now = time.monotonic()
        if self.discard_until_gap:
            # The tail of a cut utterance keeps coming until the model has
            # taken the stop in; the first pause ends it.
            if now > self.discard_deadline or now - self.discard_last_voiced > 0.6:
                self.discard_until_gap = False
            else:
                if voiced:
                    self.discard_last_voiced = now
                return
        if self.privacy_muted.is_set() or self.private_playback.is_set():
            return
        if not self.output_active.is_set():
            if not voiced:
                return
            self._begin_period()
        elif self.output_done.is_set():
            # GPT-Live streams output frames all the time, silence included.
            # Once an utterance has been closed its silence must not be
            # queued: the playback worker ends the period only when the queue
            # runs dry, and a steady trickle of silent frames kept it from
            # ever doing so -- the bridge stayed "speaking", deaf to the room,
            # and the first slide timed out waiting (live 2026-09-19).
            if not voiced:
                return
            # Talking again before the queue drained: the same utterance goes on.
            self.output_done.clear()
            self.response_in_progress.set()
            self._set_state(speaking=True)
        if voiced:
            self.last_voiced_output_at = now
        self._queue_output_delta(
            {"delta": audio, "item_id": self.current_item_id, "content_index": 0}
        )

    def _interrupt_playback(self) -> None:
        now = time.monotonic()
        speaking = self.output_active.is_set() or now - self.last_voiced_output_at < 1.5
        super()._interrupt_playback()
        self.response_in_progress.clear()
        if speaking:
            self.discard_until_gap = True
            self.discard_last_voiced = now
            self.discard_deadline = now + 6.0
            self._append(
                "instructions",
                None,
                "いま話している内容はそこで打ち切り、続きは言わないでください。",
            )

    def _narration_coverage(self) -> float:
        target = plain_length(self.narration_target)
        if target <= 0:
            return 1.0
        with self.state_lock:
            spoken = str(self.state.get("lastTranscript", ""))
        return plain_length(_RESUME_SPOKEN.sub("", spoken, count=1)) / target

    def _stop_leftover_script(self, transcript: str) -> None:
        """Cut a script that is being read after the talk was interrupted.

        A read request sent just before a cut is still carried out by
        GPT-Live: it read the interrupted slide on while the room was saying
        「発表を終了してください」 (live 2026-09-19)."""

        script = self.cut_script
        if not script or time.monotonic() - self.cut_at > 8.0:
            return
        with self.output_purpose_lock:
            purpose = self.current_output_purpose
        if purpose.startswith("presentation_") or not self.output_active.is_set():
            return
        said = _NON_SPOKEN.sub("", transcript)
        if len(said) < 8 or said[-8:] not in _NON_SPOKEN.sub("", script):
            return
        self.cut_script = ""
        self._log("leftover_script_cut", said=transcript[-80:])
        self._interrupt_playback()

    def _spoken_so_far(self) -> str:
        return self.narration_prefix + super()._spoken_so_far()

    # -------------------------------------------------------------- inbound

    def _on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            self._write_state(lastError="received malformed JSON")
            return
        kind = str(event.get("type", ""))
        self._set_state(lastEvent=kind)
        nested = event.get("event") if kind == "response.event" else None
        nested_type = str(nested.get("type", "")) if isinstance(nested, dict) else ""
        if kind not in _UNLOGGED_EVENTS and not nested_type.endswith(".delta"):
            item = (nested or {}).get("item") if isinstance(nested, dict) else None
            self._log(
                "recv",
                kind=kind,
                ack=event.get("client_event_id"),
                nested=(nested or {}).get("type") if isinstance(nested, dict) else None,
                name=item.get("name") if isinstance(item, dict) else None,
                arguments=(
                    str(item.get("arguments"))[:200]
                    if isinstance(item, dict) and item.get("type") == "function_call"
                    else None
                ),
                error=(event.get("error") if kind == "error" else None),
            )
        if kind == "session.output_audio.delta":
            audio = event.get("delta")
            if isinstance(audio, str) and audio:
                self._on_output_audio(audio)
        elif kind == "session.output_transcript.delta":
            if not self.discard_until_gap:
                delta = str(event.get("delta", ""))
                with self.state_lock:
                    transcript = str(self.state.get("lastTranscript", "")) + delta
                self._set_state(lastTranscript=transcript[-4000:])
                self._stop_leftover_script(transcript)
        elif kind == "session.input_transcript.delta":
            self._on_user_delta(str(event.get("delta", "")))
        elif kind == "response.event":
            self._on_backend_event(event)
        elif kind == "session.delegation.created":
            self.delegations += 1
            self._write_state(delegations=self.delegations)
        elif kind == "session.started":
            session = event.get("session") or {}
            self.session_id = str(session.get("id", "")) or None
            self.ws_ready.set()
            self.session_started_at = time.monotonic()
            self._write_state(
                status="connected",
                connected=True,
                sessionId=self.session_id,
                lastError=None,
            )
        elif kind == "session.usage.updated":
            usage = event.get("usage") or {}
            window = event.get("context_window") or {}
            self._write_state(
                usageSeconds=usage.get("seconds", 0),
                contextUsage=window.get("usage_ratio"),
            )
        elif kind == "session.closed":
            self.ws_ready.clear()
            self._write_state(closeReason=event.get("reason"), connected=False)
            with self.ws_lock:
                ws = self.ws
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        elif kind == "error":
            detail = event.get("error", event)
            self._write_state(lastError=json.dumps(detail, ensure_ascii=False)[:400])
            if isinstance(detail, dict):
                # A refused append must not leave a waiter hanging.
                self._ack(detail.get("client_event_id"))
            self._ack(event.get("client_event_id"))
        elif kind.endswith(".appended"):
            self._ack(event.get("client_event_id"))

    def _emit(self, event_type: str) -> None:
        """Feed a Realtime-shaped event to the Realtime handler."""

        RealtimeBridge._on_message(self, None, json.dumps({"type": event_type}))

    def _on_user_delta(self, delta: str) -> None:
        if not delta:
            return
        now = time.monotonic()
        if not self.user_turn_open:
            self.user_turn_open = True
            self.user_text = ""
            self._emit("input_audio_buffer.speech_started")
        self.user_text += delta
        self.last_user_delta_at = now
        self._set_state(lastUserText=self.user_text[-2000:])

    def _on_backend_event(self, envelope: dict[str, Any]) -> None:
        nested = envelope.get("event") or {}
        kind = str(nested.get("type", ""))
        if kind == "response.created":
            with self.live_lock:
                self.backend_active = True
            self.delegations += 1
            self._write_state(backendActive=True, delegations=self.delegations)
        elif kind in ("response.output_text.delta", "response.content_part.added") or (
            kind == "response.output_item.added"
            and (nested.get("item") or {}).get("type") == "message"
        ):
            self.backend_text_at = time.monotonic()
        elif kind == "response.output_item.done":
            item = nested.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = str(item.get("call_id", ""))
                if call_id and call_id not in self.handled_tool_calls:
                    with self.live_lock:
                        self.backend_calls_pending += 1
                self._write_state(lastDelegation=item.get("name"))
                self._handle_response_done({"response": {"output": [item]}})
        elif kind in (
            "response.completed",
            "response.done",
            "response.failed",
            "response.incomplete",
            "response.cancelled",
        ):
            with self.live_lock:
                self.backend_active = False
            updates: dict[str, Any] = {"backendActive": False}
            if kind == "response.failed":
                response = nested.get("response") or {}
                updates["lastError"] = "backend: " + json.dumps(
                    response.get("error") or response, ensure_ascii=False
                )[:300]
            self._write_state(**updates)
        elif kind == "error":
            self._write_state(
                lastError="backend: " + json.dumps(nested, ensure_ascii=False)[:300]
            )

    # ------------------------------------------------------------ housekeeping

    def _live_housekeeping(self) -> None:
        while not self.stop_event.wait(0.05):
            try:
                self._housekeeping_tick()
            except Exception as exc:  # keep going; say what broke
                self._write_state(lastError=f"housekeeping: {type(exc).__name__}: {exc}")

    def _housekeeping_tick(self) -> None:
        now = time.monotonic()
        # A tool result nobody asked a reply for still has to reach the backend.
        with self.live_lock:
            flush = bool(self.backend_items) and now >= self.backend_flush_at
        if flush:
            self._flush_backend()
        # End of the room's utterance.
        if self.user_turn_open and now - self.last_user_delta_at >= 0.9:
            self.user_turn_open = False
            self.user_turn_ended_at = now
            self._remember("user", self.user_text)
            self._log("heard", text=self.user_text[-200:])
            self.recent_user_turns.append((now, self.user_text))
            intent = presentation_control_intent(self.user_text)
            if intent is not None and self._presentation_running():
                # Not left to the model: it answered 「発表を終了します」
                # and never called the tool (live 2026-09-19).
                acted = self._apply_voice_control(intent, self.user_text, source="transcript")
                self._log("voice_control", action=intent, acted=acted, text=self.user_text[-120:])
            self._emit("input_audio_buffer.speech_stopped")
            self._emit("input_audio_buffer.committed")
            if self.presentation_qa_open.is_set() and not self.output_active.is_set():
                self.awaiting_reply_since = now
        # End of an utterance of ours.
        if (
            self.output_active.is_set()
            and not self.output_done.is_set()
            and not self.private_playback.is_set()
            and self.response_in_progress.is_set()
        ):
            quiet = now - self.last_voiced_output_at
            idle_ms = (
                self._opt("live_narration_idle_ms", 1400)
                if self.period_narration
                else self._opt("live_idle_ms", 900)
            )
            if self.period_narration and self._narration_coverage() >= 0.95:
                # The whole piece has been said: hand over to the next one
                # without the long pause meant for a breath mid-script.
                idle_ms = min(float(idle_ms), 600.0)
            if quiet >= float(idle_ms) / 1000.0:
                stalled = quiet >= float(self._opt("live_narration_stall_ms", 6000)) / 1000.0
                if (
                    not self.period_narration
                    or stalled
                    or self._narration_coverage() >= LIVE_NARRATION_COVERAGE
                ):
                    self._end_period()
        # A closed utterance whose playback never drains (the worker ends the
        # period only on an empty queue) must not hold the bridge "speaking":
        # everything else -- the room's audio, the next script -- waits on it.
        if self.output_active.is_set() and self.output_done.is_set():
            if self._done_since <= 0.0:
                self._done_since = now
            elif now - self._done_since > 4.0:
                _clear_queue(self.output_audio)
                self._write_state(livePlaybackDrained=time.time())
                self._done_since = now
        else:
            self._done_since = 0.0
        # Requested speech that never started.
        expired: list[dict[str, Any]] = []
        resend: list[str] = []
        with self.live_lock:
            for entry in list(self.expected):
                if entry["backend"] and entry["deadline"] is None and self._entry_ready(entry):
                    entry["deadline"] = now + 12.0
                if (
                    entry["narration"]
                    and entry["resend"]
                    and not entry["resent"]
                    and now - entry["created"] > 5.0
                    and not self.output_active.is_set()
                ):
                    entry["resent"] = True
                    resend.append(str(entry["resend"]))
                if (
                    entry["deadline"] is not None
                    and now >= entry["deadline"]
                    and not self.output_active.is_set()
                ):
                    self.expected.remove(entry)
                    expired.append(entry)
        for text in resend:
            self._append("instructions", None, text)
        for entry in expired:
            purpose = str(entry["purpose"])
            self._write_state(liveSpeechTimeout=purpose)
            if entry["narration"]:
                with self.state_lock:
                    silent = list(self.state.get("presentationSilentNarrations") or [])
                self._write_state(presentationSilentNarrations=(silent + [purpose])[-20:])
            self._emulate_empty(purpose)
        # A question in a window the model chose not to answer (not addressed
        # to it, or noise): the window must not wait three minutes for it.
        if (
            self.awaiting_reply_since > 0.0
            and self.presentation_qa_open.is_set()
            and not self.output_active.is_set()
            and not self.sourcevault_waiting.is_set()
            and self._backend_idle()
            and not self.expected
            and now - self.awaiting_reply_since >= 5.0
        ):
            self.awaiting_reply_since = 0.0
            self._emulate_empty("normal")

    def _wait_for_answer(self, budget_seconds: float) -> None:
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
            if (
                self.response_in_progress.is_set()
                or self.sourcevault_waiting.is_set()
                or not self._backend_idle()
                or any(entry["backend"] for entry in list(self.expected))
            ):
                continue
            return

    # ------------------------------------------------------------------ input

    def _input_sender(self) -> None:
        """One frame per chunk: the room when allowed, silence otherwise."""

        frame = max(1, int(OPENAI_INPUT_RATE * self.args.chunk_ms / 1000))
        silence = base64.b64encode(bytes(frame * 2)).decode("ascii")
        period = self.args.chunk_ms / 1000.0
        next_at = time.monotonic()
        while not self.stop_event.is_set():
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_at = time.monotonic()
            if not self.ws_ready.is_set():
                continue
            try:
                event = self.input_events.get_nowait()
            except queue.Empty:
                event = None
            blocked = (
                self.muted.is_set()
                or self.privacy_muted.is_set()
                or self.sourcevault_waiting.is_set()
                or self.local_handoff_active.is_set()
            )
            audio = silence if event is None or blocked else event.get("audio", silence)
            with self.send_lock:
                self._send_raw({"type": "session.input_audio.append", "audio": audio})

    def _send_talk_preroll(self) -> None:
        # How to answer it is in the session instructions already, so the
        # utterance goes straight in -- in one piece, never interleaved
        # with the clocked silence.
        with self.send_lock:
            super()._send_talk_preroll()

    def _end_question_context(self) -> None:
        # Nothing to close: every script is asked for with an explicit
        # instruction to read it, so the window's "say nothing" rule cannot
        # silence the next slide -- and the closing note was the thing the
        # room kept hearing (「質問時間は終わりました」 after every slide).
        # The talk moves on without a word.
        return

    def _spoken_request_text(self) -> str:
        """The utterance the backend is acting on, from the input transcript.

        The backend tool call can arrive before the transcript has caught
        up (measured: call at +0.00 s, transcript complete at +0.06 s), so
        wait briefly for the turn to end."""

        deadline = time.monotonic() + 1.5
        while self.user_turn_open and time.monotonic() < deadline:
            time.sleep(0.05)
        now = time.monotonic()
        text = "".join(
            turn for when, turn in list(self.recent_user_turns) if now - when <= 12.0
        ).strip()
        if not text:
            text = self.user_text.strip()
        if not text:
            for entry in reversed(list(self.history)):
                if entry["role"] == "user":
                    text = entry["text"]
                    break
        return text

    def _full_duplex(self) -> bool:
        # GPT-Live is built to listen while it speaks and to judge by itself
        # whether it was addressed; closing the room while it talked (and
        # for 1.2 s after) made it the Realtime half duplex.  VRChat's room
        # audio carries no echo of the avatar.  A talk keeps its own rules
        # (muted narration, question windows).
        return bool(self._opt("live_full_duplex", True)) and not self._presentation_running()

    def _presentation_finished(self) -> None:
        self._append("instructions", None, TALK_OVER)
        self._log("talk_over")

    def _local_speech_gate_ready(self) -> bool:
        # GPT-Live has no automatic-response switch to turn off, and judges by
        # itself whether it was addressed; windows are not gated.
        return False

    # ----------------------------------------------------------- presentation

    def _cut_presentation_audio(self) -> None:
        self.cut_script = self.narration_target
        self.cut_at = time.monotonic()
        with self.live_lock:
            waiting = [
                entry for entry in self.expected
                if str(entry["purpose"]).startswith("presentation_")
            ]
            for entry in waiting:
                self.expected.remove(entry)
        if waiting and not self.output_active.is_set():
            self._append(
                "instructions",
                None,
                "さきほど渡した原稿は読まないでください。" + ONE_TIME_SCOPE,
            )
        super()._cut_presentation_audio()

    def _stop_presentation(self) -> None:
        with self.live_lock:
            for entry in list(self.expected):
                if str(entry["purpose"]).startswith("presentation_"):
                    self.expected.remove(entry)
        super()._stop_presentation()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 30.0
        while not self.ws_ready.is_set():
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            if time.monotonic() >= deadline:
                raise SlidenError("GPT-Live connection was not ready for narration")
            time.sleep(0.1)

    def _narration_direction(
        self,
        *,
        first: bool,
        resume_phrase: str | None,
        extra: str = "",
        piece: str = "",
    ) -> str:
        if first and resume_phrase:
            head = f"今すぐ「{resume_phrase}」と言ってから、続けて"
        elif first:
            head = "今すぐ"
        else:
            head = "今すぐ続けて、"
        direction = (
            head + "次の原稿を、書かれたとおりに一字一句そのまま、最後まで声に出して"
            "読み上げてください。前置き・言い換え・感想は付けず、読み終えたら"
            "付け足さずにその話を終えてください。"
        )
        if extra.strip():
            direction += "話し方 (声に出さない): " + extra.strip()
        return direction + ONE_TIME_SCOPE + "\n原稿:\n" + piece

    def _wait_model_quiet(self, quiet_seconds: float, cap_seconds: float) -> None:
        """Wait until GPT-Live has not produced voiced audio for a while."""

        deadline = time.monotonic() + cap_seconds
        while time.monotonic() < deadline:
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            if (
                not self.output_active.is_set()
                and time.monotonic() - self.last_voiced_output_at >= quiet_seconds
            ):
                return
            time.sleep(0.05)

    def _narration_piece_limit(self, extra: str) -> int:
        """Characters of script one instruction can carry with its words."""

        overhead = estimate_tokens(
            self._narration_direction(
                first=True, resume_phrase="それでは、説明に戻ります", extra=extra
            )
        )
        return max(80, min(LIVE_NARRATION_CHUNK_CHARS, LIVE_APPEND_MAX_TOKENS - overhead - 8))

    def _read_script(
        self,
        purpose: str,
        text: str,
        *,
        resume_phrase: str | None = None,
        extra: str = "",
        seconds_hint: float = 30.0,
    ) -> None:
        """Have GPT-Live read *text*, piece by piece, each heard in full."""

        pieces = narration_chunks(text, self._narration_piece_limit(extra)) or (
            [text] if text.strip() else []
        )
        self.narration_prefix = ""
        for index, piece in enumerate(pieces):
            first = index == 0
            if self._presentation_cancelled():
                raise SlidenCancelled("presentation was cancelled")
            _clear_queue(self.presentation_audio_done)
            self.narration_target = piece
            # No private copy first: GPT-Live spoke it (「読み上げる原…」 from
            # its label) and then ignored the request to read (live
            # 2026-09-19).  The instruction alone carries the words, and is
            # sent once the model has fallen quiet -- one arriving while it
            # was still talking went unanswered.
            self._wait_model_quiet(0.8, 3.0)
            direction = self._narration_direction(
                first=first,
                resume_phrase=resume_phrase if first else None,
                extra=extra,
                piece=piece,
            )
            self._expect(purpose, timeout=30.0, narration=True, resend=direction)
            self._append("instructions", None, direction)
            share = len(piece) / max(1, len(text))
            self._await_presentation_audio(
                purpose, max(60.0, seconds_hint * 5.0 * share), interruptible=True
            )
            self.narration_prefix += piece

    def _narrate_presentation_segment(
        self,
        kind: str,
        text: str,
        run_id: str,
        seconds_hint: float,
        *,
        resume_phrase: str | None = None,
    ) -> None:
        self._ensure_vrchat_microphone_live()
        self._wait_ready()
        self._read_script(
            f"presentation_{kind}:{run_id}",
            text,
            resume_phrase=resume_phrase,
            seconds_hint=seconds_hint,
        )

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
        self._ensure_vrchat_microphone_live()
        self._wait_ready()
        purpose = f"presentation_slide:{run_id}:{slide_number}"
        if script is None:
            # GPT-Live cannot see the slide: the backend writes what to say.
            generated = self._slide_script(
                path, slide_number, slide_count, seconds_per_slide, narration_instructions
            )
            text = (
                resume_text(generated, already_said)
                if resume_phrase is not None
                else generated
            )
        else:
            text = script
        self._prefetch_slide_script(
            slide_number + 1, seconds_per_slide, narration_instructions
        )
        if not text.strip():
            return
        self._read_script(
            purpose,
            text,
            resume_phrase=resume_phrase,
            extra=narration_instructions,
            seconds_hint=seconds_per_slide,
        )

    # -------------------------------------------------- scripts from images

    def _responses_text(self, body: dict[str, Any], timeout: float = 90.0) -> str:
        request = urllib.request.Request(
            RESPONSES_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return response_output_text(payload)

    def _generate_slide_script(
        self,
        path: Path,
        number: int,
        count: int,
        seconds: float,
        extra: str,
    ) -> str:
        key = str(path)
        prompt = (
            f"VRChatで行う発表のスライド (全{count}枚中の第{number}枚) です。"
            f"この1枚を日本語で{seconds:g}秒前後で説明する発表原稿を書いてください。"
            "画像に実際に書かれている内容だけを根拠にし、全文の読み上げではなく"
            "聴衆に分かる自然な話し言葉にしてください。画像にない事実・数値を"
            "作らないでください。記号・URL・箇条書き記号は使わず、原稿の本文だけを"
            "出力してください。"
        )
        if extra.strip():
            prompt += "\n追加指示: " + extra.strip()
        try:
            data_uri, _width, _height = self._presentation_image_data_uri(Path(path))
            body: dict[str, Any] = {
                "model": self._opt("live_backend_model", "gpt-5.6-luna"),
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_uri, "detail": "high"},
                        ],
                    }
                ],
            }
            effort = str(self._opt("live_backend_reasoning", "low") or "")
            if effort:
                body["reasoning"] = {"effort": effort}
            script = self._responses_text(body).strip()
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            self._write_state(
                presentationScriptError=f"slide {number}: {type(exc).__name__}: {exc}"[:300]
            )
            return f"第{number}枚のスライドをご覧ください。"
        except Exception as exc:  # SlidenError from the image reader, and the rest
            self._write_state(
                presentationScriptError=f"slide {number}: {type(exc).__name__}: {exc}"[:300]
            )
            return f"第{number}枚のスライドをご覧ください。"
        if not script:
            return f"第{number}枚のスライドをご覧ください。"
        with self.live_lock:
            self.slide_scripts[key] = script
            count_scripts = len(self.slide_scripts)
        self._write_state(presentationGeneratedScripts=count_scripts)
        return script

    def _slide_script(
        self, path: Path, number: int, count: int, seconds: float, extra: str
    ) -> str:
        key = str(path)
        with self.live_lock:
            cached = self.slide_scripts.get(key)
            thread = self.slide_script_threads.get(key)
        if cached:
            return cached
        if thread is not None and thread.is_alive():
            thread.join(timeout=90.0)
            with self.live_lock:
                cached = self.slide_scripts.get(key)
            if cached:
                return cached
        return self._generate_slide_script(path, number, count, seconds, extra)

    def _prefetch_slide_script(self, number: int, seconds: float, extra: str) -> None:
        """Write the next unscripted slide's script while this one is read."""

        slides = list(self.presentation_slides)
        if not 1 <= number <= len(slides):
            return
        talk = self.presentation_talk
        if talk and number in talk.get("slides", {}):
            return
        path = slides[number - 1]
        key = str(path)
        with self.live_lock:
            if key in self.slide_scripts or key in self.slide_script_threads:
                return
            thread = threading.Thread(
                target=self._generate_slide_script,
                args=(path, number, len(slides), seconds, extra),
                daemon=True,
                name=f"VRCRealtimeSlideScript{number}",
            )
            self.slide_script_threads[key] = thread
        thread.start()
