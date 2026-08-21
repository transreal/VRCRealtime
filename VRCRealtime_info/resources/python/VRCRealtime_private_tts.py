"""Private SourceVault search and local TTS broker for VRCRealtime.

This process deliberately has no OpenAI credential.  It is the only process
that receives SourceVault text whose privacy level may be 0.5 or higher.  Such
text is converted to a WAV file by a persistent local Piper Plus process; only
the WAV path and routing metadata are returned to the Realtime worker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def _wav_seconds(path: Path) -> float:
    """Length of a synthesized clip, or 0 when it cannot be read."""

    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _protocol_write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class PiperSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.audio_dir = Path(args.audio_dir).resolve()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        env = dict(os.environ)
        # This child performs local inference only.  Strip credentials and
        # proxy variables so private text cannot be sent to a cloud endpoint.
        for key in list(env):
            upper = key.upper()
            if (
                upper.startswith(("OPENAI_", "ANTHROPIC_", "AWS_", "AZURE_", "GOOGLE_"))
                or upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
            ):
                env.pop(key, None)
        command = [
            self.args.piper_executable,
            "--model",
            self.args.piper_model,
        ]
        if getattr(self.args, "piper_config", ""):
            command += ["--config", self.args.piper_config]
        command += ["--json-input", "--quiet"]
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

    def synthesize(self, text: str, *, prefix: str = "private") -> Path:
        with self.lock:
            return self._synthesize_locked(text, prefix=prefix)

    def _synthesize_locked(self, text: str, *, prefix: str = "private") -> Path:
        self.start()
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("piper unavailable")
        target = self.audio_dir / f"{prefix}-{uuid.uuid4().hex}.wav"
        request = {
            "text": text,
            "output_file": str(target),
            "speaker_id": 0,
            "noise_scale": self.args.piper_noise_scale,
            "length_scale": self.args.piper_length_scale,
        }
        language = getattr(self.args, "piper_language", "")
        if language:
            request["language"] = language
        process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
        process.stdin.flush()
        # Piper emits the completed output path as one line per JSON request.
        completed = process.stdout.readline()
        if not completed or process.poll() is not None or not target.is_file():
            raise RuntimeError("local synthesis failed")
        return target

    _SENTENCE_ENDS = "。．.!?！？\n"

    def synthesize_bounded(
        self, text: str, max_seconds: float, *, prefix: str = "private"
    ) -> tuple[Path, bool]:
        """Synthesize *text*, kept under *max_seconds* of speech.

        The cut is measured, not guessed: the first pass gives the actual
        characters-per-second of this voice at this length scale, and the text
        is trimmed to fit from that.  A search hit can be thousands of
        characters, and a local readout nobody can interrupt is a bad way to
        spend five minutes of a conversation.
        """

        path = self.synthesize(text, prefix=prefix)
        duration = _wav_seconds(path)
        if max_seconds <= 0.0 or duration <= max_seconds or duration <= 0.0:
            return path, False
        per_second = max(1.0, len(text) / duration)
        # 8% margin: the trimmed text is not spoken at exactly the same rate.
        keep = max(40, int(max_seconds * per_second * 0.92))
        if keep >= len(text):
            return path, False
        head = text[:keep]
        cut = max(head.rfind(ch) for ch in self._SENTENCE_ENDS)
        if cut >= 40:
            head = head[: cut + 1]
        head += "以下は長いため省略しました。"
        try:
            path.unlink()
        except OSError:
            pass
        return self.synthesize(head, prefix=prefix), True

    def warm(self) -> None:
        warm_path = self.synthesize("準備完了。", prefix="warm")
        try:
            warm_path.unlink()
        except OSError:
            pass

    def close(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            if process is None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.close()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


@dataclass(frozen=True)
class SearchTicket:
    command_id: str
    request_path: Path
    temporary_path: Path
    done_path: Path


class DeferredSearch(TimeoutError):
    def __init__(self, ticket: SearchTicket) -> None:
        super().__init__("SourceVault search continues in background")
        self.ticket = ticket


class SourceVaultQueue:
    def __init__(self, service_dir: str, timeout: float) -> None:
        self.service_dir = Path(service_dir).resolve()
        self.commands = self.service_dir / "commands"
        self.done = self.commands / "done"
        self.timeout = timeout
        if not (self.service_dir / "status.json").is_file():
            raise RuntimeError("SourceVault service directory is unavailable")
        self.done.mkdir(parents=True, exist_ok=True)

    def submit_command(self, command: dict[str, Any]) -> SearchTicket:
        """Queue any service command.  The caller owns the command payload."""

        command_id = "voice:" + str(uuid.uuid4())
        filename = command_id.replace(":", "_") + ".json"
        request_path = self.commands / filename
        temporary = request_path.with_suffix(request_path.suffix + ".tmp")
        done_path = self.done / filename
        payload = dict(command)
        payload["CommandId"] = command_id
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, request_path)
        return SearchTicket(command_id, request_path, temporary, done_path)

    def submit(self, search_spec: dict[str, Any], args: argparse.Namespace) -> SearchTicket:
        command_id = "voice:" + str(uuid.uuid4())
        filename = command_id.replace(":", "_") + ".json"
        request_path = self.commands / filename
        temporary = request_path.with_suffix(request_path.suffix + ".tmp")
        done_path = self.done / filename
        command: dict[str, Any] = {
            "CommandId": command_id,
            "Command": "LocalVoiceSearch",
            "Query": str(search_spec.get("query", "")),
            "Kinds": search_spec.get("kinds", args.sourcevault_kinds),
            "Limit": int(search_spec.get("limit", args.sourcevault_limit)),
        }
        filters = search_spec.get("filters")
        if isinstance(filters, dict) and filters:
            command["Filters"] = filters
        if args.sourcevault_kb:
            command["KB"] = args.sourcevault_kb
        if args.sourcevault_group:
            command["Group"] = args.sourcevault_group
        if args.sourcevault_release_context:
            command["ReleaseContext"] = args.sourcevault_release_context
        temporary.write_text(
            json.dumps(command, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, request_path)
        return SearchTicket(command_id, request_path, temporary, done_path)

    @staticmethod
    def _cleanup(ticket: SearchTicket) -> None:
        for path in (ticket.request_path, ticket.temporary_path, ticket.done_path):
            try:
                path.unlink()
            except OSError:
                pass

    def await_result(self, ticket: SearchTicket, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ticket.done_path.is_file():
                try:
                    envelope = json.loads(ticket.done_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                result = envelope.get("Result", {})
                if not isinstance(result, dict):
                    self._cleanup(ticket)
                    raise RuntimeError("invalid SourceVault result")
                voice = result.get("Voice", result)
                if not isinstance(voice, dict):
                    self._cleanup(ticket)
                    raise RuntimeError("invalid SourceVault voice result")
                self._cleanup(ticket)
                return voice
            time.sleep(0.05)
        # Leave an in-flight request/done path intact. The credential-free
        # background collector owns it after the conversational timeout.
        raise TimeoutError("SourceVault search timed out")

    def await_envelope(self, ticket: SearchTicket, timeout: float) -> dict[str, Any]:
        """Await a command result without the LocalVoiceSearch "Voice" unwrapping."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ticket.done_path.is_file():
                try:
                    envelope = json.loads(ticket.done_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                result = envelope.get("Result", {})
                self._cleanup(ticket)
                if not isinstance(result, dict):
                    raise RuntimeError("invalid SourceVault result")
                return result
            time.sleep(0.05)
        self._cleanup(ticket)
        raise TimeoutError("SourceVault command timed out")

    def slide_deck(self, title: str, timeout: float) -> dict[str, Any]:
        """Look up a registered presentation.  Metadata only -- the registry
        release gate withholds the narration text of high-privacy entries, so
        what comes back here is safe for the OpenAI-connected parent."""

        command: dict[str, Any] = {"Command": "SlideDeckLookup", "IncludeTalk": True}
        if title:
            command["Title"] = title
        result = self.await_envelope(self.submit_command(command), timeout)
        deck = result.get("SlideDeck")
        if not isinstance(deck, dict):
            raise RuntimeError("invalid SlideDeckLookup result")
        return deck

    def web_search(self, query: str, timeout: float) -> dict[str, Any]:
        """Local SearXNG metasearch.  Titles/URLs/snippets only, no bodies."""

        result = self.await_envelope(
            self.submit_command({"Command": "LocalVoiceWebSearch", "Query": query}),
            timeout,
        )
        web = result.get("Web")
        if not isinstance(web, dict):
            raise RuntimeError("invalid LocalVoiceWebSearch result")
        return web

    def search(self, search_spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
        ticket = self.submit(search_spec, args)
        try:
            return self.await_result(ticket, self.timeout)
        except TimeoutError as exc:
            # Do not delete the in-flight command. A private background thread
            # will collect it and prepare a result for a later voice turn.
            raise DeferredSearch(ticket) from exc

    def cancel(self, ticket: SearchTicket) -> None:
        self._cleanup(ticket)


class PrivateBroker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sourcevault = SourceVaultQueue(args.sourcevault_service_dir, args.timeout_seconds)
        self.piper = PiperSession(args)
        self.cache_lock = threading.Lock()
        self.pending: dict[str, SearchTicket] = {}
        self.ready: dict[str, dict[str, Any]] = {}
        #: Private text offered but not yet accepted, keyed by token.  It never
        #: leaves this process; the parent only ever sees the token.
        self.pending_private: dict[str, tuple[str, float]] = {}
        self.last_search_key: str | None = None
        self.closing = threading.Event()

    def start(self) -> None:
        self.piper.warm()
        _protocol_write({"type": "ready", "pid": os.getpid()})

    @staticmethod
    def _followup_query(query: str) -> bool:
        lowered = query.casefold()
        return any(
            marker in lowered
            for marker in ("さっき", "先ほど", "準備", "できた", "結果を確認", "ready")
        )

    @staticmethod
    def _mail_date_range(
        query: str,
        *,
        today: date | None = None,
    ) -> tuple[str, str] | None:
        """Resolve common Japanese mail date expressions to an inclusive range."""

        base = today or datetime.now().astimezone().date()
        lowered = query.casefold()

        iso_match = re.search(r"(?<!\d)(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", lowered)
        japanese_match = re.search(
            r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日",
            lowered,
        )
        matched_date: date | None = None
        try:
            if iso_match:
                matched_date = date(*(int(value) for value in iso_match.groups()))
            elif japanese_match:
                year_text, month_text, day_text = japanese_match.groups()
                year = int(year_text) if year_text else base.year
                matched_date = date(year, int(month_text), int(day_text))
                # A yearless date far in the future usually refers to the
                # most recent occurrence, not the next calendar year.
                if not year_text and matched_date > base + timedelta(days=31):
                    matched_date = date(
                        year - 1,
                        int(month_text),
                        int(day_text),
                    )
        except ValueError:
            matched_date = None
        if matched_date is not None:
            value = matched_date.isoformat()
            return value, value

        days_ago = re.search(r"(\d{1,3})\s*日前", lowered)
        if days_ago:
            value = (base - timedelta(days=int(days_ago.group(1)))).isoformat()
            return value, value
        if any(marker in lowered for marker in ("一昨日", "おととい", "おとつい")):
            value = (base - timedelta(days=2)).isoformat()
            return value, value
        if any(marker in lowered for marker in ("昨日", "きのう", "前日", "yesterday")):
            value = (base - timedelta(days=1)).isoformat()
            return value, value
        if any(marker in lowered for marker in ("今日", "本日", "きょう", "today", "今朝")):
            value = base.isoformat()
            return value, value

        if any(marker in lowered for marker in ("先週", "前の週")):
            this_monday = base - timedelta(days=base.weekday())
            start = this_monday - timedelta(days=7)
            return start.isoformat(), (start + timedelta(days=6)).isoformat()
        if any(marker in lowered for marker in ("今週", "この週")):
            start = base - timedelta(days=base.weekday())
            return start.isoformat(), base.isoformat()
        if any(marker in lowered for marker in ("先月", "前の月")):
            first_this_month = base.replace(day=1)
            end = first_this_month - timedelta(days=1)
            start = end.replace(day=1)
            return start.isoformat(), end.isoformat()
        if any(marker in lowered for marker in ("今月", "この月")):
            return base.replace(day=1).isoformat(), base.isoformat()
        return None

    #: Kinds a question asked during a talk may be answered from: the material
    #: itself, not the owner's mail or coding session logs.  Asked what the 9th
    #: meeting covered, an unscoped search answered from a VRCPilot work log
    #: (2026-08-15) -- an audience asks about the talk, not about the bot.
    PRESENTATION_KINDS = ["search", "snapshot"]

    def _normalize_search(self, query: str, *, scope: str = "") -> dict[str, Any]:
        lowered = query.casefold()
        mail_request = any(
            marker in lowered for marker in ("メール", "mail", "受信箱", "inbox")
        )
        if mail_request:
            spec: dict[str, Any] = {
                "query": query,
                "kinds": ["mail"],
                "limit": min(5, max(1, int(self.args.sourcevault_limit))),
                "filters": {},
            }
            date_range = self._mail_date_range(query)
            if date_range is not None:
                date_from, date_to = date_range
                spec["filters"] = {
                    "dateFrom": date_from,
                    "dateTo": date_to,
                }
                list_request = any(
                    marker in lowered
                    for marker in (
                        "リスト",
                        "一覧",
                        "列挙",
                        "何件",
                        "メールを見せ",
                        "メールを確認",
                        "メールの確認",
                        "どんなメール",
                    )
                )
                if list_request:
                    # The shard-side date range does the selection; sending
                    # boilerplate such as "昨日のメールのリスト" as a semantic
                    # query can suppress otherwise valid indexed mail.
                    spec["query"] = ""
            return spec
        if scope == "presentation":
            return {
                "query": query,
                "kinds": list(self.PRESENTATION_KINDS),
                "limit": min(5, max(1, int(self.args.sourcevault_limit))),
                "filters": {},
            }
        return {
            "query": query,
            "kinds": list(self.args.sourcevault_kinds),
            "limit": int(self.args.sourcevault_limit),
            "filters": {},
        }

    @staticmethod
    def _search_key(spec: dict[str, Any]) -> str:
        return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _result_to_response(self, result: dict[str, Any]) -> dict[str, Any]:
        route = str(result.get("Route", "Error"))
        common = {
            "count": int(result.get("Count", 0) or 0),
            "privacyLevel": result.get("MaxPrivacyLevel"),
            # Which layer answered, and why the fast index did not.  Without
            # this a KB that was never built degrades invisibly into a search
            # over coding logs, and a talk question gets an unrelated answer.
            "source": result.get("Source", "KB" if result.get("KBId") else None),
            "kbStatus": result.get("KBStatus"),
        }
        if route == "OpenAI":
            safe_text = result.get("OpenAIText", "")
            if not isinstance(safe_text, str):
                safe_text = ""
            return common | {"route": "openai", "text": safe_text}
        if route == "NoResults":
            # Its own route, not an empty "openai" answer: finding nothing is
            # where the caller may want to offer a web search, and that
            # decision cannot be made from a sentence of prose.
            return common | {
                "route": "no_results",
                "text": "SourceVaultに該当する情報は見つかりませんでした。",
            }
        if route == "LocalTTS":
            wav_path, truncated = self.piper.synthesize_bounded(
                self._private_text(result), float(self.args.local_speech_max_seconds)
            )
            # Never return private text to the OpenAI-connected parent.
            return common | {
                "route": "local_tts",
                "wavPath": str(wav_path),
                "truncated": truncated,
            }
        if route == "Mixed":
            # Answer from the public material now and hold the rest.  The
            # private part is kept here, in the process with no credential, and
            # is only synthesized if the listener asks for it.
            token = uuid.uuid4().hex
            safe_text = result.get("OpenAIText", "")
            with self.cache_lock:
                self.pending_private[token] = (
                    self._private_text(result),
                    time.monotonic(),
                )
            return common | {
                "route": "mixed",
                "text": safe_text if isinstance(safe_text, str) else "",
                "privateCount": int(result.get("PrivateCount", 0) or 0),
                "privateMaxPrivacyLevel": result.get("PrivateMaxPrivacyLevel"),
                "privateToken": token,
            }
        return common | {"route": "error", "error": "SourceVault search failed"}

    def _handle_presentation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Resolve a spoken presentation title through the SourceVault registry.

        This is the only non-search request the broker serves.  It stays here
        rather than in the worker so SourceVault keeps a single ingress, and
        because the registry -- not this process -- decides what may cross into
        the OpenAI-connected parent (entries at or above the release ceiling
        come back with talkWithheld set and no narration text)."""

        title = request.get("title")
        title = title.strip() if isinstance(title, str) else ""
        if not isinstance(self.sourcevault, SourceVaultQueue):
            return {"route": "error", "error": "slide deck registry is unavailable"}
        try:
            deck = self.sourcevault.slide_deck(title, float(self.args.timeout_seconds))
        except TimeoutError:
            return {"route": "unavailable", "error": "slide deck lookup timed out"}
        except Exception:
            return {"route": "unavailable", "error": "slide deck lookup failed"}
        return {"route": "presentation", "deck": deck}

    @staticmethod
    def _private_text(result: dict[str, Any]) -> str:
        text = result.get("LocalSpeechText", "")
        if not isinstance(text, str) or not text.strip():
            return "SourceVaultに非公開情報がありますが、読み上げ可能な要約がありません。"
        return text

    def _expire_private(self) -> None:
        """Drop held private text nobody asked for.  Declining to hear it is
        also an answer, and holding it any longer serves nothing."""

        cutoff = time.monotonic() - float(self.args.private_hold_seconds)
        with self.cache_lock:
            for token in [
                key
                for key, (_text, stamp) in self.pending_private.items()
                if stamp < cutoff
            ]:
                self.pending_private.pop(token, None)

    def _handle_private_readout(self, request: dict[str, Any]) -> dict[str, Any]:
        """Play or discard the private remainder the listener was offered."""

        self._expire_private()
        token = str(request.get("token", ""))
        action = str(request.get("action", "discard"))
        with self.cache_lock:
            held = self.pending_private.pop(token, None)
        if held is None:
            return {"route": "error", "error": "no held private result"}
        if action != "play":
            return {"route": "discarded"}
        try:
            wav_path, truncated = self.piper.synthesize_bounded(
                held[0], float(self.args.local_speech_max_seconds)
            )
        except Exception:
            return {"route": "error", "error": "local synthesis failed"}
        return {
            "route": "local_tts",
            "wavPath": str(wav_path),
            "truncated": truncated,
        }

    def _handle_web(self, request: dict[str, Any]) -> dict[str, Any]:
        """Web fallback for questions the local index could not answer.

        Public search results only, so unlike a SourceVault result there is no
        privacy routing to do here -- the exposure is in the outbound query,
        which is why the worker only offers the tool when explicitly enabled."""

        query = request.get("query")
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            return {"route": "error", "error": "query required"}
        if not isinstance(self.sourcevault, SourceVaultQueue):
            return {"route": "error", "error": "web search is unavailable"}
        try:
            web = self.sourcevault.web_search(query, float(self.args.timeout_seconds))
        except TimeoutError:
            return {"route": "unavailable", "error": "web search timed out"}
        except Exception:
            return {"route": "unavailable", "error": "web search failed"}
        text = web.get("text")
        return {
            "route": "web",
            "count": int(web.get("count", 0) or 0),
            "text": text if isinstance(text, str) else "",
        }

    @staticmethod
    def _deferred_response(request_id: str, key: str) -> dict[str, Any]:
        return {
            "id": request_id,
            "route": "deferred",
            "jobId": uuid.uuid5(uuid.NAMESPACE_URL, key).hex[:16],
            "message": (
                "検索に時間がかかっているため、バックグラウンドで準備を続けています。"
                "後から結果を確認してください。"
            ),
        }

    def _complete_deferred(self, key: str, ticket: SearchTicket) -> None:
        try:
            result = self.sourcevault.await_result(
                ticket, float(self.args.background_timeout_seconds)
            )
            response = self._result_to_response(result)
        except Exception:
            self.sourcevault.cancel(ticket)
            response = {"route": "error", "error": "background search failed"}
        with self.cache_lock:
            self.pending.pop(key, None)
            if not self.closing.is_set():
                self.ready[key] = response

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("id", ""))
        kind = str(request.get("kind", "search"))
        if kind == "presentation":
            return {"id": request_id} | self._handle_presentation(request)
        if kind == "web":
            return {"id": request_id} | self._handle_web(request)
        if kind == "private_readout":
            return {"id": request_id} | self._handle_private_readout(request)
        query = request.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"id": request_id, "route": "error", "error": "query required"}
        query = query.strip()
        spec = self._normalize_search(query, scope=str(request.get("scope", "")))
        key = self._search_key(spec)
        with self.cache_lock:
            if self._followup_query(query) and self.last_search_key is not None:
                key = self.last_search_key
            if key in self.ready:
                response = self.ready.pop(key)
                return {"id": request_id} | response
            if key in self.pending:
                return self._deferred_response(request_id, key)
            self.last_search_key = key

        try:
            if isinstance(self.sourcevault, SourceVaultQueue):
                result = self.sourcevault.search(spec, self.args)
            else:
                # Small injected test adapters keep the historical interface.
                result = self.sourcevault.search(query, self.args)
            return {"id": request_id} | self._result_to_response(result)
        except DeferredSearch as deferred:
            with self.cache_lock:
                self.pending[key] = deferred.ticket
            threading.Thread(
                target=self._complete_deferred,
                args=(key, deferred.ticket),
                name="SourceVaultDeferredVoice",
                daemon=True,
            ).start()
            return self._deferred_response(request_id, key)

    def close(self) -> None:
        self.closing.set()
        with self.cache_lock:
            tickets = list(self.pending.values())
            ready = list(self.ready.values())
            self.pending.clear()
            self.ready.clear()
            self.pending_private.clear()
        for ticket in tickets:
            self.sourcevault.cancel(ticket)
        for response in ready:
            wav_path = response.get("wavPath")
            if isinstance(wav_path, str):
                try:
                    Path(wav_path).unlink()
                except OSError:
                    pass
        self.piper.close()

    def run(self) -> int:
        self.start()
        try:
            for line in sys.stdin:
                try:
                    request = json.loads(line.lstrip("\ufeff"))
                    if not isinstance(request, dict):
                        raise ValueError("invalid request")
                    response = self.handle(request)
                except Exception as exc:
                    request_id = ""
                    try:
                        request_id = str(request.get("id", ""))
                    except Exception:
                        pass
                    response = {
                        "id": request_id,
                        "route": "error",
                        "error": "private broker request failed",
                        "errorType": type(exc).__name__,
                    }
                _protocol_write(response)
        finally:
            self.close()
        return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourcevault-service-dir", required=True)
    parser.add_argument("--sourcevault-kinds", default="search,packageapi,llmlog,mail,eagle,snapshot")
    parser.add_argument("--sourcevault-group", default="")
    parser.add_argument("--sourcevault-release-context", default="")
    # Prebuilt Graph-RAG index consulted before the (much slower) adapter search.
    parser.add_argument("--sourcevault-kb", default="")
    parser.add_argument("--sourcevault-limit", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    # A local readout nobody can skip should not outlast a conversational turn.
    parser.add_argument("--local-speech-max-seconds", type=float, default=60.0)
    # How long an offered-but-unaccepted private result is held before it is
    # dropped.  Declining is an answer; holding it longer serves nothing.
    parser.add_argument("--private-hold-seconds", type=float, default=180.0)
    parser.add_argument("--background-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--piper-executable", required=True)
    parser.add_argument("--piper-model", required=True)
    # Optional: piper finds <model>.onnx.json on its own when a voice ships
    # its config that way.  Only pass --config when there is one to pass.
    parser.add_argument("--piper-config", default="")
    # Empty = do not send a "language" key at all.  Only multilingual voices
    # accept one; a single-language voice can reject the request outright, so
    # the caller resolves this from the voice config instead of assuming.
    parser.add_argument("--piper-language", default="")
    parser.add_argument("--piper-length-scale", type=float, default=1.5)
    parser.add_argument("--piper-noise-scale", type=float, default=0.667)
    parser.add_argument("--audio-dir", required=True)
    args = parser.parse_args()
    args.sourcevault_kinds = [
        item.strip() for item in args.sourcevault_kinds.split(",") if item.strip()
    ]
    args.sourcevault_limit = max(1, min(5, args.sourcevault_limit))
    args.timeout_seconds = max(1.0, min(120.0, args.timeout_seconds))
    args.local_speech_max_seconds = max(5.0, min(600.0, args.local_speech_max_seconds))
    args.private_hold_seconds = max(10.0, min(1800.0, args.private_hold_seconds))
    args.background_timeout_seconds = max(
        args.timeout_seconds, min(3600.0, args.background_timeout_seconds)
    )
    return args


def main() -> int:
    args = _parse_args()
    try:
        return PrivateBroker(args).run()
    except Exception:
        _protocol_write({"type": "fatal", "error": "private broker failed to start"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
