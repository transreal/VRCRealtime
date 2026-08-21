"""VRCRealtime_worldjoin.py -- from the Home world into a group instance.

Drives the VRChat desktop menu with VRC Pilot (screenshot -> OCR -> click)
through the sequence a person performs by hand:

    Esc -> Worlds -> <favourites list> -> <world> -> New Instance -> Group
        -> (check group / access / region) -> CREATE INSTANCE -> Join
        -> wait for the world -> walk into the room with the Sliden screen

Every step is closed-loop: it looks for the text it expects, clicks it, and
waits for the text the next page must show.  Nothing is clicked by fixed
coordinates, so a menu that opens a little differently (an old page still
open, a list already selected) recovers on its own, and a step that never
sees what it needs fails with the last screen saved for inspection.

The menu part only needs OCR.  The room part reuses the Sliden landmark
detector and the motion primitives of ``VRCRealtime_sliden``: the wall
screen shows the Sliden idle page from the moment the world loads, so it is
the one thing in the world that identifies the presentation room.

Run as a script (``--spec-file``/``--state-file``/``--control-file``) from
``VRCRealtime.wl``, or import ``WorldJoinController`` for tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from VRCRealtime_sliden import (
    Rect,
    SlidenCancelled,
    SlidenConsoleController,
    SlidenConsoleNotFound,
    _check_cancel,
    _bright_horizontal_bars,
    _display_candidates,
    console_page_rects,
    detect_sliden_screen,
    page_has_console_bars,
    screen_from_text_boxes,
)

RGBImage = NDArray[np.uint8]


class WorldJoinError(RuntimeError):
    """A menu step did not reach the screen it needed."""


# ---------------------------------------------------------------------------
# OCR text matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBox:
    """One OCR word in VRChat-window-local pixels."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class TextHit:
    """A matched run of one or more boxes on a line."""

    text: str
    rect: Rect
    score: float
    boxes: tuple[TextBox, ...]

    @property
    def center(self) -> tuple[float, float]:
        return self.rect.center


_KANA = re.compile(r"[぀-ヿｦ-ﾟ]")
#: What OCR makes of a "+" glyph in a UI font.
_PLUS_VARIANTS = {"＋": "+", "十": "+", "士": "+"}


def normalize_text(text: str, *, plus_variants: bool = False) -> str:
    """Fold OCR text for comparison: NFKC, lower case, letters/digits only.

    Spaces and punctuation are dropped, so "CREATE INSTANCE",
    "CreateInstance" and "Create  Instance." all fold to the same key.  A
    check mark or a flag glyph in front of a button label folds away too.
    "+" is kept because "Group" and "Group+" are different instance types;
    ``plus_variants`` also folds the characters OCR mistakes it for.
    """

    folded: list[str] = []
    for char in unicodedata.normalize("NFKC", text):
        if char.isspace():
            continue
        if plus_variants:
            char = _PLUS_VARIANTS.get(char, char)
        if char.isalnum() or char in "+#":
            folded.append(char.lower())
    return "".join(folded)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def group_name_matches(expected: str, observed: str, *, min_ratio: float = 0.6) -> bool:
    """Loosely compare a group name against what OCR read of it.

    The bundled OCR dictionary has no kana, so "計算と自然" comes back as
    "計算自然" or "計算七自然".  Kana are dropped from both sides and the rest
    only has to be close.
    """

    core = lambda value: _KANA.sub("", normalize_text(value))  # noqa: E731
    wanted, seen = core(expected), core(observed)
    if not wanted or not seen:
        return False
    if wanted in seen:
        return True
    return SequenceMatcher(None, wanted, seen).ratio() >= min_ratio


def text_boxes_from_words(words: Sequence[Any]) -> list[TextBox]:
    """Convert ``vrcpilot`` OCR words (anything with ``text``/``bbox``)."""

    boxes: list[TextBox] = []
    for word in words:
        x, y, width, height = word.bbox
        text = str(word.text)
        if not normalize_text(text):
            continue
        boxes.append(
            TextBox(
                text=text,
                x=int(x),
                y=int(y),
                width=max(1, int(width)),
                height=max(1, int(height)),
                confidence=float(getattr(word, "confidence", 1.0)),
            )
        )
    return boxes


def group_lines(boxes: Sequence[TextBox]) -> list[list[TextBox]]:
    """Group boxes into text lines, each sorted left to right.

    Two boxes share a line when they overlap vertically by at least half the
    smaller height and are separated horizontally by no more than one text
    height -- about a word space.  A wider gap is a neighbouring control
    (the next tile, the next button), not the rest of the same label.
    """

    lines: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda b: (b.y, b.x)):
        placed = False
        for line in lines:
            for member in line:
                overlap = min(box.y + box.height, member.y + member.height) - max(
                    box.y, member.y
                )
                if overlap < 0.5 * min(box.height, member.height):
                    continue
                gap = max(box.x - (member.x + member.width), member.x - (box.x + box.width))
                if gap <= 1.0 * max(box.height, member.height):
                    line.append(box)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b.x)
    return lines


def _union(boxes: Sequence[TextBox]) -> Rect:
    left = min(b.x for b in boxes)
    top = min(b.y for b in boxes)
    right = max(b.x + b.width for b in boxes)
    bottom = max(b.y + b.height for b in boxes)
    return Rect(left, top, right - left, bottom - top)


def _candidates(boxes: Sequence[TextBox], region: Rect | None) -> list[TextHit]:
    """Every box and every contiguous run of boxes on a line."""

    hits: list[TextHit] = []
    for line in group_lines(boxes):
        for start in range(len(line)):
            for end in range(start + 1, len(line) + 1):
                run = tuple(line[start:end])
                rect = _union(run)
                if region is not None:
                    cx, cy = rect.center
                    if not (
                        region.x <= cx <= region.right and region.y <= cy <= region.bottom
                    ):
                        continue
                hits.append(
                    TextHit(
                        text=" ".join(b.text for b in run),
                        rect=rect,
                        score=min(b.confidence for b in run),
                        boxes=run,
                    )
                )
    return hits


def find_text(
    boxes: Sequence[TextBox],
    targets: str | Sequence[str],
    *,
    mode: str = "exact",
    region: Rect | None = None,
    min_ratio: float = 0.8,
    plus_variants: bool = False,
) -> TextHit | None:
    """Find *targets* (any alias) among OCR boxes.

    ``mode`` is ``"exact"`` (folded text equal), ``"prefix"``/``"suffix"``
    (folded text starts/ends with the target and is at most three characters
    longer -- a check mark or flag glyph in front of "JP"), or ``"fuzzy"``
    (SequenceMatcher ratio at least ``min_ratio`` with identical digits, so
    "Favorite Worlds 1" can never match "Favorite Worlds 2").  Exact matches
    always win over fuzzy ones; among equals the shortest run, then the
    topmost, then the most confident, is returned -- a form label sits above
    the value that may repeat its word.
    """

    if isinstance(targets, str):
        targets = [targets]
    wanted = [normalize_text(t, plus_variants=plus_variants) for t in targets]
    wanted = [w for w in wanted if w]
    if not wanted:
        return None
    best: tuple[tuple[float, int, float, float], TextHit] | None = None
    for hit in _candidates(boxes, region):
        folded = normalize_text(hit.text, plus_variants=plus_variants)
        if not folded:
            continue
        rank: float | None = None
        for target in wanted:
            if folded == target:
                rank = 2.0
            elif mode == "prefix" and folded.startswith(target) and len(folded) <= len(target) + 3:
                rank = 1.5
            elif mode == "suffix" and folded.endswith(target) and len(folded) <= len(target) + 3:
                rank = 1.5
            elif mode == "fuzzy" and _digits(folded) == _digits(target):
                ratio = SequenceMatcher(None, folded, target).ratio()
                if ratio >= min_ratio:
                    rank = ratio
            if rank is not None:
                break
        if rank is None:
            continue
        key = (rank, -len(hit.boxes), -float(hit.rect.y), hit.score)
        if best is None or key > best[0]:
            best = (key, hit)
    return None if best is None else best[1]


def value_below(
    boxes: Sequence[TextBox],
    label: TextHit,
    *,
    max_gap_factor: float = 3.0,
    x_tolerance: int = 90,
) -> TextHit | None:
    """The line directly under a form label ("GROUP" -> its value row)."""

    best: tuple[float, TextHit] | None = None
    for line in group_lines(boxes):
        rect = _union(line)
        gap = rect.y - label.rect.bottom
        if gap < -0.25 * label.rect.height or gap > max_gap_factor * label.rect.height:
            continue
        if abs(rect.x - label.rect.x) > x_tolerance and not (
            rect.x <= label.rect.x <= rect.right
        ):
            continue
        hit = TextHit(
            text=" ".join(b.text for b in line),
            rect=rect,
            score=min(b.confidence for b in line),
            boxes=tuple(line),
        )
        if best is None or gap < best[0]:
            best = (gap, hit)
    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------


@dataclass
class WorldJoinSpec:
    world: str = "NCG Gallery"
    favorites: str = "Favorite Worlds 1"
    group: str = "計算と自然"  # 計算と自然
    group_access: str = "Group+"
    region: str = "JP"
    enter_room: bool = True
    approach_console: bool = False
    step_timeout: float = 20.0
    load_timeout: float = 180.0
    navigation_timeout: float = 120.0
    poll_seconds: float = 0.6
    ocr_min_side: int = 1080

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "WorldJoinSpec":
        spec = cls()
        mapping = {
            "world": "world",
            "favorites": "favorites",
            "group": "group",
            "groupAccess": "group_access",
            "region": "region",
            "enterRoom": "enter_room",
            "approachConsole": "approach_console",
            "stepTimeoutSeconds": "step_timeout",
            "loadTimeoutSeconds": "load_timeout",
            "navigationTimeoutSeconds": "navigation_timeout",
            "pollSeconds": "poll_seconds",
            "ocrMinSide": "ocr_min_side",
        }
        for key, attribute in mapping.items():
            if key in data and data[key] is not None:
                current = getattr(spec, attribute)
                value = data[key]
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
                setattr(spec, attribute, value)
        spec.validate()
        return spec

    def validate(self) -> None:
        for name in ("world", "favorites", "group", "group_access", "region"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if not 3.0 <= self.step_timeout <= 300.0:
            raise ValueError("step timeout must be between 3 and 300 seconds")
        if not 10.0 <= self.load_timeout <= 900.0:
            raise ValueError("load timeout must be between 10 and 900 seconds")
        if not 10.0 <= self.navigation_timeout <= 600.0:
            raise ValueError("navigation timeout must be between 10 and 600 seconds")
        if not 0.1 <= self.poll_seconds <= 5.0:
            raise ValueError("poll interval must be between 0.1 and 5 seconds")
        if not 480 <= self.ocr_min_side <= 2160:
            raise ValueError("OCR minimum side must be between 480 and 2160")


#: Text that only the VRChat loading screen shows large.  The gallery hangs a
#: poster reading "NOW LOADING..." on its wall, so size matters as much as
#: the words: see ``_LOADING_MIN_HEIGHT_FRACTION``.
_LOADING_WORDS = ("loading", "joining", "downloading", "connecting", "pleasewait")
_LOADING_MIN_HEIGHT_FRACTION = 0.02
_MENU_MARKERS = ("Selected Instance", "Create New Instance", "Launch Pad", "Respawn")


class WorldJoinController(SlidenConsoleController):
    """Menu automation on top of the Sliden controller's capture and input.

    The base class supplies the foreground-window grab, focus-retrying
    clicks, Escape, mouse-look and OSC walking; this class adds OCR and the
    knowledge of which text leads where.
    """

    def __init__(
        self,
        spec: WorldJoinSpec,
        *,
        pid: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        state_callback: Callable[..., Any] | None = None,
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
        diagnostic_directory: Path | None = None,
        ocr: Callable[[RGBImage], Sequence[TextBox]] | None = None,
        **callbacks: Any,
    ) -> None:
        super().__init__(
            pid=pid,
            timeout_seconds=max(3.0, spec.step_timeout),
            poll_seconds=spec.poll_seconds,
            cancelled=cancelled,
            state_callback=state_callback,
            osc_host=osc_host,
            osc_port=osc_port,
            diagnostic_directory=diagnostic_directory,
            # The panel approach finds the wall screen by its OCR'd wordmark
            # where the template detector cannot (oblique or close views).
            text_locator=self._text_boxes_for_sliden,
            **callbacks,
        )
        self.spec = spec
        self._ocr_callback = ocr
        self._ocr_engine: Any | None = None
        self.steps: list[str] = []
        self.warnings: list[str] = []
        self.last_texts: list[str] = []
        self.frame_size: tuple[int, int] = (0, 0)
        self._shots_written = 0
        self._started = time.monotonic()

    _SHOT_LIMIT = 120
    #: Menu animation allowance after a click, and the loading-screen poll.
    #: Class attributes so a test with a fake client can shrink them.
    _SETTLE_SCALE = 1.0
    _LOAD_MINIMUM_SECONDS = 8.0
    _LOAD_POLL_SECONDS = 2.0

    def _settle(self, seconds: float) -> None:
        time.sleep(seconds * self._SETTLE_SCALE)

    # -- observation -------------------------------------------------------

    def _ensure_ocr(self) -> Callable[[RGBImage], Sequence[TextBox]]:
        if self._ocr_callback is not None:
            return self._ocr_callback
        if self._ocr_engine is None:
            from vrcpilot.ocr import RapidOCREngine

            self._ocr_engine = RapidOCREngine(
                params={
                    # Menu labels are 12-16 px tall at 720p; scaling the short
                    # side up before detection is what makes them legible.
                    "Det.limit_side_len": self.spec.ocr_min_side,
                    "Global.text_score": 0.45,
                    "Global.log_level": "warning",
                }
            )
        engine = self._ocr_engine

        def recognize(frame: RGBImage) -> Sequence[TextBox]:
            return text_boxes_from_words(engine.recognize(frame))

        self._ocr_callback = recognize
        return recognize

    def _text_boxes_for_sliden(self, frame: RGBImage) -> list[tuple[str, Rect]]:
        return [(box.text, box.rect) for box in self._ensure_ocr()(frame)]

    def _menu_observe(
        self, label: str, *, snapshot: bool = True
    ) -> tuple[RGBImage, list[TextBox]]:
        """Capture the window and read every text on it."""

        frame = None
        deadline = time.monotonic() + max(3.0, 4 * self.poll_seconds)
        while frame is None:
            frame = self._capture()
            if frame is None:
                if time.monotonic() > deadline:
                    raise WorldJoinError("the VRChat window could not be captured")
                time.sleep(self.poll_seconds)
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        boxes = list(self._ensure_ocr()(frame))
        self.last_texts = [b.text for b in boxes][:60]
        self._state(worldJoinLastTexts=self.last_texts)
        if snapshot:
            self._menu_snapshot(label, frame)
        return frame, boxes

    def _menu_snapshot(self, label: str, frame: RGBImage) -> None:
        directory = self.diagnostic_directory
        if directory is None or self._shots_written >= self._SHOT_LIMIT:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            index = self._shots_written
            self._shots_written += 1
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:40]
            cv2.imwrite(
                str(directory / f"{index:02d}-{safe}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
            with (directory / "texts.log").open("a", encoding="utf-8") as log:
                log.write(f"{index:02d} {label}: {self.last_texts!r}\n")
        except Exception:
            self.diagnostic_directory = None

    def _state(self, **updates: Any) -> None:
        if self.state_callback is not None:
            self.state_callback(
                worldJoinElapsedSeconds=round(time.monotonic() - self._started, 1),
                **updates,
            )

    def _phase(self, phase: str, **extra: Any) -> None:
        self._state(worldJoinPhase=phase, worldJoinStep=len(self.steps), **extra)

    def _done_step(self, name: str) -> None:
        self.steps.append(name)
        self._state(worldJoinSteps=list(self.steps), worldJoinStep=len(self.steps))

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        self._state(worldJoinWarnings=list(self.warnings))

    # -- waiting and clicking ---------------------------------------------

    def _menu_wait(
        self,
        label: str,
        targets: str | Sequence[str],
        *,
        mode: str = "exact",
        timeout: float | None = None,
        absent: bool = False,
        region: Rect | None = None,
        plus_variants: bool = False,
    ) -> tuple[TextHit | None, list[TextBox]]:
        """Poll until *targets* is on screen (or gone, with ``absent``)."""

        budget = self.spec.step_timeout if timeout is None else timeout
        deadline = time.monotonic() + budget
        while True:
            _check_cancel(self.cancelled)
            _frame, boxes = self._menu_observe(label)
            hit = find_text(boxes, targets, mode=mode, region=region, plus_variants=plus_variants)
            if absent and hit is None:
                return None, boxes
            if not absent and hit is not None:
                return hit, boxes
            if time.monotonic() >= deadline:
                shown = ", ".join(self.last_texts[:25])
                wanted = targets if isinstance(targets, str) else " / ".join(targets)
                raise WorldJoinError(
                    f"{label}: "
                    + (
                        f"'{wanted}' did not disappear"
                        if absent
                        else f"'{wanted}' was not found on screen"
                    )
                    + f" within {budget:.0f}s (seen: {shown})"
                )
            time.sleep(self.poll_seconds)

    def _menu_click(
        self,
        label: str,
        targets: str | Sequence[str],
        *,
        mode: str = "exact",
        timeout: float | None = None,
        settle: float = 0.8,
        region: Rect | None = None,
        park: bool = True,
    ) -> TextHit:
        hit, _boxes = self._menu_wait(label, targets, mode=mode, timeout=timeout, region=region)
        assert hit is not None
        self._click_hit(hit, park=park)
        self._settle(settle)
        return hit

    def _click_hit(self, hit: TextHit, *, park: bool = True) -> None:
        x, y = hit.center
        self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText=hit.text)
        self._click_ui((x, y))
        if park:
            self._park_cursor()

    def _park_cursor(self) -> None:
        """Move the cursor off the menu after a click.

        Left where it clicked, the pointer covers the next label that appears
        there -- "JP" read as "!P" with the cursor over the J -- and OCR of
        the following page reads around a hole.  The bottom-left corner of
        the window is world view, where hovering does nothing.
        """

        if self._click_callback is not None:
            return
        width, height = self.frame_size
        if width <= 0 or height <= 0:
            return
        try:
            import vrcpilot

            self._retrying_focus(
                lambda: vrcpilot.mouse.move(24, max(24, height - 40), pid=self.pid)
            )
        except Exception:
            # Cosmetic; never a reason to stop.
            pass

    # -- the steps ----------------------------------------------------------

    def open_menu(self) -> None:
        """Get the main menu up with a "Worlds" entry in view.

        Escape toggles the menu, and closes a dialog before it closes the
        menu, so a few presses from any starting state end on a menu page
        that has "Worlds" somewhere -- the Launch Pad tile or the bottom bar.
        """

        self._phase("opening_menu")
        for attempt in range(5):
            _frame, boxes = self._menu_observe("menu")
            dialog_open = find_text(boxes, "Create New Instance") is not None
            if not dialog_open and find_text(boxes, "Worlds") is not None:
                self._done_step("menu")
                return
            self._press_escape()
            self._settle(1.0 if attempt == 0 else 0.8)
        raise WorldJoinError(
            "the VRChat menu did not show a Worlds entry after pressing Escape "
            f"(seen: {', '.join(self.last_texts[:25])})"
        )

    def open_worlds(self) -> None:
        self._phase("opening_worlds")
        self._menu_click("worlds", "Worlds")
        # Sidebar labels that only the Worlds page has.  Exact on purpose: a
        # fuzzy "My Worlds" is satisfied by the bottom bar's "Worlds", which
        # is on every page.
        self._menu_wait(
            "worlds_page",
            ["Search Worlds", "Recently Visited", "My Worlds", "Uploaded"],
        )
        self._done_step("worlds")

    def select_favorites(self) -> None:
        self._phase("selecting_favorites")
        # The favourites lists live inside the sidebar's "My Worlds" section,
        # which the client remembers as collapsed or expanded.  Collapsed, the
        # list names are not on screen at all: open the section first.
        for attempt in range(3):
            try:
                self._menu_click(
                    "favorites", self.spec.favorites, mode="fuzzy",
                    timeout=min(4.0, self.spec.step_timeout) if attempt < 2 else None,
                )
                break
            except WorldJoinError:
                if attempt == 2:
                    raise
                _frame, boxes = self._menu_observe("my_worlds")
                section = find_text(boxes, "My Worlds")
                if section is None:
                    raise
                self._state(worldJoinMyWorldsExpanded=attempt + 1)
                self._click_hit(section)
                self._settle(0.9)
        self._menu_wait("favorites_grid", self.spec.world, mode="fuzzy")
        self._done_step("favorites")

    def select_world(self) -> None:
        self._phase("selecting_world")
        self._menu_click("world", self.spec.world, mode="fuzzy")
        self._menu_wait("world_page", "New Instance")
        self._done_step("world")

    def open_new_instance(self) -> None:
        self._phase("opening_new_instance")
        self._menu_click("new_instance", "New Instance")
        self._menu_wait("instance_dialog", ["Create New Instance", "Instance Type"])
        self._done_step("new_instance")

    def choose_group_type(self) -> None:
        self._phase("choosing_group_type")
        # Exact on purpose: the bottom bar says "Groups" and the row's own
        # description says "owned by a group".
        self._menu_click("group_type", "Group")
        self._menu_wait("group_dialog", "Create Instance")
        self._done_step("group_type")

    def verify_group_settings(self) -> dict[str, Any]:
        """Check group, access level and region; fix what can be fixed.

        Returns what was read.  Raises when the group or the access level
        cannot be brought to the requested value -- creating an instance for
        the wrong group is worse than stopping.
        """

        self._phase("verifying_group_settings")
        result: dict[str, Any] = {}
        result["group"] = self._verify_row("GROUP", "group", self._group_ok, self._pick_group)
        result["access"] = self._verify_row(
            "INSTANCE TYPE", "access", self._access_ok, self._pick_access
        )
        result["region"] = self._select_region()
        self._state(
            worldJoinGroup=result["group"],
            worldJoinGroupAccess=result["access"],
            worldJoinRegion=result["region"],
        )
        self._done_step("verify")
        return result

    def _group_ok(self, text: str) -> bool:
        return group_name_matches(self.spec.group, text)

    def _access_ok(self, text: str) -> bool:
        return normalize_text(text, plus_variants=True) == normalize_text(
            self.spec.group_access, plus_variants=True
        )

    def _pick_group(self, boxes: Sequence[TextBox]) -> TextHit | None:
        best: tuple[float, TextHit] | None = None
        for hit in _candidates(boxes, None):
            if not group_name_matches(self.spec.group, hit.text):
                continue
            ratio = SequenceMatcher(
                None,
                _KANA.sub("", normalize_text(self.spec.group)),
                _KANA.sub("", normalize_text(hit.text)),
            ).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, hit)
        return None if best is None else best[1]

    def _pick_access(self, boxes: Sequence[TextBox]) -> TextHit | None:
        return find_text(boxes, self.spec.group_access, plus_variants=True)

    def _verify_row(
        self,
        label_text: str,
        name: str,
        accept: Callable[[str], bool],
        pick: Callable[[Sequence[TextBox]], TextHit | None],
    ) -> str:
        """Read the value under a dialog label; open the row and choose if wrong."""

        for attempt in range(2):
            _frame, boxes = self._menu_observe(f"verify_{name}")
            label = find_text(boxes, label_text)
            if label is None:
                raise WorldJoinError(
                    f"the group instance dialog shows no '{label_text}' label "
                    f"(seen: {', '.join(self.last_texts[:25])})"
                )
            value = value_below(boxes, label)
            observed = "" if value is None else value.text
            if value is not None and accept(observed):
                return observed
            if attempt == 1 or value is None:
                break
            # Open the row and pick the wanted entry from the list.
            self._click_hit(value)
            self._settle(0.9)
            deadline = time.monotonic() + self.spec.step_timeout
            chosen: TextHit | None = None
            while chosen is None:
                _frame, listing = self._menu_observe(f"{name}_list")
                chosen = pick(listing)
                if chosen is None and time.monotonic() >= deadline:
                    break
                if chosen is None:
                    time.sleep(self.poll_seconds)
            if chosen is None:
                self._press_escape()
                self._settle(0.8)
                raise WorldJoinError(
                    f"{name}: no entry matching the requested value was listed "
                    f"(seen: {', '.join(self.last_texts[:25])})"
                )
            self._click_hit(chosen)
            self._settle(0.9)
            self._menu_wait("group_dialog", "Create Instance")
        wanted = {"group": self.spec.group, "access": self.spec.group_access}[name]
        raise WorldJoinError(f"{name} is '{observed}', expected '{wanted}'")

    def _select_region(self) -> str:
        """Click the region button; a radio button, so already-selected is fine."""

        _frame, boxes = self._menu_observe("verify_region")
        label = find_text(boxes, "Region")
        region: Rect | None = None
        if label is not None:
            region = Rect(0, label.rect.bottom, 10_000, int(3.5 * label.rect.height))
        hit = find_text(boxes, self.spec.region, mode="suffix", region=region)
        if hit is None:
            hit = find_text(boxes, self.spec.region, mode="suffix")
        if hit is None:
            self._warn(f"region button '{self.spec.region}' was not found; left as shown")
            return ""
        self._click_hit(hit)
        self._settle(0.5)
        return hit.text

    def create_instance(self) -> None:
        self._phase("creating_instance")
        self._menu_click("create_instance", "Create Instance", settle=1.2)
        self._menu_wait("instance_created", "Create Instance", absent=True)
        _hit, boxes = self._menu_wait("world_page_join", "Join")
        # The card under "Selected Instance" should now describe the new
        # group instance rather than whatever was selected before.
        top = find_text(boxes, "Selected Instance")
        bottom = find_text(boxes, "Actions")
        if top is not None and bottom is not None and bottom.rect.y > top.rect.bottom:
            # The card only, not the "Instances" list to its right, which
            # shows the new instance whether or not it is the selected one.
            card = Rect(
                max(0, top.rect.x - 20),
                top.rect.bottom,
                max(200, int(0.45 * self.frame_size[0])),
                bottom.rect.y - top.rect.bottom,
            )
            deadline = time.monotonic() + min(8.0, self.spec.step_timeout)
            while find_text(boxes, "Group", mode="prefix", region=card) is None:
                if time.monotonic() >= deadline:
                    self._warn("selected instance card did not read as a group instance")
                    break
                time.sleep(self.poll_seconds)
                _frame, boxes = self._menu_observe("instance_card")
        self._done_step("create_instance")

    def join_instance(self) -> None:
        self._phase("joining")
        for attempt in range(2):
            # No cursor parking: the menu closes on this click and an
            # absolute pointer move in mouse-look mode would swing the view.
            self._menu_click("join", "Join", settle=1.5, park=False)
            try:
                self._menu_wait(
                    "menu_closed", "Join", absent=True, timeout=min(8.0, self.spec.step_timeout)
                )
                self._done_step("join")
                return
            except WorldJoinError:
                if attempt == 1:
                    raise
        raise WorldJoinError("the Join button stayed on screen")

    def wait_for_world(self) -> dict[str, Any]:
        """Wait for the loading screen to give way to the world.

        Loading is recognised by large "Loading"-style text or the world's
        name on screen; the world by the Sliden landmark or its wordmark, or
        simply by two polls in a row without any loading text after a
        minimum wait.  Never fatal: past the timeout the run carries on and
        lets the room search discover whether the world is really there.
        """

        self._phase("loading_world")
        started = time.monotonic()
        minimum = self._LOAD_MINIMUM_SECONDS
        quiet = 0
        polls = 0
        loaded_by = "timeout"
        while True:
            _check_cancel(self.cancelled)
            elapsed = time.monotonic() - started
            frame, boxes = self._menu_observe("loading", snapshot=polls % 5 == 0)
            polls += 1
            height = frame.shape[0]
            screen = detect_sliden_screen(frame)
            if screen.rect is not None and screen.confidence >= 0.55:
                loaded_by = "sliden_screen"
                break
            if find_text(boxes, "Sliden") is not None and elapsed >= minimum:
                loaded_by = "sliden_wordmark"
                break
            loading = any(
                b.height >= _LOADING_MIN_HEIGHT_FRACTION * height
                and any(w in normalize_text(b.text) for w in _LOADING_WORDS)
                for b in boxes
            ) or find_text(boxes, self.spec.world, mode="fuzzy") is not None
            if find_text(boxes, "Join") is not None:
                # The menu is still up: the join did not take.
                loading = True
            self._state(
                worldJoinLoadingSeen=loading,
                worldJoinLoadSeconds=round(elapsed, 1),
            )
            if loading:
                quiet = 0
            elif elapsed >= minimum:
                quiet += 1
                if quiet >= 2:
                    loaded_by = "no_loading_text"
                    break
            if elapsed >= self.spec.load_timeout:
                self._warn("world load could not be confirmed before the timeout")
                break
            time.sleep(self._LOAD_POLL_SECONDS)
        seconds = round(time.monotonic() - started, 1)
        self._state(worldJoinLoadedBy=loaded_by, worldJoinLoadSeconds=seconds)
        self._done_step("world_loaded")
        return {"loadedBy": loaded_by, "seconds": seconds}

    def enter_sliden_room(
        self,
        *,
        screen_confidence: float = 0.55,
        target_width_fraction: float = 0.32,
        maximum_width_fraction: float = 0.55,
        full_turn_seconds: float = 2.2,
    ) -> dict[str, Any]:
        """Turn until the Sliden wall screen is seen and walk up to it.

        This is the landmark phase of ``locate_and_approach`` on its own:
        arriving means the screen spans ``target_width_fraction`` of the
        view, which puts the avatar inside the room in front of it.  While
        the screen is too small or too oblique for the detector, the OCR'd
        "Sliden" wordmark on it stands in as the target.  The default stops
        with the real screen at roughly half the view: from further back the
        operation panel's edge-on bar beside the screen is too thin to be
        seen (a live run from 0.24 never found it), and the panel approach
        that follows backs off by itself if the screen is clipped.
        """

        self._phase("entering_room")
        started = time.monotonic()
        deadline = started + self.spec.navigation_timeout
        sweep_seconds = 0.0
        sweep_index = 0
        align_attempts = 0
        stall_count = 0
        best_seen_width = 0.0
        strafes = 0
        best_confidence = 0.0
        arrival_rejects = 0
        steer_attempts = 0
        rejected_candidates: list[tuple[float, float]] = []
        bands = self._SCREEN_PITCH_BANDS
        self.level_camera()
        # Look all the way round first and turn to the biggest Sliden page in
        # sight.  Every Sliden surface in this world reads "Sliden" -- the
        # wall screen, its sub-screens, the panel, and displays in the
        # gallery outside -- and walking at the first one seen took the
        # avatar to a small far-away sign, or to the wall screen through the
        # glass wall it cannot pass.  Only pages seen at this pitch count; a
        # page never seen in the survey is left to the sweep below.
        survey_steps = 12
        survey_counts = self._YAW_COUNTS_PER_360 / survey_steps
        best_step: int | None = None
        best_survey_width = 0
        for step in range(survey_steps):
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is not None:
                boxes = self._ensure_ocr()(frame)
                estimated = screen_from_text_boxes(
                    [(b.text, b.rect) for b in boxes], frame.shape[1], frame.shape[0],
                    exclude=console_page_rects(frame),
                    bars=[bar for bar, _l, _t in _bright_horizontal_bars(frame)],
                )
                if estimated is not None and estimated.width > best_survey_width:
                    best_survey_width = estimated.width
                    best_step = step
                self._state(
                    worldJoinRoomStatus="surveying",
                    worldJoinRoomSurveyStep=step,
                    worldJoinRoomSurveyBest=None if best_step is None else [best_step, best_survey_width],
                )
            self._move_relative((survey_counts, 0.0))
            time.sleep(max(0.15, self.poll_seconds))
        if best_step is not None:
            # A full revolution later the view is where it started; turn back
            # to the best step by the same exact counts.
            self._move_relative((survey_counts * best_step, 0.0))
            time.sleep(max(0.2, self.poll_seconds))
        while time.monotonic() < deadline:
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is None:
                time.sleep(self.poll_seconds)
                continue
            height, width = frame.shape[:2]
            viewport = self._viewport_center(frame)
            screen = detect_sliden_screen(frame)
            best_confidence = max(best_confidence, screen.confidence)
            target: Rect | None = None
            source = ""
            if screen.rect is not None and screen.confidence >= screen_confidence:
                target, source = screen.rect, "screen"
            else:
                boxes = self._ensure_ocr()(frame)
                self._menu_snapshot("room_search", frame)
                # The wordmark inflated to its page -- with the sub-screens
                # ("Local" under the page) and the operation panel ("Enter
                # Slide URL" above it) left out, so that standing next to
                # one of those does not pass for having reached the room.
                estimated = screen_from_text_boxes(
                    [(b.text, b.rect) for b in boxes], width, height,
                    exclude=console_page_rects(frame),
                    bars=[bar for bar, _l, _t in _bright_horizontal_bars(frame)],
                )
                if estimated is not None:
                    target = estimated
                    source = "wordmark"
            if target is not None:
                fraction = target.width / width
                error_x = float(target.center[0] - viewport[0])
                error_y = float(target.center[1] - viewport[1])
                self._state(
                    worldJoinRoomLandmark=source,
                    worldJoinRoomWidthFraction=round(fraction, 3),
                    worldJoinRoomCenterErrorX=round(error_x, 1),
                    worldJoinRoomStatus="approaching",
                )
                if fraction > maximum_width_fraction:
                    self._walk_backward(0.28)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if fraction >= target_width_fraction:
                    if abs(error_y) > 0.14 * height and align_attempts < 8:
                        align_attempts += 1
                        self._move_relative(
                            (0.0, self._pitch_step(max(-55.0, min(55.0, 0.22 * error_y))))
                        )
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    # Not the wall screen if, from this close, the page shows
                    # console or sub-screen furniture: the operation panel and
                    # the sub-screens render the same page.  Turn away from
                    # it and keep looking.
                    pages = console_page_rects(frame)
                    tx, ty = target.center
                    if (
                        any(
                            p.x - 0.1 * p.width <= tx <= p.right + 0.1 * p.width
                            and p.y - 0.1 * p.height <= ty <= p.bottom + 0.1 * p.height
                            for p in pages
                        )
                        or page_has_console_bars(
                            target, [bar for bar, _l, _t in _bright_horizontal_bars(frame)]
                        )
                    ) and arrival_rejects < 4:
                        arrival_rejects += 1
                        self._state(worldJoinRoomStatus="rejected_console_page",
                                    worldJoinRoomRejects=arrival_rejects)
                        self._scan_turn(0.30, 1)
                        time.sleep(max(0.18, self.poll_seconds))
                        continue
                    if source != "screen":
                        self._warn("arrived on the Sliden wordmark; the wall screen detector never confirmed it")
                    result = {
                        "landmark": source,
                        "widthFraction": round(fraction, 3),
                        "confidence": round(float(screen.confidence), 3),
                        "seconds": round(time.monotonic() - started, 1),
                    }
                    self._state(
                        worldJoinRoomStatus="arrived",
                        worldJoinRoomLandmark=source,
                        worldJoinRoomWidthFraction=result["widthFraction"],
                        worldJoinRoomSeconds=result["seconds"],
                    )
                    self._done_step("sliden_room")
                    return result
                move_x = max(-55.0, min(55.0, 0.24 * error_x)) if abs(error_x) > 0.08 * width else 0.0
                move_y = max(-55.0, min(55.0, 0.22 * error_y)) if abs(error_y) > 0.10 * height else 0.0
                if (move_x != 0.0 or move_y != 0.0) and align_attempts < 12:
                    align_attempts += 1
                    self._move_relative((move_x, self._pitch_step(move_y)))
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                align_attempts = 0
                if fraction <= best_seen_width + 0.002:
                    stall_count += 1
                else:
                    stall_count = 0
                best_seen_width = max(best_seen_width, fraction)
                if stall_count >= 4:
                    # Walking is not bringing it closer: the glass wall of
                    # the room, a door frame, furniture.  Slide sideways --
                    # in one direction for a while, then the other for
                    # longer -- so the avatar drifts along the obstacle to
                    # where it ends (the doorway) rather than shuffling on
                    # the spot, which alternating steps amount to.
                    stall_count = 0
                    best_seen_width = fraction
                    strafes += 1
                    leg = 1
                    while strafes > 3 * leg * (leg + 1) // 2:
                        leg += 1
                    direction = 1 if leg % 2 else -1
                    self._state(worldJoinRoomStatus="sidestepping",
                                worldJoinRoomSidesteps=strafes, worldJoinRoomSidestepDirection=direction)
                    self._walk_backward(0.20)
                    self._strafe(0.55, direction)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                self._walk_forward(0.42 if fraction < 0.18 else 0.22)
                time.sleep(max(0.18, self.poll_seconds))
                continue
            # Nothing recognisable yet.  A dark page candidate in view -- the
            # wall screen with its wordmark out of frame, typically -- is
            # worth centring first, so its text can be read; otherwise sweep
            # to the right, one pitch band per turn.
            candidates = [
                c for c in _display_candidates(frame, minimum_area_fraction=0.004)
                if not page_has_console_bars(
                    c.rect, [bar for bar, _l, _t in _bright_horizontal_bars(frame)]
                )
                and not any(
                    abs(c.rect.center[0] - rx) < 0.1 * width and abs(c.rect.center[1] - ry) < 0.1 * height
                    for rx, ry in rejected_candidates
                )
            ]
            if candidates and steer_attempts < 6:
                steer_attempts += 1
                cand = candidates[0].rect
                error_x = float(cand.center[0] - viewport[0])
                error_y = float(cand.center[1] - viewport[1])
                self._state(worldJoinRoomStatus="steering_to_page",
                            worldJoinRoomCandidate=[cand.x, cand.y, cand.width, cand.height])
                if abs(error_x) > 0.08 * width or abs(error_y) > 0.10 * height:
                    self._move_relative((
                        max(-55.0, min(55.0, 0.24 * error_x)),
                        self._pitch_step(max(-55.0, min(55.0, 0.22 * error_y)))))
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                # Centred and still no wordmark: not a Sliden page after all.
                rejected_candidates.append(cand.center)
                steer_attempts = 0
                continue
            steer_attempts = 0
            self._state(
                worldJoinRoomStatus="searching",
                worldJoinRoomSweep=sweep_index + 1,
                worldJoinRoomBestConfidence=round(best_confidence, 3),
            )
            best_seen_width = 0.0
            stall_count = 0
            if sweep_seconds + 0.16 > full_turn_seconds:
                sweep_seconds = 0.0
                sweep_index += 1
                self._set_pitch(bands[sweep_index % len(bands)])
                if sweep_index % len(bands) == 0:
                    # A full revolution at every pitch: this spot cannot see
                    # the screen.  Leave it for open floor and sweep again.
                    self._escape_to_open_space()
                    rejected_candidates.clear()
            else:
                self._set_pitch(bands[sweep_index % len(bands)])
                self._scan_turn(0.16, 1)
                sweep_seconds += 0.16
            time.sleep(max(0.18, self.poll_seconds))
        self._set_pitch(0.0)
        raise WorldJoinError(
            "the Sliden wall screen was not reached within "
            f"{self.spec.navigation_timeout:.0f}s (best screen confidence "
            f"{best_confidence:.2f}, {sweep_index} sweeps)"
        )

    def respawn_via_menu(self) -> bool:
        """Return to the world's spawn point through the Launch Pad.

        Escape toggles the menu; the Launch Pad, with its Respawn button, is
        what opens from a closed menu.  Read rather than clicked blind, so a
        different window size or a menu left on another page does not send
        the click somewhere else.
        """

        self._phase("respawning")
        for attempt in range(6):
            _frame, boxes = self._menu_observe("respawn")
            hit = find_text(boxes, "Respawn")
            if hit is not None:
                self._click_hit(hit)
                self._settle(2.5)
                self._menu_wait("respawned", "Respawn", absent=True, timeout=min(10.0, self.spec.step_timeout))
                self.level_camera()
                self._done_step("respawn")
                return True
            self._press_escape()
            self._settle(0.9)
        return False

    def approach_console(self) -> dict[str, Any]:
        self._phase("approaching_console")
        observation = self.locate_and_approach(timeout_seconds=self.spec.navigation_timeout)
        self._done_step("console")
        return {"confidence": round(float(observation.confidence), 3)}

    def run(self) -> dict[str, Any]:
        """The whole sequence.  Returns a summary; raises on failure."""

        summary: dict[str, Any] = {}
        self.open_menu()
        self.open_worlds()
        self.select_favorites()
        self.select_world()
        self.open_new_instance()
        self.choose_group_type()
        summary["settings"] = self.verify_group_settings()
        self.create_instance()
        self.join_instance()
        summary["load"] = self.wait_for_world()
        if self.spec.enter_room:
            summary["room"] = self.enter_sliden_room()
            if self.spec.approach_console:
                summary["console"] = self.approach_console()
        summary["steps"] = list(self.steps)
        summary["warnings"] = list(self.warnings)
        return summary


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


class _StateFile:
    """Atomic JSON state, merged on every write."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.state: dict[str, Any] = {}
        self.lock = threading.Lock()

    def write(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)
            self.state["worldJoinUpdatedAt"] = time.time()
            if self.path is None:
                return
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            try:
                temporary.write_text(
                    json.dumps(self.state, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                temporary.replace(self.path)
            except OSError:
                pass


def _watch_control_file(path: Path | None, stop: threading.Event) -> None:
    if path is None:
        return
    while not stop.wait(0.2):
        if not path.is_file():
            continue
        try:
            command = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            command = {}
        try:
            path.unlink()
        except OSError:
            pass
        if isinstance(command, dict) and str(command.get("command", "")).lower() == "stop":
            stop.set()
            return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--control-file", type=Path, default=None)
    parser.add_argument("--diagnostic-dir", type=Path, default=None)
    parser.add_argument("--vrchat-pid", type=int, default=0)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    args = parser.parse_args(argv)

    data: dict[str, Any] = {}
    if args.spec_file is not None:
        data = json.loads(args.spec_file.read_text(encoding="utf-8"))
    spec = WorldJoinSpec.from_json(data)
    pid = int(data.get("vrchatPid", args.vrchat_pid) or 0) or None
    osc_host = str(data.get("oscHost", args.osc_host))
    osc_port = int(data.get("oscPort", args.osc_port))

    state = _StateFile(args.state_file)
    stop = threading.Event()
    threading.Thread(
        target=_watch_control_file, args=(args.control_file, stop), daemon=True
    ).start()
    state.write(
        worldJoinStatus="running",
        worldJoinPhase="starting",
        worldJoinStartedAt=time.time(),
        worldJoinPid=os.getpid(),
        worldJoinSpec=asdict(spec),
        worldJoinDiagnosticDirectory=str(args.diagnostic_dir) if args.diagnostic_dir else None,
    )
    try:
        from vrcpilot import process

        resolved = process.resolve_pid(pid)
        state.write(worldJoinVRChatPID=resolved)
    except Exception as exc:
        state.write(
            worldJoinStatus="failed",
            worldJoinPhase="starting",
            worldJoinError=f"VRChat is not available: {exc}",
        )
        return 1

    controller = WorldJoinController(
        spec,
        pid=resolved,
        cancelled=stop.is_set,
        state_callback=state.write,
        osc_host=osc_host,
        osc_port=osc_port,
        diagnostic_directory=args.diagnostic_dir,
    )
    try:
        summary = controller.run()
    except SlidenCancelled:
        state.write(worldJoinStatus="cancelled", worldJoinError="cancelled")
        return 2
    except (WorldJoinError, SlidenConsoleNotFound, ValueError) as exc:
        state.write(
            worldJoinStatus="failed",
            worldJoinError=str(exc),
            worldJoinSteps=list(controller.steps),
            worldJoinWarnings=list(controller.warnings),
        )
        return 1
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        state.write(
            worldJoinStatus="failed",
            worldJoinError=f"{type(exc).__name__}: {exc}",
            worldJoinSteps=list(controller.steps),
        )
        return 1
    finally:
        controller.close()
    state.write(worldJoinStatus="done", worldJoinPhase="done", worldJoinSummary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
