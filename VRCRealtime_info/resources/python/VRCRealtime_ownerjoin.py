"""Find the owner through the VRChat menu and join their instance.

Used by the owner-follow engine when the room search keeps coming up empty:

1. ``check_owner_in_instance``: open the main menu, press the *Location*
   icon under the Launch Pad (the "Here" page), scroll its list down and read
   the **Users** section.  If the owner's name is there they are somewhere in
   this world and the search should simply continue.
2. ``join_owner``: otherwise open the right *Wing*, press its hamburger, pick
   *Friends*, page through the friend lists with the arrows until the
   **Online** list shows, click the owner, and on their page press **Join**
   when it is offered (disabled/absent means the instance cannot be joined
   from here).  Then wait for the new world to load.

Everything is local menu automation on OCR anchors (same machinery as
``VRCRealtime_worldjoin``); the only network effect is the VRChat join itself.
Measured layout anchors (1284x767 capture, menu is a fixed-size panel):

* bottom icon bar sits 41 px under the Launch Pad's "Respawn" label; the
  Location pin is the third icon, at Respawn's x (-5 px);
* wing hamburger is 56 px left of the wing title; the page arrows sit 30 px
  below the title (left arrow -46 px, right arrow -7 px from the title x);
* the "Here" page's scroll track is the panel's right edge (x = panel right
  - 12 px), its lower end 30 px above the panel bottom.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from VRCRealtime_worldjoin import (
    TextBox,
    TextHit,
    WorldJoinController,
    WorldJoinError,
    WorldJoinSpec,
    find_text,
    normalize_text,
)
from VRCRealtime_sliden import Rect


@dataclass
class OwnerLocateResult:
    outcome: str  # "in_instance" | "joined" | "not_online" | "join_unavailable" | "error"
    detail: str = ""
    owner_listed_in_instance: bool | None = None
    online_seen: bool | None = None
    join_enabled: bool | None = None
    seconds: float = 0.0
    steps: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "detail": self.detail,
            "ownerListedInInstance": self.owner_listed_in_instance,
            "onlineSeen": self.online_seen,
            "joinEnabled": self.join_enabled,
            "seconds": round(self.seconds, 1),
            "steps": list(self.steps),
        }


def looks_like(observed: str, target: str, *, min_ratio: float = 0.75) -> bool:
    """OCR-tolerant label test on the alphabetic part ("Onllne (1)" ~ "Online")."""

    seen = "".join(ch for ch in normalize_text(observed) if ch.isalpha())
    wanted = "".join(ch for ch in normalize_text(target) if ch.isalpha())
    if not seen or not wanted:
        return False
    if seen == wanted or (seen.startswith(wanted) and len(seen) <= len(wanted) + 3):
        return True
    return SequenceMatcher(None, seen, wanted).ratio() >= min_ratio


def find_like(
    boxes: Sequence[TextBox], target: str, *, region: Rect | None = None, min_ratio: float = 0.75
) -> TextBox | None:
    """First OCR box whose alphabetic text looks like *target* (topmost wins)."""

    best: TextBox | None = None
    for box in boxes:
        if region is not None:
            cx, cy = box.center
            if not (region.x <= cx <= region.right and region.y <= cy <= region.bottom):
                continue
        if looks_like(box.text, target, min_ratio=min_ratio):
            if best is None or box.y < best.y:
                best = box
    return best


def owner_name_matches(owner: str, observed: str, *, min_ratio: float = 0.7) -> bool:
    """OCR-tolerant comparison of a user name ("nconc" vs "nconco", "ncnc")."""

    wanted = normalize_text(owner)
    seen = normalize_text(observed)
    if not wanted or not seen:
        return False
    if wanted == seen or wanted in seen:
        return True
    if len(wanted) >= 4 and len(seen) >= 4 and seen in wanted:
        return True
    return SequenceMatcher(None, wanted, seen).ratio() >= min_ratio


class OwnerLocator(WorldJoinController):
    """Menu automation: is the owner here, and if not, join them."""

    #: Vertical offset from the Launch Pad's Respawn label to the bottom icon
    #: bar, and the Location pin's horizontal offset from it (px at 1x).
    _PIN_DY = 41.0
    _PIN_DX = -5.0
    #: Wing title -> hamburger / page arrows (px at 1x).
    _HAMBURGER_DX = -56.0
    _ARROW_DY = 30.0
    _LEFT_ARROW_DX = -46.0
    _RIGHT_ARROW_DX = -7.0
    #: Panel size reference: Respawn.y - clock.y measured 370 px.
    _PANEL_REFERENCE = 370.0

    def __init__(
        self,
        owner_account: str,
        *,
        pid: int | None = None,
        cancelled: Callable[[], bool] | None = None,
        state_callback: Callable[..., Any] | None = None,
        diagnostic_directory: Path | None = None,
        load_timeout: float = 180.0,
        step_timeout: float = 20.0,
        **callbacks: Any,
    ) -> None:
        spec = WorldJoinSpec(
            step_timeout=step_timeout,
            load_timeout=load_timeout,
            enter_room=False,
            approach_console=False,
        )
        super().__init__(
            spec,
            pid=pid,
            cancelled=cancelled,
            state_callback=state_callback,
            diagnostic_directory=diagnostic_directory,
            **callbacks,
        )
        self.owner_account = owner_account.strip()
        if not self.owner_account:
            raise ValueError("owner_account must not be empty")
        self.dry_run = False

    # -- state ------------------------------------------------------------
    def _state(self, **updates: Any) -> None:  # noqa: D401 - same contract
        if self.state_callback is not None:
            self.state_callback(
                ownerLocateElapsedSeconds=round(time.monotonic() - self._started, 1),
                **{("ownerLocate" + k[len("worldJoin"):] if k.startswith("worldJoin") else k): v for k, v in updates.items()},
            )

    # -- geometry helpers -------------------------------------------------
    def _panel_scale(self, boxes: Sequence[TextBox]) -> float:
        clock = self._clock_hit(boxes)
        respawn = find_text(boxes, "Respawn", mode="fuzzy", min_ratio=0.7)
        if clock is None or respawn is None:
            return 1.0
        distance = respawn.center[1] - clock.center[1]
        if distance < 200 or distance > 600:
            return 1.0
        return distance / self._PANEL_REFERENCE

    @staticmethod
    def _clock_hit(boxes: Sequence[TextBox]) -> TextHit | None:
        for box in boxes:
            text = box.text.strip()
            if len(text) in (4, 5) and ":" in text and text.replace(":", "").isdigit():
                return TextHit(text=text, rect=box.rect, score=box.confidence, boxes=(box,))
        return None

    @staticmethod
    def _right_half(frame_size: tuple[int, int]) -> Rect:
        width, height = frame_size
        return Rect(width // 2, 0, width - width // 2, height)

    def _wing_title(self, boxes: Sequence[TextBox]) -> tuple[TextHit, str] | None:
        """The right wing's title ("Explore" / "Friends" / "Wings" ...) and its canonical name."""

        width, height = self.frame_size
        region = Rect(width // 2, 0, width - width // 2, int(height * 0.5))
        for name in ("Friends", "Explore", "Wings", "Social", "Groups", "Worlds", "Avatars"):
            box = find_like(boxes, name, region=region, min_ratio=0.8)
            if box is not None:
                hit = TextHit(text=box.text, rect=box.rect, score=box.confidence, boxes=(box,))
                return hit, name
        return None

    # -- steps --------------------------------------------------------------
    def open_here_page(self) -> None:
        """Launch Pad -> Location pin -> "Here" page."""

        self._phase("opening_here")
        for attempt in range(3):
            _frame, boxes = self._menu_observe("launch_pad")
            if find_text(boxes, "Here") is not None and find_text(boxes, "Launch Pad") is None:
                self._done_step("here")
                return
            respawn = find_text(boxes, "Respawn", mode="fuzzy", min_ratio=0.7)
            if respawn is None:
                self.open_menu()
                _frame, boxes = self._menu_observe("launch_pad_again")
                respawn = find_text(boxes, "Respawn", mode="fuzzy", min_ratio=0.7)
            if respawn is None:
                raise WorldJoinError("the Launch Pad's Respawn label was not found")
            scale = self._panel_scale(boxes)
            x = respawn.center[0] + self._PIN_DX * scale
            y = respawn.center[1] + self._PIN_DY * scale
            self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText="location-pin")
            self._click_ui((x, y))
            self._park_cursor()
            self._settle(1.2)
            try:
                self._menu_wait("here_page", "Here", timeout=6.0)
                self._done_step("here")
                return
            except WorldJoinError:
                if attempt == 2:
                    raise

    def read_instance_users(self) -> tuple[list[str], bool]:
        """Scroll the "Here" page down and return (user names, owner_listed)."""

        self._phase("reading_users")
        names: list[str] = []
        listed = False
        for attempt in range(4):
            frame, boxes = self._menu_observe(f"here_scroll_{attempt}")
            users_box = find_like(
                boxes,
                "Users",
                region=Rect(int(frame.shape[1] * 0.3), 0, int(frame.shape[1] * 0.4), frame.shape[0]),
            )
            users = (
                TextHit(text=users_box.text, rect=users_box.rect, score=users_box.confidence, boxes=(users_box,))
                if users_box is not None
                else None
            )
            if users is not None:
                region = Rect(
                    int(users.rect.x - 40),
                    int(users.rect.bottom),
                    int(frame.shape[1] * 0.35),
                    int(frame.shape[0] - users.rect.bottom),
                )
                for box in boxes:
                    cx, cy = box.center
                    if not (region.x <= cx <= region.right and region.y <= cy <= region.bottom):
                        continue
                    text = box.text.strip()
                    if not text or normalize_text(text) in {"online", "activeonline", "joinme", "askme", "busy"}:
                        continue
                    names.append(text)
                    if owner_name_matches(self.owner_account, text):
                        listed = True
                self._state(ownerLocateInstanceUsers=names[:20], ownerLocateOwnerListed=listed)
                self._done_step("users_read")
                return names, listed
            # Not scrolled far enough yet: click the lower end of the scroll
            # track at the panel's right edge (the "Here" header gives the
            # panel's top; "Respawn"/clock give its size).
            here = find_text(boxes, "Here")
            clock = self._clock_hit(boxes)
            if here is None or clock is None:
                raise WorldJoinError("the Here page is not on screen")
            scale = self._panel_scale(boxes)
            panel_right = clock.center[0] + 183.0 * scale
            panel_bottom = clock.center[1] + 395.0 * scale
            x = panel_right - 12.0 * scale
            y = panel_bottom - 30.0 * scale - attempt * 6.0
            self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText="scroll-track")
            self._click_ui((x, y))
            self._park_cursor()
            self._settle(0.9)
        raise WorldJoinError("the Users section of the Here page did not appear")

    def open_friends_online(self) -> bool:
        """Right wing -> hamburger -> Friends -> arrows until "Online (N)"."""

        self._phase("opening_friends")
        frame, boxes = self._menu_observe("wing")
        found = self._wing_title(boxes)
        if found is None:
            raise WorldJoinError("the right wing's title was not found")
        title, name = found
        scale = self._panel_scale(boxes)
        region = self._right_half(self.frame_size)
        header_region = Rect(region.x, int(title.rect.bottom), region.width, int(self.frame_size[1] * 0.25))
        online = find_like(boxes, "Online", region=header_region)
        if not (name == "Friends" and online is not None):
            if name != "Wings":
                # Hamburger -> "Wings" list.
                x = title.center[0] + self._HAMBURGER_DX * scale
                y = title.center[1]
                self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText="hamburger")
                self._click_ui((x, y))
                self._park_cursor()
                self._settle(1.0)
            self._menu_click("wing_friends", "Friends", mode="fuzzy", region=region, settle=1.2)
            frame, boxes = self._menu_observe("friends_wing")
            found = self._wing_title(boxes)
            if found is None or found[1] != "Friends":
                raise WorldJoinError("the Friends wing did not open")
            title, name = found
        for attempt in range(10):
            frame, boxes = self._menu_observe(f"friends_page_{attempt}")
            found = self._wing_title(boxes)
            if found is not None and found[1] == "Friends":
                title = found[0]
            header_region = Rect(region.x, int(title.rect.bottom), region.width, int(self.frame_size[1] * 0.25))
            online = find_like(boxes, "Online", region=header_region)
            if online is not None:
                # The "Online (N)" header (a row's "● Online" badge sits lower
                # and only appears on this page anyway).
                self._done_step("friends_online")
                return True
            x = title.center[0] + self._RIGHT_ARROW_DX * scale
            y = title.center[1] + self._ARROW_DY * scale
            self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText="friends-next")
            self._click_ui((x, y))
            self._park_cursor()
            self._settle(0.9)
        return False

    def open_owner_page(self) -> bool:
        """Click the owner's row in the Online list; True when their page shows."""

        self._phase("opening_owner")
        region = self._right_half(self.frame_size)
        frame, boxes = self._menu_observe("online_list")
        row: TextBox | None = None
        for box in boxes:
            cx, cy = box.center
            if cx < region.x or cy < self.frame_size[1] * 0.3:
                continue
            if owner_name_matches(self.owner_account, box.text):
                row = box
                break
        if row is None:
            self._state(ownerLocateOnlineSeen=False)
            return False
        self._state(ownerLocateOnlineSeen=True)
        hit = TextHit(text=row.text, rect=row.rect, score=row.confidence, boxes=(row,))
        self._click_hit(hit)
        self._settle(1.4)
        self._menu_wait("owner_page", ["Send Invite", "Unfriend", "Block User", "Favorite Friend"], mode="fuzzy", timeout=8.0)
        self._done_step("owner_page")
        return True

    def join_button(self, frame: np.ndarray, boxes: Sequence[TextBox]) -> tuple[TextHit | None, bool]:
        """The Join button and whether it looks enabled (teal, not grey)."""

        region = Rect(int(self.frame_size[0] * 0.3), int(self.frame_size[1] * 0.35), int(self.frame_size[0] * 0.4), int(self.frame_size[1] * 0.5))
        hit = find_text(boxes, "Join", region=region)
        if hit is None:
            return None, False
        x0 = max(0, int(hit.rect.x - 10))
        x1 = min(frame.shape[1], int(hit.rect.right + 10))
        y0 = max(0, int(hit.rect.y - 40))
        y1 = min(frame.shape[0], int(hit.rect.y - 8))
        patch = frame[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0:
            return hit, True
        red, green, blue = (float(patch[..., c].mean()) for c in range(3))
        # Enabled tiles are teal (#2a7c85-ish): green/blue well above red.
        enabled = (green + blue) / 2.0 - red > 18.0 and (green + blue) / 2.0 > 70.0
        return hit, enabled

    def press_join(self) -> OwnerLocateResult:
        self._phase("joining_owner")
        frame, boxes = self._menu_observe("owner_page_join")
        hit, enabled = self.join_button(frame, boxes)
        if hit is None:
            request = find_text(boxes, ["Request Invite", "Req Invite"], mode="fuzzy")
            detail = "Join is not offered" + (" (Request Invite only)" if request else "")
            return OwnerLocateResult("join_unavailable", detail, online_seen=True, join_enabled=False)
        self._state(ownerLocateJoinEnabled=enabled)
        if not enabled:
            return OwnerLocateResult("join_unavailable", "Join is disabled", online_seen=True, join_enabled=False)
        if self.dry_run:
            return OwnerLocateResult("joined", "dry run: Join not pressed", online_seen=True, join_enabled=True)
        self._click_hit(hit, park=True)
        self._settle(1.2)
        # VRChat asks "Are you sure you want to join <name>?" with YES / NO.
        confirmed = self._confirm_join_dialog()
        if not confirmed:
            return OwnerLocateResult(
                "error", "the join confirmation dialog could not be answered", online_seen=True, join_enabled=True
            )
        self._done_step("join")
        loaded = self.wait_for_world()
        return OwnerLocateResult("joined", f"loaded by {loaded.get('loadedBy')}", online_seen=True, join_enabled=True)

    #: YES/NO buttons of the join confirmation: 121 px below the dialog's
    #: "Join" title, 90 px left/right of it (1x panel scale).
    _CONFIRM_DY = 121.0
    _CONFIRM_DX = 90.0

    def _confirm_join_dialog(self) -> bool:
        """Answer the "Are you sure you want to join ...?" dialog with YES."""

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            frame, boxes = self._menu_observe("join_confirm")
            asked = find_text(boxes, "Are you sure you want to join", mode="fuzzy", min_ratio=0.6)
            if asked is None:
                asked = next(
                    (
                        TextHit(text=b.text, rect=b.rect, score=b.confidence, boxes=(b,))
                        for b in boxes
                        if "sure" in normalize_text(b.text) and "join" in normalize_text(b.text)
                    ),
                    None,
                )
            if asked is None:
                # No dialog (or already gone): the join proceeds on its own.
                if find_text(boxes, "Send Invite", mode="fuzzy") is None:
                    return True
                time.sleep(self.poll_seconds)
                continue
            yes = find_text(boxes, ["YES", "VES", "YE5"], mode="fuzzy", min_ratio=0.6)
            no = find_text(boxes, "NO")
            if yes is not None and (no is None or yes.center[0] < no.center[0]):
                self._click_hit(yes, park=False)
            elif no is not None:
                x = no.center[0] - 2.0 * self._CONFIRM_DX * self._panel_scale(boxes)
                self._state(worldJoinLastClick=[round(x), round(no.center[1])], worldJoinLastClickText="YES(by NO)")
                self._click_ui((x, no.center[1]))
            else:
                scale = self._panel_scale(boxes)
                x = asked.center[0] - self._CONFIRM_DX * scale
                y = asked.center[1] + 0.55 * self._CONFIRM_DY * scale
                self._state(worldJoinLastClick=[round(x), round(y)], worldJoinLastClickText="YES(by prompt)")
                self._click_ui((x, y))
            self._settle(1.5)
            _frame, boxes = self._menu_observe("join_confirm_after", snapshot=False)
            if find_text(boxes, "Are you sure you want to join", mode="fuzzy", min_ratio=0.6) is None:
                return True
        return False

    def close_menu(self) -> None:
        """Escape until no menu marker text remains (at most three presses)."""

        for _ in range(3):
            _frame, boxes = self._menu_observe("closing", snapshot=False)
            markers = ("Launch Pad", "Here", "Respawn", "Send Invite", "Wings")
            if all(find_text(boxes, m) is None for m in markers):
                return
            self._press_escape()
            self._settle(0.8)

    # -- the whole procedure ------------------------------------------------
    def run(self, *, join: bool = True) -> OwnerLocateResult:
        """Check the instance, and join the owner elsewhere if allowed."""

        started = time.monotonic()
        result: OwnerLocateResult
        try:
            self.open_menu()
            self.open_here_page()
            _names, listed = self.read_instance_users()
            if listed:
                result = OwnerLocateResult("in_instance", "owner is listed in this instance", owner_listed_in_instance=True)
            elif not join:
                result = OwnerLocateResult("not_online", "owner not in this instance (join disabled)", owner_listed_in_instance=False)
            else:
                if not self.open_friends_online():
                    result = OwnerLocateResult("not_online", "the Online friends list did not appear", owner_listed_in_instance=False, online_seen=False)
                elif not self.open_owner_page():
                    result = OwnerLocateResult("not_online", "owner is not in the Online list", owner_listed_in_instance=False, online_seen=False)
                else:
                    result = self.press_join()
                    result.owner_listed_in_instance = False
        except WorldJoinError as exc:
            result = OwnerLocateResult("error", str(exc))
        except Exception as exc:  # never let menu trouble kill the follower
            result = OwnerLocateResult("error", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                if not (self.dry_run and False):
                    self.close_menu()
            except Exception:
                pass
        result.seconds = time.monotonic() - started
        result.steps = tuple(self.steps)
        self._state(
            ownerLocateOutcome=result.outcome,
            ownerLocateDetail=result.detail,
            ownerLocateSeconds=round(result.seconds, 1),
        )
        return result
