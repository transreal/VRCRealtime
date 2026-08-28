"""Sliden console automation and slide extraction for :mod:`VRCRealtime`.

The VRChat console is used only as a control surface.  Presentation images are
read from the source MP4 at their original resolution so the Realtime model is
never asked to infer slide content from the small in-world preview.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


RGBImage = NDArray[np.uint8]
Point = tuple[float, float]


class SlidenError(RuntimeError):
    """Base class for recoverable presentation failures."""


class SlidenCancelled(SlidenError):
    """Raised when presentation cancellation is requested."""


class SlidenConsoleNotFound(SlidenError):
    """Raised when the Sliden control surface cannot be recognized."""


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Point:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class SlidenScreenObservation:
    """Large in-room Sliden display used as a navigation landmark."""

    rect: Rect | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class SlidenConsoleObservation:
    """Recognized Sliden controls in VRChat-window-local coordinates."""

    slide_quad: NDArray[np.float32] | None = None
    home_button: Point | None = None
    url_field: Point | None = None
    #: Each control's own box, not just its centre.  Aiming has to be judged
    #: against the thing being aimed at: the URL bar is 15 pixels high across
    #: the room and 32 close up, so one fixed pixel figure is either
    #: unreachable at one distance or misses the control at the other -- and
    #: a point alone cannot answer the only question that matters before
    #: clicking, which is whether the reticle is actually on the control.
    url_field_rect: Rect | None = None
    home_button_rect: Rect | None = None
    previous_button: Point | None = None
    previous_button_rect: Rect | None = None
    next_button: Point | None = None
    next_button_rect: Rect | None = None
    modal_input: Point | None = None
    modal_clear: Point | None = None
    modal_submit: Point | None = None
    confidence: float = 0.0

    @property
    def modal_open(self) -> bool:
        return self.modal_input is not None and self.modal_submit is not None

    @property
    def console_found(self) -> bool:
        return self.slide_quad is not None or self.url_field is not None


#: Hard ceiling, in true-relative mouse pixels, on how far the search may let
#: the camera drift from the attitude it framed the wall screen at.  Without it
#: the incremental corrections that keep a low target centred accumulate into a
#: view of the floor.
_MAXIMUM_PITCH_OFFSET = 150.0

#: How far, as a share of the capture width, a console may have travelled
#: since it was last seen and still be taken for the same one.  A correction
#: sweeps it by well under a tenth of the frame; anything further away is a
#: different object.
_CONSOLE_TRACKING_RADIUS = 0.18
#: And how much it may have changed size.  Aiming turns the camera, it does
#: not walk, so the panel's width is near enough constant between frames --
#: a display of a different size in the same place is a different display.
_CONSOLE_TRACKING_SCALE = (0.65, 1.55)

#: How far, as a share of the panel's width, a recognized button may sit from
#: where the layout says it should be and still be that button.
_CONTROL_MATCH_FRACTION = 0.20
#: How far from the button row's centre, as a share of the panel's width, the
#: previous/next pair may sit.  Measured on the live console: Previous at 0.41
#: and Next at 0.73, with the fullscreen toggle at 0.20 and the lock at 0.88 --
#: the outer tray controls are what these keep out of a navigation click.  The
#: two sides differ because the tray is not symmetric.
_NAVIGATION_SPAN_PREVIOUS = 0.26
_NAVIGATION_SPAN_NEXT = 0.30
#: How much of the panel's width the URL bar must span.  The bar runs almost
#: the full width of the console; a short bright strip above the preview is
#: something else -- a caption, a reflection, the top of the frame.
_URL_FIELD_MINIMUM_SPAN = 0.45


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise SlidenCancelled("presentation was cancelled")


def _state(
    callback: Callable[..., Any] | None,
    **updates: Any,
) -> None:
    if callback is not None:
        callback(**updates)


def download_slide_video(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float = 120.0,
    max_bytes: int = 256 * 1024 * 1024,
    cancelled: Callable[[], bool] | None = None,
    state_callback: Callable[..., Any] | None = None,
) -> Path:
    """Download an HTTP(S) Sliden MP4 with a bounded streaming reader."""

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SlidenError("slide URL must be an absolute HTTP or HTTPS URL")
    if timeout_seconds <= 0:
        raise SlidenError("download timeout must be positive")
    if max_bytes <= 0:
        raise SlidenError("maximum download size must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "VRCRealtime-Sliden/1.0"},
    )
    received = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            final_url = urllib.parse.urlparse(response.geturl())
            if final_url.scheme not in {"http", "https"}:
                raise SlidenError("slide URL redirected to an unsupported scheme")
            length_header = response.headers.get("Content-Length")
            if length_header:
                try:
                    announced = int(length_header)
                except ValueError:
                    announced = 0
                if announced > max_bytes:
                    raise SlidenError(
                        f"slide video is larger than the {max_bytes} byte limit"
                    )
            with temporary.open("wb") as output:
                while True:
                    _check_cancel(cancelled)
                    if time.monotonic() >= deadline:
                        raise SlidenError("slide video download timed out")
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > max_bytes:
                        raise SlidenError(
                            f"slide video exceeded the {max_bytes} byte limit"
                        )
                    output.write(block)
                    _state(
                        state_callback,
                        presentationDownloadedBytes=received,
                    )
        if received == 0:
            raise SlidenError("slide video download returned an empty file")
        temporary.replace(destination)
        return destination
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _read_frame(capture: cv2.VideoCapture, index: int) -> RGBImage:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = capture.read()
    if not ok or bgr is None:
        raise SlidenError(f"could not decode video frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _sliden_frame_indices(frame_count: int, fps: float) -> list[int] | None:
    """Return slide starts for NotebookExtensions' 29+30n layout."""

    if abs(fps - 30.0) > 0.05 or frame_count < 59:
        return None
    remainder = frame_count - 29
    if remainder <= 0 or remainder % 30 != 0:
        return None
    slide_count = remainder // 30
    # The final 30-frame segment repeats the last slide.  Frame 29 begins
    # slide two, so only slide_count-1 starts are appended after frame zero.
    return [0] + [29 + 30 * i for i in range(max(0, slide_count - 1))]


def _scene_frame_indices(
    capture: cv2.VideoCapture,
    frame_count: int,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> list[int]:
    """Fallback scene-change extraction for non-standard MP4 timing."""

    indices: list[int] = []
    previous: NDArray[np.uint8] | None = None
    last_selected = -10
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for index in range(frame_count):
        if index % 10 == 0:
            _check_cancel(cancelled)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            break
        gray = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
            (192, 108),
            interpolation=cv2.INTER_AREA,
        )
        if previous is None:
            indices.append(index)
            last_selected = index
        else:
            difference = float(
                np.mean(cv2.absdiff(gray, previous), dtype=np.float64) / 255.0
            )
            if difference >= 0.0025 and index - last_selected >= 3:
                indices.append(index)
                last_selected = index
        previous = gray
    if not indices:
        raise SlidenError("no frames could be extracted from the slide video")
    return indices


def extract_slide_images(
    video_path: Path,
    output_directory: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
    state_callback: Callable[..., Any] | None = None,
) -> list[Path]:
    """Extract source-resolution slide images from a Sliden-compatible MP4."""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise SlidenError(f"could not open slide video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise SlidenError("slide video has invalid frame metadata")
        if width < 320 or height < 180:
            raise SlidenError("slide video resolution is too small")
        indices = _sliden_frame_indices(frame_count, fps)
        layout = "Sliden29+30n"
        if indices is None:
            layout = "SceneDetection"
            indices = _scene_frame_indices(
                capture,
                frame_count,
                cancelled=cancelled,
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        result: list[Path] = []
        # Long-GOP MP4 files make one random seek per slide surprisingly slow.
        # Decode the stream once and retain only the requested frame starts.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        wanted = iter(enumerate(indices, start=1))
        current = next(wanted, None)
        for frame_index in range(frame_count):
            _check_cancel(cancelled)
            if not capture.grab():
                break
            if current is None or frame_index != current[1]:
                continue
            ok, bgr = capture.retrieve()
            if not ok or bgr is None:
                break
            number = current[0]
            path = output_directory / f"{number:03d}.jpg"
            if not cv2.imwrite(
                str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 94]
            ):
                raise SlidenError(f"could not write extracted slide: {path}")
            result.append(path)
            _state(
                state_callback,
                presentationExtractionProgress=number / len(indices),
            )
            current = next(wanted, None)
            if current is None:
                break
        if len(result) != len(indices):
            raise SlidenError(
                f"decoded {len(result)} of {len(indices)} expected slide frames"
            )
        _state(
            state_callback,
            presentationVideoFPS=fps,
            presentationVideoFrames=frame_count,
            presentationSlideWidth=width,
            presentationSlideHeight=height,
            presentationExtractionLayout=layout,
            presentationSlideCount=len(result),
            presentationExtractionProgress=1.0,
        )
        return result
    finally:
        capture.release()


def prepare_slide_images(
    url: str,
    workspace: Path,
    *,
    timeout_seconds: float = 120.0,
    max_bytes: int = 256 * 1024 * 1024,
    cancelled: Callable[[], bool] | None = None,
    state_callback: Callable[..., Any] | None = None,
) -> list[Path]:
    """Download *url* and extract all slide images into *workspace*."""

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    video_path = workspace / f"slides-{digest}.mp4"
    slides_directory = workspace / "slides"
    _state(state_callback, presentationPhase="downloading")
    download_slide_video(
        url,
        video_path,
        timeout_seconds=timeout_seconds,
        max_bytes=max_bytes,
        cancelled=cancelled,
        state_callback=state_callback,
    )
    _check_cancel(cancelled)
    _state(state_callback, presentationPhase="extracting")
    return extract_slide_images(
        video_path,
        slides_directory,
        cancelled=cancelled,
        state_callback=state_callback,
    )


def _contour_rectangles(mask: NDArray[np.uint8]) -> list[tuple[Rect, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[tuple[Rect, Any]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 1 or height <= 1:
            continue
        result.append((Rect(x, y, width, height), contour))
    return result


def _ordered_quad(contour: Any) -> NDArray[np.float32]:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approximation) == 4:
        points = approximation.reshape(4, 2).astype(np.float32)
    else:
        points = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def _quad_point(quad: NDArray[np.float32], u: float, v: float) -> Point:
    top_left, top_right, bottom_right, bottom_left = quad
    top = top_left + u * (top_right - top_left)
    bottom = bottom_left + u * (bottom_right - bottom_left)
    point = top + v * (bottom - top)
    return (float(point[0]), float(point[1]))


@dataclass(frozen=True)
class _ChannelStats:
    """Per-frame channel projections shared by every mask.

    The navigation loop runs these detectors several times a second on a
    full-resolution capture, where per-pixel numpy expressions cost hundreds of
    milliseconds and starve the control loop.  Computing the projections once
    with OpenCV keeps a full observation inside a single poll interval.
    """

    red: NDArray[np.uint8]
    green: NDArray[np.uint8]
    blue: NDArray[np.uint8]
    maximum: NDArray[np.uint8]
    minimum: NDArray[np.uint8]


def _channel_stats(image: RGBImage) -> _ChannelStats:
    red, green, blue = cv2.split(image)
    return _ChannelStats(
        red=red,
        green=green,
        blue=blue,
        maximum=cv2.max(cv2.max(red, green), blue),
        minimum=cv2.min(cv2.min(red, green), blue),
    )


def _at_most(channel: NDArray[np.uint8], value: int) -> NDArray[np.uint8]:
    return cv2.threshold(channel, value, 255, cv2.THRESH_BINARY_INV)[1]


def _at_least(channel: NDArray[np.uint8], value: int) -> NDArray[np.uint8]:
    return cv2.threshold(channel, value - 1, 255, cv2.THRESH_BINARY)[1]


def _cyan_mask(
    image: RGBImage,
    stats: _ChannelStats | None = None,
) -> NDArray[np.uint8]:
    stats = _channel_stats(image) if stats is None else stats
    mask = cv2.bitwise_and(_at_least(stats.green, 105), _at_least(stats.blue, 125))
    mask = cv2.bitwise_and(mask, _at_least(cv2.subtract(stats.green, stats.red), 45))
    mask = cv2.bitwise_and(mask, _at_least(cv2.subtract(stats.blue, stats.red), 55))
    mask = cv2.bitwise_and(
        mask, _at_most(cv2.absdiff(stats.blue, stats.green), 100)
    )
    return mask


def _display_mask(
    image: RGBImage,
    stats: _ChannelStats | None = None,
) -> NDArray[np.uint8]:
    """Pixels that plausibly belong to an idle Sliden display surface.

    The wall screen and the small operation panel render the same neutral
    ``#313131`` idle page, so one mask serves both.  Measured in the live room
    the page sits at 49 from every viewpoint while the warm beige wall next to
    it sits at 84, so a tight absolute ceiling separates them cleanly; raising
    the ceiling to also catch the screen seen through the world's glass doors
    (up to 123) merges it with the wall and destroys the bounding box, which
    matters far more than acquiring the landmark one room early.
    """

    stats = _channel_stats(image) if stats is None else stats
    # The hue term keeps dark *warm* furniture -- this room's wooden floor and
    # brown sofa also fall under the ceiling -- from fusing with the screen and
    # inflating its bounding box.
    mask = cv2.bitwise_and(
        _at_most(stats.maximum, 66),
        _at_most(cv2.subtract(stats.maximum, stats.minimum), 45),
    )
    mask = cv2.bitwise_and(
        mask, cv2.compare(stats.blue, cv2.subtract(stats.red, 10), cv2.CMP_GE)
    )
    # Do not close this mask: a clipped wall screen can otherwise merge with
    # dark furniture through the capture edge and become one frame-sized blob.
    # Opening is the opposite operation and is required here -- without it the
    # screen bleeds into the credenza below it and into dark avatar clothing
    # through a few-pixel bridge, and the merged bounding box then points
    # navigation at the floor instead of the screen.
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )


def _darker_than_surroundings(
    value: NDArray[np.uint8],
    rect: Rect,
    *,
    margin: float = 12.0,
) -> bool:
    """Require *rect* to be genuinely darker than the ring enclosing it.

    This is what keeps the adaptive ceiling honest.  In a dim capture the
    ceiling admits most of the frame, and without a local check the background
    itself becomes a candidate display.
    """

    height, width = value.shape[:2]
    pad_x = max(6, int(0.18 * rect.width))
    pad_y = max(6, int(0.18 * rect.height))
    outer_x = max(0, rect.x - pad_x)
    outer_y = max(0, rect.y - pad_y)
    outer_right = min(width, rect.right + pad_x)
    outer_bottom = min(height, rect.bottom + pad_y)
    outer = value[outer_y:outer_bottom, outer_x:outer_right]
    if outer.size == 0:
        return False
    ring = np.ones(outer.shape, dtype=bool)
    ring[
        rect.y - outer_y : rect.bottom - outer_y,
        rect.x - outer_x : rect.right - outer_x,
    ] = False
    if np.count_nonzero(ring) < 40:
        return False
    inside = value[rect.y : rect.bottom, rect.x : rect.right]
    return float(np.median(outer[ring])) - float(np.median(inside)) >= margin


def _longest_run(profile: NDArray[np.float64], threshold: float) -> tuple[int, int]:
    """Longest contiguous index range whose coverage stays above *threshold*."""

    best_start = best_end = 0
    start: int | None = None
    for index, value in enumerate(profile):
        if value >= threshold:
            if start is None:
                start = index
            if index + 1 - start > best_end - best_start:
                best_start, best_end = start, index + 1
        else:
            start = None
    return best_start, best_end


def _tighten_display_rect(mask: NDArray[np.uint8], rect: Rect) -> Rect:
    """Shrink *rect* onto the solid body of the region it bounds.

    Opening severs most spurious bridges, but a screen standing directly on a
    dark credenza still shares a wide contact edge with it.  Keeping only the
    rows and columns that the region actually fills puts the box back on the
    screen, which is what the approach controller steers by.
    """

    roi = mask[rect.y : rect.bottom, rect.x : rect.right] > 0
    if roi.size == 0:
        return rect
    rows = roi.mean(axis=1)
    columns = roi.mean(axis=0)
    row_start, row_end = _longest_run(rows, 0.45 * float(rows.max()))
    column_start, column_end = _longest_run(columns, 0.45 * float(columns.max()))
    if row_end - row_start < 8 or column_end - column_start < 8:
        return rect
    # Occlusion -- an avatar standing in front of the screen, or the capture
    # edge cutting it -- also thins the profile.  Only trust the trim when it
    # removes an attached neighbour rather than most of the region itself.
    if (
        row_end - row_start < 0.5 * rect.height
        or column_end - column_start < 0.5 * rect.width
    ):
        return rect
    return Rect(
        x=rect.x + column_start,
        y=rect.y + row_start,
        width=column_end - column_start,
        height=row_end - row_start,
    )


def _rect_quad(rect: Rect) -> NDArray[np.float32]:
    return np.asarray(
        [
            [rect.x, rect.y],
            [rect.right, rect.y],
            [rect.right, rect.bottom],
            [rect.x, rect.bottom],
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class _DisplayCandidate:
    rect: Rect
    quad: NDArray[np.float32]
    aspect: float
    area_fraction: float
    bright_ratio: float


def _display_candidates(
    image: RGBImage,
    stats: _ChannelStats | None = None,
    *,
    minimum_width: int = 38,
    minimum_height: int = 18,
    minimum_area_fraction: float = 0.0006,
    maximum_area_fraction: float = 0.72,
) -> list[_DisplayCandidate]:
    """Return idle-page display candidates ordered from largest to smallest.

    The operation panel is the wall screen in miniature, so the same geometry
    test finds both; callers separate them by size and by the surrounding
    console furniture.
    """

    height, width = image.shape[:2]
    frame_area = float(width * height)
    stats = _channel_stats(image) if stats is None else stats
    mask = _display_mask(image, stats)
    value = stats.maximum
    # The white wordmark is the one feature every Sliden idle page carries.
    bright_mark = stats.minimum >= 150
    candidates: list[_DisplayCandidate] = []
    for raw_rect, contour in _contour_rectangles(mask):
        rect = _tighten_display_rect(mask, raw_rect)
        quad = (
            _ordered_quad(contour)
            if rect == raw_rect
            else _rect_quad(rect)
        )
        aspect = rect.width / rect.height
        area = rect.width * rect.height
        touches_side = rect.x <= 3 or rect.right >= width - 3
        # A screen seen from an angle, clipped by the capture edge or partly
        # hidden behind a door frame is no longer 16:9.  Large candidates are
        # allowed a much wider aspect band because the brightness, hue,
        # wordmark and local-contrast tests already carry the decision there.
        forgiving = touches_side or area >= 0.05 * frame_area
        if not (
            1.15 <= aspect <= 2.70 or (forgiving and 0.62 <= aspect <= 2.95)
        ):
            continue
        if not (
            minimum_area_fraction * frame_area
            <= area
            <= maximum_area_fraction * frame_area
        ):
            continue
        if rect.width < minimum_width or rect.height < minimum_height:
            continue
        if not (0.04 * height <= rect.center[1] <= 0.90 * height):
            continue
        roi = bright_mark[rect.y : rect.bottom, rect.x : rect.right]
        bright_ratio = float(np.count_nonzero(roi)) / max(1.0, float(area))
        if not 0.0015 <= bright_ratio <= 0.35:
            continue
        # A display fills its bounding box; a silhouette does not.  Without
        # this test an avatar in dark clothing standing near any bright fixture
        # reads as a panel, and the approach chases a person around the room.
        # Measured: unoccluded screen 0.84, screen with an avatar in front of
        # it 0.51, avatar alone 0.34.
        fill = float(cv2.contourArea(contour)) / max(
            1.0, float(raw_rect.width * raw_rect.height)
        )
        if fill < 0.45:
            continue
        if not _darker_than_surroundings(value, rect):
            continue
        candidates.append(
            _DisplayCandidate(
                rect=rect,
                quad=quad,
                aspect=aspect,
                area_fraction=area / frame_area,
                bright_ratio=bright_ratio,
            )
        )
    candidates.sort(key=lambda item: -item.area_fraction)
    return candidates


#: Page width, in pixels, at which the stored wordmark is life size.
_WORDMARK_PAGE_WIDTH = 345
_WORDMARK_PATH = Path(__file__).with_name("VRCRealtime_sliden_wordmark.png")
_WORDMARK_CACHE: list[NDArray[np.uint8] | None] = []


def _wordmark_template() -> NDArray[np.uint8] | None:
    if not _WORDMARK_CACHE:
        _WORDMARK_CACHE.append(
            cv2.imread(str(_WORDMARK_PATH), cv2.IMREAD_GRAYSCALE)
        )
    return _WORDMARK_CACHE[0]


def sliden_wordmark_score(page: NDArray[np.uint8]) -> float:
    """How well *page* carries the Sliden wordmark, from -1 to 1.

    Every generic test for "a dark rectangle with bright marks on it" also
    matches this room's other furniture -- an audio filter panel, a notice
    board, an avatar in dark clothing -- and each one in turn captured the
    search.  Matching the wordmark itself is the positive identification the
    room actually offers.  Measured: 0.97-0.98 for the wall screen head-on,
    0.73 for a sub-screen, against 0.43 for the audio panel, 0.38 for a notice
    board and 0.15 for the screen seen edge-on.
    """

    template = _wordmark_template()
    if template is None or page.size == 0:
        return 0.0
    gray = page if page.ndim == 2 else cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    best = -1.0
    for scale in (0.75, 0.9, 1.0, 1.15, 1.35):
        width = int(_WORDMARK_PAGE_WIDTH * scale)
        height = max(8, int(width * gray.shape[0] / max(1, gray.shape[1])))
        if width < template.shape[1] + 2 or height < template.shape[0] + 2:
            continue
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        best = max(
            best,
            float(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED).max()),
        )
    return max(0.0, best)


def detect_sliden_page_fill(
    image: RGBImage,
    stats: _ChannelStats | None = None,
) -> float:
    """Fraction of the capture covered by a Sliden page carrying its wordmark.

    Standing close to the wall screen its rectangle no longer fits in the
    frame, and every geometric test -- aspect, area cap, local contrast against
    a surround that is no longer there -- rejects it, so the search behaves as
    though the screen had vanished and sweeps on the spot.  The page itself is
    still unmistakable: a large neutral dark field with the white wordmark and
    QR code on it.  That is enough to know the avatar must step back before the
    flanking bars can be read.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 RGB numpy array")
    stats = _channel_stats(image) if stats is None else stats
    mask = _display_mask(image, stats)
    components, labels, component_stats, _ = cv2.connectedComponentsWithStats(mask)
    if components <= 1:
        return 0.0
    largest = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
    area = float(component_stats[largest, cv2.CC_STAT_AREA])
    frame_area = float(image.shape[0] * image.shape[1])
    if area < 0.12 * frame_area:
        return 0.0
    x = int(component_stats[largest, cv2.CC_STAT_LEFT])
    y = int(component_stats[largest, cv2.CC_STAT_TOP])
    w = int(component_stats[largest, cv2.CC_STAT_WIDTH])
    h = int(component_stats[largest, cv2.CC_STAT_HEIGHT])
    del labels
    # The wordmark and QR are holes punched in the dark region, not part of
    # it, so they have to be counted over the bounding box.  Counting them
    # inside the component itself finds nothing, by construction.
    marks = stats.minimum[y : y + h, x : x + w] >= 150
    mark_ratio = float(np.count_nonzero(marks)) / max(1.0, float(w * h))
    if not 0.0008 <= mark_ratio <= 0.35:
        return 0.0
    return area / frame_area


def detect_sliden_screen(image: RGBImage) -> SlidenScreenObservation:
    """Detect the large Sliden wall screen, including edge clipping.

    Sized for a navigation landmark rather than a click target: the screen must
    be recognizable from across the room, long before it fills the view.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 RGB numpy array")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    stats = _channel_stats(image)
    candidates = _display_candidates(
        image,
        stats,
        minimum_width=96,
        minimum_height=52,
        minimum_area_fraction=0.004,
        maximum_area_fraction=0.70,
    )
    # The room's three Sliden surfaces render the same page and differ only in
    # what is attached to them: the operation panel has a URL field above, the
    # sub-screens a control tray directly below, and the wall screen nothing at
    # all.  Without this the avatar framed a sub-screen from close up, called
    # it the wall screen, and read the flanking bars off the wrong object.
    furniture_bright = cv2.morphologyEx(
        cv2.bitwise_and(
            _at_least(stats.minimum, 190),
            _at_most(cv2.subtract(stats.maximum, stats.minimum), 65),
        ),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    furniture_rectangles = _contour_rectangles(furniture_bright)
    marks = stats.minimum >= 150
    scored: list[tuple[float, Rect]] = []
    for candidate in candidates:
        thin, _square, tray = _console_furniture(candidate.rect, furniture_rectangles)
        if thin or tray:
            continue
        # The wordmark and QR sit in the middle of the page.  Seen head-on they
        # land near the centre of the detected rectangle; seen edge-on from
        # beside the screen they squash against one edge, and that grazing view
        # was being accepted as "the screen framed", producing flanking-bar
        # geometry that meant nothing.  Measured: 0.40-0.65 head-on, 0.05 for
        # the grazing view that stalled a live run.
        rect = candidate.rect
        rows, columns = np.nonzero(marks[rect.y : rect.bottom, rect.x : rect.right])
        if rows.size == 0:
            continue
        if not 0.18 <= float(columns.mean()) / rect.width <= 0.82:
            continue
        if not 0.12 <= float(rows.mean()) / rect.height <= 0.90:
            continue
        if (
            sliden_wordmark_score(
                image[rect.y : rect.bottom, rect.x : rect.right]
            )
            < 0.48
        ):
            continue
        area_score = min(1.0, candidate.area_fraction / 0.18)
        text_score = min(1.0, candidate.bright_ratio / 0.035)
        aspect_score = max(0.0, 1.0 - abs(candidate.aspect - 1.78) / 1.2)
        score = 0.45 * area_score + 0.35 * text_score + 0.20 * aspect_score
        scored.append((score, candidate.rect))
    if not scored:
        return SlidenScreenObservation()
    score, rect = max(scored, key=lambda item: item[0])
    return SlidenScreenObservation(rect=rect, confidence=min(1.0, score))


def page_has_console_bars(page: Rect, bars: Sequence[Rect]) -> bool:
    """Whether a wide bright bar sits just under or just above *page*.

    The progress bar under the sub-screens and the operation panel, and the
    URL field above the panel, are bright bars at least as wide as half the
    page; the wall screen has nothing of the kind on either side.
    """

    for bar in bars:
        if bar.width < 0.4 * page.width:
            continue
        overlap = min(bar.right, page.right) - max(bar.x, page.x)
        if overlap < 0.3 * page.width:
            continue
        cy = bar.center[1]
        if page.bottom - 0.25 * page.height <= cy <= page.bottom + 1.0 * page.height:
            return True
        if page.y - 0.8 * page.height <= cy <= page.y + 0.25 * page.height:
            return True
    return False


def wordmark_width_scale(key: str) -> float | None:
    """Read a folded OCR key as the Sliden wordmark, typos included.

    The wordmark is read off a surface that is usually oblique and often
    clipped -- by the window edge, by a door frame -- so its six letters come
    back short or with the logo glyph counted as one more.  Live captures on
    2026-08-21 gave "Slider", "Slide", "liden" and "V Slide" for the same wall
    screen within seconds of each other; demanding the exact spelling made the
    estimate below fall through to whatever other Sliden surface was in frame
    (a sub-screen across the room reads the wordmark just the same), and the
    approach chased the two in turn without ever moving.  Returns the factor
    that brings a box of this many glyphs back to the width of the full six,
    so that a truncated read does not shrink the page estimate with it, or
    ``None`` when the text is not the wordmark at all.
    """

    if not key or "app" in key or "url" in key:
        # "sliden.app" under the QR code and "Enter Slide URL" over the
        # operation panel are on the page, but neither is the wordmark.
        return None
    forms = {key}
    if len(key) >= 6:
        # A stray leading letter is the logo read as a glyph.
        forms.add(key[1:])
    if not any(
        len(form) >= 5 and SequenceMatcher(None, form, "sliden").ratio() >= 0.8
        for form in forms
    ):
        return None
    return 6.0 / float(max(4, min(7, len(key))))


def screen_from_text_boxes(
    boxes: Sequence[tuple[str, Rect]],
    width: int,
    height: int,
    *,
    exclude: Sequence[Rect] = (),
    bars: Sequence[Rect] = (),
) -> Rect | None:
    """Estimate the wall screen from OCR'd text, for when the detector cannot.

    ``detect_sliden_screen`` matches the wordmark as a template, which needs
    the page roughly head-on and at a moderate distance; from beside the
    screen, or close enough that it is clipped by the capture, the match
    scores 0.3-0.4 and the landmark is simply gone.  OCR reads "Sliden" from
    almost any angle.  Every Sliden surface carries the wordmark, so the
    sub-screens are told apart by the "Local" caption under their page and
    the operation panel by the "Enter Slide URL" field above its page; what
    remains, largest first, is the wall screen.  ``exclude`` lists page
    rectangles already recognised as console or sub-screen by the detectors,
    for when the placeholder text in the URL field is too faint to read, and
    ``bars`` the bright horizontal bars in the frame: a wide one just under a
    page is the progress bar every sub-screen and the panel carry, and one
    just above it is the panel's URL field -- the wall screen has neither.
    The returned rectangle is the wordmark inflated to about the page it
    sits on.
    """

    def folded(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    marks: list[tuple[Rect, float]] = []
    captions: list[Rect] = []
    fields: list[Rect] = []
    for text, rect in boxes:
        key = folded(text)
        if not key:
            continue
        scale = wordmark_width_scale(key)
        if scale is not None:
            marks.append((rect, scale))
        elif key == "local":
            captions.append(rect)
        elif "url" in key or key.startswith("enterslide"):
            fields.append(rect)
    best: Rect | None = None
    for mark, scale in marks:
        cx, cy = mark.center
        if any(
            other.x - 0.1 * other.width <= cx <= other.right + 0.1 * other.width
            and other.y - 0.1 * other.height <= cy <= other.bottom + 0.1 * other.height
            for other in exclude
        ):
            continue
        # Centred on the wordmark, and widened as if every letter had been
        # read: the page is what the approach measures its distance by, so a
        # box that lost a letter must not shrink it.
        span = mark.width * scale
        page = Rect(
            int(cx - 1.2 * span),
            int(mark.y - 1.0 * mark.height),
            int(2.4 * span),
            int(3.5 * mark.height),
        )
        near_x = lambda other: page.x - 0.5 * page.width <= other.center[0] <= page.right + 0.5 * page.width  # noqa: E731
        if any(near_x(c) and page.bottom - 0.5 * page.height <= c.center[1] <= page.bottom + 1.2 * page.height for c in captions):
            continue
        if any(near_x(f) and page.y - 1.0 * page.height <= f.center[1] <= page.y + 0.4 * page.height for f in fields):
            continue
        if page_has_console_bars(page, bars):
            continue
        clipped = Rect(
            max(0, page.x),
            max(0, page.y),
            min(width, page.right) - max(0, page.x),
            min(height, page.bottom) - max(0, page.y),
        )
        if clipped.width <= 0 or clipped.height <= 0:
            continue
        if best is None or clipped.width * clipped.height > best.width * best.height:
            best = clipped
    return best


def console_page_rects(image: RGBImage) -> list[Rect]:
    """Pages in view that carry console or sub-screen furniture.

    Used to keep the OCR wall-screen estimate off the operation panel and the
    sub-screens, whose pages read "Sliden" just the same.
    """

    rects: list[Rect] = []
    observation = detect_sliden_console(image)
    if observation.slide_quad is not None:
        quad = observation.slide_quad
        rects.append(
            Rect(
                int(quad[:, 0].min()),
                int(quad[:, 1].min()),
                int(quad[:, 0].max() - quad[:, 0].min()),
                int(quad[:, 1].max() - quad[:, 1].min()),
            )
        )
    page = detect_sliden_panel_page(image, minimum_score=0.34)
    if page is not None:
        rects.append(page[0])
    return rects


def _detect_modal(
    image: RGBImage,
    stats: _ChannelStats | None = None,
) -> tuple[Point | None, Point | None, Point | None, float]:
    height, width = image.shape[:2]
    frame_area = float(width * height)
    stats = _channel_stats(image) if stats is None else stats
    # The input well is nearly black while the enclosing modal is dark blue.
    # Keeping this threshold deliberately low prevents both regions from
    # merging into one contour.
    dark = cv2.bitwise_and(
        _at_most(stats.maximum, 30),
        _at_most(cv2.subtract(stats.maximum, stats.minimum), 25),
    )
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
    )
    cyan = _cyan_mask(image, stats)
    candidates: list[tuple[float, Rect]] = []
    for rect, _contour in _contour_rectangles(dark):
        aspect = rect.width / rect.height
        area = rect.width * rect.height
        if not (4.0 <= aspect <= 12.0):
            continue
        if not (0.008 * frame_area <= area <= 0.10 * frame_area):
            continue
        if rect.center[1] < 0.2 * height or rect.center[1] > 0.85 * height:
            continue
        right = min(width, rect.right + max(40, int(0.18 * rect.width)))
        top = max(0, rect.y - int(0.2 * rect.height))
        bottom = min(height, rect.bottom + int(0.2 * rect.height))
        submit_roi = cyan[top:bottom, rect.right:right]
        cyan_pixels = int(np.count_nonzero(submit_roi))
        if cyan_pixels < max(20, int(0.002 * area)):
            continue
        score = area / frame_area + min(0.1, cyan_pixels / max(1.0, area))
        candidates.append((score, rect))
    if not candidates:
        return None, None, None, 0.0
    score, input_rect = max(candidates, key=lambda item: item[0])

    # A non-empty Sliden input well shows a cyan circular X button in its
    # rightmost portion. It is absent for an empty field, so keep it optional.
    clear_left = input_rect.x + int(0.70 * input_rect.width)
    clear_roi = cyan[input_rect.y : input_rect.bottom, clear_left : input_rect.right]
    clear_components, _clear_labels, clear_stats, clear_centroids = (
        cv2.connectedComponentsWithStats(clear_roi)
    )
    clear: Point | None = None
    clear_candidates: list[int] = []
    minimum_clear_area = max(20, int(0.002 * input_rect.width * input_rect.height))
    for component in range(1, clear_components):
        if int(clear_stats[component, cv2.CC_STAT_AREA]) >= minimum_clear_area:
            clear_candidates.append(component)
    if clear_candidates:
        largest_clear = max(
            clear_candidates,
            key=lambda component: int(clear_stats[component, cv2.CC_STAT_AREA]),
        )
        clear = (
            clear_left + float(clear_centroids[largest_clear, 0]),
            input_rect.y + float(clear_centroids[largest_clear, 1]),
        )

    search_left = input_rect.right
    search_right = min(width, input_rect.right + max(50, int(0.22 * input_rect.width)))
    search_top = max(0, input_rect.y - int(0.25 * input_rect.height))
    search_bottom = min(height, input_rect.bottom + int(0.25 * input_rect.height))
    roi = cyan[search_top:search_bottom, search_left:search_right]
    components, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
    submit: Point | None = None
    if components > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if int(stats[largest, cv2.CC_STAT_AREA]) >= 15:
            submit = (
                search_left + float(centroids[largest, 0]),
                search_top + float(centroids[largest, 1]),
            )
    if submit is None:
        submit = (
            input_rect.right + 0.09 * input_rect.width,
            input_rect.center[1],
        )
    return input_rect.center, clear, submit, min(1.0, 0.6 + score)


def _edge_height(left: Point, right: Point, x: float) -> float:
    """Where an edge running ``left`` to ``right`` sits at abscissa *x*."""

    if abs(right[0] - left[0]) < 1e-6:
        return (left[1] + right[1]) / 2.0
    ratio = (x - left[0]) / (right[0] - left[0])
    ratio = max(-0.25, min(1.25, ratio))
    return left[1] + ratio * (right[1] - left[1])


def _console_furniture(
    rect: Rect,
    rectangles: Sequence[tuple[Rect, Any]],
    quad: NDArray[np.float32] | None = None,
) -> tuple[list[Rect], list[Rect], list[Rect]]:
    """Find the URL bar, navigation buttons and tray around a preview *rect*.

    Returns ``(thin, square, tray)``: the bright bars above the preview, the
    small square buttons below it, and the wide button/progress tray that runs
    along its lower edge.  The tray is what separates the operation panel from
    the wall screen, which shows the identical idle page with nothing under it.
    """

    thin: list[Rect] = []
    square: list[Rect] = []
    tray: list[Rect] = []
    # A console is almost never seen square-on, and its rows tilt with it: on
    # a live capture the far end of the button row sat 92 pixels higher than
    # the near end.  Measuring "below the preview" from the bounding box then
    # drops every control on the far side -- the Next button among them, with
    # the panel in plain view -- so the bands follow the panel's own edges.
    top_edge = None if quad is None else (quad[0], quad[1])
    bottom_edge = None if quad is None else (quad[3], quad[2])
    for other, _ in rectangles:
        if other == rect:
            continue
        other_aspect = other.width / other.height
        center_x, center_y = other.center
        horizontally_near = (
            rect.x - 0.15 * rect.width <= center_x <= rect.right + 0.15 * rect.width
        )
        top = (
            float(rect.y)
            if top_edge is None
            else _edge_height(top_edge[0], top_edge[1], center_x)
        )
        bottom = (
            float(rect.bottom)
            if bottom_edge is None
            else _edge_height(bottom_edge[0], bottom_edge[1], center_x)
        )
        if (
            2.5 <= other_aspect <= 18.0
            and 0.30 * rect.width <= other.width <= 1.15 * rect.width
            and horizontally_near
            and top - 0.50 * rect.height <= center_y <= top + 0.06 * rect.height
        ):
            thin.append(other)
        if (
            0.55 <= other_aspect <= 1.65
            and 0.04 * rect.width <= other.width <= 0.20 * rect.width
            and horizontally_near
            # Strictly below the preview: the QR code on the idle page is a
            # bright square of exactly the right size and would otherwise be
            # counted as a navigation button.
            and bottom - 0.06 * rect.height
            <= center_y
            <= bottom + 0.40 * rect.height
        ):
            square.append(other)
        if (
            other_aspect >= 4.0
            and 0.45 * rect.width <= other.width <= 1.30 * rect.width
            and abs(center_x - rect.center[0]) <= 0.35 * rect.width
            and bottom - 0.04 * rect.height
            <= center_y
            <= bottom + 0.55 * rect.height
        ):
            tray.append(other)
    return thin, square, tray


def _bright_surfaces(stats: _ChannelStats) -> NDArray[np.uint8]:
    """Mask of near-white, unsaturated surfaces, separated from each other.

    Closing first fills the text and icons inside a page so it comes back as
    one region.  Opening afterwards is what keeps that region *its own*: a
    lit page in a sunlit room touches the window behind it along a few
    pixels, and a bridge that thin turns the panel and the atrium into a
    single contour covering 99% of the frame.  The console then vanishes
    from a capture it is plainly visible in.  Both surfaces are hundreds of
    pixels across, so eroding and dilating by two costs nothing and parts
    them.
    """

    mask = cv2.bitwise_and(
        _at_least(stats.minimum, 190),
        _at_most(cv2.subtract(stats.maximum, stats.minimum), 65),
    )
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)


def _quad_below(
    quad: NDArray[np.float32],
    fraction: float,
) -> NDArray[np.float32]:
    """Drop a quad's top edge to *fraction* of the way down towards its base.

    Sliding the corners along the sides rather than cropping a bounding box
    keeps the perspective of a panel seen from an angle.
    """

    top_left, top_right, bottom_right, bottom_left = quad
    return np.asarray(
        [
            top_left + fraction * (bottom_left - top_left),
            top_right + fraction * (bottom_right - top_right),
            bottom_right,
            bottom_left,
        ],
        dtype=np.float32,
    )


def _tighten_url_bar(mask: NDArray[np.uint8], rect: Rect) -> Rect:
    """Shrink a URL-bar box onto the bar itself.

    The bar is often joined by a hairline to whatever bright scenery sits
    behind the panel, and the contour's bounding box then reaches far above
    it: measured live, a 32-pixel bar came back as an 82-pixel box whose
    centre was 21 pixels above the control.  Aiming at that centre puts the
    reticle over the wall behind the panel, so the click lands on nothing --
    the dialog never opens and the miss is invisible, because the console is
    still recognized.  The bar is the lowest well-covered band in the box:
    the preview starts immediately below it.
    """

    region = mask[rect.y : rect.bottom, rect.x : rect.right]
    if region.size == 0:
        return rect
    rows = (region > 0).mean(axis=1)
    end = next(
        (row + 1 for row in range(rows.shape[0] - 1, -1, -1) if rows[row] >= 0.55),
        None,
    )
    if end is None:
        return rect
    start = end
    while start > 0 and rows[start - 1] >= 0.55:
        start -= 1
    columns = (region[start:end] > 0).mean(axis=0)
    left, right = _longest_run(columns.astype(np.float64), 0.5)
    # Trimming, not truncating.  A bar seen at an angle has no single row
    # spanning its whole width, so this measurement collapses to a fraction
    # of it -- and a bar reported as half its length fails every later test
    # of whether it is the URL field at all.  Losing that much means the row
    # profile is describing a tilt, not a bounding box inflated by scenery.
    if right - left < 0.70 * rect.width:
        return rect
    return Rect(rect.x + left, rect.y + start, right - left, end - start)


def _split_url_bar(
    mask: NDArray[np.uint8],
    rect: Rect,
) -> tuple[Rect, float] | None:
    """Recover a URL bar that merged into the preview blob.

    The bar and the loaded page are both white and separated by a hairline,
    so whether they survive as two contours comes down to a couple of
    antialiased pixels.  Live captures flip between the two forms on any
    camera movement, and in the merged form the panel lost the one feature
    that identifies it as drivable -- the console vanished mid-aim and the
    servo was left correcting towards something it could no longer see.

    Returns the recovered bar and the fraction of *rect*'s height at which
    the page proper begins, or ``None`` when the blob has no such structure
    (an ordinary uniform page, which is what a wall screen shows).
    """

    region = mask[rect.y : rect.bottom, rect.x : rect.right]
    if region.size == 0:
        return None
    coverage = (region > 0).mean(axis=1)
    height = int(coverage.shape[0])
    high, low = 0.55, 0.30
    top = next((row for row in range(height) if coverage[row] >= high), None)
    if top is None:
        return None
    bar_bottom = top
    while bar_bottom < height and coverage[bar_bottom] >= low:
        bar_bottom += 1
    bar_height = bar_bottom - top
    if not 5 <= bar_height <= max(6, int(0.30 * height)):
        return None
    page_top = bar_bottom
    while page_top < height and coverage[page_top] < low:
        page_top += 1
    gap = page_top - bar_bottom
    if not 1 <= gap <= max(2, int(0.10 * height)):
        return None
    body = coverage[page_top:]
    if body.size < 0.45 * height or float(body.mean()) < 0.55:
        return None
    columns = (region[top:bar_bottom] > 0).mean(axis=0)
    start, end = _longest_run(columns.astype(np.float64), 0.5)
    if end - start < 0.30 * rect.width:
        return None
    return (
        Rect(rect.x + start, rect.y + top, end - start, bar_height),
        page_top / float(height),
    )


def detect_sliden_console(
    image: RGBImage,
    *,
    anchor: Rect | None = None,
) -> SlidenConsoleObservation:
    """Recognize the Sliden UI without relying on absolute screen coordinates.

    Two preview appearances have to be accepted.  Once slides are loaded the
    preview is a bright page, but before that -- exactly when navigation and
    URL entry happen -- the panel shows the same dark idle page as the wall
    screen.  Detecting only the bright form made the console invisible for the
    whole approach phase.

    ``anchor`` is where a console was last positively identified.  Picking one
    out of a furnished room needs the URL bar, because the wall screen and the
    sub-screens show the same page and only the panel can be driven -- but
    *following* one already found does not, and insisting on it is what broke
    the Next button.  Aiming at a control low on the panel lifts the bar to
    the top of the capture, where it merges into whatever bright scenery it
    reaches, and the console the bot was mid-click on vanished from a frame it
    fills a fifth of.  Near a known console, the button tray identifies it.
    """

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 RGB numpy array")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    frame_area = float(width * height)
    stats = _channel_stats(image)
    modal_input, modal_clear, modal_submit, modal_confidence = _detect_modal(
        image, stats
    )

    bright = _bright_surfaces(stats)
    rectangles = _contour_rectangles(bright)
    cyan = _cyan_mask(image, stats)

    previews: list[tuple[Rect, NDArray[np.float32]]] = []
    for rect, contour in rectangles:
        aspect = rect.width / rect.height
        area = rect.width * rect.height
        if not (1.20 <= aspect <= 2.25):
            continue
        # The upper bound is only here to throw out "the whole capture is
        # bright".  It used to sit at 0.22, which is precisely the size a
        # panel reaches at the distance the bot drives it from: a live run
        # measured 0.219 and 0.221 on consecutive frames, so a single pixel
        # of growth deleted the console mid-aim.  A panel worth clicking is
        # allowed to be large.
        if not (0.012 * frame_area <= area <= 0.55 * frame_area):
            continue
        if rect.width < 140 or rect.height < 80:
            continue
        if not (0.12 * height <= rect.center[1] <= 0.90 * height):
            continue
        previews.append((rect, _ordered_quad(contour)))
    for candidate in _display_candidates(image, stats):
        previews.append((candidate.rect, candidate.quad))

    slide_candidates: list[
        tuple[float, Rect, NDArray[np.float32], list[Rect], list[Rect]]
    ] = []
    for rect, preview_quad in previews:
        thin, square, tray = _console_furniture(rect, rectangles, preview_quad)
        if not thin:
            # A loaded page can swallow the URL bar into its own contour.
            # Look inside the blob for the bar before giving the candidate up,
            # and hand back a preview that starts below it.
            split = _split_url_bar(bright, rect)
            if split is not None:
                bar, page_top = split
                thin = [bar]
                rect = Rect(
                    rect.x,
                    rect.y + int(page_top * rect.height),
                    rect.width,
                    rect.height - int(page_top * rect.height),
                )
                preview_quad = _quad_below(preview_quad, page_top)
                _thin, square, tray = _console_furniture(
                    rect, rectangles, preview_quad
                )
        area = rect.width * rect.height
        # Two independent features are required, because either alone matches
        # something else in a furnished world.  The URL field above the preview
        # is what the sub-screens and the wall screen lack -- they render the
        # same idle page but cannot be driven.  The button tray or navigation
        # buttons below it are what a captioned notice board lacks, and a board
        # with a title strip over it otherwise reads as a loaded console.
        # Sitting where a console was just identified stands in for the first
        # of those: nothing else in the room is there.
        tracked = (
            anchor is not None
            and math.hypot(
                rect.center[0] - anchor.center[0],
                rect.center[1] - anchor.center[1],
            )
            <= _CONSOLE_TRACKING_RADIUS * width
            and _CONSOLE_TRACKING_SCALE[0] * anchor.width
            <= rect.width
            <= _CONSOLE_TRACKING_SCALE[1] * anchor.width
        )
        if not (thin or tracked) or not (tray or len(square) >= 2):
            continue
        below_top = min(height, max(0, int(rect.bottom - 0.05 * rect.height)))
        below_bottom = min(height, int(rect.bottom + 0.45 * rect.height))
        below_left = max(0, int(rect.x - 0.1 * rect.width))
        below_right = min(width, int(rect.right + 0.1 * rect.width))
        cyan_ratio = float(
            np.count_nonzero(cyan[below_top:below_bottom, below_left:below_right])
        ) / max(1.0, float(area))
        score = (
            2.0 * min(1.0, area / (0.07 * frame_area))
            + 1.4 * min(1, len(thin))
            + 1.2 * min(1, len(tray))
            + 0.45 * min(4, len(square))
            + min(1.0, cyan_ratio * 60.0)
        )
        slide_candidates.append((score, rect, preview_quad, thin, square))

    if not slide_candidates:
        return SlidenConsoleObservation(
            modal_input=modal_input,
            modal_clear=modal_clear,
            modal_submit=modal_submit,
            confidence=modal_confidence,
        )

    score, _slide_rect, quad, thin, square = max(
        slide_candidates, key=lambda item: item[0]
    )
    url_field_rect = (
        _tighten_url_bar(bright, max(thin, key=lambda rect: rect.width))
        if thin
        else None
    )
    if url_field_rect is not None and not (
        url_field_rect.width >= 5 * url_field_rect.height
        and url_field_rect.width
        >= _URL_FIELD_MINIMUM_SPAN
        * float(
            np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
        )
        / 2.0
    ):
        # A strip that is not the shape of the URL bar is not the URL bar,
        # whatever else it may be.  Clicking one because it was the widest
        # bright thing above the preview lands on the panel's frame.
        url_field_rect = None
    url_field = None if url_field_rect is None else url_field_rect.center

    panel_width = max(
        1.0,
        float(np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
        / 2.0,
    )

    def along_bottom(point: Point) -> float:
        """Where *point* sits across the panel, 0 at its left edge and 1 at its
        right, measured along the tilted bottom edge rather than in screen x."""

        base = quad[3]
        direction = quad[2] - base
        length = float(direction[0] ** 2 + direction[1] ** 2)
        if length <= 0.0:
            return 0.5
        return float(
            ((point[0] - base[0]) * direction[0] + (point[1] - base[1]) * direction[1])
            / length
        )

    def button(expected_u: float) -> Rect | None:
        """The navigation button at *expected_u*, or nothing.

        Where the row cannot be made out, this used to answer with the place
        a button would be if there were one.  A guessed point is indis-
        tinguishable from a recognized one downstream, so the aim drove
        confidently onto blank panel and the click went nowhere -- with no
        sign of it anywhere, because everything reported success.  Say so
        instead: a caller that waits for the real thing loses a few frames,
        which is the cheaper mistake by far.
        """

        expected = _quad_point(quad, expected_u, 1.03)
        if not square:
            return None
        nearest = min(
            square,
            key=lambda rect: (rect.center[0] - expected[0]) ** 2
            + (rect.center[1] - expected[1]) ** 2,
        )
        offset = math.hypot(
            nearest.center[0] - expected[0], nearest.center[1] - expected[1]
        )
        if offset > _CONTROL_MATCH_FRACTION * panel_width:
            return None
        # Nearest-to-a-model-point cannot tell neighbouring controls apart when
        # they sit closer together than the tolerance: measured live, Next and
        # the lock button beside it were 62 px apart while the tolerance was 80,
        # so a small error in the quad picked the lock -- and the presentation
        # locked the panel instead of advancing (2026-08-15).  The navigation
        # pair is the pair that straddles the row's centre; the tray's outer
        # controls do not, so refuse anything that far out.
        position = along_bottom(nearest.center)
        if expected_u >= 0.5:
            if not 0.5 < position <= 0.5 + _NAVIGATION_SPAN_NEXT:
                return None
        elif not 0.5 - _NAVIGATION_SPAN_PREVIOUS <= position < 0.5:
            return None
        return nearest

    def home() -> Rect | None:
        """The home button: the small square just left of the URL bar."""

        if url_field_rect is None:
            return None
        candidates = [
            rect
            for rect, _contour in rectangles
            # Left of the bar, on the same row, and about as tall as it: the
            # home button is the only thing in the console that fits.
            if rect.right <= url_field_rect.x + 0.05 * url_field_rect.width
            and rect.right >= url_field_rect.x - 0.25 * url_field_rect.width
            and abs(rect.center[1] - url_field_rect.center[1])
            <= 1.2 * url_field_rect.height
            and 0.4 * url_field_rect.height <= rect.height <= 3.0 * url_field_rect.height
            and rect.width <= 0.30 * url_field_rect.width
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda rect: rect.right)

    next_rect = button(0.72)
    previous_rect = button(0.34)
    home_rect = home()

    confidence = min(1.0, 0.25 + 0.12 * score)
    return SlidenConsoleObservation(
        slide_quad=quad,
        home_button=None if home_rect is None else home_rect.center,
        home_button_rect=home_rect,
        url_field=url_field,
        url_field_rect=url_field_rect,
        previous_button=None if previous_rect is None else previous_rect.center,
        previous_button_rect=previous_rect,
        next_button=None if next_rect is None else next_rect.center,
        next_button_rect=next_rect,
        modal_input=modal_input,
        modal_clear=modal_clear,
        modal_submit=modal_submit,
        confidence=max(confidence, modal_confidence),
    )


def detect_sliden_panel_page(
    image: RGBImage,
    *,
    minimum_score: float = 0.48,
) -> tuple[Rect, NDArray[np.float32]] | None:
    """The operation panel's page, even when only part of it is in view.

    ``detect_sliden_console`` demands the URL field and a button row because it
    has to hand back click targets.  Deciding *"the panel is in front of me"*
    needs far less: a Sliden page carrying its wordmark with any console
    furniture attached.  Without this the avatar completed the manoeuvre, ended
    up facing the panel, and kept turning because one feature had not resolved.
    """

    stats = _channel_stats(image)
    rectangles = _contour_rectangles(_bright_surfaces(stats))
    best: tuple[float, Rect, NDArray[np.float32]] | None = None
    for candidate in _display_candidates(image, stats):
        thin, square, tray = _console_furniture(
            candidate.rect, rectangles, candidate.quad
        )
        if not (thin or tray or len(square) >= 2):
            continue
        rect = candidate.rect
        if (
            sliden_wordmark_score(image[rect.y : rect.bottom, rect.x : rect.right])
            < minimum_score
        ):
            continue
        if best is None or candidate.area_fraction > best[0]:
            best = (candidate.area_fraction, rect, candidate.quad)
    return None if best is None else (best[1], best[2])


@dataclass(frozen=True)
class SlidenEdgePanelObservation:
    """A flanking Sliden panel seen edge-on, as a bright horizontal bar."""

    rect: Rect
    side: int
    gap: float
    openness: float
    length: float


def _bright_horizontal_bars(
    image: RGBImage,
    stats: _ChannelStats | None = None,
) -> list[tuple[Rect, float, float]]:
    """Bright elongated bars as ``(box, length, thickness)``.

    Both figures come from the rotated rectangle.  Length identifies a bar
    across viewpoints -- a diagonal bar's bounding box swings wildly as the
    avatar orbits, and matching on it let tracking jump onto an unrelated
    bright line 234 px away mid-approach.  Thickness over length is how open
    the panel is, which is the only honest measure of orbit progress: the
    bounding box grows just as fast from walking closer.
    """

    stats = _channel_stats(image) if stats is None else stats
    # The bar is self-illuminated and sits well clear of everything else in the
    # room: measured core pixels run (227,249,253) here and (229,243,244) in a
    # wider shot, while the lamp-lit bench edges that kept being mistaken for
    # it only reach (204,193,175).  A threshold that admitted both left the
    # decision to shape alone, which does not survive a change of viewpoint.
    bright = cv2.bitwise_and(
        _at_least(stats.minimum, 205),
        _at_most(cv2.subtract(stats.maximum, stats.minimum), 60),
    )
    # Close along x only: the bar is one thin run of pixels and must not be
    # allowed to merge with whatever sits above or below it.
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
    )
    height = image.shape[0]
    bars: list[tuple[Rect, float, float]] = []
    width = image.shape[1]
    for rect, contour in _contour_rectangles(bright):
        if (rect.bottom >= height - 6 or rect.y <= 6) and rect.width >= 0.5 * width:
            # The capture is the whole Win32 window: the Windows taskbar strip
            # at the bottom and the white title bar at the top.  The title bar
            # is a 1290 px bright horizontal bar by every other test here and,
            # sorted longest first, it was the bar the search steered at.
            # Only full-width strips are those; a short bar cut by the frame
            # edge is a real bar the avatar is standing right over.
            continue
        # Measure elongation on the rotated rectangle.  Seen from an angle the
        # panel's bar runs diagonally across the view, and its axis-aligned box
        # is nowhere near flat -- a 145 px bar tilted 25 degrees measures 2.07
        # wide-to-high and was thrown away as "not a bar" on exactly the frames
        # where the panel was in sight.
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        edges = [(box[index], box[(index + 1) % 4]) for index in range(4)]
        start, end = max(
            edges, key=lambda edge: math.hypot(*(edge[1] - edge[0]))
        )
        long_side = math.hypot(*(end - start))
        short_side = min(
            math.hypot(*(edge[1] - edge[0]))
            for edge in edges
            if math.hypot(*(edge[1] - edge[0])) > 0.0
        )
        if short_side <= 0.0 or long_side < 5.0 * short_side:
            continue
        tilt = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        tilt = ((tilt + 90.0) % 180.0) - 90.0
        # Generous now that brightness carries the identification: orbiting the
        # panel swings its bar past 45 degrees long before it opens up, and a
        # 40 degree limit dropped it at exactly the moment it mattered.
        if abs(tilt) > 62.0:
            continue
        bars.append((rect, long_side, short_side))
    return bars


def detect_sliden_edge_panels(
    image: RGBImage,
    screen: Rect,
    *,
    maximum_gap_fraction: float = 1.7,
) -> list[SlidenEdgePanelObservation]:
    """Find the bright horizontal bars flanking the wall screen.

    A Sliden panel viewed from the side collapses to its progress bar: a few
    pixels of bright horizontal line next to the screen.  Until the avatar has
    walked around to the panel's face that line is the only evidence of where
    the panel is, so navigation steers by it instead of waiting for a panel it
    cannot yet see.  Both flanks produce one -- in the presentation room the
    left bar is a sub-screen and the right one the operation panel -- so the
    caller tries a side, checks for a URL field, and falls back to the other.
    """

    if screen.width <= 0 or screen.height <= 0:
        return []
    limit = maximum_gap_fraction * screen.width
    found: list[SlidenEdgePanelObservation] = []
    for rect, length, thickness in _bright_horizontal_bars(image):
        # The panel's bar is a substantial object next to the screen.  A floor
        # of 8% let a 26 px speck win the selection and the approach then spent
        # its whole budget walking at a highlight -- but from in front of the
        # wall screen, where the recipe starts, the real bar measures 45 px
        # against a 570 px screen (0.08), so 12% never saw it at all.
        if length < max(30.0, 0.06 * screen.width) or length > 1.6 * screen.width:
            continue
        center_x, center_y = rect.center
        # The panel hangs low beside the screen -- bottom right in this room.
        # Admitting the band above the screen let a ceiling light tube, which
        # stays perfectly thin from every angle, win the selection: a measured
        # orbit held its openness at 0.04 for the entire budget while the real
        # panel had gone from 0.15 to 0.92 in an earlier run.
        if not (
            screen.y + 0.45 * screen.height
            <= center_y
            <= screen.bottom + 0.95 * screen.height
        ):
            continue
        # Classify by which way the bar extends past the screen, not by where
        # its centre falls: a bar beside the screen still has its midpoint
        # within the screen's horizontal span whenever the two overlap, and
        # requiring the midpoint to clear the edge dropped the real one.
        screen_center_x = screen.center[0]
        overlap = min(rect.right, screen.right) - max(rect.x, screen.x)
        if overlap >= 0.6 * rect.width:
            # The bar lies within the screen's own span.  A long one under
            # the page is a progress bar; a short one below the page's foot
            # is a panel standing in front of the screen, seen from an
            # angle -- from the left of the room the operation panel's bar
            # projects onto the screen's right half.
            if rect.width >= 0.4 * screen.width or center_y < screen.y + 0.6 * screen.height:
                continue
            side, gap = (1 if center_x >= screen_center_x else -1), 0.0
        elif center_x >= screen_center_x and rect.right > screen.right:
            side, gap = 1, max(0.0, float(rect.x - screen.right))
        elif center_x < screen_center_x and rect.x < screen.x:
            side, gap = -1, max(0.0, float(screen.x - rect.right))
        else:
            continue
        if gap > limit:
            continue
        found.append(
            SlidenEdgePanelObservation(
                rect=rect,
                side=side,
                gap=gap,
                openness=thickness / max(1.0, length),
                length=length,
            )
        )
    # Longest first, ties broken by proximity to the screen.  The panel's bar
    # is the biggest bright line beside the screen; sorting by distance alone
    # promoted whatever small highlight happened to sit closest.
    found.sort(key=lambda item: (-item.length, item.gap))
    return found


def detect_sliden_bars(
    image: RGBImage,
    *,
    minimum_length: float = 60.0,
) -> list[tuple[Rect, float, float]]:
    """Every panel-like bar in view, longest first, without any other context.

    Recognising the bar only in relation to a squarely framed wall screen meant
    the search turned straight past a bar that was plainly in the capture,
    because the screen happened to be clipped or off to one side.  The bar is
    the brightest thin thing in this room and stands on its own.
    """

    bars = [
        (rect, length, thickness)
        for rect, length, thickness in _bright_horizontal_bars(image)
        if length >= minimum_length
    ]
    bars.sort(key=lambda item: -item[1])
    return bars


def track_sliden_edge_panel(
    image: RGBImage,
    point: Point,
    expected_length: float,
    *,
    radius_fraction: float = 0.25,
    scale_tolerance: float = 1.7,
) -> tuple[Rect, float, float] | None:
    """Follow a flanking bar once the wall screen has left the view.

    Orbiting the panel takes the screen out of frame long before the panel
    opens up, so the bar has to be trackable on its own.  Identity is carried
    by the bar's length rather than the width of its bounding box: a diagonal
    bar's box changes shape as fast as the viewpoint does.
    """

    if expected_length <= 0.0:
        return None
    limit = radius_fraction * image.shape[1]
    best: tuple[float, Rect, float, float] | None = None
    for rect, length, thickness in _bright_horizontal_bars(image):
        center = rect.center
        distance = math.hypot(center[0] - point[0], center[1] - point[1])
        if distance > limit:
            continue
        ratio = length / expected_length
        if not 1.0 / scale_tolerance <= ratio <= scale_tolerance:
            continue
        if best is None or distance < best[0]:
            best = (distance, rect, length, thickness)
    return None if best is None else (best[1], best[2], best[3])


def track_sliden_display(
    image: RGBImage,
    point: Point,
    expected_width: float,
    *,
    radius_fraction: float = 0.22,
    scale_tolerance: float = 2.2,
) -> Rect | None:
    """Follow a display already identified as the console.

    Identity comes from the URL field, but that strip is only a few pixels
    tall at approach range and drops out of individual frames.  The page
    itself never does, so once a panel has been confirmed the approach tracks
    the page near its last known position instead of re-proving identity every
    frame -- otherwise the search turns away each time it nearly arrives.
    """

    if expected_width <= 0.0:
        return None
    limit = radius_fraction * image.shape[1]
    best: tuple[float, Rect] | None = None
    for candidate in _display_candidates(image):
        center = candidate.rect.center
        distance = math.hypot(center[0] - point[0], center[1] - point[1])
        if distance > limit:
            continue
        ratio = candidate.rect.width / expected_width
        if not 1.0 / scale_tolerance <= ratio <= scale_tolerance:
            continue
        if best is None or distance < best[0]:
            best = (distance, candidate.rect)
    return None if best is None else best[1]


#: Downscale applied before matching the console page.  A quarter-size match
#: locates it to within a few pixels -- far finer than the aim needs -- for a
#: fraction of the cost of matching at full resolution inside a control loop.
_CONSOLE_MATCH_SHRINK = 4


def track_sliden_console(
    previous_image: RGBImage,
    anchor: Rect,
    image: RGBImage,
    *,
    radius_fraction: float = 0.15,
    minimum_score: float = 0.72,
    minimum_peak: float = 0.15,
) -> Point | None:
    """How far the console page has moved since *previous_image*.

    Recognition segments the frame by brightness, and in a sunlit room that
    is a coin toss the panel loses every few frames: its white page touches
    the window behind it, the two become one contour covering the whole
    capture, and a console filling a fifth of the view simply is not there
    any more.  Matching the page against the frame it was last seen in asks a
    different question -- *where did this go?* rather than *what is this?* --
    and has no segmentation to fail.  It only ever reports a translation, so
    it is a way to keep aiming through a dropout, not a way to identify a
    console in the first place.
    """

    step = _CONSOLE_MATCH_SHRINK
    height, width = image.shape[:2]
    if previous_image.shape != image.shape:
        return None
    margin_x = max(1, anchor.width // 10)
    margin_y = max(1, anchor.height // 10)
    left = max(0, anchor.x + margin_x)
    top = max(0, anchor.y + margin_y)
    right = min(width, anchor.right - margin_x)
    bottom = min(height, anchor.bottom - margin_y)
    if right - left < 4 * step or bottom - top < 4 * step:
        return None
    radius = int(radius_fraction * width)
    search_left = max(0, left - radius)
    search_top = max(0, top - radius)
    search_right = min(width, right + radius)
    search_bottom = min(height, bottom + radius)
    template = cv2.cvtColor(
        previous_image[top:bottom, left:right], cv2.COLOR_RGB2GRAY
    )
    search = cv2.cvtColor(
        image[search_top:search_bottom, search_left:search_right], cv2.COLOR_RGB2GRAY
    )
    template = cv2.resize(
        template,
        (max(2, template.shape[1] // step), max(2, template.shape[0] // step)),
        interpolation=cv2.INTER_AREA,
    )
    search = cv2.resize(
        search,
        (max(2, search.shape[1] // step), max(2, search.shape[0] // step)),
        interpolation=cv2.INTER_AREA,
    )
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _minimum, score, _minimum_at, maximum_at = cv2.minMaxLoc(scores)
    # A high score alone is not a location.  Where the panel has actually
    # left the search window, a page that is mostly flat white correlates
    # respectably with any pale patch, and the answer is confident nonsense.
    # A real match stands out from everywhere else it was tried.
    if score < minimum_score or score - float(scores.mean()) < minimum_peak:
        return None
    return (
        float(search_left + maximum_at[0] * step - left),
        float(search_top + maximum_at[1] * step - top),
    )


def _shift_observation(
    observation: SlidenConsoleObservation,
    delta: Point,
) -> SlidenConsoleObservation:
    """Move every world-space control of *observation* by *delta*.

    The modal is a screen-space overlay rather than part of the panel, so it
    is dropped instead of carried along.
    """

    def moved(point: Point | None) -> Point | None:
        return None if point is None else (point[0] + delta[0], point[1] + delta[1])

    rect = observation.url_field_rect
    quad = observation.slide_quad
    return SlidenConsoleObservation(
        slide_quad=None
        if quad is None
        else quad + np.asarray(delta, dtype=np.float32),
        home_button=moved(observation.home_button),
        url_field=moved(observation.url_field),
        url_field_rect=None
        if rect is None
        else Rect(
            rect.x + int(delta[0]), rect.y + int(delta[1]), rect.width, rect.height
        ),
        previous_button=moved(observation.previous_button),
        next_button=moved(observation.next_button),
        confidence=observation.confidence,
    )


def _warped_slide(
    image: RGBImage,
    quad: NDArray[np.float32] | None,
) -> RGBImage | None:
    if quad is None:
        return None
    target = np.asarray(
        [[0, 0], [319, 0], [319, 179], [0, 179]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(image, transform, (320, 180))


def slide_preview_difference(
    before_image: RGBImage,
    before_quad: NDArray[np.float32] | None,
    after_image: RGBImage,
    after_quad: NDArray[np.float32] | None,
) -> float:
    """Measure preview change after a navigation click, independent of pose."""

    before = _warped_slide(before_image, before_quad)
    after = _warped_slide(after_image, after_quad)
    if before is None or after is None:
        return 0.0
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    return float(
        np.mean(cv2.absdiff(before_gray, after_gray), dtype=np.float64) / 255.0
    )


#: Cache for :func:`_relative_step_limit`; the pointer settings are read
#: from the OS once per process rather than before every correction.
_RELATIVE_STEP_LIMIT: int | None = None


def _relative_step_limit() -> int:
    """Largest relative mouse delta Windows forwards without scaling it.

    ``SendInput`` relative motion is put through the pointer ballistics the
    user configured in Mouse Properties: with "Enhance pointer precision"
    on, any delta whose magnitude passes the first threshold is *doubled*
    before the application sees it, and past the second threshold it is
    doubled again.  A visual servo that models one pixel of command as one
    unit of motion therefore over-corrects exactly on the large steps and
    tracks correctly on the small ones -- the view shoots past the console
    on the first correction and the target leaves the frame entirely.
    Splitting each command into sub-threshold steps restores a linear,
    predictable response on any machine.
    """

    global _RELATIVE_STEP_LIMIT
    if _RELATIVE_STEP_LIMIT is not None:
        return _RELATIVE_STEP_LIMIT
    limit = 1 << 30
    try:
        import sys

        if sys.platform == "win32":
            import ctypes

            #: ``SPI_GETMOUSE`` fills three integers: the two acceleration
            #: thresholds and the acceleration enable flag.
            values = (ctypes.c_int * 3)()
            if ctypes.windll.user32.SystemParametersInfoW(0x0003, 0, values, 0):
                thresholds = [int(value) for value in values]
                if thresholds[2] > 0 and thresholds[0] > 0:
                    limit = thresholds[0]
    except Exception:
        limit = 6
    _RELATIVE_STEP_LIMIT = max(1, limit)
    return _RELATIVE_STEP_LIMIT


def _ballistics_free_steps(dx: int, dy: int) -> list[tuple[int, int]]:
    """Split ``(dx, dy)`` into steps the OS will not accelerate.

    The pieces sum exactly to the requested delta, so a caller that tracks
    how far it has turned stays correct.
    """

    limit = _relative_step_limit()
    span = max(abs(dx), abs(dy))
    if span <= limit:
        return [(dx, dy)]
    count = -(-span // limit)
    steps: list[tuple[int, int]] = []
    sent_x = sent_y = 0
    for index in range(1, count + 1):
        step_x = int(round(dx * index / count)) - sent_x
        step_y = int(round(dy * index / count)) - sent_y
        sent_x += step_x
        sent_y += step_y
        if step_x or step_y:
            steps.append((step_x, step_y))
    return steps


class SlidenConsoleController:
    """Closed-loop UI controller backed by VRC Pilot screenshots and input."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 0.25,
        cancelled: Callable[[], bool] | None = None,
        screenshot_provider: Callable[[], RGBImage | None] | None = None,
        click: Callable[[Point], None] | None = None,
        interact: Callable[[], None] | None = None,
        relative_move: Callable[[Point], None] | None = None,
        viewport_center_provider: Callable[[RGBImage], Point] | None = None,
        scan_turn: Callable[[float], None] | None = None,
        move_forward: Callable[[float], None] | None = None,
        strafe: Callable[[float, int], None] | None = None,
        escape: Callable[[], None] | None = None,
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
        clear_text: Callable[[], None] | None = None,
        paste: Callable[[str], None] | None = None,
        state_callback: Callable[..., Any] | None = None,
        diagnostic_directory: Path | None = None,
        text_locator: Callable[[RGBImage], Sequence[tuple[str, Rect]]] | None = None,
    ) -> None:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("console timeout and polling interval must be positive")
        self.pid = pid
        #: Optional OCR: every text on a frame with its box.  Used to find the
        #: wall screen where the template detector cannot (see
        #: ``screen_from_text_boxes``); navigation works without it.
        self._text_locator = text_locator
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.cancelled = cancelled
        self.state_callback = state_callback
        self._screenshot_provider = screenshot_provider
        self._click_callback = click
        self._interact_callback = interact
        self._relative_move_callback = relative_move
        self._viewport_center_provider = viewport_center_provider
        self._scan_turn_callback = scan_turn
        self._move_forward_callback = move_forward
        self._strafe_callback = strafe
        self._escape_callback = escape
        self.osc_host = osc_host
        self.osc_port = osc_port
        self._motion_controller: Any | None = None
        self._pitch_offset = 0.0
        #: The motion issued most recently, so it can be undone.  The bar is
        #: part of the room and never moves: if a step loses it, reversing that
        #: step puts it back where it was.
        self._last_action: tuple[str, float, float] | None = None
        #: Sub-pixel part of the last aim command, carried forward so a long
        #: run of small corrections is not rounded away to nothing.
        self._move_residue: Point = (0.0, 0.0)
        #: Measured capture pixels of view motion per relative mouse count,
        #: per axis.  Signed, so a client configured with an inverted axis is
        #: learned rather than fought.  ``None`` until the servo has probed.
        self._aim_gain: list[float | None] = [None, None]
        #: Size of the exploratory command used until that measurement lands.
        self._aim_probe = self._AIM_PROBE_START
        #: Where and how large a console was last recognized, so following
        #: one does not have to re-prove its identity from scratch every
        #: frame.  Only a full recognition can create it, and losing sight of
        #: the target clears it.  The frame and observation that produced it
        #: are kept alongside, as the template for matching it forward.
        self._console_anchor: Rect | None = None
        self._console_frame: RGBImage | None = None
        self._console_observation: SlidenConsoleObservation | None = None
        self.diagnostic_directory = diagnostic_directory
        self._diagnostics_written = 0
        #: Cached window handle and screen grabber for the fast capture path.
        self._window_handle: int | None = None
        self._screen_grabber: Any | None = None
        #: When the VRChat foreground check was last made for a mouse move.
        self._last_target_check = 0.0
        #: Text boxes read by the last OCR pass, so bar searches can leave
        #: white lettering on a page alone.
        self._last_text_boxes: list[Rect] = []
        self._clear_text_callback = clear_text
        self._paste_callback = paste

    #: Attempts and spacing for an input that Windows refused to deliver
    #: because VRChat was not the foreground window.  Something else holding
    #: the foreground for a moment -- a notification, the notebook the run is
    #: being watched from -- used to end the presentation outright.
    _FOCUS_RETRIES = 3
    _FOCUS_RETRY_SECONDS = 0.3

    def _retrying_focus(self, action: Callable[[], Any]) -> Any:
        """Run *action*, giving a momentary loss of foreground a second try."""

        for attempt in range(self._FOCUS_RETRIES):
            try:
                return action()
            except Exception as exc:
                # vrcpilot is imported lazily throughout this module so the
                # detectors stay usable without VRChat installed; match its
                # focus error by name rather than dragging in the import.
                if type(exc).__name__ != "VRChatNotFocusedError":
                    raise
                if attempt == self._FOCUS_RETRIES - 1:
                    raise
                _check_cancel(self.cancelled)
                time.sleep(self._FOCUS_RETRY_SECONDS)

    def _detect(self, frame: RGBImage) -> SlidenConsoleObservation:
        """Recognize the console, remembering where this run last saw one.

        The first recognition of a run has no anchor and so must pass the
        full test, URL bar included.  Everything after it is following the
        same object across a few pixels of camera movement, which is a much
        weaker claim and survives the bar drifting out of the capture.
        """

        observation = detect_sliden_console(frame, anchor=self._console_anchor)
        quad = observation.slide_quad
        if quad is not None:
            left, right = float(quad[:, 0].min()), float(quad[:, 0].max())
            top, bottom = float(quad[:, 1].min()), float(quad[:, 1].max())
            self._console_anchor = Rect(
                int(left), int(top), int(right - left), int(bottom - top)
            )
            self._console_frame = frame
            self._console_observation = observation
        return observation

    def _track_console(
        self,
        frame: RGBImage,
        template: tuple[RGBImage, SlidenConsoleObservation, Rect] | None = None,
    ) -> SlidenConsoleObservation | None:
        """Carry a recognized console forward into a frame it is not
        recognized in, by matching where its page went.

        The template may be given explicitly, and an aim that cares about one
        particular control should do so: the console's own record is replaced
        by every recognition, including the ones that resolved the page but
        not the button, and carrying one of those forward produces a console
        with nothing to aim at -- which reads as the panel being gone.
        """

        if template is not None:
            previous, observation, anchor = template
        else:
            anchor = self._console_anchor
            previous = self._console_frame
            observation = self._console_observation
        if anchor is None or previous is None or observation is None:
            return None
        delta = track_sliden_console(previous, anchor, frame)
        if delta is None:
            return None
        if template is not None:
            return _shift_observation(observation, delta)
        # Carry the anchor along too.  Recognition's relaxed path needs one
        # near the panel, and an anchor pinned to the last frame recognition
        # managed goes stale exactly when it is needed most.  The template
        # stays put: re-cutting it from a matched frame would let small
        # errors compound into a drift with nothing to correct it.
        self._console_anchor = Rect(
            anchor.x + int(delta[0]),
            anchor.y + int(delta[1]),
            anchor.width,
            anchor.height,
        )
        return _shift_observation(observation, delta)

    def _forget_console(self) -> None:
        """Drop the tracking state so the next recognition starts cold."""

        self._console_anchor = None
        self._console_frame = None
        self._console_observation = None

    def _capture(self) -> RGBImage | None:
        """Grab the VRChat window, cheaply when it is already in front.

        ``take_screenshot`` re-focuses the window on every call, and its
        Win32 path taps Alt to defeat the foreground lock.  Inside a servo
        that is both a needless delay per frame and a stream of stray Alt
        presses into the running client, so the ordinary case -- VRChat
        already foreground, which it must be for the input to land anyway --
        skips straight to the grab.
        """

        _check_cancel(self.cancelled)
        if self._screenshot_provider is not None:
            frame = self._screenshot_provider()
            return None if frame is None else frame.copy()
        import vrcpilot

        frame = self._capture_foreground()
        if frame is not None:
            return frame
        shot = self._retrying_focus(
            lambda: vrcpilot.take_screenshot(settle_seconds=0.05, pid=self.pid)
        )
        return None if shot is None else shot.image

    def _capture_foreground(self) -> RGBImage | None:
        """Grab the window without focusing it, or ``None`` to use the slow path."""

        import sys

        if sys.platform != "win32":
            return None
        try:
            import mss
            import win32gui
            from vrcpilot import process
            from vrcpilot.windows import find_vrchat_hwnd, get_window_rect

            hwnd = self._window_handle
            if hwnd is None or not win32gui.IsWindow(hwnd):
                hwnd = find_vrchat_hwnd(process.resolve_pid(self.pid))
                self._window_handle = hwnd
            if hwnd is None or win32gui.GetForegroundWindow() != hwnd:
                return None
            rect = get_window_rect(hwnd)
            if rect is None:
                return None
            if self._screen_grabber is None:
                self._screen_grabber = mss.MSS()
            x, y, width, height = rect
            shot = self._screen_grabber.grab(
                {"left": x, "top": y, "width": width, "height": height}
            )
            return (
                np.frombuffer(shot.rgb, dtype=np.uint8)
                .reshape(shot.size.height, shot.size.width, 3)
                .copy()
            )
        except Exception:
            # Any Win32/mss surprise falls back to the supported helper
            # rather than stopping a presentation.
            self._window_handle = None
            self._screen_grabber = None
            return None

    #: Most before/after pairs written per run.  Enough to see a pattern,
    #: few enough that a long presentation cannot fill a disk.
    _DIAGNOSTIC_LIMIT = 8

    def _save_diagnostic(
        self,
        reason: str,
        before: RGBImage | None,
        after: RGBImage | None,
        command: Point | None,
        target: Point | None,
    ) -> None:
        """Record the frames around an aim failure for later inspection.

        Losing the target is invisible from the outside -- the state file
        only ever shows the last frame that *worked* -- and a run that ends
        "it moved a few times and stopped" cannot be diagnosed from numbers
        alone.  One before/after pair says immediately whether the panel left
        the frame, whether the capture stopped showing VRChat, or whether the
        view never moved at all.
        """

        directory = self.diagnostic_directory
        if directory is None or self._diagnostics_written >= self._DIAGNOSTIC_LIMIT:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            index = self._diagnostics_written
            self._diagnostics_written += 1
            for name, image in (("before", before), ("after", after)):
                if image is None:
                    continue
                cv2.imwrite(
                    str(directory / f"aim-{index:02d}-{reason}-{name}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
            with (directory / "aim.log").open("a", encoding="utf-8") as log:
                log.write(
                    f"{index:02d} {reason} command={command} target={target} "
                    f"probe={self._aim_probe} gain={self._aim_gain}\n"
                )
        except Exception:
            # Diagnostics must never be the reason a presentation stops.
            self.diagnostic_directory = None

    def close(self) -> None:
        """Release the screen grabber held open by the fast capture path."""

        grabber, self._screen_grabber = self._screen_grabber, None
        self._window_handle = None
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass

    def _click_ui(self, point: Point) -> None:
        """Click a screen-space control while a VRChat menu is open."""

        _check_cancel(self.cancelled)
        if self._click_callback is not None:
            self._click_callback(point)
            return
        import vrcpilot

        x, y = (int(round(point[0])), int(round(point[1])))
        self._retrying_focus(lambda: vrcpilot.mouse.move(x, y, pid=self.pid))
        self._retrying_focus(
            lambda: vrcpilot.mouse.click(
                vrcpilot.MouseButton.LEFT,
                duration=0.06,
                pid=self.pid,
            )
        )

    #: Where the Launch Pad's respawn button sits, as a fraction of the
    #: capture.  Measured at 610, 643 in a 1308x799 window -- the bottom row
    #: of the Launch Pad, between "home" and "select".
    _RESPAWN_POINT = (0.466, 0.805)

    def respawn(self) -> bool:
        """Return the avatar to the world's spawn point.

        Navigation results are only comparable from a known starting place.
        Left where the previous attempt happened to end, the same code reaches
        the panel one run and never sees it the next.
        """

        _check_cancel(self.cancelled)
        before = self._capture()
        if before is None:
            return False
        height, width = before.shape[:2]
        self._press_escape()
        time.sleep(0.8)
        menu = self._capture()
        if menu is not None and self._frame_motion(before, menu) < 0.01:
            # Escape toggles: a menu that was already open has just been
            # closed.  Open it again rather than giving up.
            self._press_escape()
            time.sleep(0.8)
            menu = self._capture()
        if menu is None or self._frame_motion(before, menu) < 0.01:
            return False
        self._click_ui(
            (
                self._RESPAWN_POINT[0] * width,
                self._RESPAWN_POINT[1] * height,
            )
        )
        time.sleep(1.2)
        after = self._capture()
        if after is None:
            return False
        if self._frame_motion(menu, after) < 0.02:
            # Still on the menu: the click missed.  Close it and carry on from
            # wherever the avatar is rather than leaving a menu in the way.
            self._press_escape()
            time.sleep(0.5)
            return False
        self._last_action = None
        self._pitch_offset = 0.0
        return True

    def _press_escape(self) -> None:
        _check_cancel(self.cancelled)
        if self._escape_callback is not None:
            self._escape_callback()
            return
        import vrcpilot

        vrcpilot.keyboard.press(vrcpilot.Key.ESCAPE, focus=True, pid=self.pid)

    def _undo_last_action(self) -> bool:
        """Reverse the most recent motion, restoring the previous viewpoint."""

        action = self._last_action
        if action is None:
            return False
        self._last_action = None
        kind, first, second = action
        if kind == "look":
            self._pitch_offset -= second
            self._move_relative((-first, -second))
        elif kind == "turn":
            self._scan_turn(first, -second)
        elif kind == "strafe":
            self._strafe(first, -int(second))
        elif kind == "forward":
            self._walk_backward(first)
        elif kind == "back":
            self._walk_forward(first)
        else:
            return False
        self._last_action = None
        return True

    def _move_relative(self, delta: Point) -> None:
        _check_cancel(self.cancelled)
        self._last_action = ("look", float(delta[0]), float(delta[1]))
        if self._relative_move_callback is not None:
            self._relative_move_callback(delta)
            return
        import sys
        import vrcpilot

        # Commands arrive as floats.  Truncating each one on its own threw
        # away every correction below half a pixel, so a servo that was one
        # or two pixels out could never close the gap.  Carry the remainder
        # into the next command instead.
        wanted_x = float(delta[0]) + self._move_residue[0]
        wanted_y = float(delta[1]) + self._move_residue[1]
        dx, dy = (int(round(wanted_x)), int(round(wanted_y)))
        self._move_residue = (wanted_x - dx, wanted_y - dy)
        if dx == 0 and dy == 0:
            return
        if sys.platform == "win32":
            # vrcpilot's Win32 backend currently calls
            # pydirectinput.moveRel(dx, dy) without its ``relative=True``
            # switch.  In VRChat mouse-look mode that round-trips through the
            # recentered OS cursor, swallowing small corrections and turning
            # larger ones into jumps.  Use the same backend's true relative
            # SendInput path until the upstream wrapper exposes it.
            from vrcpilot.controls import ensure_target
            from vrcpilot.controls.mouse.windows import pydirectinput

            # The foreground/PID check behind ensure_target costs about a
            # quarter of a second (process enumeration); an aim loop issues
            # a correction every frame or two, so re-checking each time
            # doubled the loop period.  Once a second is plenty to notice
            # the window losing focus.
            now = time.monotonic()
            if now - self._last_target_check > 1.0:
                self._retrying_focus(lambda: ensure_target(pid=self.pid))
                self._last_target_check = now
            # pydirectinput sleeps its module-level PAUSE (0.1 s by default)
            # after every call.  A 100-count look split into sub-threshold
            # steps is 17 calls, so a single correction took 1.7 s and the
            # navigation spent most of its time waiting between mouse
            # events.  VRChat reads raw relative motion; the events can
            # follow each other closely.
            previous_pause = getattr(pydirectinput, "PAUSE", 0.1)
            pydirectinput.PAUSE = 0.004
            try:
                for step_x, step_y in _ballistics_free_steps(dx, dy):
                    pydirectinput.moveRel(step_x, step_y, relative=True)
            finally:
                pydirectinput.PAUSE = previous_pause
            return
        vrcpilot.mouse.move(dx, dy, relative=True, pid=self.pid)

    def _interact(self, viewport_center: Point) -> None:
        _check_cancel(self.cancelled)
        if self._interact_callback is not None:
            self._interact_callback()
            return
        if self._click_callback is not None:
            self._click_callback(viewport_center)
            return
        import vrcpilot

        self._retrying_focus(
            lambda: vrcpilot.mouse.click(
                vrcpilot.MouseButton.LEFT,
                duration=0.06,
                pid=self.pid,
            )
        )

    def _viewport_center(self, image: RGBImage) -> Point:
        if self._viewport_center_provider is not None:
            return self._viewport_center_provider(image)
        # VRC Pilot captures the complete Win32 window, including its title
        # bar.  Mouse-look aims through the client-area centre, not the full
        # image centre, so use the actual client rectangle when available.
        try:
            import sys

            if sys.platform == "win32":
                import win32gui
                from vrcpilot import process
                from vrcpilot.windows import find_vrchat_hwnd, get_window_rect

                pid = process.resolve_pid(self.pid)
                hwnd = find_vrchat_hwnd(pid)
                outer = None if hwnd is None else get_window_rect(hwnd)
                if hwnd is not None and outer is not None:
                    client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
                    _left, _top, client_right, client_bottom = win32gui.GetClientRect(
                        hwnd
                    )
                    height, width = image.shape[:2]
                    center = (
                        client_left - outer[0] + client_right / 2.0,
                        client_top - outer[1] + client_bottom / 2.0,
                    )
                    # The client area is the window minus its frame, so its
                    # centre is a few tens of pixels from the capture's.
                    # Anything further apart means the two measurements are in
                    # different units -- a DPI-scaled client rect against a
                    # physical-pixel window rect -- and a reticle in the wrong
                    # place is worse than a slightly offset one.
                    if (
                        abs(center[0] - width / 2.0) <= 0.25 * width
                        and abs(center[1] - height / 2.0) <= 0.25 * height
                    ):
                        return center
        except Exception:
            pass
        height, width = image.shape[:2]
        return (width / 2.0, height / 2.0)

    def _ensure_motion_controller(self) -> Any:
        if self._motion_controller is None:
            from vrcpilot import OscSender

            self._motion_controller = OscSender(
                self.osc_host,
                self.osc_port,
            ).controller()
        return self._motion_controller

    def _scan_right(self, seconds: float) -> None:
        """Rotate one bounded search tile without translating the avatar."""

        self._scan_turn(seconds, 1)

    def _scan_turn(self, seconds: float, direction: int) -> None:
        _check_cancel(self.cancelled)
        duration = max(0.05, min(0.30, float(seconds)))
        step = 1 if direction >= 0 else -1
        self._last_action = ("turn", duration, float(step))
        if self._scan_turn_callback is not None:
            self._scan_turn_callback(duration)
            return
        controller = self._ensure_motion_controller()
        turn = controller.look_right if step > 0 else controller.look_left
        turn(True)
        try:
            time.sleep(duration)
        finally:
            turn(False)

    def _walk_forward(self, seconds: float) -> None:
        """Walk toward the centred console in a short, bounded pulse."""

        _check_cancel(self.cancelled)
        duration = max(0.08, min(0.60, float(seconds)))
        self._last_action = ("forward", duration, 0.0)
        if self._move_forward_callback is not None:
            self._move_forward_callback(duration)
            return
        controller = self._ensure_motion_controller()
        controller.vertical(0.45)
        try:
            time.sleep(duration)
        finally:
            controller.vertical(0.0)

    def _strafe(self, seconds: float, direction: int) -> None:
        """Side-step without turning.

        The operation panel is a flat surface: from most of the room it is a
        few pixels of edge and cannot be recognized no matter how long the
        avatar spins.  Only translation changes the viewing angle enough for it
        to open up, so searching for it has to include side steps.
        """

        _check_cancel(self.cancelled)
        duration = max(0.08, min(0.70, float(seconds)))
        step = 1 if direction >= 0 else -1
        self._last_action = ("strafe", duration, float(step))
        if self._strafe_callback is not None:
            self._strafe_callback(duration, step)
            return
        controller = self._ensure_motion_controller()
        controller.horizontal(0.45 * step)
        try:
            time.sleep(duration)
        finally:
            controller.horizontal(0.0)

    def _walk_backward(self, seconds: float) -> None:
        """Back off when the wall screen has grown past the frame edges.

        The flanking bars live just outside the screen's left and right edges,
        so they are only in the capture while the whole screen is.  Walking to
        the screen until it fills the view hides the very thing the approach
        needs to read next.
        """

        _check_cancel(self.cancelled)
        duration = max(0.08, min(0.60, float(seconds)))
        self._last_action = ("back", duration, 0.0)
        if self._move_forward_callback is not None:
            self._move_forward_callback(-duration)
            return
        controller = self._ensure_motion_controller()
        controller.vertical(-0.45)
        try:
            time.sleep(duration)
        finally:
            controller.vertical(0.0)

    def _pitch_step(self, move_y: float) -> float:
        """Apply a pitch correction, bounded against cumulative drift."""

        allowed = max(
            -_MAXIMUM_PITCH_OFFSET - self._pitch_offset,
            min(_MAXIMUM_PITCH_OFFSET - self._pitch_offset, float(move_y)),
        )
        self._pitch_offset += allowed
        return allowed

    @staticmethod
    def _frame_motion(before: RGBImage, after: RGBImage) -> float:
        """How much the view changed, 0 for identical frames."""

        shrink = lambda frame: cv2.resize(  # noqa: E731 - local, single use
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            (128, 72),
            interpolation=cv2.INTER_AREA,
        )
        return float(
            np.mean(cv2.absdiff(shrink(before), shrink(after)), dtype=np.float64)
            / 255.0
        )

    def _escape_to_open_space(self, attempts: int = 8) -> bool:
        """Walk out of a corner and into the open part of the room.

        Sweeping from a corner can never reveal the wall screen, and turning
        further cannot fix it -- the avatar has to leave.  Rather than reason
        about the geometry of the room, try a heading and check whether the
        view actually moved: a wall gives back an unchanged frame no matter how
        long the walk is.
        """

        self._set_pitch(0.0)
        for attempt in range(attempts):
            _check_cancel(self.cancelled)
            before = self._capture()
            self._walk_forward(0.50)
            time.sleep(max(0.15, self.poll_seconds))
            after = self._capture()
            if before is None or after is None:
                continue
            if self._frame_motion(before, after) >= 0.02:
                # Moving freely: carry on a little to clear the corner.
                self._walk_forward(0.55)
                time.sleep(max(0.15, self.poll_seconds))
                return True
            # Wedged against something.  Try a different heading.
            self._scan_right(0.25)
            time.sleep(max(0.12, self.poll_seconds))
        return False

    #: Relative mouse counts per degree of desktop mouse-look, both axes.
    #: Measured live on 2026-08-18: a full yaw turn is about 3600 counts and
    #: the pitch clamps (+-90 degrees) are about 1800 apart.  Scales with the
    #: client's mouse sensitivity setting; ``measure_pitch_range`` re-derives
    #: it from the clamps.
    _YAW_COUNTS_PER_360 = 3600.0
    _PITCH_COUNTS_PER_180 = 1800.0
    #: Counts from the bottom pitch clamp (looking straight down) up to a
    #: level view.  Measured live 2026-08-18 against the room's furniture at
    #: eye height; the bottom clamp is used because the floor is textured and
    #: unmistakable, where the ceiling above the top clamp is featureless.
    _LEVEL_FROM_BOTTOM_COUNTS = 880.0

    def level_camera(self) -> None:
        """Bring the camera to a level attitude from any pitch, absolutely.

        Nothing reports the camera's pitch, and every phase that inherits an
        unknown one -- a fresh controller, a respawn, a search abandoned
        half-way -- used to sweep "level" while looking at the floor and
        never see a wall screen that sits above the horizon.  VRChat clamps
        the pitch at straight down, so pushing the view down by more than
        the whole range lands on the clamp regardless of where it started,
        and a measured number of counts back up is level.  From then on the
        offset bookkeeping is measured from a known attitude.
        """

        span = float(self._PITCH_COUNTS_PER_180)
        self._move_relative((0.0, span * 1.5))
        time.sleep(0.05)
        self._move_relative((0.0, -float(self._LEVEL_FROM_BOTTOM_COUNTS)))
        self._pitch_offset = 0.0
        self._move_residue = (0.0, 0.0)
        self._last_action = None

    def measure_pitch_range(self, *, step: float = 100.0) -> float:
        """Measure ``_PITCH_COUNTS_PER_180`` on this client and adopt it.

        Looks up to the clamp, then steps down until the frame stops
        changing; the steps in between are the range.  For re-calibrating
        after a change of mouse sensitivity; takes a few seconds and leaves
        the camera level.
        """

        self._move_relative((0.0, -(self._PITCH_COUNTS_PER_180 * 1.5)))
        time.sleep(0.15)
        previous = self._capture()
        motions: list[float] = []
        for _ in range(int(6000 / step)):
            self._move_relative((0.0, step))
            time.sleep(0.12)
            frame = self._capture()
            if previous is None or frame is None:
                previous = frame
                continue
            motions.append(self._frame_motion(previous, frame))
            previous = frame
            moving = [m for m in motions if m >= 0.02]
            if len(moving) >= 3 and len(motions) >= 5:
                reference = float(np.median(moving))
                if motions[-1] < 0.25 * reference and motions[-2] < 0.25 * reference:
                    break
        moving_steps = sum(1 for m in motions if m >= 0.02)
        span = max(400.0, moving_steps * step)
        type(self)._PITCH_COUNTS_PER_180 = span
        type(self)._YAW_COUNTS_PER_360 = 2.0 * span
        self._move_relative((0.0, -span / 2.0))
        self._pitch_offset = 0.0
        return span

    def _set_pitch(self, offset: float) -> None:
        """Move the camera to a pitch band measured from the entry attitude.

        Tracking the accumulated offset keeps the sweep from drifting into the
        floor over a long search, which previously left the avatar spinning
        over floor texture with the room out of frame.
        """

        delta = float(offset) - self._pitch_offset
        if abs(delta) < 1.0:
            return
        self._move_relative((0.0, delta))
        self._pitch_offset = float(offset)

    @staticmethod
    def _console_width_fraction(
        observation: SlidenConsoleObservation,
        image_width: int,
    ) -> float:
        quad = observation.slide_quad
        if quad is None or image_width <= 0:
            return 0.0
        top = float(np.linalg.norm(quad[1] - quad[0]))
        bottom = float(np.linalg.norm(quad[2] - quad[3]))
        return (top + bottom) / (2.0 * image_width)

    #: Pitch bands, in true-relative mouse pixels from the reference attitude,
    #: swept while hunting for each target.  Positive looks down.  The console
    #: bands start level rather than tilted down: an operation panel can be
    #: mounted anywhere, and assuming it sits below eye level pointed the
    #: search at the floor in a room where it hangs at head height.
    _SCREEN_PITCH_BANDS = (0.0, 70.0, -70.0)
    #: Progressively downward: the panel and its bar hang low, so a revolution
    #: that finds nothing is followed by one taken with the view lowered.
    _CONSOLE_PITCH_BANDS = (0.0, 50.0, 100.0)

    @staticmethod
    def _screen_skew(frame: RGBImage, screen: Rect) -> float | None:
        """How obliquely the screen is seen, from the slope of its bottom
        edge in the capture: positive when the edge rises to the right,
        i.e. the avatar stands left of the screen's middle (the near end of
        an edge below eye level sits lower in the frame).  ``None`` when the
        page is not segmented.  Uses the bottom edge only because the top is
        often cut by the capture from where the recipe starts."""

        cx, cy = screen.center
        stats = _channel_stats(frame)
        for raw_rect, contour in _contour_rectangles(_display_mask(frame, stats)):
            if not (raw_rect.x <= cx <= raw_rect.right and raw_rect.y <= cy <= raw_rect.bottom):
                continue
            if raw_rect.width < 0.5 * screen.width:
                continue
            points = contour.reshape(-1, 2)
            bins = 8
            xs: list[float] = []
            ys: list[float] = []
            for index in range(bins):
                left = raw_rect.x + raw_rect.width * (index + 0.5) / (bins + 1)
                right = raw_rect.x + raw_rect.width * (index + 1.5) / (bins + 1)
                inside = points[(points[:, 0] >= left) & (points[:, 0] < right)]
                if inside.size == 0:
                    continue
                xs.append(0.5 * (left + right))
                ys.append(float(inside[:, 1].max()))
            if len(xs) < 4:
                return None
            slope = float(np.polyfit(xs, ys, 1)[0])
            return -slope
        return None

    def _is_text_bar(self, rect: Rect) -> bool:
        """Whether a bright bar is really lettering read by the last OCR."""

        for text in self._last_text_boxes:
            overlap_x = min(rect.right, text.right) - max(rect.x, text.x)
            overlap_y = min(rect.bottom, text.bottom) - max(rect.y, text.y)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            area = overlap_x * overlap_y
            # Both ways: the bar must be mostly lettering and the lettering
            # mostly this bar.  From the left of the room the panel's real
            # bar happens to project onto "sliden.app" and a one-sided test
            # threw it away.
            if area >= 0.7 * rect.width * rect.height and area >= 0.5 * text.width * text.height:
                return True
        return False

    def _observe_screen(self, frame: RGBImage) -> tuple[Rect | None, str]:
        """The wall screen's rectangle and how it was found ("detector",
        "text" or "")."""

        screen = detect_sliden_screen(frame)
        if screen.rect is not None and screen.confidence >= 0.55:
            return screen.rect, "detector"
        if self._text_locator is not None:
            # OCR costs about half a second; only pay it when a dark page
            # candidate is in the frame at all.
            if not _display_candidates(frame, minimum_area_fraction=0.003):
                self._last_text_boxes = []
                return None, ""
            try:
                boxes = list(self._text_locator(frame))
            except Exception:
                boxes = []
            # Kept for the bar search: white text on a page ("sliden.app"
            # under the QR code) is a bright horizontal bar by every other
            # test and was steered at as the panel.
            self._last_text_boxes = [rect for _text, rect in boxes]
            rect = screen_from_text_boxes(
                boxes, frame.shape[1], frame.shape[0],
                exclude=console_page_rects(frame),
                bars=[bar for bar, _length, _thickness in _bright_horizontal_bars(frame)],
            )
            if rect is not None:
                # The wordmark only says roughly where the page is; the
                # dark page itself, when segmented, says exactly.  Flank
                # classification needs the real edges and centre.
                cx, cy = rect.center
                for candidate in _display_candidates(frame, minimum_area_fraction=0.003):
                    page = candidate.rect
                    if (
                        page.x <= cx <= page.right
                        and page.y <= cy <= page.bottom
                        and page.width >= 0.6 * rect.width
                    ):
                        return page, "text+page"
                return rect, "text"
        return None, ""

    #: The bar must reach this far down the capture before the avatar is
    #: taken to be beside the panel: "keep it centred, looking further down
    #: as you go, until it sits at the very bottom of the view".
    _BAR_BOTTOM_FRACTION = 0.90
    #: How far the camera may be pitched down, in relative mouse pixels, while
    #: following the bar.  Far past the sweep limit on purpose: at the end of
    #: the approach the bar is right beneath the avatar.
    _BAR_PITCH_LIMIT = 900.0
    #: Pitch, in mouse counts below level, to look at after the about-face:
    #: about 40 degrees, where a waist-high panel one or two steps away sits.
    _FACE_PITCH_COUNTS = 600.0
    #: Widest the panel's page may be, as a share of the capture, when the
    #: approach declares arrival; wider and the URL bar leaves the frame as
    #: soon as the aim drops to the button row.
    _ARRIVAL_MAX_WIDTH_FRACTION = 0.32
    #: Backward steps the facing stage may spend getting to that distance
    #: before it settles for where it stands.
    _FACE_BACK_OFF_LIMIT = 10
    #: Pitches (mouse counts below level) visited in turn while the page is
    #: not in view after the about-face.
    _FACE_PITCH_LADDER = (760.0, 480.0, 300.0, 120.0, 0.0, 600.0, 900.0)
    #: How the avatar moves while hunting for the wanted side's bar: back
    #: away first for the wide view, then keep stepping towards that side --
    #: from the far side of the room the panel's bar is hidden behind the
    #: furniture in front of it -- looking twice after each move.
    _SIDE_HUNT_MOVES = (
        "back", "look", "back", "look",
        "side", "look", "side", "look", "side", "look",
        "side", "look", "side", "look", "side", "look",
        "back", "look", "side", "look",
    )

    def approach_operation_panel(
        self,
        *,
        timeout_seconds: float = 120.0,
        prefer_side: int = 1,
        rejected_sides: set[int] | None = None,
    ) -> SlidenConsoleObservation:
        """Reach the operation panel by way of its edge-on bar.

        The owner's recipe for this room, applied literally.  Find the black
        Sliden wall screen; the bright bars beside it are the flanking Sliden
        surfaces seen edge-on -- the sub-screen on one side, the operation
        panel on the other (right, in this room).  From behind the bar walk
        at it, keeping it in the centre of the view and looking further and
        further down as it nears, until it sits at the very bottom of the
        view; take two more steps past it; turn about, and the panel's face
        is in view.  A bar that turns out to have no URL field on its face
        was the sub-screen: its side is rejected and the other is tried.
        """

        if timeout_seconds <= 0.0:
            raise ValueError("navigation timeout must be positive")
        rejected = set() if rejected_sides is None else rejected_sides
        deadline = time.monotonic() + timeout_seconds
        stage = "framing_screen"
        screen: Rect | None = None
        screen_source = ""
        sweep_steps = 0
        back_offs = 0
        strafes = 0
        side_hunts = 0
        lateral_moves = 0
        close_ins = 0
        close_in_turns = 0
        side = 0
        bar_center: Point | None = None
        bar_length = 0.0
        pitch_down = 0.0
        lost = 0
        approach_steps = 0
        best_bottom = 0.0
        stall = 0
        cross_steps = 0
        face_steps = 0
        face_pitch_nudges = 0
        confirm_frames = 0
        face_back_offs = 0
        face_last_width = 0.0
        face_last_center: Point | None = None
        face_relook = 0
        # Level the camera absolutely: the phase before this one (the room
        # entry, an earlier attempt, whatever the operator left) may have
        # left it pitched, and a sweep run "level" from a downward view
        # never sees a wall screen that sits above the horizon.
        self.level_camera()
        #: Sweep attitude per revolution: level first, then up, then down,
        #: then further up -- the camera may have been left pitched by an
        #: earlier phase and the bookkeeping cannot know by how much.
        framing_bands = (0.0, -80.0, 80.0, -160.0)

        def report(**updates: Any) -> None:
            fields: dict[str, Any] = {
                "presentationPhase": "approaching_sliden_panel",
                "presentationNavigationStatus": stage,
                "presentationNavigationPanelSide": side,
            }
            # A caller may name a sub-status of the stage ("backing_off");
            # it overrides rather than collides.
            fields.update(updates)
            _state(self.state_callback, **fields)

        while time.monotonic() < deadline:
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is None:
                time.sleep(self.poll_seconds)
                continue
            height, width = frame.shape[:2]
            viewport = self._viewport_center(frame)

            if stage == "framing_screen":
                # Too close to any Sliden page to see its surroundings.
                fill = detect_sliden_page_fill(frame)
                if fill >= 0.30 and back_offs < 12:
                    back_offs += 1
                    report(presentationNavigationBackOffs=back_offs, presentationSlidenPageFill=round(fill, 3))
                    self._walk_backward(0.30)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                screen, screen_source = self._observe_screen(frame)
                if screen is None:
                    sweep_steps += 1
                    report(presentationNavigationScanStep=sweep_steps)
                    if sweep_steps > 16 * len(framing_bands):
                        raise SlidenConsoleNotFound(
                            "the wall screen never came into view while looking for the panel bar"
                        )
                    self._set_pitch(framing_bands[(sweep_steps // 16) % len(framing_bands)])
                    self._scan_turn(0.16, 1)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                fraction = screen.width / width
                error_x = float(screen.center[0] - viewport[0])
                report(
                    presentationSlidenScreenBox=[screen.x, screen.y, screen.width, screen.height],
                    presentationSlidenScreenWidthFraction=round(fraction, 3),
                    presentationSlidenScreenSource=screen_source,
                )
                clipped = screen.x <= 4 or screen.right >= width - 4
                if (fraction > 0.55 or clipped) and back_offs < 12:
                    back_offs += 1
                    self._walk_backward(0.30)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if abs(error_x) > 0.15 * width and sweep_steps < 40:
                    sweep_steps += 1
                    self._move_relative((max(-45.0, min(45.0, 0.26 * error_x)), 0.0))
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                # Stand in front of the screen's middle, not off to one side:
                # from the left of the room the operation panel's bar hides
                # behind the furniture and the sub-screen fills the view.
                # The page's corners say which side we are on.
                skew = self._screen_skew(frame, screen)
                report(presentationSlidenScreenSkew=None if skew is None else round(skew, 3))
                if skew is not None and abs(skew) > 0.12 and lateral_moves < 10:
                    lateral_moves += 1
                    self._strafe(0.40, 1 if skew > 0 else -1)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                stage = "finding_bar"
                sweep_steps = 0
                continue

            if stage == "finding_bar":
                assert screen is not None
                fresh_screen, source = self._observe_screen(frame)
                if fresh_screen is not None:
                    screen, screen_source = fresh_screen, source
                    # Keep the screen roughly centred: the hunt below moves
                    # the avatar, and the flank test is measured from it.
                    error_x = float(screen.center[0] - viewport[0])
                    # While closing in on one side the screen is meant to sit
                    # off-centre; only keep it from leaving the frame.
                    if abs(error_x) > (0.45 if close_ins else 0.15) * width:
                        self._move_relative((max(-45.0, min(45.0, 0.26 * error_x)), 0.0))
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                pages = [
                    candidate.rect
                    for candidate in _display_candidates(frame, minimum_area_fraction=0.002)
                ]
                flanks = [
                    panel
                    for panel in detect_sliden_edge_panels(frame, screen)
                    if panel.side not in rejected
                    and not self._is_text_bar(panel.rect)
                    # A bright bar with a dark page sitting right on top of it
                    # is the progress bar under a face-on display -- the
                    # sub-screen seen from the front, typically -- not a
                    # panel seen edge-on.  There is nothing to walk around.
                    and not any(
                        page.bottom >= panel.rect.y - 2.0 * panel.rect.height
                        and page.bottom <= panel.rect.y + 0.5 * panel.rect.height
                        and min(page.right, panel.rect.right) - max(page.x, panel.rect.x)
                        >= 0.5 * panel.rect.width
                        for page in pages
                        if page != screen
                    )
                ]
                report(
                    presentationNavigationBarsSeen=[
                        [panel.side, round(panel.length), round(panel.gap)] for panel in flanks
                    ]
                )
                chosen = next((panel for panel in flanks if panel.side == prefer_side), None)
                if chosen is None and prefer_side not in rejected and side_hunts < len(self._SIDE_HUNT_MOVES):
                    # The wanted side's bar is edge-on from here and shows
                    # nothing at all from some spots -- the other side's
                    # bar is not a substitute for it.  From further back the
                    # whole wall is in view and the bar reads as a line beside
                    # the screen; so back away and step towards the wanted
                    # side, alternately, before settling for the other.
                    move = self._SIDE_HUNT_MOVES[side_hunts]
                    side_hunts += 1
                    report(presentationNavigationSideHunts=side_hunts, presentationNavigationSideHuntMove=move)
                    # "look" steps only observe again: the edge-on bar is a
                    # few pixels and drops out of single frames.
                    if move == "back":
                        self._walk_backward(0.40)
                    elif move == "side":
                        self._strafe(0.45, prefer_side)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if chosen is None and prefer_side not in rejected and close_ins < 8:
                    # Still nothing on the wanted side.  From across the room
                    # its bar merges into the bright furniture beside the
                    # screen; from a few steps closer it separates.  Walk
                    # towards where it must be -- just past the screen's
                    # edge on that side, at the screen's foot -- and look
                    # again, keeping the screen in view as the reference.
                    aim_x = screen.right + 0.25 * screen.width if prefer_side > 0 else screen.x - 0.25 * screen.width
                    aim_y = screen.bottom + 0.20 * screen.height
                    error_x = float(aim_x - viewport[0])
                    error_y = float(aim_y - viewport[1])
                    report(presentationNavigationCloseIns=close_ins)
                    if abs(error_x) > 0.06 * width and close_in_turns < 12:
                        close_in_turns += 1
                        self._move_relative((max(-60.0, min(60.0, 0.30 * error_x)), 0.0))
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    if abs(error_y) > 0.12 * height and close_in_turns < 12:
                        close_in_turns += 1
                        move_y = max(-40.0, min(40.0, 0.22 * error_y))
                        self._pitch_offset += move_y
                        self._move_relative((0.0, move_y))
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    # Only walks count as close-in steps; the aiming above
                    # is free, so the budget is spent getting closer.
                    close_ins += 1
                    close_in_turns = 0
                    self._walk_forward(0.45)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if chosen is None and flanks:
                    chosen = flanks[0]
                if chosen is None:
                    strafes += 1
                    if strafes > 6:
                        raise SlidenConsoleNotFound(
                            "no panel bar was seen beside the wall screen"
                        )
                    # A flat panel is invisible from some angles: change where
                    # we stand, alternating sides with growing amplitude.
                    direction = 1 if strafes % 2 else -1
                    for _ in range(1 + strafes // 2):
                        self._strafe(0.30, direction)
                        time.sleep(0.05)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                side = chosen.side
                bar_center = chosen.rect.center
                bar_length = chosen.length
                stage = "approaching_bar"
                approach_steps = 0
                lost = 0
                best_bottom = 0.0
                stall = 0
                report(presentationNavigationPanelBarWidthFraction=round(chosen.rect.width / width, 3))
                continue

            if stage == "approaching_bar":
                assert bar_center is not None
                # A tight window: the bar only moves as far per frame as one
                # yaw correction sweeps it.  A 0.40 window let the tracker
                # jump onto the sub-screen's progress bar half a frame away
                # and walk the avatar into the wrong corner.
                tracked = track_sliden_edge_panel(
                    frame, bar_center, bar_length, radius_fraction=0.18, scale_tolerance=1.7
                )
                if tracked is not None and self._is_text_bar(tracked[0]):
                    tracked = None
                if tracked is not None:
                    rect_t = tracked[0]
                    pages_t = [
                        c.rect for c in _display_candidates(frame, minimum_area_fraction=0.002)
                    ]
                    if any(
                        page.bottom >= rect_t.y - 2.0 * rect_t.height
                        and page.bottom <= rect_t.y + 0.5 * rect_t.height
                        and min(page.right, rect_t.right) - max(page.x, rect_t.x) >= 0.5 * rect_t.width
                        for page in pages_t
                    ):
                        # The progress bar under a face-on page, not our bar.
                        tracked = None
                if tracked is None and screen is not None:
                    # From close range the bar can also come out of the
                    # detector's own edge search when the screen is still up.
                    fresh_screen, _source = self._observe_screen(frame)
                    if fresh_screen is not None:
                        screen = fresh_screen
                        for panel in detect_sliden_edge_panels(frame, screen):
                            if panel.side == side:
                                tracked = (panel.rect, panel.length, panel.length * panel.openness)
                                break
                if tracked is None:
                    lost += 1
                    report(presentationNavigationPanelLostFrames=lost)
                    # No undoing of the last step here: the edge-on bar is
                    # a few pixels and drops out of every other frame, and
                    # undoing each step that "lost" it walked the avatar
                    # forward and back on the spot for a whole minute.
                    if lost <= 3:
                        # Most likely a flicker; look again where it was.
                        pass
                    elif lost <= 8:
                        # It has not moved: it slipped below the view as we
                        # closed in.  Look down for it, never sideways.
                        pitch_down += 30.0
                        self._move_relative((0.0, 30.0))
                    elif lost <= 10:
                        pitch_down -= 40.0
                        self._move_relative((0.0, -40.0))
                    else:
                        # Give the frame back and start over from the screen.
                        self._move_relative((0.0, -pitch_down))
                        pitch_down = 0.0
                        stage = "framing_screen"
                        sweep_steps = 0
                    time.sleep(max(0.12, self.poll_seconds))
                    continue
                lost = 0
                rect, bar_length, thickness = tracked
                bar_center = rect.center
                approach_steps += 1
                error_x = float(rect.center[0] - viewport[0])
                error_y = float(rect.center[1] - viewport[1])
                bottom_fraction = rect.bottom / height
                report(
                    presentationNavigationPanelBarWidthFraction=round(rect.width / width, 3),
                    presentationNavigationPanelBarBottomFraction=round(bottom_fraction, 3),
                    presentationNavigationPanelBarLength=round(bar_length),
                    presentationNavigationPanelOpenness=round(thickness / max(1.0, bar_length), 3),
                    presentationNavigationPanelPitchDown=round(pitch_down),
                    presentationNavigationPanelApproachSteps=approach_steps,
                )
                if bottom_fraction >= self._BAR_BOTTOM_FRACTION or (
                    pitch_down >= self._BAR_PITCH_LIMIT and error_y > 0.25 * height
                ):
                    stage = "crossing"
                    cross_steps = 0
                    continue
                if bottom_fraction > best_bottom + 0.01:
                    best_bottom = bottom_fraction
                    stall = 0
                else:
                    stall += 1
                if approach_steps > 90 or stall > 30:
                    # Not getting closer -- a piece of furniture is in the
                    # way.  Carry on with the manoeuvre from here rather than
                    # walking into the same obstacle until the timeout.
                    stage = "crossing"
                    cross_steps = 0
                    continue
                move_x = max(-45.0, min(45.0, 0.28 * error_x)) if abs(error_x) > 0.06 * width else 0.0
                move_y = 0.0
                if abs(error_y) > 0.10 * height:
                    move_y = max(-40.0, min(40.0, 0.25 * error_y))
                    move_y = max(-pitch_down, min(self._BAR_PITCH_LIMIT - pitch_down, move_y))
                if move_x != 0.0 or abs(move_y) >= 1.0:
                    pitch_down += move_y
                    self._pitch_offset = pitch_down
                    self._move_relative((move_x, move_y))
                    self._last_action = ("look", move_x, move_y)
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                self._walk_forward(0.40 if bar_length < 0.18 * width else 0.28)
                time.sleep(max(0.18, self.poll_seconds))
                continue

            if stage == "crossing":
                cross_steps += 1
                report(presentationNavigationCrossSteps=cross_steps)
                self._walk_forward(0.35)
                time.sleep(max(0.18, self.poll_seconds))
                if cross_steps < 2:
                    continue
                # About-face.  The panel is now behind, beside and below the
                # avatar, at waist height a step or two away: about 40
                # degrees under the horizon.  Level the camera absolutely
                # (the approach's pitch bookkeeping is not trusted for this),
                # turn exactly half a revolution by mouse -- the OSC turn
                # rate drifts with the frame rate, and a live run turned a
                # quarter where 180 was meant -- and look down that far; the
                # page itself trims the rest.
                pitch_down = 0.0
                self.level_camera()
                self._move_relative((self._YAW_COUNTS_PER_360 / 2.0, 0.0))
                time.sleep(0.15)
                face_pitch = self._FACE_PITCH_COUNTS
                self._move_relative((0.0, face_pitch))
                self._pitch_offset = face_pitch
                stage = "facing_panel"
                face_steps = 0
                face_pitch_nudges = 0
                confirm_frames = 0
                face_back_offs = 0
                face_last_width = float(width)
                face_last_center = None
                face_relook = 0
                time.sleep(max(0.25, self.poll_seconds))
                continue

            if stage == "facing_panel":
                observation = detect_sliden_console(frame)
                page = detect_sliden_panel_page(frame, minimum_score=0.34)
                target: Rect | None = None
                if observation.slide_quad is not None:
                    quad = observation.slide_quad
                    target = Rect(
                        int(quad[:, 0].min()), int(quad[:, 1].min()),
                        int(quad[:, 0].max() - quad[:, 0].min()),
                        int(quad[:, 1].max() - quad[:, 1].min()),
                    )
                elif page is not None:
                    target = page[0]
                else:
                    # A bare page below eye level and not filling the frame:
                    # the wall screen, if it is in view at all after the
                    # turn, is large and sits at or above the horizon.
                    lower = [
                        candidate for candidate in _display_candidates(frame)
                        if 0.01 <= candidate.area_fraction <= 0.30
                        and candidate.rect.center[1] >= viewport[1] - 0.05 * height
                        and candidate.rect.y > 4
                    ]
                    if lower:
                        target = min(
                            lower,
                            key=lambda c: math.hypot(
                                c.rect.center[0] - viewport[0], c.rect.center[1] - viewport[1]
                            ),
                        ).rect
                    else:
                        # The panel's own bright bars -- URL field above,
                        # progress bar below -- are visible when its dark
                        # page is not segmented (glare, partial view).  Steer
                        # to the biggest one in the lower part of the view.
                        low_bars = [
                            (length, rect)
                            for rect, length, _thickness in _bright_horizontal_bars(frame)
                            if rect.center[1] >= 0.35 * height and length >= 0.08 * width
                        ]
                        if low_bars:
                            target = max(low_bars, key=lambda item: item[0])[1]
                if self.diagnostic_directory is not None and face_steps < 12:
                    self._save_diagnostic("facing", frame, None, None, None)
                report(
                    presentationConsoleConfidence=round(float(observation.confidence), 3),
                    presentationNavigationFaceSteps=face_steps,
                    presentationConsoleWidthFraction=None if target is None else round(target.width / width, 3),
                    presentationNavigationFaceTarget=None if target is None else [target.x, target.y, target.width, target.height],
                )
                if target is None:
                    face_steps += 1
                    if face_steps > 40:
                        rejected.add(side)
                        raise SlidenConsoleNotFound(
                            "turned about beside the bar but no Sliden page came into view"
                        )
                    if face_last_center is not None and face_relook < 3:
                        # It was in view a moment ago and left through one
                        # edge; look that way first rather than fanning
                        # symmetrically about a heading it is not on.
                        face_relook += 1
                        dx = float(face_last_center[0] - viewport[0])
                        dy = float(face_last_center[1] - viewport[1])
                        move = (max(-60.0, min(60.0, 0.5 * dx)), max(-60.0, min(60.0, 0.5 * dy)))
                        self._pitch_offset += move[1]
                        self._move_relative(move)
                        time.sleep(max(0.15, self.poll_seconds))
                        continue
                    # The yaw after the about-face is roughly right; the
                    # pitch is the unknown.  Alternate small yaw wiggles
                    # with visits to a ladder of pitches: a little lower
                    # first (the panel at the feet), then up towards level
                    # (a page mounted higher, or the sub-screen on its wall).
                    if face_steps % 3 == 0 and face_pitch_nudges < len(self._FACE_PITCH_LADDER):
                        wanted = self._FACE_PITCH_LADDER[face_pitch_nudges]
                        face_pitch_nudges += 1
                        delta = wanted - self._pitch_offset
                        self._move_relative((0.0, delta))
                        self._pitch_offset = wanted
                    else:
                        self._scan_turn(0.10, 1 if face_steps % 6 < 3 else -1)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                error_x = float(target.center[0] - viewport[0])
                error_y = float(target.center[1] - viewport[1])
                # Stand back far enough to drive the panel from here: aiming
                # at the Next button lifts the URL bar towards the top of the
                # capture, and from arm's length it leaves the frame -- with
                # it goes the console's identity and the click is refused
                # (measured 2026-08-15: 490 px of panel against 360 px of
                # headroom).  A page under ~0.32 of the view keeps both.
                face_last_center = target.center
                face_relook = 0
                too_close = target.width > self._ARRIVAL_MAX_WIDTH_FRACTION * width or (
                    observation.url_field_rect is not None
                    and observation.url_field_rect.y < 0.10 * height
                )
                if too_close and face_back_offs >= self._FACE_BACK_OFF_LIMIT:
                    # Cannot get further away -- the sofa is behind.  With
                    # the URL field in view the panel is still drivable
                    # (the aim steps back again before Next); stopping here
                    # beats spending the whole budget walking into a sofa.
                    if observation.url_field is not None and observation.slide_quad is not None:
                        report(presentationNavigationCloseArrival=True)
                        too_close = False
                if too_close:
                    face_back_offs += 1
                    stuck = (
                        face_back_offs >= 3
                        and target.width >= face_last_width - 0.01 * width
                    )
                    face_last_width = float(target.width)
                    report(
                        presentationNavigationStatus="backing_off",
                        presentationNavigationBackOffs=face_back_offs,
                        presentationNavigationBackOffStuck=stuck,
                    )
                    if stuck:
                        # Backing straight up gained nothing: something is
                        # behind.  Sidestep, alternating, and back off again
                        # from there.
                        self._strafe(0.40, 1 if face_back_offs % 2 else -1)
                    self._walk_backward(0.24)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if abs(error_x) > 0.08 * width or abs(error_y) > 0.14 * height:
                    move_y = max(-40.0, min(40.0, 0.22 * error_y))
                    self._pitch_offset += move_y
                    self._move_relative((max(-45.0, min(45.0, 0.28 * error_x)), move_y))
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                if observation.url_field is not None and observation.slide_quad is not None:
                    _state(
                        self.state_callback,
                        presentationPhase="at_sliden_console",
                        presentationNavigationStatus="aligned",
                        presentationConsoleDistanceReady=True,
                    )
                    return observation
                confirm_frames += 1
                if confirm_frames in (4, 8):
                    # Centred on a page without a URL field.  Before calling
                    # it the sub-screen, make sure it is not just the top of
                    # the panel cut off by the capture: step back once.
                    self._walk_backward(0.22)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if confirm_frames > 12:
                    # Centred on a Sliden page with no URL field: the
                    # sub-screen.  Its side is wrong; leave for the other one.
                    rejected.add(side)
                    report(presentationNavigationCandidateRejected=True)
                    raise SlidenConsoleNotFound(
                        f"the panel on side {side} shows no URL field; it is the sub-screen"
                    )
                time.sleep(max(0.15, self.poll_seconds))
                continue

        raise SlidenConsoleNotFound(
            f"timed out ({stage}) while approaching the operation panel by its bar"
        )

    def locate_and_approach(
        self,
        *,
        timeout_seconds: float = 120.0,
        # Measured against the live room: the panel stops growing at 0.19-0.25
        # of the frame width because furniture blocks the last step.  Demanding
        # more kept the approach walking into a sofa until the timeout.
        target_width_fraction: float = 0.18,
        # Identity is settled by the URL field the detector insists on, so the
        # score only has to clear a plausibility bar.  A run that reached the
        # panel at 0.61 used to be thrown away by a 0.62 gate.
        minimum_confidence: float = 0.55,
        approach_confidence: float = 0.33,
        screen_confidence: float = 0.55,
        screen_target_width_fraction: float = 0.30,
        #: Standing any closer hides the flanking bars outside the capture.
        screen_maximum_width_fraction: float = 0.55,
        #: The flanking bars are only readable once the screen is genuinely in
        #: front of the avatar; measuring them against a distant or off-centre
        #: screen picks bars belonging to something else entirely.
        bar_reading_width_fraction: float = 0.20,
        full_turn_seconds: float = 2.2,
        respawn_first: bool = False,
        require_wordmark: bool = True,
    ) -> SlidenConsoleObservation:
        """Search the room, approach Sliden, and stop in front of its panel.

        Four phases, all re-entered from observations rather than latched:
        find the wall screen, walk to it, search the panel around it, then
        approach the panel.  Every phase can fall back to the previous one, so
        a wrong guess costs one sweep instead of the whole run.
        """

        if timeout_seconds <= 0.0:
            raise ValueError("navigation timeout must be positive")
        if not 0.10 <= target_width_fraction <= 0.60:
            raise ValueError("target console width fraction must be between 0.10 and 0.60")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum console confidence must be between 0 and 1")
        if not 0.0 <= approach_confidence <= minimum_confidence:
            raise ValueError("approach confidence must be between 0 and the minimum")
        if full_turn_seconds <= 0.0:
            raise ValueError("full turn duration must be positive")

        if respawn_first:
            _state(
                self.state_callback,
                presentationPhase="respawning",
                presentationNavigationStatus="respawning",
            )
            _state(
                self.state_callback,
                presentationNavigationRespawned=self.respawn(),
            )
        deadline = time.monotonic() + timeout_seconds
        # The owner's recipe first: wall screen -> flanking bar -> walk at it
        # looking down -> two steps past -> about-face.  A side whose panel
        # shows no URL field is the sub-screen and the other side is tried;
        # only when the recipe cannot even find a bar does the older
        # room-wide search below get the remaining budget.
        rejected_sides: set[int] = set()
        recipe_error: str | None = None
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 5.0:
                break
            try:
                return self.approach_operation_panel(
                    timeout_seconds=min(remaining, 150.0),
                    prefer_side=1,
                    rejected_sides=rejected_sides,
                )
            except SlidenConsoleNotFound as exc:
                recipe_error = str(exc)
                _state(
                    self.state_callback,
                    presentationNavigationRecipeAttempt=attempt + 1,
                    presentationNavigationRecipeError=recipe_error,
                )
                if not rejected_sides:
                    break
        if recipe_error is not None and "never came into view" not in recipe_error:
            # The recipe found the wall screen and still failed.  The older
            # room-wide search below has nothing better to offer from here
            # -- it spent its whole budget turning on the spot in live runs
            # -- so stop with the recipe's own account of what went wrong.
            raise SlidenConsoleNotFound(recipe_error)
        phase = "searching_screen"
        scan_step = 0
        sweep_seconds = 0.0
        sweep_index = 0
        relocation_index = 0
        orbit_index = 0
        lost_frames = 0
        rejected_frames = 0
        stall_count = 0
        align_attempts = 0
        best_seen_width = 0.0
        best_confidence = 0.0
        best_screen_confidence = 0.0
        best_width = 0.0
        screen_seen = False
        console_seen = False
        tracked_center: Point | None = None
        tracked_width = 0.0
        edge_center: Point | None = None
        edge_width = 0.0
        edge_side = 0
        edge_direction = -1
        edge_best_height = 0.0
        edge_best_openness = 0.0
        edge_stage = "approach"
        edge_stage_steps = 0
        edge_stall = 0
        edge_attempts = 0
        edge_rejected: set[int] = set()
        screen_anchor: Rect | None = None
        screen_stalls = 0
        framing_attempts = 0

        def begin(next_phase: str) -> None:
            # Deliberately does not reset the sweep: re-entering the search
            # after a brief detour used to discard the turn already made, so
            # the avatar span at one pitch band for ever without completing a
            # revolution or moving on to the next band.
            nonlocal phase, lost_frames, rejected_frames
            nonlocal stall_count, best_seen_width, align_attempts
            phase = next_phase
            lost_frames = 0
            rejected_frames = 0
            stall_count = 0
            align_attempts = 0
            best_seen_width = 0.0

        def sweep(bands: Sequence[float], scan_seconds: float) -> bool:
            """Turn one search tile.  Returns True when a full turn finished.

            A revolution is the most that turning alone can ever reveal, so the
            band -- and with it the pitch -- always changes at that point.  The
            caller must move or look elsewhere; carrying on round a second time
            cannot show anything the first pass did not.
            """

            nonlocal sweep_seconds, sweep_index
            if sweep_seconds + scan_seconds > full_turn_seconds:
                sweep_seconds = 0.0
                sweep_index += 1
                self._set_pitch(bands[sweep_index % len(bands)])
                return True
            self._set_pitch(bands[sweep_index % len(bands)])
            self._scan_right(scan_seconds)
            sweep_seconds += scan_seconds
            return False

        while time.monotonic() < deadline:
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is None:
                time.sleep(self.poll_seconds)
                continue
            height, width = frame.shape[:2]
            viewport_center = self._viewport_center(frame)
            observation = detect_sliden_console(frame)
            best_confidence = max(best_confidence, observation.confidence)
            if observation.modal_open:
                _state(
                    self.state_callback,
                    presentationPhase="at_sliden_console",
                    presentationNavigationStatus="aligned",
                    presentationConsoleDistanceReady=True,
                )
                return observation

            width_fraction = self._console_width_fraction(observation, width)
            best_width = max(best_width, width_fraction)
            usable = (
                observation.slide_quad is not None
                and observation.confidence >= approach_confidence
            )
            if require_wordmark and usable and observation.slide_quad is not None:
                # Navigation happens before any slide is loaded, so the panel
                # is showing its idle page and must carry the wordmark.  A
                # sunlit patch of the gallery floor otherwise satisfied every
                # geometric test and the search "arrived" at it in 10 seconds.
                quad_x = observation.slide_quad[:, 0]
                quad_y = observation.slide_quad[:, 1]
                left = max(0, int(quad_x.min()))
                top = max(0, int(quad_y.min()))
                right = min(width, int(quad_x.max()))
                bottom = min(height, int(quad_y.max()))
                usable = right - left >= 8 and bottom - top >= 8 and (
                    sliden_wordmark_score(frame[top:bottom, left:right]) >= 0.34
                )
            if usable:
                console_seen = True
                if phase != "approaching_console":
                    begin("approaching_console")
                quad_center = np.mean(observation.slide_quad, axis=0)
                quad_height = float(
                    observation.slide_quad[:, 1].max()
                    - observation.slide_quad[:, 1].min()
                )
                error_x = float(quad_center[0] - viewport_center[0])
                error_y = float(quad_center[1] - viewport_center[1])
                tracked_center = (float(quad_center[0]), float(quad_center[1]))
                tracked_width = float(
                    observation.slide_quad[:, 0].max()
                    - observation.slide_quad[:, 0].min()
                )
                _state(
                    self.state_callback,
                    presentationConsoleConfidence=observation.confidence,
                    presentationConsoleWidthFraction=width_fraction,
                    presentationConsoleCenterErrorX=error_x,
                    presentationNavigationScanStep=scan_step,
                    presentationNavigationCandidateRejected=False,
                )
                if (
                    width_fraction >= target_width_fraction
                    and observation.confidence >= minimum_confidence
                ):
                    # Arrival is judged before alignment on purpose.  The
                    # console controller re-aims precisely with its own visual
                    # servo, so being close and confident is the whole job
                    # here; testing alignment first let the approach reach the
                    # panel and then lose it while still nudging the camera.
                    _state(
                        self.state_callback,
                        presentationPhase="at_sliden_console",
                        presentationNavigationStatus="aligned",
                        presentationConsoleDistanceReady=True,
                        presentationNavigationForwardSeconds=0.0,
                    )
                    return observation
                # Yaw has to be tight because the avatar walks where it looks.
                # Pitch only has to keep the panel on screen, so its dead band
                # is scaled to the panel itself: a fixed wide band let a panel
                # mounted above eye level climb out of the top of the frame as
                # the avatar closed in, losing the lock a step before arriving.
                vertical_tolerance = max(0.10 * height, 0.70 * quad_height)
                if abs(error_x) > 0.06 * width or abs(error_y) > vertical_tolerance:
                    if align_attempts >= 12:
                        # Aiming is not converging -- two candidates are most
                        # likely trading places.  Break the tie by turning away
                        # and letting the sweep re-acquire from a clean state.
                        self._scan_right(0.22)
                        begin("searching_console")
                        continue
                    align_attempts += 1
                    _state(
                        self.state_callback,
                        presentationPhase="aligning_sliden",
                        presentationNavigationStatus="aligning",
                        presentationConsoleDistanceReady=False,
                    )
                    move_y = self._pitch_step(max(-30.0, min(30.0, 0.20 * error_y)))
                    self._move_relative(
                        (max(-45.0, min(45.0, 0.28 * error_x)), move_y)
                    )
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                align_attempts = 0
                if width_fraction >= target_width_fraction:
                    # Close enough to read, yet still not convincing: this is
                    # something else that merely looks like a panel.  Turn away
                    # so the next sweep cannot immediately re-acquire it.
                    rejected_frames += 1
                    if rejected_frames >= 6:
                        _state(
                            self.state_callback,
                            presentationNavigationCandidateRejected=True,
                        )
                        self._scan_right(0.22)
                        begin("searching_console")
                    time.sleep(self.poll_seconds)
                    continue
                if width_fraction <= best_seen_width + 0.002:
                    stall_count += 1
                else:
                    stall_count = 0
                best_seen_width = max(best_seen_width, width_fraction)
                if stall_count >= 4:
                    if (
                        width_fraction >= 0.60 * target_width_fraction
                        and observation.confidence >= minimum_confidence
                    ):
                        # As close as the avatar can physically get: a panel
                        # mounted on a wall or behind furniture stops growing
                        # well before the nominal target.  Aimed and confident
                        # is what the console controller actually needs.
                        _state(
                            self.state_callback,
                            presentationPhase="at_sliden_console",
                            presentationNavigationStatus="aligned",
                            presentationConsoleDistanceReady=True,
                            presentationNavigationForwardSeconds=0.0,
                        )
                        return observation
                    # Blocked by furniture, or the panel is unreachable from
                    # this side.  Step aside and resume the local search.
                    self._strafe(0.35, 1 if orbit_index % 2 else -1)
                    orbit_index += 1
                    begin("searching_console")
                    continue
                pulse = (
                    0.50
                    if width_fraction < 0.10
                    else (0.30 if width_fraction < 0.18 else 0.16)
                )
                _state(
                    self.state_callback,
                    presentationPhase="approaching_sliden",
                    presentationNavigationStatus="approaching",
                    presentationConsoleDistanceReady=False,
                    presentationNavigationForwardSeconds=pulse,
                )
                self._walk_forward(pulse)
                time.sleep(max(0.18, self.poll_seconds))
                continue

            if phase == "approaching_console":
                # Recognition of a distant panel is intermittent: its URL strip
                # is only a few pixels tall, so single frames drop out.  Keep
                # driving off the panel's page, whose identity was already
                # established, instead of giving up the lock and turning away
                # every time the approach nearly arrives.
                tracked = (
                    track_sliden_display(frame, tracked_center, tracked_width)
                    if tracked_center is not None
                    else None
                )
                if tracked is not None:
                    lost_frames = 0
                    tracked_center = tracked.center
                    tracked_width = float(tracked.width)
                    tracked_fraction = tracked.width / width
                    error_x = float(tracked.center[0] - viewport_center[0])
                    error_y = float(tracked.center[1] - viewport_center[1])
                    _state(
                        self.state_callback,
                        presentationPhase="approaching_sliden",
                        presentationNavigationStatus="tracking",
                        presentationConsoleWidthFraction=tracked_fraction,
                        presentationConsoleCenterErrorX=error_x,
                    )
                    if tracked_fraction >= target_width_fraction and console_seen:
                        _state(
                            self.state_callback,
                            presentationPhase="at_sliden_console",
                            presentationNavigationStatus="aligned",
                            presentationConsoleDistanceReady=True,
                        )
                        return observation
                    tolerance_y = max(0.10 * height, 0.70 * tracked.height)
                    if abs(error_x) > 0.06 * width or abs(error_y) > tolerance_y:
                        move_y = self._pitch_step(
                            max(-30.0, min(30.0, 0.20 * error_y))
                        )
                        self._move_relative(
                            (max(-45.0, min(45.0, 0.28 * error_x)), move_y)
                        )
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    self._walk_forward(
                        0.50
                        if tracked_fraction < 0.10
                        else (0.30 if tracked_fraction < 0.18 else 0.16)
                    )
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                lost_frames += 1
                if lost_frames <= 8:
                    time.sleep(self.poll_seconds)
                    continue
                tracked_center = None
                begin("searching_console")

            screen = detect_sliden_screen(frame)
            best_screen_confidence = max(best_screen_confidence, screen.confidence)
            screen_ready = (
                screen.rect is not None and screen.confidence >= screen_confidence
            )
            if screen_anchor is not None:
                # Stay on the display the approach committed to.  Every Sliden
                # surface in this room renders the same page, so re-picking the
                # best-scoring one each frame let a nearby sub-screen take over
                # mid-walk; the approach then stopped growing, called itself
                # arrived at a screen filling 9% of the view, and read the
                # flanking bars off the wrong object.
                anchored = track_sliden_display(
                    frame,
                    screen_anchor.center,
                    float(screen_anchor.width),
                    radius_fraction=0.30,
                    scale_tolerance=2.0,
                )
                if anchored is not None:
                    screen = SlidenScreenObservation(
                        rect=anchored,
                        confidence=max(screen.confidence, screen_confidence),
                    )
                    screen_ready = True
            if screen_ready:
                screen_seen = True
                assert screen.rect is not None
                screen_width_fraction = screen.rect.width / width
                _state(
                    self.state_callback,
                    presentationSlidenScreenConfidence=screen.confidence,
                    presentationSlidenScreenBox=[
                        screen.rect.x,
                        screen.rect.y,
                        screen.rect.width,
                        screen.rect.height,
                    ],
                    presentationSlidenScreenWidthFraction=screen_width_fraction,
                )

            if phase == "approaching_edge_panel":
                assert edge_center is not None
                if edge_stage == "turning":
                    # Deliberately before the bar lookup.  Having walked past
                    # the bar and lowered the view, it is out of frame by
                    # design; treating that as "lost" sent the run into
                    # recovery and the about-face never happened.
                    # Looser here: standing beside the panel there is nothing
                    # else it could be, and a page seen steeply from above and
                    # to one side never matches the wordmark as cleanly as a
                    # head-on one does.
                    page = detect_sliden_panel_page(frame, minimum_score=0.34)
                    if page is not None:
                        # The panel is in front of the avatar.  Stop turning:
                        # waiting for every console feature to resolve kept the
                        # avatar spinning past a panel that was plainly there.
                        page_rect, page_quad = page
                        page_error_x = float(
                            page_rect.center[0] - viewport_center[0]
                        )
                        page_error_y = float(
                            page_rect.center[1] - viewport_center[1]
                        )
                        _state(
                            self.state_callback,
                            presentationPhase="at_sliden_console",
                            presentationNavigationStatus="panel_in_view",
                            presentationConsoleWidthFraction=page_rect.width / width,
                        )
                        if (
                            abs(page_error_x) > 0.10 * width
                            or abs(page_error_y) > 0.18 * height
                        ) and edge_stage_steps < 14:
                            edge_stage_steps += 1
                            move_y = self._pitch_step(
                                max(-30.0, min(30.0, 0.22 * page_error_y))
                            )
                            self._move_relative(
                                (
                                    max(-45.0, min(45.0, 0.30 * page_error_x)),
                                    move_y,
                                )
                            )
                            time.sleep(max(0.10, self.poll_seconds))
                            continue
                        if page_rect.width > 0.75 * width:
                            self._walk_backward(0.24)
                            time.sleep(max(0.18, self.poll_seconds))
                            continue
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="aligned",
                            presentationConsoleDistanceReady=True,
                        )
                        if observation.slide_quad is not None:
                            return observation
                        return SlidenConsoleObservation(
                            slide_quad=page_quad,
                            home_button=_quad_point(page_quad, 0.035, 0.04),
                            url_field=_quad_point(page_quad, 0.55, -0.08),
                            previous_button=_quad_point(page_quad, 0.34, 1.03),
                            next_button=_quad_point(page_quad, 0.72, 1.03),
                            confidence=minimum_confidence,
                        )
                    edge_stage_steps += 1
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="turning_to_panel",
                        presentationNavigationPanelTurnStep=edge_stage_steps,
                    )
                    # Half tiles: at 45 degrees per observation the panel can
                    # pass between two frames unseen.  0.12 s is about 22
                    # degrees, so sixteen steps is one revolution -- the point
                    # past which turning is pointless.
                    self._scan_right(0.12)
                    if edge_stage_steps >= 16:
                        edge_rejected.add(edge_side)
                        edge_center = None
                        begin("searching_console")
                    time.sleep(max(0.18, self.poll_seconds))
                    continue

                bar: Rect | None = None
                bar_length = 0.0
                bar_openness = 0.0
                if screen_ready and screen.rect is not None:
                    flank = next(
                        (
                            panel
                            for panel in detect_sliden_edge_panels(frame, screen.rect)
                            if panel.side == edge_side
                        ),
                        None,
                    )
                    if flank is not None:
                        bar = flank.rect
                        bar_length = flank.length
                        bar_openness = flank.openness
                if bar is None:
                    # Walking at the bar takes the screen out of frame, so the
                    # bar has to be trackable on its own.  The window has to be
                    # wide: one side step moves the bar about 335 px across the
                    # capture, and a 288 px window lost it on the first step of
                    # every orbit and never found it again.
                    tracked_bar = track_sliden_edge_panel(
                        frame,
                        edge_center,
                        edge_width,
                        radius_fraction=0.35,
                    )
                    if tracked_bar is not None:
                        bar, bar_length, thickness = tracked_bar
                        bar_openness = thickness / max(1.0, bar_length)
                if bar is None:
                    # The bar is part of the room: it cannot have gone
                    # anywhere, only out of frame or behind something.  Look
                    # back where it was instead of abandoning the target and
                    # restarting the search from scratch.
                    lost_frames += 1
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="recovering_panel_bar",
                        presentationNavigationPanelLostFrames=lost_frames,
                    )
                    if lost_frames == 1 and self._undo_last_action():
                        # The bar is part of the room and has not moved, so
                        # reversing the step that lost it restores the view
                        # that had it.
                        pass
                    elif lost_frames <= 2:
                        self._move_relative(
                            (
                                max(-60.0, min(60.0, 0.35 * (edge_center[0] - viewport_center[0]))),
                                max(-40.0, min(40.0, 0.30 * (edge_center[1] - viewport_center[1]))),
                            )
                        )
                    elif lost_frames <= 14:
                        # The panel hangs low, and closing on it walks the view
                        # over the top of it.  Look down to find it again --
                        # turning cannot help, the bar has not moved.
                        self._move_relative((0.0, self._pitch_step(24.0)))
                    elif lost_frames <= 20:
                        # Only then a narrow left-right check, well under a
                        # quarter turn in total.
                        sign = 1.0 if lost_frames % 2 else -1.0
                        self._move_relative((sign * 26.0, 0.0))
                    else:
                        begin("searching_console")
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                lost_frames = 0
                edge_center = bar.center
                edge_width = bar_length
                edge_attempts += 1
                bar_fraction = bar.width / width
                error_x = float(bar.center[0] - viewport_center[0])
                error_y = float(bar.center[1] - viewport_center[1])
                _state(
                    self.state_callback,
                    presentationPhase="approaching_sliden_panel",
                    presentationNavigationStatus="edge_panel",
                    presentationNavigationPanelSide=edge_side,
                    presentationNavigationPanelBarWidthFraction=bar_fraction,
                    presentationNavigationPanelBarHeight=bar.height,
                    presentationNavigationPanelOpenness=round(bar_openness, 3),
                    presentationNavigationPanelOrbitDirection=edge_direction,
                )
                openness = bar_openness
                if openness > edge_best_openness + 0.01:
                    edge_best_openness = openness
                if edge_attempts > 60:
                    # This flank stopped opening: it is not the panel, or it
                    # cannot be reached from here.  Give the other side a turn,
                    # and only here is it right to forget the remembered bar.
                    edge_rejected.add(edge_side)
                    edge_center = None
                    begin("searching_console")
                    continue
                # Only yaw decides where the avatar walks.  Pitch was being
                # corrected every step to keep the bar centred, and because the
                # bar sits below eye level that walked the view steadily down
                # until a bench edge or a rug highlight captured the tracker
                # and the real bar was gone.  Touch pitch only to stop the bar
                # falling out of frame, and never let it accumulate.
                bar_leaving_frame = abs(error_y) > 0.34 * height
                if abs(error_x) > 0.07 * width or bar_leaving_frame:
                    move_y = 0.0
                    if bar_leaving_frame:
                        move_y = self._pitch_step(
                            max(-25.0, min(25.0, 0.18 * error_y))
                        )
                    self._move_relative(
                        (max(-45.0, min(45.0, 0.28 * error_x)), move_y)
                    )
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                # The panel's face points away from the wall screen, so no
                # amount of orbiting from this side brings it round.  Walk onto
                # the bar, carry on a little past it, then turn about while
                # looking down -- the panel is then facing the avatar.
                if edge_stage == "approach":
                    # Measured: the bar tops out near 0.21 of the frame width
                    # before the avatar is against it, so a 0.34 threshold was
                    # never reached and the manoeuvre never started.  Move on
                    # when it stops growing as well.
                    if bar_length > edge_best_height + 2.0:
                        edge_best_height = bar_length
                        edge_stall = 0
                    else:
                        edge_stall += 1
                    # Aiming can keep the walk from ever running, so the
                    # manoeuvre also starts on a plain step budget: it is a
                    # known-good sequence and is worth attempting from roughly
                    # the right place rather than never at all.
                    if edge_attempts >= 25:
                        edge_stall = 5
                    if bar_length < 0.20 * width and edge_stall < 5:
                        if not screen_ready and edge_attempts > 16:
                            # Walking blind at a bright line the wall screen can
                            # no longer vouch for is how the avatar ended up in
                            # a corner chasing a light fixture.  Re-anchor.
                            begin("searching_console")
                            continue
                        self._walk_forward(
                            0.40 if bar_length < 0.18 * width else 0.22
                        )
                        time.sleep(max(0.18, self.poll_seconds))
                        continue
                    edge_stage = "crossing"
                    edge_stage_steps = 0
                if edge_stage == "crossing":
                    edge_stage_steps += 1
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="crossing_panel",
                    )
                    self._walk_forward(0.30)
                    if edge_stage_steps >= 2:
                        edge_stage = "turning"
                        edge_stage_steps = 0
                        # Look down at the bar before turning: the panel lies
                        # below eye level from this close.  Commanded outright
                        # rather than through the pitch bookkeeping, which had
                        # drifted to roughly this value already and so made the
                        # move a no-op -- recorded frames showed the about-face
                        # happening at eye level with the panel below the view.
                        self._move_relative((0.0, 110.0))
                        self._pitch_offset = 110.0
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                # Every stage returns above; nothing else to do here.
                time.sleep(self.poll_seconds)
                continue

            if phase == "searching_screen" and screen_anchor is not None:
                # A stale anchor from an abandoned approach must not keep
                # forcing a screen into the observation once the search has
                # restarted somewhere else in the room.
                screen_anchor = None

            if phase in {"searching_screen", "approaching_screen"} and screen_ready:
                assert screen.rect is not None
                if phase != "approaching_screen":
                    begin("approaching_screen")
                    screen_stalls = 0
                screen_anchor = screen.rect
                screen_center = screen.rect.center
                screen_error_x = float(screen_center[0] - viewport_center[0])
                screen_error_y = float(screen_center[1] - viewport_center[1])
                screen_width_fraction = screen.rect.width / width
                _state(
                    self.state_callback,
                    presentationPhase="approaching_sliden_screen",
                    presentationNavigationStatus="landmark",
                    presentationSlidenScreenCenterErrorX=screen_error_x,
                    presentationSlidenScreenCenterErrorY=screen_error_y,
                    presentationSlidenLandmarkReady=False,
                )
                if screen_width_fraction > screen_maximum_width_fraction:
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="backing_off",
                    )
                    self._walk_backward(0.28)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                if screen_width_fraction >= screen_target_width_fraction:
                    # Distance is what this phase exists for.  Testing it before
                    # yaw alignment matters: this room shows several Sliden
                    # displays at once, and aiming first let the detector
                    # alternate between two of them forever without ever moving
                    # on.  Pitch is still worth correcting, because the screen
                    # is the only absolute reference for a level view and the
                    # panel sweep that follows is measured from this attitude.
                    if abs(screen_error_y) > 0.14 * height and align_attempts < 8:
                        align_attempts += 1
                        move_y = self._pitch_step(
                            max(-55.0, min(55.0, 0.22 * screen_error_y))
                        )
                        self._move_relative((0.0, move_y))
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    _state(
                        self.state_callback,
                        presentationSlidenLandmarkReady=True,
                        presentationNavigationRepositioned=True,
                    )
                    # The camera is now centred on the wall screen, which is
                    # the most reliable level reference this room offers.  Make
                    # that the zero the panel sweep measures its bands from.
                    self._pitch_offset = 0.0
                    begin("searching_console")
                    framing_attempts = 0
                    continue
                # Centre the landmark on both axes.  This also restores a level
                # view when an earlier run left the camera pointing at the
                # floor; a yaw-only scan would otherwise circle over texture.
                move_x = (
                    max(-55.0, min(55.0, 0.24 * screen_error_x))
                    if abs(screen_error_x) > 0.08 * width
                    else 0.0
                )
                move_y = (
                    max(-55.0, min(55.0, 0.22 * screen_error_y))
                    if abs(screen_error_y) > 0.08 * height
                    else 0.0
                )
                if (move_x != 0.0 or move_y != 0.0) and align_attempts < 12:
                    align_attempts += 1
                    move_y = self._pitch_step(move_y)
                    self._move_relative((move_x, move_y))
                    time.sleep(max(0.10, self.poll_seconds))
                    continue
                align_attempts = 0
                if screen_width_fraction <= best_seen_width + 0.002:
                    stall_count += 1
                else:
                    stall_count = 0
                best_seen_width = max(best_seen_width, screen_width_fraction)
                if stall_count >= 4:
                    # Walking is getting nowhere.  Stepping aside and trying
                    # again is the answer, not declaring the landmark reached:
                    # calling a screen 9% of the view "arrived at" is what put
                    # the flanking-bar geometry on the wrong object.
                    stall_count = 0
                    best_seen_width = screen_width_fraction
                    screen_stalls += 1
                    if screen_stalls > 4:
                        screen_anchor = None
                        begin("searching_screen")
                        continue
                    self._strafe(0.40, 1 if screen_stalls % 2 else -1)
                    time.sleep(max(0.18, self.poll_seconds))
                    continue
                self._walk_forward(0.42 if screen_width_fraction < 0.18 else 0.22)
                time.sleep(max(0.18, self.poll_seconds))
                continue

            scan_step += 1
            if phase == "searching_console":
                if edge_center is None:
                    # A bar in the capture is enough to act on.  Requiring the
                    # wall screen to be squarely framed first made the search
                    # turn straight past bars that were plainly in view.
                    fresh = detect_sliden_bars(frame)
                    if fresh:
                        fresh_rect, fresh_length, _fresh_thickness = fresh[0]
                        fresh_side = (
                            1
                            if screen.rect is None
                            or fresh_rect.center[0] >= screen.rect.center[0]
                            else -1
                        )
                        if fresh_side not in edge_rejected:
                            begin("approaching_edge_panel")
                            edge_side = fresh_side
                            edge_center = fresh_rect.center
                            edge_width = fresh_length
                            edge_best_height = 0.0
                            edge_best_openness = 0.0
                            edge_stage = "approach"
                            edge_stage_steps = 0
                            edge_stall = 0
                            edge_attempts = 0
                            _state(
                                self.state_callback,
                                presentationNavigationStatus="edge_panel",
                                presentationNavigationPanelSide=edge_side,
                            )
                            continue
                if edge_center is not None and edge_side not in edge_rejected:
                    # The bar is still exactly where it was.  If a sweep step
                    # brings it back into view, resume the approach instead of
                    # rediscovering it through the wall screen.
                    resumed = track_sliden_edge_panel(
                        frame, edge_center, edge_width, radius_fraction=0.50
                    )
                    if resumed is not None:
                        begin("approaching_edge_panel")
                        edge_center = resumed[0].center
                        edge_width = resumed[1]
                        edge_stage = "approach"
                        edge_stage_steps = 0
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="edge_panel",
                            presentationNavigationPanelSide=edge_side,
                        )
                        continue
                # The bars sit just outside the screen's left and right edges,
                # so reading them needs the screen head-on with both edges in
                # frame.  Establish that actively instead of waiting for a
                # sweep to happen to land on it: the run that failed had the
                # screen 313 px off centre against a 288 px tolerance and swept
                # past it for two and a half minutes.
                if screen_ready and screen.rect is not None and framing_attempts < 24:
                    frame_fraction = screen.rect.width / width
                    screen_error_x = float(
                        screen.rect.center[0] - viewport_center[0]
                    )
                    clipped = screen.rect.x <= 4 or screen.rect.right >= width - 4
                    if frame_fraction > screen_maximum_width_fraction or clipped:
                        framing_attempts += 1
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="framing_screen",
                        )
                        self._walk_backward(0.26)
                        time.sleep(max(0.18, self.poll_seconds))
                        continue
                    if abs(screen_error_x) > 0.12 * width:
                        framing_attempts += 1
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="framing_screen",
                        )
                        self._move_relative(
                            (max(-45.0, min(45.0, 0.26 * screen_error_x)), 0.0)
                        )
                        time.sleep(max(0.10, self.poll_seconds))
                        continue
                    if frame_fraction < bar_reading_width_fraction:
                        framing_attempts += 1
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="framing_screen",
                        )
                        self._walk_forward(0.26)
                        time.sleep(max(0.18, self.poll_seconds))
                        continue
                squarely_facing = (
                    screen_ready
                    and screen.rect is not None
                    and bar_reading_width_fraction
                    <= screen.rect.width / width
                    <= screen_maximum_width_fraction + 0.10
                    and screen.rect.x > 4
                    and screen.rect.right < width - 4
                    and abs(screen.rect.center[0] - viewport_center[0])
                    <= 0.22 * width
                )
                if squarely_facing and screen.rect is not None:
                    flanking = next(
                        (
                            panel
                            for panel in detect_sliden_edge_panels(frame, screen.rect)
                            if panel.side not in edge_rejected
                        ),
                        None,
                    )
                    if flanking is not None:
                        # The panel is beside the screen, edge-on.  Steer at
                        # the bar instead of sweeping the room blindly.
                        begin("approaching_edge_panel")
                        edge_side = flanking.side
                        edge_center = flanking.rect.center
                        edge_width = flanking.length
                        edge_best_height = 0.0
                        edge_best_openness = 0.0
                        edge_stage = "approach"
                        edge_stage_steps = 0
                        # Orbit toward the flank the panel is on first.
                        edge_direction = flanking.side
                        edge_stall = 0
                        edge_attempts = 0
                        _state(
                            self.state_callback,
                            presentationNavigationStatus="edge_panel",
                            presentationNavigationPanelSide=edge_side,
                        )
                        continue
                _state(
                    self.state_callback,
                    presentationPhase="finding_sliden_panel",
                    presentationNavigationStatus="finding_panel",
                    presentationNavigationScanStep=scan_step,
                    presentationNavigationSweep=sweep_index + 1,
                    presentationNavigationPanelScanStep=orbit_index,
                    presentationConsoleConfidence=observation.confidence,
                    presentationSlidenScreenConfidence=screen.confidence,
                    presentationSlidenLandmarkReady=True,
                    presentationConsoleDistanceReady=False,
                )
                if sweep(self._CONSOLE_PITCH_BANDS, 0.12):
                    # A full turn from this spot found nothing.  The panel is a
                    # flat surface, so change where we stand rather than how we
                    # look: alternate side steps of growing amplitude.  Never
                    # translate with the wall screen out of sight, though --
                    # that is how the search used to wander out of the
                    # presentation area entirely and end up facing a corner.
                    if not screen_ready:
                        time.sleep(max(0.18, self.poll_seconds))
                        continue
                    direction = -1 if orbit_index % 2 == 0 else 1
                    for _ in range(1 + orbit_index // 2):
                        self._strafe(0.34, direction)
                        time.sleep(0.06)
                    orbit_index += 1
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="orbiting_panel",
                        presentationNavigationPanelScanStep=orbit_index,
                    )
                    if orbit_index % 6 == 0:
                        # Re-anchor on the wall screen: several failed orbits
                        # usually mean we drifted out of the presentation area.
                        begin("searching_screen")
                time.sleep(max(0.18, self.poll_seconds))
                continue

            _state(
                self.state_callback,
                presentationPhase="searching_sliden_screen",
                presentationNavigationStatus="searching_landmark",
                presentationNavigationScanStep=scan_step,
                presentationNavigationSweep=sweep_index + 1,
                presentationConsoleConfidence=observation.confidence,
                presentationSlidenScreenConfidence=screen.confidence,
                presentationSlidenLandmarkReady=False,
                presentationConsoleDistanceReady=False,
            )
            if sweep(self._SCREEN_PITCH_BANDS, 0.16):
                if sweep_index % len(self._SCREEN_PITCH_BANDS) == 0:
                    # Every pitch band has been swept from here without finding
                    # the screen.  Standing in a corner or facing a partition
                    # cannot be fixed by turning, so leave for open space.
                    _state(
                        self.state_callback,
                        presentationNavigationStatus="leaving_corner",
                        presentationNavigationRepositioned=True,
                    )
                    self._escape_to_open_space()
                    relocation_index += 1
            time.sleep(max(0.18, self.poll_seconds))

        self._set_pitch(0.0)
        detail = (
            "the wall screen was never recognized"
            if not screen_seen
            else (
                "the wall screen was reached but no operation panel came into view"
                if not console_seen
                else "an operation panel was seen but never became convincing"
            )
        )
        raise SlidenConsoleNotFound(
            f"timed out while searching for and approaching the Sliden console: "
            f"{detail}; best console confidence was {best_confidence:.2f}, best "
            f"console width fraction was {best_width:.3f}, best screen "
            f"confidence was {best_screen_confidence:.2f}, "
            f"{scan_step} scan steps over {sweep_index} sweeps, "
            f"{relocation_index} relocations, {orbit_index} panel orbits"
        )

    #: Bounds a single measurement has to fall inside to be believed.  Outside
    #: them the frame changed for some other reason (the target was
    #: re-detected elsewhere, someone walked past) and the sample is noise.
    #: The upper bound is generous against any real mouse setting but still
    #: physical.  It was briefly 400, which is not a sensitivity -- it is what
    #: you get from dividing a target that jumped to another object by a
    #: one-count command, and once believed it makes every later correction
    #: too small to move anything.
    _AIM_GAIN_RANGE = (0.02, 40.0)
    #: How far the target may move in one step and still be the same control.
    #: Beyond this the observation is about something else, and measuring the
    #: camera from it teaches the servo nonsense.
    _AIM_JUMP_FRACTION = 0.22
    #: Weight of a new measurement in the running estimate.
    _AIM_GAIN_WEIGHT = 0.55
    #: The first command issued before the response has been measured, and
    #: the range that probe is allowed to grow and shrink over.  It has to
    #: start small: a client whose sensitivity sweeps the whole frame in a few
    #: counts loses the target on the probe itself, and a loop that then
    #: repeats the same probe never learns anything.  Too small only costs an
    #: iteration, so the probe starts timid and doubles until the view
    #: measurably moves -- and halves whenever a command loses the target.
    _AIM_PROBE_START = 2.0
    _AIM_PROBE_MINIMUM = 1.0
    _AIM_PROBE_MAXIMUM = 48.0
    #: Hard ceiling on one correction, and on the share of the frame a single
    #: correction is predicted to sweep.
    _AIM_MAXIMUM_COUNTS = 90.0
    _AIM_MAXIMUM_TRAVEL = 0.40
    #: Fraction of the computed correction actually issued.  Under-driving
    #: costs an iteration; over-driving costs the target.
    _AIM_DAMPING = 0.75
    #: Motion below this is not a reading.  A recognized control's centre
    #: wanders a pixel between frames on its own, so calling a two-pixel shift
    #: a measurement let the servo conclude the camera was five times more
    #: sluggish than it is -- and answer with a ninety-count command.
    _AIM_STALL_PIXELS = 4.0
    #: Consecutive unresponsive commands before the servo stops waiting for a
    #: camera that is never going to turn.
    _AIM_STALL_LIMIT = 3
    #: How many times one aim may try to clear a VRChat menu out of the way.
    #: Escape closes an open menu and opens the main one when none is open,
    #: so a second press undoes a wrong guess; more than that is a mouse this
    #: process does not own.
    _AIM_MENU_DISMISSALS = 2
    #: Consecutive frames without the target before aiming is abandoned.  The
    #: remaining budget is better spent recognizing the console again than
    #: correcting towards something no longer in the capture.
    _AIM_LOSS_LIMIT = 8
    #: Consecutive frames the console may be carried forward by matching
    #: rather than recognized outright.  Generous on purpose: driving the
    #: console from close up puts its URL bar behind the window's title bar,
    #: and with the bar gone the panel cannot be identified cold at all --
    #: every frame of the aim is a matched one.  A run of four ended an
    #: otherwise healthy aim that the match was following to the pixel.  The
    #: real guard is the match score, which decays as the camera turns far
    #: enough for a translation to stop describing the panel; this only
    #: bounds how long a stale template may drive a click.
    _AIM_TRACK_LIMIT = 16

    def _aim_command(self, axis: int, error: float, span: float) -> float:
        """How far to push one axis to cancel *error* pixels of aim error."""

        gain = self._aim_gain[axis]
        if gain is None:
            counts = self._aim_probe * (1.0 if error >= 0.0 else -1.0)
            return counts if abs(error) >= 1.0 else 0.0
        counts = self._AIM_DAMPING * error / gain
        travel = self._AIM_MAXIMUM_TRAVEL * span
        ceiling = min(self._AIM_MAXIMUM_COUNTS, travel / abs(gain))
        return max(-ceiling, min(ceiling, counts))

    def _record_aim_response(
        self,
        axis: int,
        command: float,
        moved: float,
        floor: float,
    ) -> None:
        """Learn pixels-per-count from a command and the motion it produced."""

        if abs(command) < 1.0 or abs(moved) < floor:
            return
        # The camera turns, so the world -- and the target with it -- travels
        # the opposite way to the command.
        measured = -moved / command
        low, high = self._AIM_GAIN_RANGE
        if not low <= abs(measured) <= high:
            return
        previous = self._aim_gain[axis]
        self._aim_gain[axis] = (
            measured
            if previous is None
            else (1.0 - self._AIM_GAIN_WEIGHT) * previous
            + self._AIM_GAIN_WEIGHT * measured
        )

    def _world_click(
        self,
        target: Callable[[SlidenConsoleObservation], Point | None],
        description: str,
        *,
        tolerance_pixels: Point = (14.0, 14.0),
        bounds: Callable[[SlidenConsoleObservation], Rect | None] | None = None,
        deadline: float | None = None,
        require_recognized: bool = False,
    ) -> tuple[RGBImage, SlidenConsoleObservation]:
        """Aim a world-space control at the reticle, then Interact-click it.

        The camera responds to relative mouse input by an amount nobody here
        knows: it depends on the client's mouse sensitivity and on the
        pointer settings of the machine running the bot.  Rather than assume
        a figure, each correction is measured against the frame it produces
        and the estimate drives the next one, so the loop converges whether
        one mouse count turns the view by a fifth of a pixel or by twenty.
        """

        if deadline is None:
            deadline = time.monotonic() + self.timeout_seconds
        best_confidence = 0.0
        best_error: float | None = None
        previous_point: Point | None = None
        previous_frame: RGBImage | None = None
        frame: RGBImage | None = None
        command: Point | None = None
        losses = 0
        stalls = 0
        tracked_run = 0
        menu_dismissals = 0
        #: Frame, observation and box the target was last actually seen in.
        template: tuple[RGBImage, SlidenConsoleObservation, Rect] | None = None
        #: The position the response was last measured from, whether it came
        #: from recognition or from a match, and everything commanded since.
        measure_point: Point | None = None
        measure_carried = False
        pending_command: Point = (0.0, 0.0)
        iterations = 0
        while time.monotonic() < deadline:
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is None:
                time.sleep(self.poll_seconds)
                continue
            observation = self._detect(frame)
            best_confidence = max(best_confidence, observation.confidence)
            point = target(observation)
            carried_point = False
            if point is not None:
                tracked_run = 0
                # Remember the frame this control was last actually seen in,
                # not merely the last frame a console was recognized in: a
                # template without the control in it cannot carry it forward.
                quad = observation.slide_quad
                if quad is not None:
                    template = (
                        frame,
                        observation,
                        Rect(
                            int(quad[:, 0].min()),
                            int(quad[:, 1].min()),
                            int(quad[:, 0].max() - quad[:, 0].min()),
                            int(quad[:, 1].max() - quad[:, 1].min()),
                        ),
                    )
            elif tracked_run < self._AIM_TRACK_LIMIT:
                # Recognition dropped the panel, which in this room says more
                # about the sunlight than about where the panel is.  Carry the
                # last one forward rather than throwing the correction away.
                # Bounded, because a translation stops describing a panel once
                # the camera has turned far enough to change its perspective.
                carried = self._track_console(frame, template)
                if carried is not None:
                    candidate = target(carried)
                    if candidate is not None:
                        observation = carried
                        point = candidate
                        carried_point = True
                        tracked_run += 1
                        _state(
                            self.state_callback,
                            presentationConsoleAimTracked=tracked_run,
                        )
            if point is None:
                losses += 1
                if command is not None:
                    self._save_diagnostic(
                        "lost", previous_frame, frame, command, previous_point
                    )
                    # Only a command that actually swept the view can be
                    # blamed for the target leaving it, and whether it did is
                    # a question for the frames, not for the estimate under
                    # test: judging it by predicted travel let an inflated
                    # estimate justify inflating itself, run away to the
                    # ceiling within one aim, and commit the click ninety
                    # pixels off the button.
                    # Blaming the command needs the match to have failed too:
                    # that is what says the panel is not merely unrecognized
                    # but genuinely no longer where it was.  Running out of
                    # matched frames says nothing about the camera.
                    swept = (
                        tracked_run < self._AIM_TRACK_LIMIT
                        and previous_frame is not None
                        and self._frame_motion(previous_frame, frame) >= 0.02
                    )
                    if swept:
                        # The correction is what took the target out of view,
                        # and the target does not move on its own: put the
                        # camera back where it could still see it, and treat
                        # the disappearance as evidence that the view responds
                        # more strongly than the estimate says.  Repeating a
                        # command that just lost the target only loses it
                        # again, so the probe shrinks too.
                        self._move_relative((-command[0], -command[1]))
                        self._pitch_offset -= command[1]
                        pending_command = (
                            pending_command[0] - command[0],
                            pending_command[1] - command[1],
                        )
                        self._aim_probe = max(
                            self._AIM_PROBE_MINIMUM, 0.5 * self._aim_probe
                        )
                        for axis in (0, 1):
                            gain = self._aim_gain[axis]
                            if gain is not None:
                                self._aim_gain[axis] = max(
                                    -self._AIM_GAIN_RANGE[1],
                                    min(self._AIM_GAIN_RANGE[1], 2.0 * gain),
                                )
                    # Otherwise the view barely moved and recognition dropped
                    # the panel for reasons of its own.  Undoing a correction
                    # that was not at fault just trades one lost frame for the
                    # next, and the aim never advances at all.
                    command = None
                    previous_point = None
                    previous_frame = None
                _state(
                    self.state_callback,
                    presentationConsoleConfidence=observation.confidence,
                    presentationConsoleAimLost=losses,
                    presentationConsoleAimProbe=self._aim_probe,
                )
                if losses >= self._AIM_LOSS_LIMIT:
                    # Out of sight this long, "we know where it is" has
                    # expired: the next recognition starts from scratch.
                    self._save_diagnostic(
                        "gave-up",
                        None if template is None else template[0],
                        frame,
                        command,
                        previous_point,
                    )
                    self._forget_console()
                    raise SlidenConsoleNotFound(
                        f"lost sight of {description} while aiming at it and "
                        f"could not pick it up again after {losses} frames; "
                        f"best confidence was {best_confidence:.2f}"
                    )
                time.sleep(self.poll_seconds)
                continue
            # Calibration measures how far the panel travelled against
            # everything commanded since it was last measured.  Accumulating
            # the commands is what makes it work at all here: recognition
            # drops out on most frames, and from close up -- where the URL
            # bar sits behind the window's title bar -- on every one of them,
            # so a rule that waited for two clean recognitions in a row would
            # never calibrate and the aim would crawl at probe speed until it
            # ran out of budget.  A matched position is quantized to the scale
            # the match ran at, so any pair involving one only counts once the
            # panel has moved well past that quantum.
            if measure_point is not None and max(
                abs(pending_command[0]), abs(pending_command[1])
            ) >= 1.0:
                floor = (
                    3.0 * _CONSOLE_MATCH_SHRINK
                    if carried_point or measure_carried
                    else self._AIM_STALL_PIXELS
                )
                moved = (
                    point[0] - measure_point[0],
                    point[1] - measure_point[1],
                )
                jumped = math.hypot(moved[0], moved[1]) > (
                    self._AIM_JUMP_FRACTION * frame.shape[1]
                )
                if not jumped:
                    self._record_aim_response(0, pending_command[0], moved[0], floor)
                    self._record_aim_response(1, pending_command[1], moved[1], floor)
                responded = jumped or math.hypot(moved[0], moved[1]) >= floor
                calibrated = any(gain is not None for gain in self._aim_gain)
                if not responded and not calibrated:
                    # The probe was too gentle to read.  Push harder rather
                    # than concluding anything about the camera.
                    self._aim_probe = min(
                        self._AIM_PROBE_MAXIMUM, 2.0 * self._aim_probe
                    )
                if (
                    responded
                    or carried_point
                    or measure_carried
                    or not (calibrated or self._aim_probe >= self._AIM_PROBE_MAXIMUM)
                ):
                    # "The camera is frozen" is a claim about the client, and
                    # only clean recognitions can support it.
                    stalls = 0
                else:
                    stalls += 1
            measure_point = point
            measure_carried = carried_point
            pending_command = (0.0, 0.0)
            losses = 0
            center = self._viewport_center(frame)
            error_x = point[0] - center[0]
            error_y = point[1] - center[1]
            error = math.hypot(error_x, error_y)
            best_error = error if best_error is None else min(best_error, error)
            iterations += 1
            _state(
                self.state_callback,
                presentationConsoleConfidence=observation.confidence,
                presentationConsoleModal=observation.modal_open,
                presentationConsoleInputMode="world_aim",
                presentationConsoleAimError=error,
                presentationConsoleAimErrorX=error_x,
                presentationConsoleAimErrorY=error_y,
                presentationConsoleAimGainX=self._aim_gain[0],
                presentationConsoleAimGainY=self._aim_gain[1],
                presentationConsoleAimProbe=self._aim_probe,
                presentationConsoleAimIterations=iterations,
                presentationConsoleAimLost=0,
                presentationConsoleTarget=[point[0], point[1]],
                presentationConsoleReticle=[center[0], center[1]],
            )
            # One mouse count is the finest move the client can be asked for,
            # so half of what it sweeps is the best aim that exists.  Holding
            # out for a tolerance below that is asking for an accuracy the
            # hardware cannot express, and the loop can only spend its budget
            # oscillating over the target until it times out.
            reachable = (
                max(tolerance_pixels[0], 0.5 * abs(self._aim_gain[0] or 0.0)),
                max(tolerance_pixels[1], 0.5 * abs(self._aim_gain[1] or 0.0)),
            )
            close_enough = (
                abs(error_x) <= reachable[0] and abs(error_y) <= reachable[1]
            )
            # Being near a point the control was measured from is not the same
            # as being on the control.  Where its box is known, the reticle
            # has to be provably inside it before the click is worth making --
            # otherwise a click that lands on the panel's frame reports
            # exactly like one that lands on the field, and the run ends with
            # nothing but a dialog that never opened.
            box = None if bounds is None else bounds(observation)
            if box is not None:
                margin_x = min(0.30 * box.width, 6.0)
                margin_y = min(0.30 * box.height, 6.0)
                inside = (
                    box.x + margin_x <= center[0] <= box.right - margin_x
                    and box.y + margin_y <= center[1] <= box.bottom - margin_y
                )
                close_enough = close_enough and inside
                _state(self.state_callback, presentationConsoleAimInsideTarget=inside)
            if close_enough and require_recognized and carried_point:
                # The template match keeps the aim moving while recognition
                # drops out, but it cannot say *which* control is under the
                # reticle: it translates the whole console, and a translation
                # that is off by one button reads as a perfect aim.  Measured
                # live, an aim running 14 frames on matched positions clicked
                # the chime button two places right of Next (2026-08-15).
                # Approaching on a match is fine; committing on one is not.
                close_enough = False
                _state(
                    self.state_callback,
                    presentationConsoleAimUnverified=True,
                )
            if close_enough:
                if reachable != tuple(tolerance_pixels):
                    _state(
                        self.state_callback,
                        presentationConsoleAimQuantized=True,
                    )
                self._save_diagnostic("clicked", frame, None, command, point)
                self._interact(center)
                return frame, observation
            if stalls >= self._AIM_STALL_LIMIT:
                # Repeated commands moved nothing, so VRChat is not turning
                # the camera for us.  In practice that means one of its own
                # menus has the cursor -- the action wheel was found open
                # across a live run -- and the one thing not to do then is
                # click: the reticle is over a menu item, and "press whatever
                # is under the cursor" is how a presentation changes the
                # avatar's outfit.  Close the menu instead and carry on.
                stalls = 0
                if menu_dismissals >= self._AIM_MENU_DISMISSALS:
                    raise SlidenConsoleNotFound(
                        f"the camera stopped responding while aiming at "
                        f"{description} and did not come back after "
                        f"{menu_dismissals} attempts to clear a VRChat menu; "
                        f"something else has the mouse"
                    )
                menu_dismissals += 1
                _state(
                    self.state_callback,
                    presentationConsoleInputMode="dismissing_menu",
                    presentationConsoleAimStalled=True,
                    presentationConsoleMenuDismissals=menu_dismissals,
                )
                self._press_escape()
                time.sleep(max(0.4, self.poll_seconds * 2.0))
                self._aim_probe = self._AIM_PROBE_START
                measure_point = None
                pending_command = (0.0, 0.0)
                continue
            height, width = frame.shape[:2]
            command = (
                self._aim_command(0, error_x, float(width)),
                self._aim_command(1, error_y, float(height)),
            )
            previous_point = point
            previous_frame = frame
            pending_command = (
                pending_command[0] + command[0],
                pending_command[1] + command[1],
            )
            self._move_relative(command)
            self._pitch_offset += command[1]
            time.sleep(max(0.08, self.poll_seconds))
        detail = (
            "unknown"
            if best_error is None
            else f"{best_error:.1f} pixels"
        )
        self._save_diagnostic("timeout", previous_frame, frame, command, previous_point)
        raise SlidenConsoleNotFound(
            f"timed out while aiming at {description}; best confidence was "
            f"{best_confidence:.2f}, best reticle error was {detail} over "
            f"{iterations} corrections, {losses} of them with the target out "
            f"of view; measured view response was "
            f"{self._aim_gain[0]} horizontal / {self._aim_gain[1]} vertical "
            f"pixels per mouse count at probe {self._aim_probe}"
        )

    def _paste(self, value: str) -> None:
        _check_cancel(self.cancelled)
        if self._paste_callback is not None:
            self._paste_callback(value)
            return
        import vrcpilot

        # The modal input was focused by _click_ui immediately before this
        # call.  Re-running the Win32 foreground guard here is both redundant
        # and unreliable after VRChat releases its captured mouse for the
        # Sliden modal, so keep the verified focus and only emit Ctrl+V.
        vrcpilot.clipboard.paste(value, focus=False, pid=self.pid)

    def _clear_input_text(self) -> None:
        """Replace any residual modal text even if X-button detection fails."""

        _check_cancel(self.cancelled)
        if self._clear_text_callback is not None:
            self._clear_text_callback()
            return
        import vrcpilot

        vrcpilot.keyboard.press(
            vrcpilot.Key.CTRL,
            vrcpilot.Key.A,
            focus=False,
            pid=self.pid,
        )
        vrcpilot.keyboard.press(
            vrcpilot.Key.BACKSPACE,
            focus=False,
            pid=self.pid,
        )

    def _wait_for(
        self,
        predicate: Callable[[SlidenConsoleObservation], bool],
        description: str,
        *,
        timeout_seconds: float | None = None,
        deadline: float | None = None,
        search_pitch: bool = False,
    ) -> tuple[RGBImage, SlidenConsoleObservation]:
        """Poll until *predicate* holds, optionally looking around for it.

        With ``search_pitch`` the wait is not passive.  An operation panel
        that is simply above or below the current line of sight produced
        exactly the same "nothing recognized" frames as one in another room,
        and the difference is one glance -- so after a short grace the view
        is tilted through a few bands and returned to where it started.
        """

        if deadline is None:
            deadline = time.monotonic() + (
                self.timeout_seconds if timeout_seconds is None else timeout_seconds
            )
        entry_pitch = self._pitch_offset
        grace = time.monotonic() + max(1.0, min(4.0, self.poll_seconds * 12.0))
        bands = (0.0, -55.0, 55.0, -110.0, 110.0)
        band_index = 0
        band_started = 0.0
        best_confidence = 0.0
        try:
            while time.monotonic() < deadline:
                _check_cancel(self.cancelled)
                frame = self._capture()
                if frame is not None:
                    observation = self._detect(frame)
                    best_confidence = max(best_confidence, observation.confidence)
                    _state(
                        self.state_callback,
                        presentationConsoleConfidence=observation.confidence,
                        presentationConsoleModal=observation.modal_open,
                    )
                    if predicate(observation):
                        entry_pitch = self._pitch_offset
                        return frame, observation
                now = time.monotonic()
                if search_pitch and now >= grace and now - band_started >= 1.2:
                    band_index += 1
                    band_started = now
                    self._set_pitch(entry_pitch + bands[band_index % len(bands)])
                    _state(
                        self.state_callback,
                        presentationConsoleSearchPitch=(
                            self._pitch_offset - entry_pitch
                        ),
                    )
                time.sleep(self.poll_seconds)
        finally:
            if self._pitch_offset != entry_pitch:
                self._set_pitch(entry_pitch)
        raise SlidenConsoleNotFound(
            f"timed out while recognizing {description}; "
            f"best confidence was {best_confidence:.2f}"
        )

    @staticmethod
    def _url_field_tolerance(observation: SlidenConsoleObservation) -> Point:
        """How close the reticle has to sit to the middle of the URL bar.

        Measured off the bar in the frame rather than fixed: it is 15 pixels
        high seen from across the room and 27 from working distance, so a
        constant four-pixel demand asks the servo for an accuracy the mouse
        cannot express at one distance while accepting a miss at the other.
        Horizontally the bar is wide, and being off-centre along it still
        lands inside the control.
        """

        rect = observation.url_field_rect
        if rect is None:
            return (30.0, 6.0)
        return (
            max(12.0, min(0.40 * rect.width, 60.0)),
            max(4.0, min(0.32 * rect.height, 14.0)),
        )

    def console_in_view(
        self,
        *,
        timeout_seconds: float = 8.0,
    ) -> SlidenConsoleObservation | None:
        """Is the operation panel usable from where the avatar stands?

        The predicate is deliberately the same one ``configure_url`` waits on:
        a positive answer here has to mean that configure_url will not then
        turn around and fail.  Anything weaker would let a run skip the walk
        to the panel and die at the URL field instead.

        Glancing counts as "in view": ``search_pitch`` tilts through a few
        bands and comes back, because a panel one head-tilt above the line of
        sight is in front of the avatar in every sense that matters.  Returns
        the observation, or ``None`` when the panel is not reachable from
        here -- never raises, because "not here" is an answer, not a failure.
        """

        try:
            _frame, observation = self._wait_for(
                lambda item: item.modal_open
                or (item.console_found and item.url_field is not None),
                "the Sliden console",
                timeout_seconds=max(1.0, timeout_seconds),
                search_pitch=True,
            )
        except SlidenConsoleNotFound:
            return None
        except SlidenCancelled:
            raise
        return observation

    def configure_url(self, url: str) -> SlidenConsoleObservation:
        """Recognize the console, open its URL dialog, paste, and submit."""

        # One budget covers recognition and every click attempt.  Giving each
        # step the full console timeout meant a single run could spend three
        # minutes failing to open a dialog the caller was told to give thirty
        # seconds to.
        deadline = time.monotonic() + self.timeout_seconds
        _state(self.state_callback, presentationPhase="recognizing_console")
        _frame, observation = self._wait_for(
            lambda item: item.modal_open or (
                item.console_found and item.url_field is not None
            ),
            "the Sliden console",
            deadline=deadline,
            search_pitch=True,
        )
        if not observation.modal_open:
            dialog_grace = min(2.0, max(0.25, self.poll_seconds * 6.0))
            last_dialog_error: SlidenConsoleNotFound | None = None
            attempts = 3
            for attempt in range(1, attempts + 1):
                _state(
                    self.state_callback,
                    presentationConsoleClickAttempt=attempt,
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    _aimed, observation = self._world_click(
                        lambda item: item.url_field
                        if not item.modal_open
                        else None,
                        "the Sliden URL field",
                        tolerance_pixels=self._url_field_tolerance(observation),
                        bounds=lambda item: item.url_field_rect,
                        deadline=time.monotonic()
                        + remaining / (attempts - attempt + 1),
                    )
                except SlidenConsoleNotFound as exc:
                    # A missed aim is worth another attempt: the panel may
                    # have been re-recognized a few pixels away, or the view
                    # drifted.  Only the exhausted budget ends the loop.
                    last_dialog_error = exc
                    continue
                try:
                    _frame, observation = self._wait_for(
                        lambda item: item.modal_open,
                        "the Sliden URL input dialog",
                        timeout_seconds=min(
                            dialog_grace, max(0.0, deadline - time.monotonic())
                        ),
                    )
                    break
                except SlidenConsoleNotFound as exc:
                    # The aim was on the field and the click was made, yet no
                    # dialog.  Keep the frame: nothing in the numbers can say
                    # whether the click landed somewhere else, was swallowed,
                    # or opened something that then closed.
                    self._save_diagnostic(
                        "clicked-no-dialog", self._capture(), None, None, None
                    )
                    last_dialog_error = exc
            if not observation.modal_open:
                raise last_dialog_error or SlidenConsoleNotFound(
                    "ran out of console time before the Sliden URL dialog "
                    "could be opened"
                )
        if observation.modal_input is None or observation.modal_submit is None:
            raise SlidenConsoleNotFound(
                "the Sliden URL dialog was recognized without an input field "
                "and a submit button"
            )
        _state(self.state_callback, presentationConsoleInputMode="menu_ui")
        clear_method = "keyboard"
        if observation.modal_clear is not None:
            _state(self.state_callback, presentationPhase="clearing_console_url")
            self._click_ui(observation.modal_clear)
            clear_method = "clear_button+keyboard"
            time.sleep(max(0.08, self.poll_seconds))
        _state(
            self.state_callback,
            presentationConsoleClearDetected=observation.modal_clear is not None,
            presentationConsoleClearMethod=clear_method,
        )
        self._click_ui(observation.modal_input)
        self._clear_input_text()
        self._paste(url)
        self._click_ui(observation.modal_submit)
        _state(self.state_callback, presentationPhase="loading_console_url")
        # Submitting hands the URL to the world, which then fetches and
        # buffers the video itself.  That wait is a download, not a
        # recognition, so it is not bounded by the console timeout -- which
        # exists to stop the bot hunting for UI it cannot see.
        # What this step is waiting for is the dialog closing over a console
        # that is showing something -- not a button, which nothing here is
        # about to press.  Demanding a recognized Next button made a loaded
        # deck read as a failed load whenever the button row happened to be
        # out of frame or unresolved, and the URL entry that had just
        # succeeded was reported as a timeout.
        _frame, ready = self._wait_for(
            lambda item: not item.modal_open and item.slide_quad is not None,
            "the loaded Sliden console",
            timeout_seconds=max(self.timeout_seconds, 60.0),
        )
        return ready

    #: Clearance the console's URL bar needs above the reticle before a
    #: bottom-row button can be aimed at.
    _NAV_FIT_MARGIN = 28
    #: Backward steps allowed to buy that clearance.
    _NAV_FIT_STEPS = 3

    def _console_nav_clearance(
        self, observation: SlidenConsoleObservation, frame: RGBImage
    ) -> tuple[float, float] | None:
        """``(needed, available)`` pixels of headroom for a bottom-row button.

        Clicking Next means pitching down until the reticle sits on it, which
        lifts the whole panel in the frame by the same amount.  Standing close
        enough to fill the view with a slide lifts the URL bar out of the
        window -- and the URL bar is what identifies the panel at all, so from
        there every correction is made blind.  Returns None when there is not
        enough of a console recognized to judge.
        """

        button = observation.next_button
        if button is None:
            return None
        top: float | None = None
        if observation.url_field_rect is not None:
            top = float(observation.url_field_rect.y)
        elif observation.slide_quad is not None:
            quad = observation.slide_quad
            # The bar sits just above the preview; allow for its own height.
            height = float(quad[:, 1].max() - quad[:, 1].min())
            top = float(quad[:, 1].min()) - 0.10 * height
        if top is None:
            return None
        reticle = self._viewport_center(frame)
        return (float(button[1]) - top, float(reticle[1]) - self._NAV_FIT_MARGIN)

    def _make_navigation_reachable(self) -> bool:
        """Back off until the Next button can be aimed at without blinding the
        detector.  Returns True when the geometry works, False when it does not
        and the aim has to be attempted from where it stands anyway.

        Observed 2026-08-15: from arm's length the panel spanned ~490 px while
        only ~360 px of headroom existed, so every downward correction pushed
        the URL bar behind the title bar; the aim lost the button, undid the
        correction, lost it again, and gave up after eight frames with the
        slide never advanced.
        """

        steps = 0
        for _ in range(self._NAV_FIT_STEPS + 1):
            _check_cancel(self.cancelled)
            frame = self._capture()
            if frame is None:
                time.sleep(self.poll_seconds)
                continue
            clearance = self._console_nav_clearance(self._detect(frame), frame)
            if clearance is None:
                # Nothing to judge; let the aim loop do its own recovery.
                _state(self.state_callback, presentationConsoleNavFit=None)
                return False
            needed, available = clearance
            _state(
                self.state_callback,
                presentationConsoleNavFit=[needed, available],
                presentationConsoleStepBacks=steps,
            )
            if needed <= available:
                return True
            if steps >= self._NAV_FIT_STEPS:
                return False
            self._walk_backward(0.30)
            steps += 1
            time.sleep(max(0.20, self.poll_seconds))
        return False

    def previous_slide(self) -> float:
        """Recognize and click Previous, returning the observed preview change.

        Used for going back during questions, not during the talk itself.  On
        slide 1 the control is greyed out and not recognized at all, which is
        the right answer: there is nothing before it.
        """

        return self._navigate_slide(
            lambda item: item.previous_button,
            lambda item: item.previous_button_rect,
            "the Sliden previous button",
        )

    def next_slide(self) -> float:
        """Recognize and click Next, returning the observed preview difference."""

        return self._navigate_slide(
            lambda item: item.next_button,
            lambda item: item.next_button_rect,
            "the Sliden next button",
        )

    def _navigate_slide(self, control, bounds, description: str) -> float:
        self._make_navigation_reachable()
        before, observation = self._world_click(
            lambda item: control(item)
            if not item.modal_open and item.slide_quad is not None
            else None,
            description,
            bounds=bounds,
            # Which button is under the reticle has to be recognized, not
            # inferred from a translated template: the buttons sit ~60 px
            # apart and pressing the wrong one locks the panel or rings the
            # chime instead of advancing.
            require_recognized=True,
        )
        deadline = time.monotonic() + min(self.timeout_seconds, 5.0)
        best_difference = 0.0
        while time.monotonic() < deadline:
            _check_cancel(self.cancelled)
            time.sleep(self.poll_seconds)
            after = self._capture()
            if after is None:
                continue
            after_observation = self._detect(after)
            difference = slide_preview_difference(
                before,
                observation.slide_quad,
                after,
                after_observation.slide_quad,
            )
            best_difference = max(best_difference, difference)
            if difference >= 0.0015:
                return difference
        # The console button was recognized and clicked.  A tiny preview (or
        # two visually identical consecutive slides) can make visual change
        # confirmation inconclusive, so report that fact without double-clicking.
        return best_difference


def remove_workspace(path: Path) -> None:
    """Best-effort cleanup kept here for easy unit-test substitution."""

    shutil.rmtree(path, ignore_errors=True)
