"""Locate the SourceVault-managed local model assets from the worker side.

The ONNX vision models and the local voice assets are *not* owned by
VRCRealtime: they live under ``SourceVault_vision/`` and ``SourceVault_voice/``
in the package directory and are resolved by ``SourceVault_vision.wl`` /
``SourceVault_voice.wl``.  The Wolfram side normally hands the concrete paths
to the worker on the command line, so this module is the fallback that keeps
the Python modules importable and unit-testable on their own.

The search order mirrors the Wolfram side exactly, so a machine that overrides
a root in one place gets the same answer in the other:

  vision  $SOURCEVAULT_VISION_ROOT
          <package dir>/SourceVault_vision
          %LOCALAPPDATA%/SourceVault/vision

  voice   $SOURCEVAULT_VOICE_ROOT
          <package dir>/SourceVault_voice
          %LOCALAPPDATA%/SourceVault/voice
          %LOCALAPPDATA%/VRCRealtime          (legacy, read-only)

Nothing here raises when an asset is missing: callers decide whether the
feature that needs it degrades or fails, and they need the searched paths to
say so usefully.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "package_directory",
    "find_tts_runtime",
    "find_tts_voice",
    "voice_language",
    "vision_search_path",
    "voice_search_path",
    "find_vision_model",
    "speech_model_search_path",
    "find_speech_model",
    "default_speech_model_directory",
    "MissingAsset",
]


class MissingAsset(FileNotFoundError):
    """Raised by callers that cannot degrade; carries the searched paths."""

    def __init__(self, what: str, searched: list[Path]) -> None:
        self.what = what
        self.searched = list(searched)
        listing = "\n  ".join(str(p) for p in searched) or "(no candidate paths)"
        super().__init__(f"{what} not found. Searched:\n  {listing}")


def _local_app_data() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base)
    return Path.home() / "AppData" / "Local"


def package_directory() -> Path:
    """Return the MyPackages directory this checkout lives in.

    Walking up beats a fixed ``parents[n]`` here: the python modules moved into
    ``VRCRealtime_info/resources/python/`` once already, and a hard-coded depth
    silently resolves to the wrong directory the next time they move.
    """

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "VRCRealtime.wl").is_file():
            return candidate
        if (candidate / "SourceVault_vision").is_dir():
            return candidate
        if (candidate / "SourceVault_voice").is_dir():
            return candidate
    # Fall back to the historical layout (python next to VRCRealtime.wl).
    return here.parents[min(3, len(here.parents) - 1)]


def _roots(env_name: str, package_subdir: str, app_data_subdir: str) -> list[Path]:
    roots: list[Path] = []
    override = os.environ.get(env_name)
    if override:
        roots.append(Path(override))
    roots.append(package_directory() / package_subdir)
    roots.append(_local_app_data() / "SourceVault" / app_data_subdir)
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def vision_search_path() -> list[Path]:
    return _roots("SOURCEVAULT_VISION_ROOT", "SourceVault_vision", "vision")


def voice_search_path() -> list[Path]:
    roots = _roots("SOURCEVAULT_VOICE_ROOT", "SourceVault_voice", "voice")
    # Read-only compatibility with the pre-SourceVault_voice location.
    roots.append(_local_app_data() / "VRCRealtime")
    return roots


def find_vision_model(directory: str, filename: str) -> Path | None:
    """Return the first existing ``<root>/<directory>/<filename>``."""

    for root in vision_search_path():
        candidate = root / directory / filename
        if candidate.is_file():
            return candidate
    return None


def find_tts_runtime() -> Path | None:
    """Return the local Piper executable, or ``None``."""

    name = "piper.exe" if os.name == "nt" else "piper"
    for root in voice_search_path():
        for candidate in (
            root / "tts" / "runtime" / "piper" / "bin" / name,
            root / "tts" / "runtime" / "piper" / name,
            # Pre-SourceVault_voice layout had no tts/ level.
            root / "runtime" / "piper" / "bin" / name,
        ):
            if candidate.is_file():
                return candidate
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and (Path(entry) / name).is_file():
            return Path(entry) / name
    return None


def find_tts_voice(name: str | None = None) -> tuple[Path, Path | None] | None:
    """Return ``(model.onnx, config.json | None)`` for an installed voice.

    No voice is privileged: the name is the folder (or file) name, and with
    *name* omitted the first voice in sorted order wins, so the answer is the
    same on every machine with the same set installed.  Returns ``None`` when
    nothing is installed -- callers decide whether that is fatal.
    """

    for root in voice_search_path():
        for base in (root / "tts" / "models", root / "models"):
            if not base.is_dir():
                continue
            for directory in sorted(p for p in base.iterdir() if p.is_dir()):
                if name and directory.name != name:
                    continue
                models = sorted(directory.glob("*.onnx"))
                if models:
                    return models[0], _voice_config_for(models[0])
            for model in sorted(base.glob("*.onnx")):
                if name and model.stem != name:
                    continue
                return model, _voice_config_for(model)
    return None


def _voice_config_for(model: Path) -> Path | None:
    for candidate in (
        model.with_name(model.name + ".json"),
        model.parent / "config.json",
    ):
        if candidate.is_file():
            return candidate
    jsons = sorted(model.parent.glob("*.json"))
    return jsons[0] if len(jsons) == 1 else None


def voice_language(config: Path | None) -> str:
    """Language code to send with a synthesis request, or '' for none.

    Only multilingual voices accept one; sending a language to a
    single-language voice can be rejected outright.
    """

    if config is None or not config.is_file():
        return ""
    import json

    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    language = ""
    if isinstance(data.get("language"), dict):
        language = str(data["language"].get("code") or "")
    if not language and isinstance(data.get("espeak"), dict):
        language = str(data["espeak"].get("voice") or "")
    multilingual = data.get("phoneme_type") == "multilingual" or "-" in language
    return language if multilingual else ""


def _speech_model_present(directory: Path) -> bool:
    return (directory / "am" / "final.mdl").is_file() and (directory / "graph").is_dir()


def speech_model_search_path() -> list[Path]:
    """Directories that may contain Vosk model folders, in priority order."""

    bases: list[Path] = []
    for root in voice_search_path():
        # Canonical layout, symmetric with tts/: asr/README.md ships, asr/models/
        # is what the manifest excludes.
        bases.append(root / "asr" / "models")
        bases.append(root / "asr")
        # Legacy: %LOCALAPPDATA%/VRCRealtime/models/<model>
        bases.append(root / "models")
    return bases


def find_speech_model(name: str | None = None) -> Path | None:
    """Return an installed Vosk model directory, or ``None``.

    With *name* given only that model matches; otherwise the first complete
    model found in search order wins (name-sorted within one directory, so the
    answer does not depend on filesystem enumeration order).
    """

    for base in speech_model_search_path():
        if not base.is_dir():
            continue
        if name:
            candidate = base / name
            if _speech_model_present(candidate):
                return candidate
            continue
        for candidate in sorted(p for p in base.iterdir() if p.is_dir()):
            if _speech_model_present(candidate):
                return candidate
    return None


def default_speech_model_directory() -> Path:
    """Where a newly downloaded ASR model should go."""

    override = os.environ.get("SOURCEVAULT_VOICE_ROOT")
    root = Path(override) if override else package_directory() / "SourceVault_voice"
    return root / "asr" / "models"
