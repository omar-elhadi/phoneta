"""Local model registry — resolve cache paths, verify integrity, report missing.

Every model has:
- a **name** (stable key for the downloader / UI)
- a **description** (human-readable)
- a **`present()` method** that returns True when the model is available locally
- optional **checksums** (sha256 of key files) for integrity verification
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _xdg_cache() -> Path:
    """XDG-compliant cache directory for model weights."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        return Path(base) / "phoneta" / "models"
    base = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(base) / "phoneta" / "models"


@dataclass(frozen=True)
class ModelEntry:
    """Description of a locally managed model or artifact."""

    name: str
    description: str
    present: Callable[[], bool]
    verify: Callable[[], bool] | None = None


# ───────────────────────────────────────────────────────────────────
#  Presence checks  (pure callables — don't import heavy deps at module level)
# ───────────────────────────────────────────────────────────────────


def _whisper_present() -> bool:
    """Is the faster-whisper ``base`` model cached?"""
    try:
        from faster_whisper.utils import _get_assets_path  # pyright: ignore[reportPrivateUsage]
    except ImportError:
        return False

    assets = Path(_get_assets_path())
    model_dir = assets / "models--Systran--faster-whisper-base"
    return model_dir.is_dir() and any(
        f.suffix in (".bin", ".onnx") for f in model_dir.rglob("model.*")
    )


def _silero_present() -> bool:
    """Is the silero-vad model available?"""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False

    cache = _xdg_cache() / "silero-vad"
    return (cache / "silero_vad.jit").is_file()


def _mfa_en_present() -> bool:
    """Are the MFA English acoustic model and dictionary present?"""
    try:
        from montreal_forced_aligner.models import AcousticModel  # type: ignore[import-untyped]

        return bool(AcousticModel.pretrained_available("english_mfa"))
    except Exception:
        return False


def _mfa_fr_present() -> bool:
    """Are the MFA French acoustic model and dictionary present?"""
    try:
        from montreal_forced_aligner.models import AcousticModel  # type: ignore[import-untyped]

        return bool(AcousticModel.pretrained_available("french_mfa"))
    except Exception:
        return False


def _espeak_present() -> bool:
    """Is espeak-ng installed and working?"""
    try:
        from phonemizer.backend import EspeakBackend

        EspeakBackend("en-us")
        return True
    except Exception:
        return False


# ───────────────────────────────────────────────────────────────────
#  Registry
# ───────────────────────────────────────────────────────────────────

_REGISTRY: list[ModelEntry] = [
    ModelEntry(
        name="whisper-base",
        description="faster-whisper base (74M) — speech recognition",
        present=_whisper_present,
    ),
    ModelEntry(
        name="silero-vad",
        description="silero-vad — voice activity detection",
        present=_silero_present,
    ),
    ModelEntry(
        name="mfa-en",
        description="Montreal Forced Aligner — English acoustic model + dictionary",
        present=_mfa_en_present,
    ),
    ModelEntry(
        name="mfa-fr",
        description="Montreal Forced Aligner — French acoustic model + dictionary",
        present=_mfa_fr_present,
    ),
    ModelEntry(
        name="espeak-ng",
        description="espeak-ng — text-to-phoneme engine (system package)",
        present=_espeak_present,
    ),
]


def list_models() -> list[ModelEntry]:
    """Return the full registry."""
    return list(_REGISTRY)


def missing_models() -> list[ModelEntry]:
    """Return models that are NOT present on this machine."""
    return [m for m in _REGISTRY if not m.present()]


def all_present() -> bool:
    """True when every registered model is available."""
    return len(missing_models()) == 0


def model_cache_dir() -> Path:
    """XDG cache directory for model files (created if needed)."""
    path = _xdg_cache()
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Verify a single file's sha256 checksum."""
    if not path.is_file():
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_sha256