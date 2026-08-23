#!/usr/bin/env python3
"""One-shot model downloader for Phoneta.

Downloads every required model once, verifies each, and exits cleanly.
After this script succeeds, Phoneta runs fully offline.

Usage::

    python scripts/download_models.py          # download all missing models
    python scripts/download_models.py --verify # check only, don't download
    python scripts/download_models.py --model whisper-base  # single model

"""

from __future__ import annotations

import argparse
import sys
import time


def _report(msg: str) -> None:
    print(f"\033[1;34m[phoneta-setup]\033[0m {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"\033[1;32m  ✓\033[0m {msg}")


def _fail(msg: str) -> None:
    print(f"\033[1;31m  ✗\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m  ⚠\033[0m {msg}")


# ── per-model downloaders ────────────────────────────────────────────


def _download_whisper_base() -> bool:
    """Trigger faster-whisper to cache the ``base`` model."""
    _report("Downloading faster-whisper base model (~140 MB) …")
    try:
        from faster_whisper import WhisperModel

        start = time.monotonic()
        _model = WhisperModel("base", device="cpu", compute_type="int8")  # noqa: F841
        elapsed = time.monotonic() - start
        _ok(f"Whisper base ready ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        _fail(f"Whisper download failed: {exc}")
        return False


def _download_silero() -> bool:
    """Cache the silero-vad JIT model."""
    _report("Downloading silero-vad model …")
    try:
        from silero_vad import load_silero_vad

        start = time.monotonic()
        load_silero_vad(onnx=True)
        elapsed = time.monotonic() - start
        _ok(f"silero-vad ready ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        _fail(f"silero-vad download failed: {exc}")
        return False


def _download_mfa(lang: str) -> bool:
    """Download an MFA pretrained acoustic model + dictionary."""
    _report(f"Downloading MFA {lang} acoustic model + dictionary …")
    try:
        from montreal_forced_aligner.models import AcousticModel  # type: ignore[import-untyped]

        start = time.monotonic()
        AcousticModel.download(f"{lang}_mfa")
        elapsed = time.monotonic() - start
        _ok(f"MFA {lang} ready ({elapsed:.1f}s)")
        return True
    except Exception as exc:
        _fail(f"MFA {lang} download failed: {exc}")
        return False


def _check_espeak() -> bool:
    """Verify espeak-ng is installed (system package, not downloadable here)."""
    _report("Checking espeak-ng …")
    try:
        from phonemizer.backend import EspeakBackend

        EspeakBackend("en-us")
        _ok("espeak-ng is installed and working")
        return True
    except Exception:
        _warn(
            "espeak-ng not found. Install it via your package manager:\n"
            "  Ubuntu/Debian:  sudo apt install espeak-ng\n"
            "  Fedora:         sudo dnf install espeak-ng\n"
            "  Arch:           sudo pacman -S espeak-ng\n"
            "  Windows:        download from https://github.com/espeak-ng/espeak-ng/releases"
        )
        return False


# ── dispatch ─────────────────────────────────────────────────────────

DOWNLOADERS: dict[str, callable] = {  # type: ignore[valid-type]
    "whisper-base": _download_whisper_base,
    "silero-vad": _download_silero,
    "mfa-en": lambda: _download_mfa("english"),
    "mfa-fr": lambda: _download_mfa("french"),
    "espeak-ng": _check_espeak,
}

ORDERED_KEYS = ("whisper-base", "silero-vad", "mfa-en", "mfa-fr", "espeak-ng")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all models required by Phoneta (one-shot, offline thereafter)."
    )
    parser.add_argument("--verify", action="store_true", help="Check only; don't download.")
    parser.add_argument(
        "--model",
        choices=list(ORDERED_KEYS),
        help="Download a single model instead of all.",
    )
    return parser


def run_downloads(model: str | None, verify_only: bool) -> int:
    """Run all (or one) download step(s).  Returns 0 on success, 1 on any failure."""
    from phoneta.models.registry import missing_models

    if model:
        keys = [model]
    else:
        missing = {m.name for m in missing_models()}
        if not missing:
            _report("All models already present — nothing to download.")
            return 0
        keys = [k for k in ORDERED_KEYS if k in missing]

    if verify_only:
        _report("Verifying model availability (no downloads) …")
        failed = sum(1 for k in keys if not DOWNLOADERS.get(k, lambda: True)())  # type: ignore[operator]
        if failed:
            _fail(f"{failed} model(s) missing. Run without --verify to download.")
            return 1
        _ok("All checked models are present.")
        return 0

    errors = 0
    for key in keys:
        downloader = DOWNLOADERS.get(key)
        if downloader is None:
            _fail(f"Unknown model: {key}")
            errors += 1
            continue
        if not downloader():
            errors += 1

    if errors:
        _fail(f"{errors} download(s) failed.")
        return 1

    _ok("All models downloaded — Phoneta is ready for offline use.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_downloads(model=args.model, verify_only=args.verify)


if __name__ == "__main__":
    sys.exit(main())