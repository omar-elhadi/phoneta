# Phoneta

Offline, privacy-first pronunciation evaluator for English and French.

Record yourself reading a sentence, and Phoneta transcribes it, aligns it at the
phoneme level, and shows color-coded per-word/phoneme feedback plus pitch-contour
(prosody) analysis. Everything runs locally — no network calls, no telemetry, and
raw audio is deleted immediately after analysis.

## Status

Bootstrap: repository skeleton, packaging, and CI are in place. The engine and UI
are implemented in subsequent phases (see the project plan/tasks docs).

## Requirements

- Python 3.10+
- `espeak-ng` system binary (used by `phonemizer` for text→IPA)
- Models are downloaded once on first run (see below), then the app is fully offline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + dev tools
# Optional, full alignment stack:
pip install -e ".[align]"
```

## Run

```bash
phoneta --help
python -m phoneta --help
```

## Download models (one-time)

```bash
python scripts/download_models.py
```

## Test

```bash
pytest
ruff check .
mypy src
```

## Privacy

- Zero outbound network calls at runtime (enforced by an offline-audit test).
- Raw recordings are auto-deleted after analysis.
- Progress history is stored locally in SQLite only.
