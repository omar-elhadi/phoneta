# Phoneta

Offline, privacy-first pronunciation evaluator for English and French.

Record yourself reading a sentence — Phoneta transcribes it, aligns it at the
phoneme level, and shows color-coded per-word/phoneme feedback plus pitch-contour
(prosody) analysis. Everything runs locally: no network calls, no telemetry, and
raw audio is deleted immediately after analysis.

## Status

MVP complete: recording, ASR, forced alignment, scoring, prosody, PySide6 UI,
SQLite history, model downloader, and offline-audit gate. See the [spec](./speech-pronunciation-evaluator-spec.md),
[plan](./speech-pronunciation-evaluator-plan.md), and [tasks](./speech-pronunciation-evaluator-tasks.md) for full details.

## Requirements

- Python 3.10+
- `espeak-ng` system package (used by `phonemizer` for text→IPA)
- GPU optional — `faster-whisper` runs on CPU

## Quick Start

```bash
# 1. Create venv and install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + test/lint tools

# 2. Download models (one-time, ~200 MB)
python scripts/download_models.py

# 3. Launch the desktop app
phoneta
# or: python -m phoneta
```

## Run Commands

```bash
phoneta --help                      # CLI usage
phoneta --version                   # version info
python -m phoneta                   # GUI (same as `phoneta`)
```

## Tests

```bash
pytest                              # 127 tests (unit + integration)
pytest tests/smoke/                 # Real ML components (needs models installed)

ruff check .                        # Lint
mypy src                            # Type-check
python -m compileall -q src tests   # Syntax-check
```

## Benchmark

```bash
PYTHONPATH=src python scripts/benchmark_latency.py
```

## Privacy Guarantees

| Guarantee | How |
|---|---|
| Zero network calls at runtime | Enforced by `tests/integration/test_offline_audit.py` |
| Raw audio auto-deleted | Pipeline `finally` block removes temp WAV |
| No telemetry | No outbound TCP/UDP anywhere in the source |
| Local-only history | SQLite database, never synced |

## Architecture

```
src/phoneta/
├── app.py              # QApplication bootstrap + model check
├── core/
│   ├── audio/          # recorder (sounddevice), VAD (silero)
│   ├── alignment/      # g2p (phonemizer), ASR (faster-whisper), MFA, sequence
│   ├── metrics/        # scoring (green/yellow/red), prosody (librosa pyin)
│   └── pipeline.py     # end-to-end orchestration
├── models/             # registry + checksums
├── storage/            # SQLite practice history
└── ui/                 # PySide6: main, recorder, results, inspector, privacy badge
```

## Packaging

```bash
bash scripts/package_linux.sh       # PyInstaller standalone .tar.gz
```