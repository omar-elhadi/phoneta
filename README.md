# Phoneta

Offline, privacy-first pronunciation evaluator for English and French.

Record yourself reading a sentence — Phoneta transcribes it, aligns it at the
phoneme level, and shows color-coded per-word/phoneme feedback plus pitch-contour
(prosody) analysis. Everything runs locally: no network calls, no telemetry, and
raw audio is deleted immediately after analysis.

## How It Works

1. **Type a phrase** (remembered for next time) and pick the language.
2. **Press Record** — one big toggle button with a live countdown and a
   colour-coded volume meter.
3. Recording stops automatically (or press Stop) and analysis runs instantly.
4. **Read your feedback**: every word is a coloured card — green = good,
   yellow = close, red = needs work — with an overall score banner.
5. **Tap any word** for the phoneme inspector: reference vs. your IPA,
   per-phoneme confidence, and prosody stats.
6. Edit the phrase and hit **Re-analyse** to score a new attempt against the
   same recording.

Everything is explained in-app under **ℹ About**, and errors come with
actionable hints (e.g. how to install a missing component).

## Status

v0.2.0 — MVP complete plus production UI polish: centralised design system,
recorder/results UX, settings persistence, desktop entry, and app icon. See the
[spec](./speech-pronunciation-evaluator-spec.md), [plan](./speech-pronunciation-evaluator-plan.md),
and [tasks](./speech-pronunciation-evaluator-tasks.md) for full details.

## Requirements

- Python 3.10+
- `espeak-ng` system package (used by `phonemizer` for text→IPA)
- GPU optional — `faster-whisper` runs on CPU

## Install

**GPU (NVIDIA CUDA) — ~4 GB, recommended for real-time use:**

```bash
bash scripts/install.sh             # auto-detects GPU and picks the right torch
```

**CPU-only — ~1.5 GB, slower ASR/VAD:**

```bash
bash scripts/install.sh --cpu       # skips CUDA, uses CPU torch (~200 MB instead of ~2.5 GB)
```

**Manual — pick your extras:**

```bash
python -m venv .venv
source .venv/bin/activate

# Fastest CI/dev (no torch — tests are mocked):
pip install -e ".[dev]"                            # ~800 MB

# Full GPU desktop:
pip install -e ".[full,align]"                     # ~4 GB

# Full CPU desktop:
pip install -e ".[cpu,full,align]" --index-url https://download.pytorch.org/whl/cpu   # ~1.5 GB
```

| Extra | Includes | Size |
|---|---|---|
| `[dev]` | pytest, mypy, ruff (no torch) | ~800 MB |
| `[vad]` | torch + torchaudio + silero-vad (voice detection) | +2.5 GB |
| `[cpu]` | same as `[vad]` but CPU-only torch | +200 MB |
| `[gpu]` | same as `[vad]` but CUDA torch | +2.5 GB |
| `[align]` | Montreal Forced Aligner | +500 MB |
| `[full]` | `[dev]` + `[vad]` | ~3.3 GB |
| `[all]` | `[full]` + `[align]` | ~4 GB |

## Model Download (one-time)

```bash
python scripts/download_models.py    # ~250 MB: Whisper base, silero, MFA
```

## Run Commands

```bash
phoneta --help                      # CLI usage
phoneta --version                   # version info
python -m phoneta                   # GUI (same as `phoneta`)
```

## Tests

```bash
pytest                              # unit + integration (faked ML deps)
pytest tests/smoke/                 # Real ML components (needs models installed)

ruff check .                        # Lint
mypy src                            # Type-check
python -m compileall -q src tests   # Syntax-check
```

## Desktop Integration (Linux)

Install a launcher entry so Phoneta shows up in your app menu:

```bash
sudo cp packaging/phoneta.desktop /usr/share/applications/
sudo cp src/phoneta/assets/phoneta.svg /usr/share/icons/hicolor/scalable/apps/phoneta.svg
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
├── assets/             # app icon (SVG, bundled with the package)
├── core/
│   ├── audio/          # recorder (sounddevice), VAD (silero)
│   ├── alignment/      # g2p (phonemizer), ASR (faster-whisper), MFA, sequence
│   ├── metrics/        # scoring (green/yellow/red), prosody (librosa pyin)
│   └── pipeline.py     # end-to-end orchestration
├── models/             # registry + checksums
├── storage/            # SQLite practice history
└── ui/                 # PySide6: theme, main window, recorder, results,
                        # inspector, privacy badge, setup screen, icon
```

## Packaging

```bash
bash scripts/package_linux.sh       # PyInstaller standalone .tar.gz
```