#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────
# Phoneta — smart installer
#
# Detects whether an NVIDIA GPU is available and installs the
# appropriate PyTorch variant:
#   • GPU: torch + CUDA (~2.5 GB extra)
#   • CPU: torch CPU-only (~200 MB extra)
#
# Total install with GPU: ~4 GB   |   CPU-only: ~1.5 GB
#
# Usage:
#   bash scripts/install.sh           # auto-detect
#   bash scripts/install.sh --cpu     # force CPU
#   bash scripts/install.sh --gpu     # force GPU
# ───────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── detect GPU ────────────────────────────────────────────────────
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    HAS_GPU=true
fi

# ── parse args ────────────────────────────────────────────────────
MODE="auto"
if [[ "${1:-}" == "--cpu" ]]; then
    MODE="cpu"
elif [[ "${1:-}" == "--gpu" ]]; then
    MODE="gpu"
fi

if [[ "$MODE" == "auto" ]]; then
    if $HAS_GPU; then
        MODE="gpu"
    else
        MODE="cpu"
    fi
fi

echo "=== Phoneta installer ==="
echo "  mode : $MODE"
echo "  gpu  : $HAS_GPU (detected)"
echo ""

# ── install ───────────────────────────────────────────────────────
case "$MODE" in
    gpu)
        echo "→ Installing with GPU (CUDA) PyTorch …"
        pip install -e ".[gpu,full,align]"
        ;;
    cpu)
        echo "→ Installing with CPU-only PyTorch …"
        pip install -e ".[cpu,full,align]" --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

echo ""
echo "✓ Phoneta installed."
echo ""
echo "Next: download models (one-time, ~250 MB):"
echo "  python scripts/download_models.py"
echo ""
if [[ "$MODE" == "cpu" ]]; then
    echo "⚠  CPU mode — ASR and VAD will be slower. GPU recommended for real-time use."
fi