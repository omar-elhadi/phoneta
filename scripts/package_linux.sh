#!/usr/bin/env bash
# Build a standalone Phoneta AppImage / directory for Linux.
#
# Prerequisites:
#   - Python 3.10+ with venv
#   - espeak-ng installed (system package)
#   - pyinstaller installed (this script installs it in a venv)
#
# Usage:
#   ./scripts/package_linux.sh          # build in dist/Phoneta/
#   ./scripts/package_linux.sh --clean  # clean rebuild
#
# Output:
#   dist/Phoneta/Phoneta               # standalone executable

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[package]${NC} $*"; }
ok()    { echo -e "${GREEN}  ✓${NC} $*"; }
die()   { echo -e "${RED}  ✗${NC} $*" >&2; exit 1; }

# ── args ───────────────────────────────────────────────────────────
CLEAN=0
case "${1:-}" in
    --clean) CLEAN=1 ;;
    "")      ;;
    *)       echo "Usage: $0 [--clean]" >&2; exit 2 ;;
esac

# ── prerequisites ──────────────────────────────────────────────────
info "Checking prerequisites …"

command -v python3 >/dev/null || die "python3 not found"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "python $PYVER"

# espeak-ng check (warn only — models will be downloaded at first run)
if command -v espeak-ng >/dev/null; then
    ok "espeak-ng found"
else
    echo -e "  ${RED}⚠${NC}  espeak-ng not found — install it via your package manager"
fi

# ── venv ───────────────────────────────────────────────────────────
if [ "$CLEAN" -eq 1 ]; then
    info "Cleaning previous build …"
    rm -rf build/ dist/ .venv-build/
fi

if [ ! -d .venv-build ]; then
    info "Creating build venv …"
    python3 -m venv .venv-build
fi

# shellcheck source=/dev/null
source .venv-build/bin/activate

info "Installing build dependencies …"
pip install --upgrade pip -q
pip install pyinstaller -q
pip install -e ".[all]" -q

# ── build ──────────────────────────────────────────────────────────
info "Running PyInstaller …"
pyinstaller phoneta.spec --noconfirm --clean

# ── verify ─────────────────────────────────────────────────────────
info "Verifying output …"

OUT_DIR="dist/Phoneta"
EXECUTABLE="$OUT_DIR/Phoneta"

if [ -f "$EXECUTABLE" ]; then
    SIZE=$(du -sh "$OUT_DIR" | cut -f1)
    ok "build successful — $OUT_DIR ($SIZE)"
else
    die "executable not found at $EXECUTABLE"
fi

# Quick smoke test (--version should exit 0)
info "Smoke testing …"
if "$EXECUTABLE" --version 2>/dev/null; then
    ok "--version works"
else
    die "executable failed to run --version"
fi

deactivate

info ""
info "Build complete: $(realpath "$OUT_DIR")"
info "To run:  $EXECUTABLE"
info ""

# ── desktop entry (optional) ───────────────────────────────────────
DESKTOP_FILE="$OUT_DIR/phoneta.desktop"
cat > "$DESKTOP_FILE" << 'DESKTOPEOF'
[Desktop Entry]
Type=Application
Name=Phoneta
Comment=Offline Pronunciation Coach (English + French)
Exec=Phoneta
Path=.
Icon=phoneta
Categories=Education;Languages;
Terminal=false
DESKTOPEOF
ok "Desktop entry written to $DESKTOP_FILE"