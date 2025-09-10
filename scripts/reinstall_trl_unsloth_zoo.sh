#!/usr/bin/env bash

# Fast local reinstall of patched TRL and Unsloth-Zoo
# Use this after you overwrite the libraries with the improved code in third_party/

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Ensure we run from project root (must contain third_party/)
if [[ ! -d "third_party" ]]; then
  log_error "Run this script from the project root (third_party/ not found)."
  exit 1
fi

ROOT_DIR=$(pwd)
cleanup() { cd "$ROOT_DIR"; }
trap cleanup EXIT

log_info "Reinstalling TRL and Unsloth-Zoo in editable mode..."

# --- TRL ---
log_info "Uninstalling existing trl (if any)..."
python -m pip uninstall -y trl || log_warn "trl may not be installed"

if [[ -d "third_party/trl-0.19.1" ]]; then
  log_info "Installing trl from third_party/trl-0.19.1 (editable)..."
  cd third_party/trl-0.19.1
  python -m pip install -e . || { log_error "Installing TRL failed"; exit 1; }
  cd "$ROOT_DIR"
  log_success "TRL installed"
else
  log_error "Directory third_party/trl-0.19.1 not found"
  exit 1
fi

# --- Unsloth-Zoo ---
log_info "Uninstalling existing unsloth-zoo (if any)..."
python -m pip uninstall -y unsloth-zoo || log_warn "unsloth-zoo may not be installed"

if [[ -d "third_party/unsloth-zoo" ]]; then
  log_info "Installing unsloth-zoo from third_party/unsloth-zoo (editable)..."
  cd third_party/unsloth-zoo
  python -m pip install -e . || { log_error "Installing Unsloth-Zoo failed"; exit 1; }
  cd "$ROOT_DIR"
  log_success "Unsloth-Zoo installed"
else
  log_error "Directory third_party/unsloth-zoo not found"
  exit 1
fi

log_info "Verifying imports..."
python - <<'PY'
try:
    import trl
    print("✅ TRL import ok", getattr(trl, "__version__", ""))
except Exception as e:
    print("❌ TRL import failed:", e)

try:
    import unsloth_zoo  # package import name
    print("✅ Unsloth-Zoo import ok")
except Exception as e:
    print("❌ Unsloth-Zoo import failed:", e)
PY

log_success "Done."
