#!/usr/bin/env bash
# setup-vllm.sh: run ONCE on the cheap build box (e.g. t3a.large, x86 Ubuntu or Amazon Linux 2023).
# Installs vLLM into ~/vllm-env so you can save an AMI and launch a g5 from it.
# No GPU needed for this step.
#
# Default: pip-install a prebuilt wheel, with TMPDIR on the root disk.
#   Amazon Linux 2023's /tmp is a small tmpfs (~7.8G); leaving pip there
#   fails with "No space left on device" even when the disk has free space.
# Optional: BUILD_FROM_SOURCE=1 clones the repo and does an editable install
#   with VLLM_USE_PRECOMPILED=1 (no CUDA kernel compile).
# Pin: VLLM_VERSION=v0.34.0 (git tag; the leading v is stripped for pip).
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-v0.34.0}"
VLLM_PIP_VERSION="${VLLM_VERSION#v}"
BUILD_FROM_SOURCE="${BUILD_FROM_SOURCE:-0}"
VENV="$HOME/vllm-env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -m)" == "aarch64" ]]; then
  echo "This box is ARM (Graviton). Build on x86 (t3/t3a/c5) so the AMI can boot on a g5." >&2
  exit 1
fi

# Keep pip's temp files on disk, not on tmpfs /tmp.
export TMPDIR="${TMPDIR:-$HOME/piptmp}"
mkdir -p "$TMPDIR"
echo "==> TMPDIR=$TMPDIR (pip builds go on the root disk, not /tmp)"

echo "==> System packages"
if command -v apt-get >/dev/null; then          # Ubuntu / Deep Learning AMI (Ubuntu)
  sudo apt-get update -y
  sudo apt-get install -y git gcc python3-venv python3-pip python3-dev tmux htop
elif command -v dnf >/dev/null; then            # Amazon Linux 2023
  # gcc + python3.11-devel are required: Triton compiles a C helper at runtime
  # and needs Python.h. Without them, vLLM dies at engine start with
  # "Python.h: No such file or directory".
  sudo dnf install -y git gcc python3.11 python3.11-pip python3.11-devel tmux
  PY=python3.11
else
  echo "Unsupported OS: need apt-get or dnf" >&2; exit 1
fi
PY="${PY:-python3}"

echo "==> Python venv at $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

if [[ "$BUILD_FROM_SOURCE" == "1" ]]; then
  echo "==> vLLM $VLLM_VERSION (editable install, precompiled kernels)"
  cd "$HOME"
  if [[ ! -d vllm ]]; then
    git clone --branch "$VLLM_VERSION" --depth 1 https://github.com/vllm-project/vllm.git
  fi
  cd vllm
  VLLM_USE_PRECOMPILED=1 python -m pip install -e .
else
  echo "==> vLLM $VLLM_PIP_VERSION (prebuilt wheel)"
  python -m pip install "vllm==$VLLM_PIP_VERSION"
fi
python -m pip install "huggingface_hub[cli]"

echo "==> Install the serve script"
mkdir -p "$HOME/bin" "$HOME/vllm-logs"
if [[ -f "$SCRIPT_DIR/serve-vllm.sh" ]]; then
  cp "$SCRIPT_DIR/serve-vllm.sh" "$HOME/bin/serve-vllm.sh"
  chmod +x "$HOME/bin/serve-vllm.sh"
  echo "Installed $HOME/bin/serve-vllm.sh"
else
  echo "WARNING: serve-vllm.sh not found next to setup-vllm.sh ($SCRIPT_DIR)." >&2
  echo "Copy it onto this instance, then:" >&2
  echo "  mkdir -p ~/bin ~/vllm-logs && cp serve-vllm.sh ~/bin/ && chmod +x ~/bin/serve-vllm.sh" >&2
fi

python -c "import vllm; print('vLLM installed:', vllm.__version__)"
echo
echo "Done. Next: save an AMI of this instance (EC2 > Instances > Actions > Image and templates > Create image),"
echo "then launch a g5.xlarge/g5.2xlarge from that AMI and run: ~/bin/serve-vllm.sh"
echo "To rebuild from the git tree instead of the wheel: BUILD_FROM_SOURCE=1 bash setup-vllm.sh"
