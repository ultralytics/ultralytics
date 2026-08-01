#!/bin/sh
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Install the Ultralytics `yolo` command, including Python itself if it is missing, using uv https://docs.astral.sh/uv/
# Example usage: curl -fsSL https://raw.githubusercontent.com/ultralytics/ultralytics/main/install.sh | sh
# Options: ULTRALYTICS_PYTHON=3.12 sets the Python version, ULTRALYTICS_PACKAGE=ultralytics sets the PyPI package

set -eu

python_version=${ULTRALYTICS_PYTHON:-3.12}
package=${ULTRALYTICS_PACKAGE:-}
bin_dir=${XDG_BIN_HOME:-$HOME/.local/bin}

info() { printf '\033[1;34mUltralytics\033[0m %s\n' "$1"; }
fail() {
  printf '\033[1;31mUltralytics\033[0m %s\n' "$1" >&2
  exit 1
}
has() { command -v "$1" > /dev/null 2>&1; }

# Machines without a display get opencv-python-headless to avoid libGL errors
if [ -z "$package" ]; then
  if [ "$(uname -s)" = "Linux" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    package=ultralytics-opencv-headless
  else
    package=ultralytics
  fi
fi

# uv downloads and manages Python, so no system Python is required
if ! has uv; then
  info "installing uv..."
  if has curl; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
  elif has wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    fail "curl or wget is required"
  fi
  PATH="$bin_dir:$PATH"
  export PATH
  has uv || fail "uv was installed but is not on PATH, restart your shell and rerun"
fi

info "installing $package on Python $python_version, this may take a few minutes..."
uv tool install --upgrade --python-preference only-managed --python "$python_version" "$package"
uv tool update-shell || true # add the tool directory to PATH in your shell profile
PATH="$(uv tool dir --bin 2> /dev/null || echo "$bin_dir"):$PATH"
export PATH

yolo version || fail "installation failed"
info "run 'yolo predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg' to get started"
info "open a new terminal if the 'yolo' command is not found"
