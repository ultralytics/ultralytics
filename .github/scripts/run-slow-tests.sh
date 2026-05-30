#!/usr/bin/env bash
# Run slow tests with enough diagnostics for silent runner failures.

set -euo pipefail

uv_system_args=()
[[ "${MATRIX_OS:-}" == "cpu-latest" ]] || uv_system_args=(--system)

snapshot() {
  echo "Resource snapshot: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  df -h .
  command -v free > /dev/null 2>&1 && free -h
  [[ "$(uname -s)" == "Linux" ]] && ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -15 || true
}

restore_torch() {
  [[ "${MATRIX_TORCH:-latest}" == "latest" ]] && return
  echo "Restoring torch==${MATRIX_TORCH} torchvision==${MATRIX_TORCHVISION}"
  uv pip install "${uv_system_args[@]}" "torch==${MATRIX_TORCH}" "torchvision==${MATRIX_TORCHVISION}" \
    --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
}

snapshot
if [[ "$(uname -s)" == "Linux" ]]; then
  while sleep 60; do snapshot; done &
  trap 'kill "$!" 2>/dev/null || true' EXIT
fi

restore_torch
pytest_workers=2
pytest_cmd=(pytest -n "$pytest_workers" --dist=loadfile)
if [[ "${MATRIX_OS:-}" == "ubuntu-24.04-arm" ]]; then
  pytest_workers=1
  pytest_cmd=(pytest -n "$pytest_workers" --dist=loadfile --forked)
fi

main_args=(
  -vv -s --slow --cov=ultralytics/ --cov-report=xml --durations=50 --durations-min=1 tests/
  --deselect tests/test_exports.py::test_export_imx
  --deselect tests/test_exports.py::test_export_mnn
  --deselect tests/test_exports.py::test_export_ncnn
  --deselect tests/test_exports.py::test_export_executorch
  --deselect tests/test_exports.py::test_export_axelera
  --deselect tests/test_exports.py::test_export_deepx
)
PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 "${pytest_cmd[@]}" "${main_args[@]}"

for export_test in tests/test_exports.py::test_export_axelera tests/test_exports.py::test_export_deepx; do
  rm -rf "${TMPDIR:-/tmp}"/pytest-of-* ~/.cache/pip || true
  uv cache prune --ci || true
  python - <<'PY'
import os
from pathlib import Path

if cache_dir := os.environ.get("UV_CACHE_DIR"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
PY
  snapshot
  restore_torch
  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 pytest -vv -s --slow --cov=ultralytics/ --cov-append \
    --cov-report=xml --durations=10 --durations-min=1 "$export_test"
done
