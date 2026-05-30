#!/usr/bin/env bash
# Run slow tests with diagnostics for resource-heavy exporter failures.

set -euo pipefail

diagnose_environment() {
  python - <<'PY'
import os
import platform
import shutil
import sys

total, used, free = shutil.disk_usage(".")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CPU count: {os.cpu_count()}")
print(f"Disk: total={total // 2**30}GiB used={used // 2**30}GiB free={free // 2**30}GiB")
PY
}

dump_resources() {
  echo "Resource snapshot: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  df -h .
  if command -v free >/dev/null 2>&1; then free -h; fi
  if [[ "$(uname -s)" == "Linux" ]] && command -v ps >/dev/null 2>&1; then
    ps -eo pid,ppid,pcpu,pmem,rss,vsz,comm --sort=-rss | head -20 || true
  fi
}

start_resource_monitor() {
  dump_resources
  if [[ "$(uname -s)" == "Linux" ]]; then
    while true; do
      sleep 60
      dump_resources
    done &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
  fi
}

uv_system_args=()
if [[ "${MATRIX_OS:-}" != "cpu-latest" ]]; then
  uv_system_args=(--system)
fi

restore_torch() {
  if [[ "${MATRIX_TORCH:-latest}" != "latest" ]]; then
    echo "Restoring torch==${MATRIX_TORCH} torchvision==${MATRIX_TORCHVISION}"
    uv pip install "${uv_system_args[@]}" "torch==${MATRIX_TORCH}" "torchvision==${MATRIX_TORCHVISION}" \
      --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    python - <<'PY'
import torch
import torchvision

print(f"Restored torch={torch.__version__} torchvision={torchvision.__version__}")
PY
  fi
}

clean_disk() {
  echo "Cleaning runner caches before isolated export test"
  uv cache clean || true
  rm -rf "${TMPDIR:-/tmp}"/pytest-of-* ~/.cache/pip ~/.cache/uv || true
  dump_resources
}

run_main_pytest() {
  local pytest_workers=2
  local pytest_extra=()

  if [[ "${MATRIX_OS:-}" == "ubuntu-24.04-arm" ]]; then
    pytest_workers=1
    pytest_extra+=(--forked)
  fi

  local pytest_args=(
    -vv
    -s
    --slow
    --cov=ultralytics/
    --cov-report=xml
    --durations=50
    --durations-min=1
    tests/
    --deselect tests/test_exports.py::test_export_imx
    --deselect tests/test_exports.py::test_export_mnn
    --deselect tests/test_exports.py::test_export_ncnn
    --deselect tests/test_exports.py::test_export_executorch
    --deselect tests/test_exports.py::test_export_axelera
    --deselect tests/test_exports.py::test_export_deepx
  )

  PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 pytest -n "$pytest_workers" --dist=loadfile \
    "${pytest_extra[@]}" "${pytest_args[@]}"
}

run_isolated_export_tests() {
  local export_tests=(
    tests/test_exports.py::test_export_axelera
    tests/test_exports.py::test_export_deepx
  )

  for export_test in "${export_tests[@]}"; do
    clean_disk
    restore_torch
    PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 pytest -vv -s --slow --cov=ultralytics/ --cov-append \
      --cov-report=xml --durations=10 --durations-min=1 "$export_test"
  done
}

diagnose_environment
start_resource_monitor
restore_torch
run_main_pytest
run_isolated_export_tests
