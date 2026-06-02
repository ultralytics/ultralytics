#!/bin/bash
# Continuous nvidia-smi sampler. Started once at the start of the sweep, killed when sweep ends.
# Single CSV; the H12_SIZE_{START,END} markers in sweep.log slice it post-hoc.
#
# Usage: bash mem_logger.sh [out.csv] [interval_sec]
set -u
out="${1:-/home/rick/runs/reid/h12/mem.csv}"
interval="${2:-5}"
mkdir -p "$(dirname "$out")"

# CSV header. nvidia-smi appends rows without a header so we write our own.
echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,utilization_mem_pct,temp_c,power_w" > "$out"

trap 'echo "mem_logger exiting"; exit 0' INT TERM

while true; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
             --format=csv,noheader,nounits >> "$out" 2>/dev/null || true
  sleep "$interval"
done
