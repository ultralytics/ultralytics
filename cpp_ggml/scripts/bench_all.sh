#!/usr/bin/env bash
#
# bench_all.sh — collect the full performance matrix into benchmarks/bench.jsonl
#
# Runs yolo-cli bench for every (model x dtype) on the requested backend build
# directory and appends one JSON line per run. Re-runs overwrite nothing: the
# plot script de-duplicates by (backend, model, dtype) keeping the last entry.
#
# Usage: scripts/bench_all.sh BUILD_DIR [BACKEND_TAG] [models...]
#   scripts/bench_all.sh build-cuda cuda
#   scripts/bench_all.sh build-cpu cpu yolo26n yolov8n   # subset
#
# GPU runs get warmup 20 / iters 50 plus a 3 s cool-down between entries:
# back-to-back sweeps otherwise ride GPU clock/power transients and report
# inflated means (observed yolo26n-f16 CUDA 17 ms in-sweep vs 10 ms solo).

set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="${1:?build dir required, e.g. build-cuda}"
TAG="${2:-$BUILD}"
shift 2 || true
if (( $# )); then
    MODELS=("$@")
else
    MODELS=(yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x yolo26n-depth)
fi
read -r -a DTYPES <<< "${YOLO_BENCH_DTYPES:-f16 f32 q8_0}"
SRC="../ultralytics/assets/bus.jpg"

THREADS=0
GPU_ARGS=()
COOLDOWN=0
if [[ "$TAG" == "cpu" ]]; then THREADS=8; else GPU_ARGS=(--warmup 20 --iters 50); COOLDOWN=3; fi

mkdir -p benchmarks
OUT="${YOLO_BENCH_OUT:-benchmarks/bench.jsonl}"
echo "collecting ${#MODELS[@]} models x ${#DTYPES[@]} dtypes on $TAG -> $OUT"

for m in "${MODELS[@]}"; do
    for dt in "${DTYPES[@]}"; do
        f="models/gguf/$m-$dt.gguf"
        [[ -f "$f" ]] || { echo "[fail] $f missing" >&2; exit 1; }
        args=(--model "$f" --source "$SRC" "${GPU_ARGS[@]}")
        [[ "$THREADS" -gt 0 ]] && args+=(--threads "$THREADS")
        # CPU m/l/x are slow; trim iterations so the sweep stays tractable.
        if [[ "$TAG" == "cpu" && ("$m" == *m || "$m" == *l || "$m" == *x) ]]; then
            args+=(--iters 10 --warmup 3)
        elif [[ "$TAG" == "cpu" ]]; then
            args+=(--iters 30 --warmup 10)
        fi
        line=$(./"$BUILD"/bin/yolo-cli bench "${args[@]}" | tail -1)
        [[ "$line" == "{"* ]] || { echo "[fail] $m-$dt" >&2; exit 1; }
        # Reject silent GPU-to-CPU fallback, then normalize the verified backend tag for plots.
        line=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); tag=sys.argv[2]; actual=d['backend'].lower(); \
assert tag == 'cpu' and actual == 'cpu' or tag != 'cpu' and tag in actual, f'expected {tag}, got {actual}'; \
d['backend']=tag; print(json.dumps(d))" "$line" "$TAG")
        echo "$line" >> "$OUT"
        ms=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['e2e_ms']['mean'])" "$line")
        echo "[ok] $m-$dt e2e_ms=$ms"
        [[ "$COOLDOWN" -gt 0 ]] && sleep "$COOLDOWN"
    done
done
echo "done: $(wc -l < "$OUT") total entries in $OUT"
