set -e
set -x

MODEL=$1

trtexec --onnx="${MODEL}.onnx" \
    --fp16 \
    --saveEngine="${MODEL}.engine" \
    --timingCacheFile="${MODEL}.engine.timing.cache" \
    --warmUp=500   \
    --duration=10  \
    --useCudaGraph  \
    --useSpinWait  \
    --noDataTransfers > /dev/null

trtexec \
    --fp16 \
    --loadEngine="${MODEL}.engine" \
    --timingCacheFile="${MODEL}.engine.timing.cache" \
    --warmUp=500   \
    --duration=10  \
    --useCudaGraph  \
    --useSpinWait  \
    --noDataTransfers