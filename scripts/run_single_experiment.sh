#!/usr/bin/env bash
# ============================================================
# Run a single compression experiment — 8×H200
# ============================================================
# Usage:
#   bash scripts/run_single_experiment.sh k64_baseline
#   bash scripts/run_single_experiment.sh k128_lowcomp
#
# Or directly with torchrun:
#   torchrun --nproc_per_node=8 scripts/train.py \
#       --config configs/h200_base.yaml \
#       model.n_prompt_tokens=16 model.n_context_tokens=48
# ============================================================

set -euo pipefail

NUM_GPUS=8
CONFIG="configs/h200_base.yaml"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name>"
    echo ""
    echo "Available experiments:"
    echo "  k16_extreme    k=16  (256x compression)  n_p=4,  n_c=12"
    echo "  k32_high       k=32  (128x compression)  n_p=8,  n_c=24"
    echo "  k64_baseline   k=64  ( 64x compression)  n_p=16, n_c=48"
    echo "  k128_moderate  k=128 ( 32x compression)  n_p=32, n_c=96"
    echo "  k256_low       k=256 ( 16x compression)  n_p=64, n_c=192"
    exit 1
fi

EXPERIMENT=$1

case ${EXPERIMENT} in
    k16_extreme)
        NP=4; NC=12; K=16 ;;
    k32_high)
        NP=8; NC=24; K=32 ;;
    k64_baseline)
        NP=16; NC=48; K=64 ;;
    k128_moderate)
        NP=32; NC=96; K=128 ;;
    k256_low)
        NP=64; NC=192; K=256 ;;
    *)
        echo "Unknown experiment: ${EXPERIMENT}"
        exit 1 ;;
esac

RUN_NAME="k${K}_np${NP}_nc${NC}"
OUTPUT_DIR="outputs/ablation_compression/${RUN_NAME}"

echo "================================================================"
echo "  Experiment: ${EXPERIMENT}"
echo "  k=${K}  n_p=${NP}  n_c=${NC}"
echo "  Compression: ~$((4096 / K))x"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"

torchrun --nproc_per_node=${NUM_GPUS} \
    scripts/train.py --config ${CONFIG} \
    model.n_prompt_tokens=${NP} \
    model.n_context_tokens=${NC} \
    output_dir="${OUTPUT_DIR}" \
    wandb_run_name="${RUN_NAME}"

echo "Done: ${OUTPUT_DIR}"
