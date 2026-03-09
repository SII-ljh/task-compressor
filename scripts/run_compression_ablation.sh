#!/usr/bin/env bash
# ============================================================
# Compression Ratio Ablation — 8×H200
# ============================================================
# Runs 5 experiments with different soft prompt counts (k):
#   k=16  (256x compression)  n_p=4,  n_c=12
#   k=32  (128x compression)  n_p=8,  n_c=24
#   k=64  ( 64x compression)  n_p=16, n_c=48   ← baseline
#   k=128 ( 32x compression)  n_p=32, n_c=96
#   k=256 ( 16x compression)  n_p=64, n_c=192
#
# Usage:
#   bash scripts/run_compression_ablation.sh          # run all
#   bash scripts/run_compression_ablation.sh 64       # run k=64 only
#   bash scripts/run_compression_ablation.sh 16 32    # run k=16 and k=32
# ============================================================

set -euo pipefail

CONFIG="configs/h200_base.yaml"
NUM_GPUS=8
BASE_OUTPUT="outputs/ablation_compression"

# ── Experiment definitions ──
# Format: "k n_p n_c"
declare -A EXPERIMENTS=(
    [16]="4 12"
    [32]="8 24"
    [64]="16 48"
    [128]="32 96"
    [256]="64 192"
)

# ── Parse CLI args: which k values to run ──
if [ $# -gt 0 ]; then
    SELECTED_K=("$@")
else
    SELECTED_K=(16 32 64 128 256)
fi

echo "================================================================"
echo "  Compression Ratio Ablation — 8×H200"
echo "  Config:  ${CONFIG}"
echo "  GPUs:    ${NUM_GPUS}"
echo "  K values: ${SELECTED_K[*]}"
echo "================================================================"

for K in "${SELECTED_K[@]}"; do
    if [ -z "${EXPERIMENTS[$K]+x}" ]; then
        echo "ERROR: Unknown k=${K}. Valid values: 16, 32, 64, 128, 256"
        exit 1
    fi

    read -r NP NC <<< "${EXPERIMENTS[$K]}"
    RUN_NAME="k${K}_np${NP}_nc${NC}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Experiment: k=${K}  (n_p=${NP}, n_c=${NC})"
    echo "  Compression: ~$((4096 / K))x  (4096 context → ${K} soft prompts)"
    echo "  Output: ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────────────────"

    # Skip if final checkpoint already exists
    if [ -d "${OUTPUT_DIR}/final" ]; then
        echo "  → SKIPPING: final checkpoint already exists"
        continue
    fi

    torchrun --nproc_per_node=${NUM_GPUS} \
        scripts/train.py --config ${CONFIG} \
        model.n_prompt_tokens=${NP} \
        model.n_context_tokens=${NC} \
        output_dir="${OUTPUT_DIR}" \
        wandb_run_name="${RUN_NAME}"

    echo "  → Completed: k=${K}"
done

echo ""
echo "================================================================"
echo "  All experiments finished!"
echo "  Results in: ${BASE_OUTPUT}/"
echo "================================================================"
