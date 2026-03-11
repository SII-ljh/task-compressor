#!/usr/bin/env bash
# ============================================================
# Compression Ratio Ablation — 8×H200
# ============================================================
# Runs 8 experiments with different soft prompt counts (k):
#   k=16   (256x compression)  n_p=4,   n_c=12
#   k=32   (128x compression)  n_p=8,   n_c=24
#   k=64   ( 64x compression)  n_p=16,  n_c=48    ← baseline
#   k=128  ( 32x compression)  n_p=32,  n_c=96
#   k=256  ( 16x compression)  n_p=64,  n_c=192
#   k=512  (  8x compression)  n_p=128, n_c=384
#   k=1024 (  4x compression)  n_p=256, n_c=768
#   k=2048 (  2x compression)  n_p=512, n_c=1536
#
# If one experiment fails, it logs the failure and continues
# with the remaining experiments.
#
# Usage:
#   bash scripts/run_compression_ablation.sh              # run all
#   bash scripts/run_compression_ablation.sh 64           # run k=64 only
#   bash scripts/run_compression_ablation.sh 16 32 64     # run selected
# ============================================================

set -uo pipefail
# NOTE: no 'set -e' — we handle errors per-experiment

CONFIG="configs/h200_base.yaml"
NUM_GPUS=8
BASE_OUTPUT="outputs/ablation_compression"

# ── Experiment definitions ──
# Format: "n_p n_c per_gpu_batch_size grad_accum"
# effective_batch = per_gpu_bs × num_gpus × grad_accum = 256 for all
declare -A EXPERIMENTS=(
    [16]="4 12 32 1"
    [32]="8 24 32 1"
    [64]="16 48 32 1"
    [128]="32 96 32 1"
    [256]="64 192 32 1"
    [512]="128 384 16 2"
    [1024]="256 768 8 4"
    [2048]="512 1536 4 8"
)

# ── Parse CLI args: which k values to run ──
if [ $# -gt 0 ]; then
    SELECTED_K=("$@")
else
    SELECTED_K=(16 32 64 128 256 512 1024 2048)
fi

# Track results
PASSED=()
FAILED=()
SKIPPED=()

echo "================================================================"
echo "  Compression Ratio Ablation — 8×H200"
echo "  Config:  ${CONFIG}"
echo "  GPUs:    ${NUM_GPUS}"
echo "  K values: ${SELECTED_K[*]}"
echo "================================================================"

for K in "${SELECTED_K[@]}"; do
    if [ -z "${EXPERIMENTS[$K]+x}" ]; then
        echo "ERROR: Unknown k=${K}. Valid values: 16, 32, 64, 128, 256, 512, 1024, 2048"
        FAILED+=("k=${K} (unknown)")
        continue
    fi

    read -r NP NC BS GA <<< "${EXPERIMENTS[$K]}"
    RUN_NAME="k${K}_np${NP}_nc${NC}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Experiment: k=${K}  (n_p=${NP}, n_c=${NC})"
    echo "  Compression: ~$((4096 / K))x  (4096 context → ${K} soft prompts)"
    echo "  Batch: per_gpu=${BS}, grad_accum=${GA}, effective=$((BS * NUM_GPUS * GA))"
    echo "  Output: ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────────────────"

    # Generate per-experiment config YAML (so evaluate_qa_detailed.py can find it)
    EXP_CONFIG="configs/${RUN_NAME}.yaml"
    if [ ! -f "${EXP_CONFIG}" ]; then
        python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['model']['n_prompt_tokens'] = ${NP}
cfg['model']['n_context_tokens'] = ${NC}
cfg['training']['per_gpu_batch_size'] = ${BS}
cfg['training']['gradient_accumulation_steps'] = ${GA}
cfg['output_dir'] = '${OUTPUT_DIR}'
cfg['wandb_run_name'] = '${RUN_NAME}'
with open('${EXP_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
        echo "  → Generated ${EXP_CONFIG}"
    fi

    # Skip if final checkpoint already exists
    if [ -d "${OUTPUT_DIR}/final" ]; then
        echo "  → SKIPPING: final checkpoint already exists"
        SKIPPED+=("k=${K}")
        continue
    fi

    if torchrun --nproc_per_node=${NUM_GPUS} \
        scripts/train.py --config ${EXP_CONFIG} ; then
        echo "  → PASSED: k=${K}"
        PASSED+=("k=${K}")
    else
        echo "  → FAILED: k=${K} (exit code $?)"
        echo "  → Continuing with next experiment..."
        FAILED+=("k=${K}")
    fi
done

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Ablation Summary"
echo "================================================================"
echo "  Passed:  ${#PASSED[@]}  ${PASSED[*]:-}"
echo "  Failed:  ${#FAILED[@]}  ${FAILED[*]:-}"
echo "  Skipped: ${#SKIPPED[@]}  ${SKIPPED[*]:-}"
echo "  Results: ${BASE_OUTPUT}/"
echo "================================================================"

# Exit with error if any experiment failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
