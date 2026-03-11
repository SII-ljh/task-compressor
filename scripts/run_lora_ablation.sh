#!/usr/bin/env bash
# ============================================================
# LoRA Ablation — 8×H200, max_prompt_length=512
# ============================================================
# Ablates LoRA rank and target modules while keeping compression
# ratio fixed (k=64, n_p=16, n_c=48).
# Context=4096, prompt(Q)=512, response=512.
#
# Experiments:
#   no_lora     — rank=0   (frozen encoder, perceiver-only)
#   r8          — rank=8,   target=q,v
#   r16         — rank=16,  target=q,v
#   r32         — rank=32,  target=q,v
#   r64         — rank=64,  target=q,v
#   r128        — rank=128, target=q,v         ← baseline
#   r256        — rank=256, target=q,v
#   r128_qkvo   — rank=128, target=q,k,v,o
#   r128_all    — rank=128, target=all linear
#
# Training: total_steps=50000 with early stopping (patience=10).
#
# Usage:
#   bash scripts/run_lora_ablation.sh              # run all
#   bash scripts/run_lora_ablation.sh r128          # run baseline only
#   bash scripts/run_lora_ablation.sh no_lora r128  # run selected
# ============================================================

set -uo pipefail
# NOTE: no 'set -e' — we handle errors per-experiment

CONFIG="configs/h200_base.yaml"
NUM_GPUS=8
BASE_OUTPUT="outputs/ablation_lora"

# ── Experiment definitions ──
# Format: "rank alpha target_modules per_gpu_bs grad_accum"
# target_modules: comma-separated, converted to YAML list in Python
# effective_batch = per_gpu_bs × num_gpus × grad_accum
ALL_EXPERIMENTS="no_lora r8 r16 r32 r64 r128 r256 r128_qkvo r128_all"

get_experiment_params() {
    local name=$1
    case "$name" in
        no_lora)    echo "0 0 q_proj,v_proj 32 1" ;;
        r8)         echo "8 8 q_proj,v_proj 32 1" ;;
        r16)        echo "16 16 q_proj,v_proj 32 1" ;;
        r32)        echo "32 32 q_proj,v_proj 32 1" ;;
        r64)        echo "64 64 q_proj,v_proj 32 1" ;;
        r128)       echo "128 128 q_proj,v_proj 32 1" ;;
        r256)       echo "256 256 q_proj,v_proj 32 1" ;;
        r128_qkvo)  echo "128 128 q_proj,k_proj,v_proj,o_proj 32 1" ;;
        r128_all)   echo "128 128 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 32 1" ;;
        *)          echo "" ;;
    esac
}

# ── Parse CLI args: which experiments to run ──
if [ $# -gt 0 ]; then
    SELECTED=("$@")
else
    SELECTED=(no_lora r8 r16 r32 r64 r128 r256 r128_qkvo r128_all)
fi

# Track results
PASSED=()
FAILED=()
SKIPPED=()

echo "================================================================"
echo "  LoRA Ablation — 8×H200, prompt(Q)=512"
echo "  Config:      ${CONFIG}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Experiments: ${SELECTED[*]}"
echo "================================================================"

for EXP_NAME in "${SELECTED[@]}"; do
    PARAMS=$(get_experiment_params "$EXP_NAME")
    if [ -z "$PARAMS" ]; then
        echo "ERROR: Unknown experiment '${EXP_NAME}'. Valid: ${ALL_EXPERIMENTS}"
        FAILED+=("${EXP_NAME} (unknown)")
        continue
    fi

    read -r RANK ALPHA TARGETS BS GA <<< "$PARAMS"
    RUN_NAME="lora_${EXP_NAME}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Experiment: ${EXP_NAME}"
    echo "  LoRA: rank=${RANK}, alpha=${ALPHA}, targets=${TARGETS}"
    echo "  Batch: per_gpu=${BS}, grad_accum=${GA}, effective=$((BS * NUM_GPUS * GA))"
    echo "  Output: ${OUTPUT_DIR}"
    echo "────────────────────────────────────────────────────────────────"

    # Generate per-experiment config YAML
    EXP_CONFIG="configs/${RUN_NAME}.yaml"
    if [ ! -f "${EXP_CONFIG}" ]; then
        python3 -c "
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

# ── Model ──
cfg['model']['lora_rank'] = ${RANK}
cfg['model']['lora_alpha'] = ${ALPHA}
cfg['model']['lora_target_modules'] = '${TARGETS}'.split(',')

# ── Data: Q (prompt) length = 512 ──
cfg['data']['max_prompt_length'] = 512

# ── Training: large steps + early stopping ──
cfg['training']['per_gpu_batch_size'] = ${BS}
cfg['training']['gradient_accumulation_steps'] = ${GA}
cfg['training']['total_steps'] = 50000
cfg['training']['warmup_steps'] = 500
cfg['training']['eval_steps'] = 500
cfg['training']['save_steps'] = 2000
cfg['training']['early_stopping_patience'] = 10

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
        SKIPPED+=("${EXP_NAME}")
        continue
    fi

    if torchrun --nproc_per_node=${NUM_GPUS} \
        scripts/train.py --config "${EXP_CONFIG}" ; then
        echo "  → PASSED: ${EXP_NAME}"
        PASSED+=("${EXP_NAME}")
    else
        echo "  → FAILED: ${EXP_NAME} (exit code $?)"
        echo "  → Continuing with next experiment..."
        FAILED+=("${EXP_NAME}")
    fi
done

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  LoRA Ablation Summary"
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
