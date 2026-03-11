#!/usr/bin/env bash
# ============================================================
# Training — 8×H200, Q(prompt)=2048, LoRA r=128
# ============================================================
# Context=4096, prompt(Q)=2048, response=512.
# LoRA: rank=128, alpha=128, target=q,v.
# Compression: k=64 (n_p=16, n_c=48).
# Training: total_steps=50000 with early stopping (patience=10).
#
# Usage:
#   bash scripts/run_train_q2048.sh
# ============================================================

set -euo pipefail

CONFIG="configs/h200_base.yaml"
NUM_GPUS=8
RUN_NAME="train_q2048_r128"
OUTPUT_DIR="outputs/${RUN_NAME}"
EXP_CONFIG="configs/${RUN_NAME}.yaml"

echo "================================================================"
echo "  Training — Q=2048, LoRA r=128, 50k steps"
echo "  Config:   ${CONFIG} → ${EXP_CONFIG}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Output:   ${OUTPUT_DIR}"
echo "================================================================"

# Generate experiment config YAML
if [ ! -f "${EXP_CONFIG}" ]; then
    python3 -c "
import yaml

with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

# ── Model ──
cfg['model']['lora_rank'] = 128
cfg['model']['lora_alpha'] = 128
cfg['model']['lora_target_modules'] = ['q_proj', 'v_proj']

# ── Data: Q (prompt) length = 2048 ──
cfg['data']['max_prompt_length'] = 2048

# ── Training ──
cfg['training']['per_gpu_batch_size'] = 32
cfg['training']['gradient_accumulation_steps'] = 1
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
    echo "  → SKIPPING: final checkpoint already exists at ${OUTPUT_DIR}/final"
    exit 0
fi

torchrun --nproc_per_node=${NUM_GPUS} \
    scripts/train.py --config "${EXP_CONFIG}"

echo "  → Training complete. Output: ${OUTPUT_DIR}"
