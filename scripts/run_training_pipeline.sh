#!/usr/bin/env bash
# ============================================================
# Task Compressor — Full Training Pipeline
# ============================================================
#
# Sequentially runs sanity checks → main training → evaluation.
# Each stage gates on the previous: if a sanity check fails,
# the pipeline stops before wasting GPU hours.
#
# Usage:
#   # Full pipeline: sanity checks → train → eval (8×H200)
#   bash scripts/run_training_pipeline.sh
#
#   # Skip sanity checks, go straight to training
#   bash scripts/run_training_pipeline.sh --skip-sanity
#
#   # Only run sanity checks (no main training)
#   bash scripts/run_training_pipeline.sh --sanity-only
#
#   # Single GPU mode
#   bash scripts/run_training_pipeline.sh --gpus 1
#
#   # Custom config
#   bash scripts/run_training_pipeline.sh --config configs/h200_base.yaml
#
#   # Resume from checkpoint
#   bash scripts/run_training_pipeline.sh --resume outputs/step_2000
#
#   # Override any training config
#   bash scripts/run_training_pipeline.sh -- training.total_steps=10000
# ============================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────
NUM_GPUS=8
CONFIG="configs/h200_base.yaml"
SKIP_SANITY=false
SANITY_ONLY=false
RESUME=""
EXTRA_OVERRIDES=()
OUTPUT_DIR=""  # empty = use config default

# ── Parse arguments ───────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --config)
            CONFIG="$2"; shift 2 ;;
        --skip-sanity)
            SKIP_SANITY=true; shift ;;
        --sanity-only)
            SANITY_ONLY=true; shift ;;
        --resume)
            RESUME="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --)
            shift; EXTRA_OVERRIDES=("$@"); break ;;
        *)
            EXTRA_OVERRIDES+=("$1"); shift ;;
    esac
done

# ── Derived variables ─────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="outputs/pipeline_${TIMESTAMP}.log"
mkdir -p outputs

# Helper: log with timestamp
log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${PIPELINE_LOG}"
}

# Helper: run a command and capture exit code
run_stage() {
    local stage_name="$1"
    shift
    log "────────────────────────────────────────────────────"
    log "STAGE: ${stage_name}"
    log "CMD: $*"
    log "────────────────────────────────────────────────────"

    local start_time=$SECONDS
    if "$@" 2>&1 | tee -a "${PIPELINE_LOG}"; then
        local elapsed=$(( SECONDS - start_time ))
        log "PASS: ${stage_name} (${elapsed}s)"
        return 0
    else
        local elapsed=$(( SECONDS - start_time ))
        log "FAIL: ${stage_name} (${elapsed}s)"
        return 1
    fi
}

# ── Banner ────────────────────────────────────────────────────
echo "================================================================"
echo "  Task Compressor — Training Pipeline"
echo "================================================================"
echo "  Config:       ${CONFIG}"
echo "  GPUs:         ${NUM_GPUS}"
echo "  Skip sanity:  ${SKIP_SANITY}"
echo "  Sanity only:  ${SANITY_ONLY}"
echo "  Resume:       ${RESUME:-'(none)'}"
echo "  Extra args:   ${EXTRA_OVERRIDES[*]:-'(none)'}"
echo "  Log:          ${PIPELINE_LOG}"
echo "================================================================"
echo ""

PIPELINE_START=$SECONDS

# ══════════════════════════════════════════════════════════════
#  Stage 0: Prerequisites
# ══════════════════════════════════════════════════════════════

log "Checking prerequisites..."

# Check config exists
if [ ! -f "${CONFIG}" ]; then
    log "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# Extract data paths from config
TRAIN_FILE=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('data', {}).get('train_file', 'data/qa_train.json'))
")
DEV_FILE=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('data', {}).get('dev_file', 'data/qa_dev.json'))
")
BASE_MODEL=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('model', {}).get('base_model', 'models/Qwen3-0.6B'))
")

# Check model
if [ ! -d "${BASE_MODEL}" ]; then
    log "Model not found: ${BASE_MODEL}"
    log "Downloading model..."
    python scripts/download_models.py || {
        log "ERROR: Model download failed. Run manually:"
        log "  python scripts/download_models.py"
        exit 1
    }
fi

# Check data
if [ ! -f "${TRAIN_FILE}" ]; then
    log "ERROR: Training data not found: ${TRAIN_FILE}"
    log "Run: python scripts/prepare_data.py"
    exit 1
fi
if [ ! -f "${DEV_FILE}" ]; then
    log "ERROR: Dev data not found: ${DEV_FILE}"
    log "Run: python scripts/prepare_data.py"
    exit 1
fi

TRAIN_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${TRAIN_FILE}'))))")
DEV_SAMPLES=$(python3 -c "import json; print(len(json.load(open('${DEV_FILE}'))))")
log "Data:  ${TRAIN_FILE} (${TRAIN_SAMPLES} samples)"
log "Dev:   ${DEV_FILE} (${DEV_SAMPLES} samples)"
log "Model: ${BASE_MODEL}"

# Check CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    log "WARNING: CUDA not available. Training will be very slow."
fi

log "Prerequisites OK"
echo ""

# ══════════════════════════════════════════════════════════════
#  Stage 1: Sanity Check — Single Sample Overfit
# ══════════════════════════════════════════════════════════════

if [ "${SKIP_SANITY}" = false ]; then

    log "=== SANITY CHECK 1/2: Single sample overfit ==="

    STEP1_OUTPUT="outputs/pipeline_sanity/step1_${TIMESTAMP}"

    run_stage "Single-sample overfit (step1)" \
        python scripts/overfitting/step1_single_sample.py \
            --config "${CONFIG}" \
            --data_path "${TRAIN_FILE}" \
            --steps 300 \
            --lr 3e-4 \
            --lora_lr 3e-5 \
            --log_every 10 \
            --output_dir "${STEP1_OUTPUT}" \
    || {
        log "ABORT: Single-sample overfit failed. Pipeline stopped."
        log "Check output: ${STEP1_OUTPUT}/"
        exit 1
    }

    # Check if step1 actually converged (PPL < 50)
    STEP1_PPL=$(python3 -c "
import csv, sys
with open('${STEP1_OUTPUT}/learning_curve.csv') as f:
    rows = list(csv.DictReader(f))
    if rows:
        ppl = float(rows[-1]['train_ppl'])
        print(f'{ppl:.1f}')
    else:
        print('999')
")
    log "Step 1 final PPL: ${STEP1_PPL}"

    # Lenient threshold — just needs to show learning
    if python3 -c "exit(0 if float('${STEP1_PPL}') < 50 else 1)"; then
        log "Step 1 PASS (PPL=${STEP1_PPL} < 50)"
    else
        log "WARNING: Step 1 PPL=${STEP1_PPL} >= 50, model may not be learning well."
        log "Continuing anyway — check diagnostics if main training fails."
    fi

    echo ""

    # ══════════════════════════════════════════════════════════════
    #  Stage 2: Sanity Check — Small Batch Memorization
    # ══════════════════════════════════════════════════════════════

    log "=== SANITY CHECK 2/2: Small batch memorization ==="

    STEP2_OUTPUT="outputs/pipeline_sanity/step2_${TIMESTAMP}"

    run_stage "Small-batch memorization (step2)" \
        python scripts/overfitting/step2_memorize_tiny.py \
            --config "${CONFIG}" \
            --data_path "${TRAIN_FILE}" \
            --num_samples 32 \
            --steps 2000 \
            --lr 3e-4 \
            --lora_lr 3e-5 \
            --batch_size 32 \
            --log_every 20 \
            --output_dir "${STEP2_OUTPUT}" \
    || {
        log "ABORT: Small-batch memorization failed. Pipeline stopped."
        log "Check output: ${STEP2_OUTPUT}/"
        exit 1
    }

    STEP2_PPL=$(python3 -c "
import csv
with open('${STEP2_OUTPUT}/learning_curve.csv') as f:
    rows = list(csv.DictReader(f))
    if rows:
        ppl = float(rows[-1]['train_ppl'])
        print(f'{ppl:.1f}')
    else:
        print('999')
")
    log "Step 2 final PPL: ${STEP2_PPL}"

    if python3 -c "exit(0 if float('${STEP2_PPL}') < 50 else 1)"; then
        log "Step 2 PASS (PPL=${STEP2_PPL} < 50)"
    else
        log "WARNING: Step 2 PPL=${STEP2_PPL} >= 50, model struggles to memorize."
        log "Consider investigating before full training."
    fi

    echo ""
    log "Sanity checks complete."

    if [ "${SANITY_ONLY}" = true ]; then
        elapsed=$(( SECONDS - PIPELINE_START ))
        log "================================================================"
        log "  Sanity checks done (${elapsed}s). --sanity-only mode, exiting."
        log "================================================================"
        exit 0
    fi
fi

# ══════════════════════════════════════════════════════════════
#  Stage 3: Main Training
# ══════════════════════════════════════════════════════════════

log "=== MAIN TRAINING ==="

# Build training command
TRAIN_CMD=()
if [ "${NUM_GPUS}" -gt 1 ]; then
    TRAIN_CMD+=(torchrun --nproc_per_node="${NUM_GPUS}")
else
    TRAIN_CMD+=(python)
fi
TRAIN_CMD+=(scripts/train.py --config "${CONFIG}")

# Resume from checkpoint
if [ -n "${RESUME}" ]; then
    TRAIN_CMD+=(training.resume_from="${RESUME}")
fi

# Custom output dir
if [ -n "${OUTPUT_DIR}" ]; then
    TRAIN_CMD+=(output_dir="${OUTPUT_DIR}")
fi

# Extra overrides
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    TRAIN_CMD+=("${EXTRA_OVERRIDES[@]}")
fi

run_stage "Main training" "${TRAIN_CMD[@]}" || {
    log "ERROR: Main training failed!"
    log "Check logs and consider resuming from latest checkpoint."
    exit 1
}

echo ""

# ══════════════════════════════════════════════════════════════
#  Stage 4: Evaluation
# ══════════════════════════════════════════════════════════════

# Determine output dir for finding the final checkpoint
FINAL_OUTPUT="${OUTPUT_DIR}"
if [ -z "${FINAL_OUTPUT}" ]; then
    FINAL_OUTPUT=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', 'outputs'))
")
fi

FINAL_CKPT="${FINAL_OUTPUT}/final"

if [ -d "${FINAL_CKPT}" ]; then
    log "=== EVALUATION ==="
    run_stage "Evaluation" \
        python scripts/evaluate.py \
            --checkpoint "${FINAL_CKPT}" \
            --data "${DEV_FILE}" \
    || {
        log "WARNING: Evaluation failed, but training completed successfully."
    }
else
    log "No final checkpoint found at ${FINAL_CKPT}, skipping evaluation."
fi

# ══════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════

elapsed=$(( SECONDS - PIPELINE_START ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))

echo ""
echo "================================================================"
echo "  Pipeline Complete!"
echo "================================================================"
echo "  Total time:   ${hours}h ${minutes}m (${elapsed}s)"
echo "  Config:       ${CONFIG}"
echo "  Output:       ${FINAL_OUTPUT}/"
echo "  Pipeline log: ${PIPELINE_LOG}"
echo "================================================================"
