#!/usr/bin/env bash
# ============================================================
# Generate per-experiment config YAMLs for existing ablation checkpoints.
#
# Scans outputs/ablation_compression/k*_np*_nc*/ directories,
# parses n_p/n_c from the directory name, and creates
# configs/{experiment_name}.yaml based on h200_base.yaml.
#
# Usage:
#   bash scripts/generate_ablation_configs.sh
# ============================================================

set -euo pipefail

BASE_CONFIG="configs/h200_base.yaml"
ABLATION_DIR="outputs/ablation_compression"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -d "$ABLATION_DIR" ]; then
    echo "ERROR: Ablation directory not found: $ABLATION_DIR"
    exit 1
fi

COUNT=0
SKIPPED=0

for dir in "$ABLATION_DIR"/k*; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    cfg="configs/${name}.yaml"

    if [ -f "$cfg" ]; then
        echo "  EXISTS  $cfg"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Parse n_p and n_c from directory name (k512_np128_nc384)
    np=$(echo "$name" | grep -oP 'np\K\d+')
    nc=$(echo "$name" | grep -oP 'nc\K\d+')

    if [ -z "$np" ] || [ -z "$nc" ]; then
        echo "  SKIP    $name (cannot parse np/nc)"
        continue
    fi

    python3 -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['model']['n_prompt_tokens'] = $np
cfg['model']['n_context_tokens'] = $nc
cfg['output_dir'] = '$dir'
cfg['wandb_run_name'] = '$name'
with open('$cfg', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "  CREATED $cfg  (n_p=$np, n_c=$nc)"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Done: $COUNT created, $SKIPPED already existed."
