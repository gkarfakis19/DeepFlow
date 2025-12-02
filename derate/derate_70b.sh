#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DERATE_FILE="${DERATE_CONFIG_OVERRIDE:-$SCRIPT_DIR/derate_70b.yaml}"
HARDWARE_CONFIG="$SCRIPT_DIR/hardware-config/a100_flat_tp8_cp1.yaml"
MODEL_CONFIG="$ROOT_DIR/configs/model-config/Llama3.1-70B.yaml"

cd "$ROOT_DIR"
uv run python run_perf.py \
  --hardware_config "$HARDWARE_CONFIG" \
  --model_config "$MODEL_CONFIG" \
  --derate_config "$DERATE_FILE"
