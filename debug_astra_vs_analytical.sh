#!/usr/bin/env bash
set -euo pipefail

export DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1
export DEEPFLOW_VISUALIZE_GRAPHS=1
export DEEPFLOW_PERSIST_ARTIFACT_VIZ=1

HW_CONFIG="configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG="configs/model-config/Llama2-7B_inf.yaml"

get_model() {
  python3 - "$HW_CONFIG" <<'PY'
import sys
path = sys.argv[1]
with open(path) as fh:
    lines = fh.readlines()
seen_exec = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('execution_backend:'):
        seen_exec = True
        continue
    if seen_exec and stripped.startswith('model:'):
        value = stripped.split(':', 1)[1].strip().split()[0]
        print(value)
        break
PY
}

set_model() {
  local value="$1"
  python3 - "$HW_CONFIG" "$value" <<'PY'
import sys
path, value = sys.argv[1:3]
with open(path) as fh:
    lines = fh.readlines()
seen_exec = False
for idx, line in enumerate(lines):
    stripped = line.lstrip()
    if stripped.startswith('execution_backend:'):
        seen_exec = True
        continue
    if seen_exec and stripped.startswith('model:'):
        prefix, rest = line.split('model:', 1)
        comment = ''
        if '#' in rest:
            before_comment, comment = rest.split('#', 1)
            comment = '#' + comment.rstrip('\n')
        else:
            before_comment = rest.rstrip('\n')
        new_line = f"{prefix}model: {value}"
        if comment:
            if not comment.startswith(' '):
                new_line += ' '
            new_line += comment
        elif before_comment.strip():
            new_line += before_comment.rstrip()
        lines[idx] = new_line + '\n'
        break
with open(path, 'w') as fh:
    fh.writelines(lines)
PY
}

ORIGINAL_MODEL=$(get_model)
trap 'set_model "$ORIGINAL_MODEL"' EXIT

run_and_capture() {
  local mode="$1"
  set_model "$mode"
  tmpfile=$(mktemp)
  if ! uv run run_perf.py --hardware_config "$HW_CONFIG" --model_config "$MODEL_CONFIG" | tee "$tmpfile"; then
    rm -f "$tmpfile"
    return 1
  fi
  local inf_line throughput_line ttft_line
  inf_line=$(grep -F 'LLM inference time:' "$tmpfile" | tail -n1 || true)
  throughput_line=$(grep -F 'Decode throughput tok/s:' "$tmpfile" | tail -n1 || true)
  ttft_line=$(grep -F 'LLM time to first token:' "$tmpfile" | tail -n1 || true)
  rm -f "$tmpfile"
  case "$mode" in
    analytical)
      ANALYTICAL_INF_LINE="$inf_line"
      ANALYTICAL_TPUT_LINE="$throughput_line"
      ANALYTICAL_TTFT_LINE="$ttft_line"
      ;;
    astra)
      ASTRA_INF_LINE="$inf_line"
      ASTRA_TPUT_LINE="$throughput_line"
      ASTRA_TTFT_LINE="$ttft_line"
      ;;
  esac
}
run_and_capture analytical
if [ -d "./output/LLM" ]; then
  rm -rf "./output/LLM_analytical"
  mv "./output/LLM" "./output/LLM_analytical"
  rm -rf "./output_graph/decode_samples_analytical"
  mv "./output_graph/decode_samples" "./output_graph/decode_samples_analytical"
fi

run_and_capture astra
if [ -d "./output/LLM" ]; then
  rm -rf "./output/LLM_astra"
  mv "./output/LLM" "./output/LLM_astra"
  rm -rf "./output_graph/decode_samples_astra"
  mv "./output_graph/decode_samples" "./output_graph/decode_samples_astra"
fi

printf 'ANALYTICAL: %s\n' "${ANALYTICAL_INF_LINE:-LLM inference time not found}"
printf 'ANALYTICAL: %s\n' "${ANALYTICAL_TPUT_LINE:-Decode throughput tok/s not found}"
printf 'ANALYTICAL: %s\n' "${ANALYTICAL_TTFT_LINE:-LLM time to first token not found}"
printf 'ASTRA: %s\n' "${ASTRA_INF_LINE:-LLM inference time not found}"
printf 'ASTRA: %s\n' "${ASTRA_TPUT_LINE:-Decode throughput tok/s not found}"
printf 'ASTRA: %s\n' "${ASTRA_TTFT_LINE:-LLM time to first token not found}"
