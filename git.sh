#!/usr/bin/env bash
# Helper script to create branch llm-compare-latest on top of llm-dev and
# pull in the STG/MLSynth wrapper plus required configs. Run from repo root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLD="${ROOT}/../DeepFlow-old"

cd "$ROOT"

echo "==> Checking out llm-dev and creating llm-compare-latest"
git checkout llm-dev
git checkout -B llm-compare-latest

echo "==> Copying wrapper + comparator from DeepFlow-old"
git --git-dir="${OLD}/.git" show HEAD:tools/wrapper.py > tools/wrapper.py
git --git-dir="${OLD}/.git" show HEAD:tools/compare_et_timing.py > tools/compare_et_timing.py

echo "==> Copying configs used by wrapper"
git --git-dir="${OLD}/.git" show HEAD:configs/hardware-config/a100_80GB_NEMO.yaml > configs/hardware-config/a100_80GB_NEMO.yaml
# Uncomment if you also want the AA-scale variant:
git --git-dir="${OLD}/.git" show HEAD:configs/hardware-config/a100_80GB_AA_scale.yaml > configs/hardware-config/a100_80GB_AA_scale.yaml

echo "==> Copying optional model configs (uncomment as needed)"
git --git-dir="${OLD}/.git" show HEAD:configs/model-config/Llama1-13B_2K.yaml > configs/model-config/Llama1-13B_2K.yaml
git --git-dir="${OLD}/.git" show HEAD:configs/model-config/Llama3.1-70B.yaml > configs/model-config/Llama3.1-70B.yaml
git --git-dir="${OLD}/.git" show HEAD:configs/model-config/Llama3.1-70B_128k.yaml > configs/model-config/Llama3.1-70B_128k.yaml
git --git-dir="${OLD}/.git" show HEAD:configs/model-config/Llama3.1-70B_32k.yaml > configs/model-config/Llama3.1-70B_32k.yaml

echo "==> Copying MLSynth directory (comment out if not needed)"
git --git-dir="${OLD}/.git" archive HEAD MLSynth | tar -xC "$ROOT"

echo "==> Optional wrapper docs (comment out if not needed)"
git --git-dir="${OLD}/.git" show HEAD:WRAPPER_PLAN.md > WRAPPER_PLAN.md
git --git-dir="${OLD}/.git" show HEAD:STG_BYTES_PROOF.md > STG_BYTES_PROOF.md

echo "==> Staging new files"
git add tools/wrapper.py tools/compare_et_timing.py
git add configs/hardware-config/a100_80GB_NEMO.yaml
git add MLSynth || true  # ignore if MLSynth was not extracted
# Add any optional files you enabled above:
git add configs/hardware-config/a100_80GB_AA_scale.yaml
git add configs/model-config/Llama1-13B_2K.yaml configs/model-config/Llama3.1-70B.yaml \
        configs/model-config/Llama3.1-70B_128k.yaml configs/model-config/Llama3.1-70B_32k.yaml
git add WRAPPER_PLAN.md STG_BYTES_PROOF.md

echo "==> Status"
git status --short

echo "Done. Review changes, fix imports if needed, then commit."
