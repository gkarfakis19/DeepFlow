#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/LLM.yaml
# Setup hardware parameter in configs/hardware-config/a100_80GB.yaml

python run_perf.py --hardware_config configs/hardware-config/a100_80GB_example.yaml --model_config configs/model-config/LLM.yaml