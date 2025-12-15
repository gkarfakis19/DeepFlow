#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/Llama2-7B_inf.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB.yaml

python run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/LLM_inf.yaml