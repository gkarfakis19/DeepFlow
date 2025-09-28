#!/bin/bash

# Runs performance model in GEMM mode
# Setup model parameter in configs/model-config/GEMM.yaml
# Setup hardware parameter in configs/hardware-config/a100_80GB.yaml

python run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/GEMM.yaml