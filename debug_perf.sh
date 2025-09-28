export DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1
export DEEPFLOW_VISUALIZE_GRAPHS=1
export DEEPFLOW_PERSIST_ARTIFACT_VIZ=1


uv run run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/Llama2-7B_inf.yaml