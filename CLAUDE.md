# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFlow is a hardware-first end-to-end performance simulator for distributed LLM training and inference. It generates synthetic Chakra execution traces with annotated compute times and executes them through either an analytical network model or AstraSim for congestion-aware simulation. The tool enables detailed hardware/software co-design space exploration for modern LLM workloads at scale (hundreds to thousands of nodes).

## Common Commands

### Environment Setup

Using `uv` (recommended):
```bash
pip install uv
uv venv [/path/to/new/virtual/environment]
source [/path/to/new/virtual/environment]/bin/activate
uv sync
```

Using `pip`:
```bash
python3 -m venv [/path/to/new/virtual/environment]
source [/path/to/new/virtual/environment]/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Building AstraSim (Optional)

AstraSim is required for advanced network simulation modes (hybrid, full_astrasim_hierarchical, full_astrasim_flattened):

```bash
git submodule update --init --recursive
git submodule update --remote -- astra-sim
ASTRA_SIM=$(realpath ./astra-sim)
cd ${ASTRA_SIM}
./build/astra_analytical/build.sh
cd ..
```

If protobuf failures occur:
```bash
pip uninstall protobuf
pip install protobuf==3.20.3
```

### Running Simulations

**LLM mode (analytical backend - fast, no congestion modeling):**
```bash
./examples/llm.sh
# or
python run_perf.py --hardware_config configs/hardware-config/a100_80GB_example.yaml --model_config configs/model-config/LLM_inf.yaml
```

**LLM mode (AstraSim backend - slower, congestion-aware):**
```bash
./examples/llm_astra.sh
# or
DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1 DEEPFLOW_VISUALIZE_GRAPHS=1 python run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/LLM.yaml
```

**GEMM mode:**
```bash
python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/GEMM.yaml
```

**LSTM mode:**
```bash
python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LSTM.yaml
```

### Architecture Search

```bash
# Single parallelism strategy
python GD_search.py --exp_config configs/[config.yaml] --exp_dir [output_dir] --debug False --index [index] --batch_size [batch] --hidden_dim [dim] --data_scale [scale] --dp [dp] --lp [lp] --kp_type [0|1] --kp1 [kp1] --kp2 [kp2] --inter_derate [factor] --intra_derate [factor] --kp1_inter [False|True] --kp2_inter [False|True] --dp_inter [False|True] --lp_inter [False|True] --wafer_dim [dim]

# All parallelism strategies
python main.py arch_search --exp_dir [output_dir] --exp_config configs/[config.yaml]
```

### Environment Variables

**Cache Control:**
- `DEEPFLOW_ASTRA_CACHE_MODE`: Controls AstraSim result caching
  - `NO_CACHE`: Disable caching entirely
  - `CACHE_READONLY`: Read-only mode (for multi-threaded runs)
  - `CACHE_READWRITE`: Normal caching (default)

**Artifact Generation:**
- `DEEPFLOW_VISUALIZE_GRAPHS=1`: Generate graph visualizations (PNG files)
- `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1`: Save AstraSim execution traces to disk
- `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`: Generate PNG/TXT dumps of ET files (very slow for large graphs)

**Important:** Do not set `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1` for multi-threaded runs.

## Architecture Overview

### Execution Flow

1. **Entry point:** `run_perf.py` parses hardware and model YAML configs, validates dependencies, and dispatches to the appropriate time calculator
2. **Time calculators:**
   - `TimeCalculationLLM` (training): Builds per-pipeline Chakra graphs
   - `TimeCalculationLLMInference` (inference): Handles prefill + decode phases
   - `TimeCalculation` (LSTM/GEMM): Legacy LSTM and GEMM workloads
3. **Graph building:** `simulate_LLM.py` defines `Node`, `Edge`, `Graph` objects capturing compute kernels and collectives with annotated durations
4. **Execution backend selection:** Configured via `execution_backend.model` in hardware YAML
   - `analytical`: Fast analytical scheduler, ring-only networks, no congestion
   - `astra`: AstraSim integration with multiple modes and network topologies
5. **Output:** Results written to `output/<mode>/`, with optional artifacts in `astra_cache/`

### Execution Backends

**Analytical (default):**
- Very fast but inaccurate
- Only supports ring network topology
- No congestion modeling
- Set `execution_backend.model: analytical`

**AstraSim Hybrid:**
- DeepFlow executes pipeline graph
- AstraSim executes transformer block graphs
- Models congestion inside transformer blocks only
- ~2-3x slower than analytical
- Set `execution_backend.model: astra` and `execution_backend.astra.mode: hybrid`

**AstraSim Full Hierarchical:**
- Separate pipeline and transformer graphs
- Models congestion within each layer separately
- Assumes no congestion between pipeline/tensor parallelism dimensions
- Set `execution_backend.model: astra` and `execution_backend.astra.mode: full_astrasim_hierarchical`

**AstraSim Full Flattened:**
- Single flattened graph combining pipeline and transformer operations
- Most accurate congestion modeling
- Very slow for large systems
- Set `execution_backend.model: astra` and `execution_backend.astra.mode: full_astrasim_flattened`

### Key Modules

**Core simulation:**
- `run_perf.py`: Main entry point
- `time_calculation_LLM.py`: LLM training time calculator
- `time_calculation_inf.py`: LLM inference time calculator
- `simulate_LLM.py`: Graph construction for LLM workloads
- `tile.py`: Tiled GEMM representation and memory-aware kernel timing
- `LLM_util.py`: LLM-specific utilities (attention, FFN shapes)

**Hardware modeling:**
- `hw_component.py`: Device, memory hierarchy, network characteristics
- `config.py`: YAML configuration parsing for hardware and model params
- `energy.py`: Power/energy modeling

**AstraSim integration:**
- `astrasim_lib/bootstrap.py`: Chakra protobuf dependencies
- `astrasim_lib/config_generation.py`: Generate AstraSim network/system configs
- `astrasim_lib/et_utils.py`: Chakra ET node creation and writing
- `astrasim_lib/integration.py`: Cache orchestration and AstraSim binary invocation
- `astrasim_lib/executor.py`: Convert DeepFlow graphs to Chakra ET bundles

**Architecture search:**
- `GD_search.py`: Gradient descent-based hardware search
- `deviceMapping.py`: Device projection and topology mapping
- `topology.py` / `topology_hack.py`: Network topology abstractions

### Configuration Files

**Hardware configs** (`configs/hardware-config/*.yaml`):
- Device counts, memory hierarchy, network topology
- Parallelism strategy (`scheduling_param`: dp, lp, kp1, kp2, mb)
- Execution backend selection
- Network topology (`network_topology`: ring, fc, switch for inter/intra node)
- System hierarchy (devices per node, inter/intra bandwidth derate)

**Model configs** (`configs/model-config/*.yaml`):
- `mode`: "LLM", "LSTM", or "GEMM"
- `run_type`: "training" or "inference"
- LLM: batch_size, seq_len, hidden_dim, attention config, ffn_dim/ffn_mult, vocab_size, num_layers
- Inference: sample_every parameter for decode step sampling

### Submodules

**astra-sim:** Extended AstraSim simulator for network modeling
- Build with `./build/astra_analytical/build.sh`
- Provides Chakra protobuf definitions

**symbolic_tensor_graph (STG):** Alternative Chakra ET generator
- Generates synthetic transformer workloads without compute annotations
- Supports DP, TP, PP, SP parallelism strategies
- Used for comparison studies

**SimAI:** Alibaba's full-stack LLM training simulator
- Alternative simulator for comparison
- Analytical and full-simulation modes

## Output Locations

- `output/LLM/`: LLM simulation results and summary files
- `output/GEMM/`: GEMM performance results
- `output/LSTM/`: LSTM simulation results
- `astra_cache/`: Cached AstraSim runs and generated workloads
  - `network_analytical_<NPUS>.yml`: Generated network configs
  - `system_native_collectives.json`: AstraSim system config
  - `cache.json`: Cache metadata
- `output_graph/`: Graph visualizations when `DEEPFLOW_VISUALIZE_GRAPHS=1`

## Important Notes

### Network Topology Support

- Analytical backend only supports ring topology for both inter and intra node
- AstraSim backend supports ring, fully-connected (fc), and switch topologies
- Non-ring topologies require AstraSim installation

### Cache Management

- AstraSim results are cached by default in `./astra_cache/`
- Clear cache when modifying AstraSim binary or network parameters
- Use `DEEPFLOW_ASTRA_CACHE_MODE=NO_CACHE` to disable caching
- Use `DEEPFLOW_ASTRA_CACHE_MODE=CACHE_READONLY` for multi-threaded runs

### Parallelism Strategy

Configured in hardware YAML under `scheduling_param`:
- `dp`: Data parallel dimension
- `lp`: Pipeline parallel stages (layer parallel)
- `mb`: Number of micro-batches for pipeline parallelism
- `kp1`, `kp2`: Tensor parallel dimensions
- `t`: Tensor parallelism type ("CR" for Column-Row, "RC" for Row-Column)

Total devices = dp × lp × kp1 × kp2

### Inference vs Training

Set `run_type` in model config:
- `"training"`: Full forward + backward + optimizer
- `"inference"`: Prefill + decode phases with KV cache

Inference mode uses `TimeCalculationLLMInference` and reports:
- Time to first token (TTFT)
- Prefill time
- Decode time and throughput (tokens/s)
- Per-step decode rates at start, middle, end

### Artifact Visualization

When enabled (`DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`):
- Generates `.png` graph visualizations
- Creates `.txt` dumps of ET files
- Very slow for large graphs with many nodes
- Not recommended for production runs

### Python Version

Requires Python 3.11+ (specified in `pyproject.toml`)
