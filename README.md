## Installation Guide ##

Pre-requirement: Python3

**Step 1**. git clone https://github.com/gkarfakis19/DeepFlow/

**Step 2**. cd DeepFlow

**Step 3**. Setup the environment:

**Option A: Using uv (recommended)**
	* pip install uv
	* uv venv [/path/to/new/virtual/environment]
	* source [/path/to/new/virtual/environment]/bin/activate
	* uv sync

**Option B: Using pip**
	* python3 -m venv [/path/to/new/virtual/environment]
	* source [/path/to/new/virtual/environment]/bin/activate
	* pip install --upgrade pip
	* pip install -r requirements.txt

**Step 4**. (Optional) Setup AstraSim for advanced network simulation:

	* git submodule update --init --recursive
	* git submodule update --remote -- astra-sim
	* ASTRA_SIM=$(realpath ./astra-sim)
	* cd ${ASTRA_SIM}
	* ./build/astra_analytical/build.sh
	* cd ..

**Note**: If you encounter protobuf failures while building AstraSim, try:
	* pip uninstall protobuf
	* pip install protobuf==3.20.3

**Step 5**. Test if the installation has been successful:

	* ./examples/llm.sh # For analytical backend
	* ./examples/llm_astra.sh # For AstraSim backend (requires AstraSim installation)

### Execution Backend Configuration

DeepFlow supports 4 execution backends with different accuracy and performance characteristics. Configure the backend in your hardware config file under `execution_backend`:

**(1) Analytical DeepFlow** (Default - no AstraSim needed)
- **Accuracy**: Very fast but inaccurate - only ring network model and no congestion modeling.
- **Configuration**:
```yaml
execution_backend:
  model: analytical
```

**(2) Hybrid** (AstraSim needed)
- **Accuracy**: More accurate - models congestion in transformer blocks (only) but roughly 2-3x slower.
- **Execution**: DeepFlow executes pipeline graph, AstraSim executes transformer block graph
- **Configuration**:
```yaml
execution_backend:
  model: astra
  astra:
    mode: hybrid
```

**(3) Full AstraSim Hierarchical** (AstraSim needed)
- **Accuracy**: Even more accurate - models congestion in transformer and pipeline graphs separately. Assumes no congestion between pipeline/data parallelism and tensor parallelism (optimistic). Roughly as fast as Hybrid for small systems, increasingly slower for larger systems.
- **Execution**: AstraSim executes both pipeline and transformer block graphs separately
- **Configuration**:
```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_hierarchical
```

**(4) Full AstraSim Flattened** (AstraSim needed)
- **Accuracy**: Most accurate - models congestion between all collectives with no separate network assumptions. Very slow for large systemsbut most comprehensive.
- **Execution**: AstraSim executes one big flattened graph combining pipeline and transformer operations
- **Configuration**:
```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_flattened
```
## Execution Modes ##

DeepFlow can be used in 6 different modes:

### Model Prediction Modes

(1) Peformance Prediction Mode (GEMM) 
 **When to use**: Use for distributed GEMM prediction  
 **How**:   
* Specify the GEMM parameters in configs/model-config/GEMM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/GEMM.yaml

(2) Performance Prediction Mode (LLM mode)
 **When to use**: Use for End-2-End LLM prediction  
 **How**:   
* Specify the LLM parameters in configs/model-config/LLM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LLM.yaml

(3) Performance Prediction Mode (LSTM End-2-End Application)
 **When to use**: Use for End-2-End LSTM prediction  
 **Note**: This mode has not been tested/validated with AstraSim backend.
 **How**:   
* Specify the LSTM parameters in configs/model-config/LSTM.yaml
* Specify the Hardware parameters in configs/hardware-config/[config.yaml]
* python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LSTM.yaml
        
LLM Mode is WIP. Limited validation has been performed. Results are saved under `output/<mode>`.

The below 3 modes are only tested/validated for LSTM (porting to LLM is WIP).

(4) Performance Prediction Mode (using main.py standalone argument; this is somewhat equivalent of option 2, for running on slurm)
* python main.py stand_alone --exp_dir [/path/to/output/result/directory] --exp_config configs/[config.yaml]

(5) Architecture search for a fixed parallelism strategy
* python GD_search.py --exp_config configs/[config.yaml] --exp_dir [/path/to/output/directory] --debug False --index [index] --batch_size [batch] --hidden_dim [lstm_dim] --data_scale [dataset_scaling_factor] --dp [data parallel dim.] --lp [layer parallel dim.] --kp_type [0|1] --kp1 [kp1 dim.] --kp2 [kp2 dim.] --inter_derate [derate_factor_for_inter_package_bandwidth] --intra_derate [derate_factor_for_intra_package_bandwidth] --kp1_inter [False|True] --kp2_inter [False|True] --dp_inter [False|True] --lp_inter [False|True] --wafer_dim [package dim.]
* **Example**: python GD_search.py --exp_config configs/exp_config.yaml --exp_dir output --debug False --index 40 --batch_size 256 --hidden_dim 19968 --data_scale 1 --dp 64 --lp 1 --kp_type 1 --kp1 1 --kp2 1 --inter_derate 0 --intra_derate 2 --kp1_inter False --kp2_inter False --dp_inter False --lp_inter False --wafer_dim 8

(6) Architecture Search mode for all types of parallelism strategies
* python main.py arch_search --exp_dir [/path/to/output/directory] --exp_config configs/[config.yaml]


## AstraSim Artifact and Graph Visualization ##

DeepFlow can generate and visualize network communication artifacts when using AstraSim execution backend.

**Environment Flags:**
* `DEEPFLOW_VISUALIZE_GRAPHS=1`: Generate graph visualizations of DeepFlow computation graphs executed (no AstraSim artifact visualization)
* `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1`: Enable artifact persistence to disk (both AstraSim and DeepFlow artifacts)
* `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`: Generate PNG visualizations and text dumps for persisted ET files (*very* slow for many nodes!)
* Do not set `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1` for multi-threaded runs.

**Artifact Output Locations:**
* Flattened execution mode: `output/LLM/astra_flat/`
* Hierarchical/Hybrid modes: `output/LLM/astra_hier/`

**Generated Files:**
* `.et` files: Chakra execution traces for AstraSim replay
* `.png` files: Rendered PNG visualizations (when `DEEPFLOW_VISUALIZE_GRAPHS=1` or `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`)
* `.txt` files: Human-readable text dumps of ET files (when `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`)

**Usage Example:**
```bash
DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1 DEEPFLOW_VISUALIZE_GRAPHS=1 DEEPFLOW_PERSIST_ARTIFACT_VIZ=1 python run_perf.py \
  --hardware_config configs/hardware-config/a100_80GB.yaml \
  --model_config configs/model-config/LLM.yaml
```

## Tips ##

* AstraSim backend caches runs by default in ./astra_cache/. To disable caching, set the environment variable `DEEPFLOW_ASTRA_CACHE_MODE` to `NO_CACHE` or `CACHE_READONLY` (intended for multi-threaded runs) in your environment. `NO_CACHE` (or manual cache flushing) is necessary if the AstraSim binary itself is modified.
* Check config directory for  different architecture templates and technology node configurations
