## Installation Guide

Prerequisite: Python 3

### Step 1. Clone the repository

```bash
git clone https://github.com/gkarfakis19/DeepFlow/
cd DeepFlow
```

### Step 2. Set up the environment

#### Option A: Using uv (recommended)

- `pip install uv`
- `uv venv [/path/to/new/virtual/environment]`
- `source [/path/to/new/virtual/environment]/bin/activate`
- `uv sync`

If uv is used, make sure to use `uv run` instead of `python` to execute python scripts within the uv environment.

#### Option B: Using pip

- `python3 -m venv [/path/to/new/virtual/environment]`
- `source [/path/to/new/virtual/environment]/bin/activate`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`

### Step 3. (Optional) Set up AstraSim for advanced network simulation

- `git submodule update --init --recursive`
- `git submodule update --remote -- astra-sim`
- `ASTRA_SIM=$(realpath ./astra-sim)`
- `cd ${ASTRA_SIM}`
- `git submodule update --remote -- extern/network_backend/analytical`
- `./build/astra_analytical/build.sh`
- `cd ..`

### Troubleshooting

If you encounter protobuf failures while building AstraSim, try:

- `pip uninstall protobuf`
- `pip install protobuf==3.20.3`

### Step 4. Verify the installation

- `./examples/llm.sh` or `./examples/llm_inference.sh` (analytical backend, training & inference of Llama2-7B)
- `./examples/llm_astra.sh` or `./examples/llm_astra_inference.sh` (AstraSim backend, training & inference of Llama2-7B. Requires AstraSim installation)

## Execution Backend Configuration

DeepFlow supports four execution backends with different accuracy and performance characteristics. Configure the backend in your hardware config file under `execution_backend`.

### 1. Analytical DeepFlow (Default - no AstraSim needed)

- **Accuracy:** Very fast but inaccurate; only ring network model and no congestion modeling.
- **Configuration:**

```yaml
execution_backend:
  model: analytical
```

### 2. Hybrid (AstraSim needed)

- **Accuracy:** More accurate; models congestion in transformer blocks only but roughly 2-3× slower.
- **Execution:** DeepFlow executes the pipeline graph; AstraSim executes the transformer block graph.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: hybrid
```

### 3. Full AstraSim Hierarchical (AstraSim needed)

- **Accuracy:** Even more accurate; models congestion in transformer and pipeline graphs separately. Assumes no congestion between pipeline/data parallelism and tensor parallelism (optimistic). Roughly as fast as Hybrid for small systems, increasingly slower for larger systems.
- **Execution:** AstraSim executes both pipeline and transformer block graphs separately.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_hierarchical
```

### 4. Full AstraSim Flattened (AstraSim needed)

- **Accuracy:** Most accurate; models congestion between all collectives with no separate network assumptions. Very slow for large systems but most comprehensive.
- **Execution:** AstraSim executes one big flattened graph combining pipeline and transformer operations.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_flattened
```

## Execution Modes

DeepFlow can be used in 2 different modes.

### Model Prediction Modes

1. **Performance Prediction Mode (GEMM)**
   - **When to use:** Distributed GEMM prediction.
   - **How:**
     - Specify GEMM parameters in `configs/model-config/GEMM.yaml`.
     - Specify hardware parameters in `configs/hardware-config/[config.yaml]`.
     - Run `python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/GEMM.yaml`.

2. **Performance Prediction Mode (LLM)**
   - **When to use:** End-to-end LLM prediction.
   - **How:**
     - Specify LLM parameters in `configs/model-config/LLM.yaml`.
     - Specify hardware parameters in `configs/hardware-config/[config.yaml]`.
     - Run `python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LLM.yaml`.

LLM mode is a work in progress with limited validation. Results are saved under `output/<mode>`.

## AstraSim Artifact and Graph Visualization

DeepFlow can generate and visualize graphs, and when using the AstraSim network backend, can also generate and visualize network communication artifacts.

### Environment Flags

- `DEEPFLOW_VISUALIZE_GRAPHS=1`: Generate graph visualizations of DeepFlow computation graphs (no AstraSim artifact visualization).
- `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1`: Enable artifact persistence to disk (for both AstraSim and DeepFlow artifacts).
- `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`: Generate PNG visualizations and text dumps for persisted AstraSim ET files (very slow for many nodes).
- Do not set `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1` for multi-threaded runs.

### Artifact Output Locations

- Flattened execution mode: `output/LLM/astra_flat/`
- Hierarchical/Hybrid modes: `output/LLM/astra_hier/`


### Generated Files

- `.et` files: Chakra execution traces for AstraSim replay.
- `.png` files: Rendered PNG visualizations (when `DEEPFLOW_VISUALIZE_GRAPHS=1` or `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`).
- `.txt` files: Human-readable text dumps of ET files (when `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`).

### Debugging: Usage Example

```bash
DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1 DEEPFLOW_VISUALIZE_GRAPHS=1 DEEPFLOW_PERSIST_ARTIFACT_VIZ=1 python run_perf.py \
  --hardware_config configs/hardware-config/a100_80GB.yaml \
  --model_config configs/model-config/LLM.yaml
```

## Tips

- AstraSim backend caches runs by default in `./astra_cache/`. To disable caching, set the environment variable `DEEPFLOW_ASTRA_CACHE_MODE` to `NO_CACHE` or `CACHE_READONLY` (for multi-threaded runs). `NO_CACHE` (or manual cache flushing) is necessary if the AstraSim binary itself is modified.
- Check the `configs` directory for architecture templates and technology node configurations.


## Example case study: Tensor parallelism on inference runtime for a new LLM config

This is a short step-by-step guide on how to use DeepFlow to estimate the inference runtime of a LLM model with varying degrees of tensor parallelism on the default A100 80GB hardware config.

1. Start by creating a model config yaml file. You can do so manually, but for this example we will use the helper configs/model-config/hf_to_config.py script to generate the config file from a HuggingFace model (in this case, Qwen/Qwen2.5-3B).

```bash
python configs/model-config/hf_to_config.py Qwen/Qwen2.5-3B --run-type inference --batch-size 32 --seq-len 65536 --decode-len 1024 --use-flashattention true --flash-tile-size 256 -o configs/model-config/Qwen2.5-3B.yaml
```
 The config file for Qwen2.5-3B is generated under `configs/model-config`

2. Open the provided A100 hardware file and set tensor parallel degree to 1:

Edit `configs/hardware-config/a100_80GB.yaml` so that `parallelism.tp: 1`. This models one GPU without tensor parallelism.

Run the inference estimation using the example configs:

```bash
python run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/Qwen2.5-3B.yaml
```

3. To see the effects of tensor-parallelism, modify the hardware file, switching to two devices per node, and tp = 2.

Edit `configs/hardware-config/a100_80GB.yaml` so that `parallelism.tp: 2`. This models two GPU-inference with tensor parallelism degree of 2.

Re-run the same inference command with the updated hardware config.

Comparing the two runs will show how increasing tensor parallelism changes the predicted inference runtime for this model (in this case, by around 50%).

## Current Support and feature status

### AstraSim Integation
- **Supported:** AstraSim integration is supported for all modes and model types.
- **Work in progress:** AstraSim integration is not fully validated, and only supports 1 dimensional network topologies due to inherent limitations of the AstraSim network simulator. We are working on extending support to multi-dimensional network topologies.

### FlashAttention
- **Current support:**
  - Forward pass for **training**
  - **Prefill** phase in inference
- **Work in progress:**
  - Attention tile size is currently manually defined. We will add support for automatically determining the optimal tile size based on **SRAM size** as done in optimized GPU kernels.
  - For inference **decode**, FlashAttention is theoretically supported but not enabled as memory-bound decode does not typically benefit from FlashAttention. Instead, FlashDecoding-style techniques are typically used that are not implemented.

### Data Parallelism
- **Supported:** training and inference  
- For inference, acts as **replica-count**, as separate data-parallel chains of replicas act as independent instances and do not communicate. Hence, only the final decode throughput is multiplied by the number of replicas. For training full DDP & ZeRO-style (stage 1,2 or 3) Data Parallelism is supported.

### Tensor Parallelism
- **Supported:** training and inference  
- Implements **Megatron-LM–style tensor parallelism**, with **sequence parallelism** optionally enabled or disabled

### Pipeline Parallelism
- **Supported:** training and inference  
- Each data batch is divided into **microbatches** (mb) that dictate pipeline
- Supports **GPipe-style pipeline parallelism** only  
- Other pipeline styles (e.g. PipeDream) are **work in progress**

### Context Parallelism
- **Supported:** Deepspeed-style context parallelism (CP) for training only.
- **Inference path** is under development (WIP)

### Hybrid Parallelism
- **Supported:** Hybrid parallelism is supported for both training and inference for all supported parallelism and model types.

### Model Types
- **Supported:** `gpt`, `llama`, `qwen2`, `phi3` for both training and inference  
- **Note:** optional sliding-window attention configurations (in `qwen2`, `phi3`) currently **fall back to dense attention**.

### Attention Types
- **Supported:** **MHA**,**GQA** (`num_kv_heads`) for training and inference. **MQA** == **GQA** with `num_kv_heads` set to 1, so is also supported.
- **Work in progress:** **MLA**, potentially **sliding-window** attention

### Mixture of Experts (MoE)
- **Supported:** Single-GPU case in both training and inference (not tested/validated, heavily WIP)
- **Work in progress:** Multi-GPU expert parallelism, validation and refinement.

### Memory Estimation
- **Status:** Not enabled in the mainstream workflow (WIP)
- **Supported:** Transformer activation (including KV cache) and static memory estimation with **hybrid parallelism** in training mode, and **inference** with **Tensor Parallelism**  
- **Work in progress:** Will be fully released after further refinement

### KV-Cache
- **Supported:** KV-cache runtime impact is modeled for all supported parallelism and model types.
- **Work in progress:** KV-cache memory estimation is not yet supported.

### Energy Estimation
- **Supported:** Very rudimentary, non validated model. Does not consider static power. Inference only for now (Training is WIP). Energy due to prefill and decode is reported separately.
- **Work in progress:** Training support, and heavy refinement of current implementation.

### Mixed Precision
- **Supported:** Mixed precision is supported for all supported parallelism and model types.
- **Work in progress:** KV Cache precision cannot yet be set (config value is ignored).

### Validation
- **Status:** Validation scripts are under development. Limited inference and training validation has been performed.
- **Supported:** For inference, we validate against Koyeb's single NVIDIA A100 data on LLama3.1-8B. The relevant script can be found in `validation_scripts/koyeb.py`. Below is the resulting plot, comparing DeepFlow's predicted decode throughput with Koyeb's reported decode throughput, available at https://www.koyeb.com/docs/hardware/gpu-benchmarks. Average absolute error is ~6%.

![Koyeb Validation](validation_scripts/koyeb_a100_sxm_no_parallelism.png)
