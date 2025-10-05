# DeepFlowSim Repo Primer

Read paper_primer.md first. It describes the goal of the paper.
Then, read this.

## Big Picture
- `run_perf.py` is the main entry point; it loads hardware and model YAMLs, validates AstraSim dependencies, and dispatches to `TimeCalculationLLM` for training runs or `TimeCalculationLLMInference` for inference (LLM mode).
- `TimeCalculationLLM` builds DeepFlow's per-pipeline Chakra graph, chooses the execution backend via `ExecutionMode`, and either runs the analytical scheduler or calls the AstraSim path builder to obtain congestion-aware timings.
- `simulate_LLM.py` defines the graph objects (`Node`, `Edge`, `Graph`) that capture compute kernels and collectives with annotated durations and message sizes.
- The `astrasim_lib` package converts those graphs into Chakra ET protobufs, writes AstraSim configs from the hardware description, and replays the traces through the extended AstraSim binary.
- Hardware/memory/network characteristics live in `hw_component.py`, while `tile.py` and `LLM_util.py` translate model shapes into tiled GEMMs whose runtimes feed the graph annotations.

## Execution Flow
1. Configure the hardware in `configs/hardware-config/*.yaml` (device counts, memory hierarchy, network topology, execution backend).
2. Configure the workload in `configs/model-config/LLM.yaml` (batch size, sequence length, attention type, layer count, run type).
3. `run_perf.py --hardware_config ... --model_config ...` parses configs with `config.parse_config`, instantiates `TimeCalculationLLM`, and emits an initial pipeline graph.
4. Depending on `execution_backend`:
   - `analytical`: DeepFlow's internal scheduler computes per-node times using tiling + bandwidth models only.
   - `astra` + `hybrid`/`full_astrasim_hierarchical`/`full_astrasim_flattened`: DeepFlow emits detailed Chakra ETs, invokes the AstraSim wrapper, ingests per-rank timelines, and stitches them back into the pipeline iteration time.
5. Results, optional ET dumps, and visualizations are written under `output/LLM/` and `astra_cache/` (configurable via env flags like `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS`).

## Mapping to Paper Outline
- **Synthetic Chakra ET with annotated compute times** → `TimeCalculationLLM` + `simulate_LLM.Graph` populate per-op `duration` fields using `TiledGEMM` and memory-aware kernels before handing traces to AstraSim.
- **Extended AstraSim runtime** → `astrasim_lib/config_generation.py` derives intra/inter-dimension bandwidth/latency from the hardware config, while `astrasim_lib/integration.py` manages cache signatures, multi-topology collectives, and trace emission; `astrasim_lib/executor.py` performs the Chakra ET conversion.
- **Support for modern LLMs** → `configs/model-config/LLM.yaml` plus helpers in `LLM_util.py` and `model.py` cover dense transformer parameters (attention variants, FFN scaling) and feed the graph builder; the `transformer_cfg` stubs inside `simulate_LLM` mark where MoE or other algorithm-specific kernels plug in.
- **Head-to-head comparisons** → `run_perf.py` / `TimeCalculationLLM` run both analytical and AstraSim backends from the same graph, making it straightforward to sweep STG-like (graph-only) vs full DeepFlowSim runs; the caching layer in `astrasim_lib` accelerates repeated comparisons.
- **Case studies (3D L2, FlashAttention, faulty links, hybrid SSM)** → Hardware sweeps (e.g., editing L2 size/bandwidth in `configs/hardware-config/waferscale_*.yaml`) and algorithm toggles (e.g., `LLM_util` GEMM shapes, optional FlashAttention kernels) can be realized by modifying the YAML configs or GEMM templates; faulty-link modeling is still marked TODO inside the AstraSim comments.

## Key Config Knobs
- `execution_backend` block: selects backend (`analytical` vs `astra`), AstraSim mode, collectives, and system options (`active_chunks_per_dimension`, `endpoint_delay`, etc.).
- `scheduling_param`: sets data/pipeline/tensor parallel degrees (`dp`, `lp`, `kp1`, `kp2`, `mb`) that define the topology for pipeline and transformer graphs.
- `network_topology`: specifies intra/inter-node fabric (`ring`, `fc`, `switch`) and drives the AstraSim topology generator.
- Environment flags: `DEEPFLOW_VISUALIZE_GRAPHS`, `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS`, `DEEPFLOW_ASTRA_CACHE_MODE` control artifact dumps and cache policy.

## Current Gaps vs. Outline
- Faulty-link modeling and richer multidimensional network abstractions are noted as WIP in the outline; the current code base exposes hooks (`network_topology`, per-comm metadata) but does not yet inject fault-aware latency/throughput adjustments.
- Inference support is partially scaffolded (`simulate_inf.py`, `TimeCalculationLLMInference`), but the bulk of the pipeline focuses on training runs.
- MLSynth/Multiverse integration is presently manual; DeepFlowSim can emit compute-annotated traces, yet glue code for direct Multiverse comparison remains to be written.

## Usage Tips
- Use `examples/llm.sh` (analytical) or `examples/llm_astra.sh` (AstraSim) as starting points.
- Clear `astra_cache/` when modifying the AstraSim binary or network parameters to avoid stale cache reuse.
- Enable graph visualization sparingly; large-scale flattened runs can generate huge ETs and PNGs.
