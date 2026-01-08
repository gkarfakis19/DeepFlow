# Wrapper Plan

## Scope
Design a reusable harness that compares three modeling flows on the same DeepFlow hardware/model configuration:

1. DeepFlow + AstraSim with DeepFlow’s annotated compute times.
2. DeepFlow ablation that suppresses compute annotations so AstraSim runs in roofline mode.
3. Symbolic Tensor Graph (STG) generated traces replayed through AstraSim.

Outputs should land under ./gen_et/ split by mode, in both dry-run (graph generation only) and full-run (complete AstraSim execution) paths.

---

## Inputs & Dependencies
- Hardware YAML (e.g., configs/hardware-config/a100_80GB.yaml): provides scheduling_param (dp, lp, kp1, kp2, mb), network topology, AstraSim config options, and execution backend mode.
- Model YAML (configs/model-config/LLM.yaml): provides transformer dimensions (batch_size, seq_len, hidden_dim, ffn_dim/ffn_mult, attention heads, num_layers, run_type).
- AstraSim binary: default astra-sim/build/astra_analytical/bin/astra_analytical, overridable through the wrapper CLI.
- symbolic_tensor_graph environment: Python 3.11-compatible execution to run main.py with same parallelism degrees as DeepFlow.
- DeepFlow modules: run_perf.py, time_calculation_LLM.py, simulate_LLM.py, astrasim_lib/*.

---

## Architecture Overview
- Implement wrapper as a new script (e.g., tools/compare_workloads.py) that relies on module-level configuration constants (no CLI parsing).
- Provide a global selector (e.g., RUN_SELECTION) to choose between single-mode execution (`deepflow`, `deepflow_ablation`, `stg`) or `all`.
- Define per-mode handler functions; only the DeepFlow annotated handler has working logic initially, while the other two raise NotImplementedError placeholders.
- Write artifacts into directories located next to the wrapper script (e.g., wrapper_outputs/<mode>), flattening top-level files rather than preserving intermediate folders.
- Reuse parsed hardware/model metadata across runs to avoid redundant work.

---

## Mode Execution Details
### 1. DeepFlow (Annotated)
- Launch run_perf.py via subprocess using parsed YAML paths.
- Force artifact persistence with environment variables:
  * DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1
  * DEEPFLOW_PERSIST_ARTIFACT_VIZ=1 when --dry-run is set.
- Add new env DEEPFLOW_ASTRA_SKIP_EXEC so _run_full_astrasim_flattened exits the process immediately after ET emission, skipping AstraSim execution entirely.
- Collect ET bundle (llm_graph.*.et, manifest.json, comm_groups.json, PNG/TXT) from DeepFlow output directories and copy into ./gen_et/deepflow.
- In full runs parse summary_LLM.txt and run_cache_astrasim output for per-rank totals.

### 2. DeepFlow Ablation (Roofline)
- Controlled by env DEEPFLOW_ANNOTATE_COMPUTE=0.
- With flag off, zero per-node durations when building pipeline/transformer graphs while preserving FLOP and tensor size metadata from compute_all_gemm_and_node_times.
- Extend simulate_LLM.Node (and flattening helpers) to carry optional ops/tensor_bytes metadata.
- Update convert_deepflow_graph_to_chakra_et to emit num_ops/tensor_size attrs when durations are zero, enabling AstraSim roofline path.
- When ablation flag active, generate_astrasim_configs_from_hw should set "roofline-enabled": 1 in system json.
- Copy ET bundle into ./gen_et/deepflow_abl and record AstraSim outputs like in annotated mode.

### 3. STG + AstraSim
- Derive STG generator arguments directly from DeepFlow configs: map dp/lp/kp[*] to --dp/--pp/--tp, propagate batch/micro_batch/seq_len/hidden_dim/num_layers/head counts, and compute dff when only ffn_mult is provided.
- Execute symbolic_tensor_graph/main.py via the repo virtualenv (default ../.venv/bin/python3) into a staging directory (no comm-group flag needed; it emits workload.json alongside ETs), then flatten top-level artifacts into wrapper_outputs/stg/.
- When not in dry-run mode, synthesize a manifest from the ET bundle, call generate_astrasim_configs_from_hw with the wrapper-local output directory, and replay the traces through run_cache_astrasim, reusing the returned per-rank timings in the wrapper report. Visualizations (when enabled) reuse astrasim_lib.executor helpers on the copied ET files.
- Clean temporary directories after copying to keep symbolic_tensor_graph/generated free of wrapper residue.

---

## Dry-Run Behaviour
- All modes generate ETs and optional visualizations without running AstraSim.
- Wrapper prints summary table of produced artifacts per mode (ET prefix, manifest path, viz assets).
- DeepFlow dry-run should still output cached graph PNG/TXT and exit with success.

## Full-Run Behaviour
- Execute AstraSim (DeepFlow paths via integrated call, STG via standalone run_cache_astrasim).
- Aggregate per-mode metrics: total iteration time, per-rank times, number of ranks, cache hits/misses.
- Emit comparison report (JSON + console table) highlighting runtime deltas between annotated DeepFlow, ablation, and STG.

---

## Implementation Tasks
1. Prototype wrapper script with global configuration blocks (no CLI) and set up environment handling for DeepFlow annotated mode.
2. Add new environment switches inside DeepFlow codebase:
   - DEEPFLOW_ASTRA_SKIP_EXEC gate in time_calculation_LLM.py.
   - DEEPFLOW_ANNOTATE_COMPUTE handling across compute timing + ET conversion.
   - Roofline toggle inside generate_astrasim_configs_from_hw.
3. Extend ET conversion and Graph data structures to carry FLOP/tensor metadata.
4. Build manifest generation utility reusable by both DeepFlow and STG outputs.
5. Implement visualization helper invocation for dry-run outputs.
6. Integrate standalone AstraSim execution for STG path.
7. Write reporting layer summarizing outputs and checking for rank/metadata mismatches.

---

## Validation Plan
- Dry-run sanity test on existing configs (flattened mode) to ensure ./gen_et directories populated with ETs and visual artifacts.
- Full-run test on small-scale config to capture runtimes from all three modes and verify roofline mode hits AstraSim (requires non-zero tensor_size/num_ops).
- Regression check: ensure annotated DeepFlow path remains unchanged when new env flags are unset.
- Add documentation (README snippet) describing wrapper usage, environment variables, and expected outputs.
