# AstraSim Library Guide

## Module reference
- `astrasim_lib/__init__.py` re-exports the library's public helpers so callers can import bootstrap, configuration, ET, and execution utilities from a single namespace.
- `astrasim_lib/bootstrap.py` appends the Chakra protobuf and helper directories from the checked-out `astra-sim` submodule to `sys.path` and verifies the generated modules (`et_def_pb2`, `protolib`) can be imported.
- `astrasim_lib/config_generation.py` derives the analytical network YAML and system JSON required by AstraSim. It computes intra-node bandwidth/latency via `hw_component.Network`, chooses collective implementations from the DeepFlow hardware object, and writes deterministic files in the AstraSim cache directory.
- `astrasim_lib/et_utils.py` imports the Chakra protobuf definitions and exposes helpers to create compute/communication nodes, write ET records, and emit microbenchmark traces for collectives or pipeline traffic.
- `astrasim_lib/integration.py` owns cache orchestration for analytical runs. It normalizes environment overrides, generates workload traces (either single-collective or concurrent sequences), locates the bundled remote-memory JSON, and wraps the AstraSim binary invocation with cache lookups and retries.
- `astrasim_lib/executor.py` converts DeepFlow graphs into Chakra ET bundles, constructs communicator groups, and reuses the caching helpers to run the analytical binary. It also offers visualization/debug routines and cleans generated artifacts when temporary directories are reused.

## Analytical execution flow
1. **Bootstrap Chakra dependencies.** Modules call `ensure_chakra_available()` so the protobuf schema and helper utilities shipped with the `astra-sim` submodule can be imported before any trace emission or decoding occurs.
2. **Generate AstraSim configs.** `generate_astrasim_configs_from_hw()` requires an explicit rank count. It selects the effective topology (ring configurations with ≤2 ranks are promoted to fully-connected), converts the measured throughput/latency into GB/s and ns, writes `network_analytical_<NPUS>.yml`, and emits `system_native_collectives.json` with either backend-provided overrides or topology-driven defaults.
3. **Produce workload ET bundles.**
   - Microbenchmarks call `generate_workload_et()` for single collectives or `generate_concurrent_collectives_et()` for scripted sequences. Both helpers emit `<prefix>.<rank>.et` files and store them under `astra_cache/workload/...`.
   - Graph-driven runs go through `executor.convert_deepflow_graph_to_chakra_et()`, which traverses the DeepFlow DAG, expands pipeline send/recv pairs, tracks collective/control dependencies, and writes `llm_graph.<rank>.et` per rank. During this conversion a `manifest.json` is generated alongside the ET files summarizing the per-rank operations; the manifest is **not** consumed by AstraSim, but it fingerprints the graph for caching.
4. **Launch the analytical binary with caching.** `run_cache_astrasim()` prepares a cache signature that records the communication type, rank count, message size, topology, derived bandwidth/latency, collective policy choices, and any system overrides. When a `manifest.json` path is provided, the cache key hashes the manifest contents together with the generated network/system configs (and optional communicator groups) so graph edits invalidate cached results. The helper materializes missing workloads, resolves the bundled remote-memory configuration (`astra-sim/examples/remote_memory/analytical/no_memory_expansion.json`), and finally calls `run_astrasim_analytical()` with:
   - `--workload-configuration=<prefix>` pointing to the ET bundle prefix.
   - `--system-configuration=<system_native_collectives.json>`.
   - `--network-configuration=<network_analytical_<NPUS>.yml>`.
   - `--comm-group-configuration=<comm_groups.json>` when `_write_comm_groups_json()` produced stage groupings for model-parallel graphs.
   The manifest never reaches the command line; it only influences cache lookups.
5. **Collect and persist results.** `run_astrasim_analytical()` parses wall-time lines from the AstraSim output (converting cycles to seconds), and `run_cache_astrasim()` stores the per-rank timings, maximum duration, canonical signature, workload prefix, and config paths in `cache.json` when caching is writable.

## Cache behaviour and controls
- **Storage layout.** Cache metadata lives at `<ASTRA_CACHE_DIR>/cache.json` (default `./astra_cache/cache.json`). Generated workloads and configuration files are stored under the same directory so repeated runs can reuse them.
- **Modes & environment variables.** `DEEPFLOW_ASTRA_CACHE_MODE` selects `NO_CACHE`, `CACHE_READONLY`, or `CACHE_READWRITE`. `ASTRA_CACHE_DIR` relocates both the cache file and any generated workload/configuration artifacts.
- **Entry structure.** Each cache record captures the normalized signature, hashed canonical string, per-rank wall times, max runtime, and file locations. When a manifest is involved, its path is recorded for transparency; the manifest’s bytes were already folded into the cache key to guard against stale graph workloads.
- **Retries and validation.** Analytical launches retry up to five times if the simulator reports zero durations. Persistent failures raise an error so upstream callers can diagnose network or configuration issues instead of silently returning zero timings.
