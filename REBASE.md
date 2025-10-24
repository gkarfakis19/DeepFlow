## Rebase Notes – `llm-compare` onto `llm-training`

### High-Level Outcome
- Rebased `llm-compare` on top of the latest `llm-training` (which already carries tensor/context/sequence parallel support, schema updates, etc.).
- Restored the llm-compare “deepflow ablation” functionality so wrapper tooling can emit FLOP/tensor-byte metadata and roofline-aware ET artifacts without regressing the new features added in `llm-training`.
- No tests were run in this session; follow-up wrapper runs are still required.

### Code Changes
1. **`time_calculation_LLM.py`**
   - Reintroduced metadata bookkeeping (FLOPs/tensor bytes) for all GEMMs and pointwise ops.
   - Added `DEEPFLOW_ANNOTATE_COMPUTE` gating so durations can be zeroed when generating roofline traces.
   - Threaded metadata into transformer entries, pipeline `comp_times`, and node breakdowns used later by AstraSim.
   - Added helper `_compute_gemm_flops_bytes` and expanded node_breakdown/comp_times structures with `*_flops` / `*_bytes` entries.

2. **`LLM_excution.py`**
   - Pipeline flattener now preserves `flops`/`tensor_bytes` when cloning and expanding nodes.
   - `_copy_metadata` copies those attributes so flattened ETs retain roofline annotations.

3. **`simulate_inf.py`** & **`time_calculation_inf.py`**
   - Recreated the decode-step GEMM measurement path so inference samples emit FLOP/byte metadata.
   - Updated to compute per-op FLOPs/bytes matching the training path and pass them into the shared graph builder.

4. **`run_perf.py`**
   - Defaulted `cache_handling` to `NO_CACHE` to align with wrapper expectations (avoids stale ET reuse).

5. **`astrasim_lib` support**
   - Existing `new_comp_node` already accepts `flops`/`tensor_bytes`; `convert_deepflow_graph_to_chakra_et` now feeds it real metadata coming from the restored code paths.

6. **Miscellaneous**
   - Removed temporary TODO scaffolding once functionality was reintroduced.
   - Ensured all touched modules compile via `python3 -m compileall`.

### Clean-Up / Push Instructions
- Working tree contains staged changes above; remote `origin/llm-compare` still has old history.
- To publish the rebased branch: ensure the tree is clean, then `git push --force-with-lease origin llm-compare`.
- Wrapper validation (deepflow annotated & ablation modes) still needs to be run manually.
