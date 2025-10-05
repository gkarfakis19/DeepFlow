# SimAI Analytical Integration Follow-Up Plan

## Scope
This document drills into the first three limitations called out in the initial SimAI analytical integration:

1. Missing backward/input-gradient compute coverage.
2. Absent dependency reconstruction (`dependency = -1`).
3. Heuristic communication label synthesis.

For each area we outline the required DeepFlow data sources, conversion logic, and wrapper interfaces that must change to close the fidelity gap.

---

## 1. Populate Backward and Input-Gradient Compute Columns

### Current behaviour and constraints
- `SimAIWorkloadBuilder` collapses all backward compute into the weight-gradient column and leaves input-gradient compute at zero because the flattened DeepFlow nodes only expose a Boolean `fwd` flag and a coarse `direction` string.【F:tools/simai_analytical.py†L226-L279】
- DeepFlow’s GEMM timing utilities already compute separate activation-gradient and weight-gradient durations inside `_distributed_gemm_backward`, but those values are summed before the results reach the transformer graph builder.【F:time_calculation_LLM.py†L535-L563】

### Plan of record
1. **Expose gradient components in timing results.**
   - Extend `_distributed_gemm_backward` to return a structured object (or augment the existing dict in `compute_all_gemm_and_node_times`) that surfaces `grad_act_time` and `grad_wt_time` alongside the combined total for each GEMM.
   - For CR/RC tensor-parallel cases, `getDistGEMM_b_kp1`/`getDistGEMM_b_kp2` already compute both pieces; thread them through instead of collapsing immediately.【F:time_calculation.py†L1567-L1698】
2. **Thread metadata into transformer graph construction.**
   - Include the two backward components in `transformer_gemm_entries` so each `simulate_LLM.Node` created for backward passes carries both activation- and weight-gradient durations, either by attaching custom attributes (e.g., `bwd_grad_act_duration`) or by splitting a single node into two sequential nodes per GEMM.【F:time_calculation_LLM.py†L1325-L1372】【F:simulate_LLM.py†L556-L612】
3. **Aggregate accurately in the workload builder.**
   - Update `_LayerAccumulator` to maintain `input_grad_compute_s` separately from `weight_grad_compute_s`.
   - When walking the flattened graph, consume the new metadata: map activation-gradient components into SimAI’s input-gradient columns and weight-gradient components into the weight-gradient columns while leaving optimizer/update slots untouched.
4. **Validation checkpoints.**
   - Add sanity checks that the sum of forward/backward components per layer matches DeepFlow’s original `transformer_time_f`/`transformer_time_b` outputs to avoid drift.【F:time_calculation_LLM.py†L1035-L1108】
   - Unit-test the builder against a small config to ensure SimAI columns are populated and consistent.

### Open considerations
- If the backward component split requires changing `simulate_LLM.Node` semantics, guard with feature flags so existing DeepFlow analytical/Astra flows remain unaffected.
- Ensure any additional nodes maintain `micro_batch_index`, `layer_index`, and `stage_id` to keep the aggregator functioning.

---

## 2. Reconstruct Record Dependencies

### Current behaviour and constraints
- Every SimAI record is emitted with `dependency = -1`, ignoring the true ordering encoded by the flattened graph’s parent/child relationships.【F:tools/simai_analytical.py†L268-L283】
- The flattened graph already captures precise dependencies via `parents`/`children` lists on `simulate_LLM.Node`/`Edge` objects.【F:simulate_LLM.py†L24-L64】

### Plan of record
1. **Track record membership while iterating.**
   - While walking `_iter_graph`, build a map from each graph object to its `(stage, layer, micro_batch)` key so dependencies can later be resolved at record granularity.
2. **Compute topological order and parent references.**
   - After aggregation, traverse each record’s constituent nodes to identify upstream records (`parent_key != current_key`). Collect the latest dependency (e.g., max topological index) or maintain the full set if SimAI can consume multiple dependencies.
   - Derive a deterministic record ordering (micro-batch → layer → stage) and compute the integer index SimAI expects. If SimAI requires a single dependency ID, choose the highest-index predecessor or encode a sentinel when multiple exist.
3. **Encode dependencies in workload output.**
   - Replace the hard-coded `-1` with the resolved dependency index before writing `SimAIRecord` lines.
4. **Validation checkpoints.**
   - Assert that dependencies never reference future records (guarding the topological ordering).
   - Optionally emit a DOT/JSON debug dump for a small config to cross-verify with SimAI’s original workloads.

### Open considerations
- Confirm SimAI’s parser semantics (single integer vs. list). If only a single integer is supported, document the policy for resolving many-to-one dependencies (e.g., pick the last predecessor in the emission order).
- Handle cases where a layer has no external parents (should remain `-1`).

---

## 3. Replace Heuristic Communication Labeling

### Current behaviour and constraints
- Labels are currently synthesized via `_COMM_BASE_LABELS`/`_COMM_SUFFIXES` lookups on `comm_type` and `comm_interconnect_type`, causing ambiguous cases to collapse to `NONE` when multiple collectives contribute to the same record.【F:tools/simai_analytical.py†L210-L221】【F:tools/simai_analytical.py†L244-L267】
- DeepFlow already constructs per-GEMM communication metadata with precise keys and participant counts (`transformer_comm_metadata`), and those keys remain attached to nodes through `comm_keys` when the transformer graph is built.【F:time_calculation_LLM.py†L1171-L1213】【F:simulate_LLM.py†L556-L612】

### Plan of record
1. **Capture comm-key provenance.**
   - While iterating the flattened graph, record the exact metadata key (e.g., `qkv_proj_backward_wt_all_gather`) attached to each `Edge` instead of relying on lowercase string conversions.
   - Enrich `SimAIWorkloadBuilder` with a lookup into the original `transformer_comm_metadata` so we can recover the canonical collective type, participants, and the tensor-parallel axis associated with the edge.
2. **Define deterministic mapping rules.**
   - Translate DeepFlow’s metadata fields (`type`, `interconnect_type`) into SimAI’s vocabulary with a structured mapping table that distinguishes DP/TP/EP collectives and combined cases (e.g., DP+EP all-reduces) without losing detail.
   - When multiple communication events feed the same column (e.g., both `wt_all_gather` and `act_all_gather` for RC backward), emit composite labels (`ALLGATHER_TP_WT+ACT`) if SimAI accepts custom tokens; otherwise, split the record into separate rows so each SimAI phase carries a single well-defined collective.
3. **Augment aggregation strategy.**
   - Instead of blindly summing bytes per direction, accumulate per-collective statistics in a structured list and only collapse to a single label when the set is homogeneous. Provide fallbacks (e.g., choose the dominant byte-count collective) with explicit warnings when heterogeneity remains.
4. **Validation checkpoints.**
   - Cross-check emitted labels against SimAI workloads generated by AICB for equivalent configs to ensure naming conventions match.
   - Add assertions ensuring every communication edge with non-zero bytes is represented in exactly one SimAI column, preventing silent drops.

### Open considerations
- Investigate whether SimAI analytical mode accepts multi-token suffixes or requires the exact `ALLGATHER[_TP]` schema; adapt mapping accordingly.
- For non-transformer components (embeddings, optimizer), decide whether to include their collectives or continue skipping them with documented rationale.

---

## Next Steps
- Prototype the backward/IG compute split first since it feeds into both dependency timing accuracy and communication attribution.
- Stage changes behind a wrapper flag (`enable_simai_detailed_metrics`) to allow incremental rollout.
- Update documentation and wrapper CLI help once fidelity improvements land.
