# MLSynth Data Parallel Behavior

This note documents why MLSynth currently produces identical per-rank execution traces regardless of the requested data-parallel degree, and contrasts that behavior with Symbolic Tensor Graph (STG).

## MLSynth: DP scaling is not applied

The generator copies the global batch size straight into the model and never adjusts it per replica. Key paths:

- `MLSynth/synthesise_workload.py` only validates `batch_size >= dp_size` and `batch_size / dp_size >= num_microbatches`, but it never updates the batch stored in the config:
  ```python
  # MLSynth/synthesise_workload.py:43-49
  def validate_config(cfg):
      if cfg["model"]["batch_size"] < cfg["parallelism"]["dp_size"]:
          raise ValueError(...)
      batch_size = cfg["model"]["batch_size"] // cfg["parallelism"]["dp_size"]
      if batch_size < cfg["model"]["num_microbatches"]:
          raise ValueError(...)
  ```

- `Transformer.__init__` stores the global batch unmodified:
  ```python
  # MLSynth/Model/Transformer.py:29-38
  self.batch_size = int(config["model"]["batch_size"])
  ...
  self.layers.append(TransformerLayer(..., scale=self.scale))
  ```

- When emitting ET nodes, the orchestrator still uses the full global batch `B` for every rank:
  ```python
  # MLSynth/Orchestrator/MegatronLM.py:61-109
  B = self.model.get_batch_size()
  ...
  for b in range(self.num_microbatches):
      cmp_nodes = self.model.fwd(..., num_batches=B/self.num_microbatches, ...)
  ```

  The only DP-specific logic is at the end, where it optionally appends an all-reduce:
  ```python
  if self.dp_size > 1:
      dp_comm_node = allreduce(dp_comm_size, ...)
  ```

Because `num_batches` is computed from the global batch `B` and never divided by `dp_size`, every replica’s ET bundle reflects the entire global batch. Changing `dp_size` only changes how many copies are emitted, not the per-rank workload. AstraSim’s totals therefore remain unchanged.

## STG: DP scaling baked into node shapes

STG’s generator does rewrite tensor dimensions by the data-parallel factor when it converts the symbolic graph to Chakra traces. Evidence:

- The main driver embeds `MicroBatch/dp` into every operator’s shape before converting to ETs (`symbolic_tensor_graph/main.py:208-343`). The emitted CSVs confirm the division; the example below comes from the default Llama trace:
  ```text
  symbolic_tensor_graph/llama.csv:2  ... "MicroBatch/dp, Seq/(cp*tp), Dmodel"
  symbolic_tensor_graph/llama.csv:14 ... "MicroBatch/dp, Seq/cp, Dmodel/Head, Head/tp"
  ```

- Those CSVs are translated directly into Chakra nodes, so the per-rank FLOPs, tensor sizes, and comm volumes all scale with `1/dp`. Increasing `dp` genuinely shrinks the per-rank workload and AstraSim reports a lower iteration time.

## Summary

- MLSynth: global batch is never divided by `dp_size`; per-rank ETs are clones. Fixing this requires modifying MLSynth’s generator to halve (or generalize) `B` before creating nodes.
- STG: tensor shapes include `1/dp`—all per-rank workloads shrink as expected when data parallelism increases.

Until MLSynth’s generator is updated, wrapper-level workarounds (e.g., renaming comm groups) cannot change this behavior.
## MLSynth TP FLOPs Workaround

MLSynth's Transformer templates ignore the configured tensor-parallel degree when computing GEMM FLOPs: `_attention_compute` and `_ffwd_compute` always use the full `batch_size * seq_len * hidden_size` operands regardless of `tp_size`. When the wrapper runs AstraSim in roofline mode, those inflated FLOP counts produce runtimes roughly `tp` times longer than DeepFlow's analytic timings.

Until MLSynth fixes the generator, our wrapper scales the per-rank batch size to compensate. With scheduling params `dp`, `tp`, `lp=1`, `mb=1`:

- We compute `scaled_batch = max(1, (global_batch_size / dp) / tp)` and write that into `wrapper_tmp/wrapper_input.yaml`.
- This divides the GEMM FLOPs by the tensor-parallel degree so `num_ops` in the ET matches the work each rank actually executes, allowing roofline replay to line up with DeepFlow/DeepFlow ablation.

This is only a temporary hack: it implicitly shrinks the problem size for MLSynth (and its reported communication volumes). Remove this once MLSynth's `TransformerLayer` accounts for `tp_size` in its FLOP formulas.