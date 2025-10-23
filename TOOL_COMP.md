# Tool Behavior Comparison

This note captures the divergence we have observed between DeepFlow, Symbolic Tensor Graph (STG), and MLSynth while chasing apples-to-apples comparisons.

## DeepFlow

- **FFN width** – Uses the model-config `ffn_dim`, so FLOPs/bytes reflect actual transformer MLP widths (Llama 2‑7B’s 11008 intermediate instead of a fixed 4× hidden).
- **Backward accounting** – Models both FFN backprop GEMMs (grad‑activation + grad‑weight), doubling FLOPs and bytes exactly as the kernels do.
- **Data-parallel scaling** – Tensor shapes divide by `dp`; per-rank workloads and comm volumes shrink correctly when `dp_size` increases.
- **Unit conventions** – Emits `comm_size`/`tensor_size` in **bytes** and counts GEMM MACs as `2·m·k·n`, matching AstraSim’s byte-based/TFLOP expectations.
- **Pointwise kernels** – LayerNorm/residual/logits softmax are priced with multi-pass memory footprints instead of “one-attr” placeholders.
- **Attention softmax caveat** – Still relies on the naïve QKᵀ + softmax implementation; without FlashAttention modeling the score/softmax tensors carry inflated memory traffic compared to STG/MLSynth.

## Symbolic Tensor Graph (STG)

- **FFN width** – Derives GEMM shapes from the symbolic graph, so widths mirror the model description.
- **FFN backward** – Only charges a single GEMM’s worth of bytes in backprop; grad‑act/grad‑weight traffic is not both represented.
- **Data parallel** – CSV tensors embed `MicroBatch/dp`, so per-rank FLOPs, tensors, and comm sizes scale with `1/dp`.
- **Unit conventions** – `comm_size`/`tensor_size` record element counts; AstraSim interprets them as bytes, undercutting traffic unless scaled by dtype size.
- **GEMM MACs** – Einsum handler counts one op per multiply-add; FLOPs land at half the 2× convention used by peak TFLOPs.
- **Pointwise kernels** – LayerNorm/residual templates are tiny Element/Add ops, yielding unrealistically small FLOP/memory totals.
- **Attention modeling** – Had to be patched back to the non-fused CSV template to get unfused attention kernels; the fused version masked per-kernel detail.
- **Logits softmax** – No final softmax/projection nodes are emitted; downstream consumers must supply them.

## MLSynth

- **FFN width** – Assumes the MLP intermediate is exactly 4× hidden; workloads with narrower FFNs (e.g., Llama 2‑7B² ≈ 2.69× hidden) appear ~50 % too heavy.
- **Data-parallel scaling** – Never divides the global batch by `dp_size`; each rank receives the full batch, so per-rank ETs are identical regardless of data parallelism.
- **Unit conventions** – Already multiplies by bytes-per-element, so AstraSim sees byte-accurate traffic.
- **GEMM MACs** – Uses the `2·m·k·n` convention (when populated) but largely emits monolithic `COMP_NODE_*_ffwd_compute` blocks without per-kernel metadata.
- **Attention modeling** – No awareness of grouped-query attention; `num_heads` is always treated as `kv_heads`, so GQA layouts are inaccurate.
- **Placeholders** – Default ET nodes are generic attention/FFN compute stubs; they need model-specific scaling to mirror full kernel details.

These differences motivated the neutralization and scaling experiments in the wrapper; the list above documents the root causes for each tool. In particular, DeepFlow currently leads on realistic FFN/backward accounting and data-parallel scaling, while STG/MLSynth require dtype-aware sizing and richer kernel modeling to line up with DeepFlow’s fidelity.
