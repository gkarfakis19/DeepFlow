## STG vs DeepFlow Discrepancy (Llama2-7B, DP=2)

### What We Expected
- HuggingFace Llama2-7B config (batch 2048, seq 4096, num_heads 32, hidden 4096).
- QK matmul FLOPs per rank should be `(batch/dp) × heads × seq² × head_dim × 2` (mul+add)  
  → `1024 × 32 × 4096² × 128 × 2 = 1.407×10¹⁴` FLOPs.
- Softmax scaling adds another `3.30×10¹²` FLOPs per layer.

### What DeepFlow Emits
- Graph nodes:
  - `attention_score_forward_mb*` carry `num_ops=1.407×10¹⁴` (correct Seq² scaling).
  - `pt_attention_scale_softmax_*` carry `num_ops=3.30×10¹²`.
  - Embedding lookup `embedding0_1` = `1.72×10¹⁰` FLOPs (gather, not dense matmul).
- AstraSim roofline runtime:  
  - `seq_len = 4096` → `~998 s`  
  - `seq_len = 100k` → `~3.0×10⁵ s`  
  - `seq_len = 16` → `~59 s`

### What STG Emits
- ET nodes derived from `sharding_spreadsheets/module3/tpsp/group_query_attention_*.csv`.
- `attn_kernel.qkv` CUSTOM node encodes only `Batch × Seq × hidden` (linear projection) **no `Seq²` term**:
  - `num_ops=5.15×10¹⁰` for Llama2-7B (3 orders of magnitude too small).
- No softmax nodes anywhere in the ET (`grep "softmax"` → empty).
- Embedding modeled as dense GEMM:
  - `mb0.in_emb.y/dw/dx` each report `1.10×10¹⁵` FLOPs (32k× larger than DeepFlow).
- AstraSim runtime reflects the metadata:
  - `seq_len = 4096` → `~438 s`
  - `seq_len = 100k` → `~1.1×10⁴ s` (stays ∝ `seq`, not `seq²`)
  - `seq_len = 16` → `~1.7 s` (now close to DeepFlow)

### Consequences
1. **Missing attention score cost**: main matmul is undercounted by ~2700× for seq=4096, so long sequences run far too fast under STG.
2. **Missing softmax/scaling kernels**: further underestimates compute/bytes.
3. **Embedding treated as matmul**: huge over-count of FLOPs that do not occur in DeepFlow’s gather-based embedding, skewing totals.

Overall, STG’s metadata omits the quadratic attention score workload and softmax entirely, while inflating embedding costs. DeepFlow preserves both, hence the large runtime divergence that grows with sequence length. Narrow sequences mask the issue because the missing Seq² term is small; large sequences expose it immediately.

Follow-up digging confirmed the root cause: the STG stage-1 LLaMA model was switched to `group_query_attention_kernel_fused.csv`, whose CUSTOM node only scales with `Batch×Seq×Hidden` and drops the Seq² matmuls plus softmax bookkeeping. Reverting to the original `group_query_attention_kernel.csv` restores the quadratic attention ops in the ETs, bringing the roofline scaling back in line with DeepFlow. The fused kernel might only be defensible if someone injects empirical latency data for that CUSTOM op, which is not happening in the current STG flow.
