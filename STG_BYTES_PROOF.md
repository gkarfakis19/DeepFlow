# STG `comm_size` Is Measured In Elements, While AstraSim Expects Bytes

This note documents the evidence that symbolic tensor graph (STG) workloads emit `comm_size` values in **elements**, but AstraSim interprets them as **bytes**. The mismatch explains the communication-size disagreement you are observing between DeepFlow and STG.

---

## 1. STG Emits Raw Element Counts

- STG builds Chakra communication nodes by evaluating tensor shapes with `Tensor.eval_size`, which returns the product of the symbolic dimensions without any dtype scaling. The value is written directly to `node.comm_size`:
  - Collective edges: `symbolic_tensor_graph/symbolic_tensor_graph/graph/convert_chakra.py:119-181`
  - Pipeline send/recv edges: `symbolic_tensor_graph/symbolic_tensor_graph/graph/convert_chakra.py:465-490`
- `Tensor.eval_size` is explicitly “number of elements” (`symbolic_tensor_graph/symbolic_tensor_graph/tensor.py:120-124`); there is no path that multiplies by bytes-per-element.
- The `.et` artifacts confirm this. Example (`tools/wrapper_outputs/stg/workload.1.et.txt:1-40`):
  - Node `mb0.transformer.2.input_norm.dx@0_Y_SEND` reports `comm_size=274877906944`.
  - The tensor shape is `MicroBatch/dp,Seq/(cp*tp),Dmodel`. With the active config (`micro_batch=2048`, `seq_len=16384`, `hidden_dim=8192`) the element count is `2048 × 16384 × 8192 = 274,877,906,944`. The number matches exactly—no 2-byte (fp16) scaling is present.

**Conclusion:** STG’s ET files encode communication payloads purely as element counts.

---

## 2. AstraSim Treats `comm_size` As Bytes

Once a Chakra ET is replayed, AstraSim never rescales `comm_size`—it forwards the value as the byte count that the network backend must transport.

1. **Workload ingestion (`Workload.cc`):**
   - Collectives: `comm_size = node->comm_size<uint64_t>()`; the same value is passed to `sys->generate_all_reduce(comm_size, …)` and recorded for statistics (`astra-sim/astra-sim/workload/Workload.cc:337-368`).
   - Pipeline edges: send/recv nodes call `sys->front_end_sim_send(…, size, UINT8, …)` with `size = node->comm_size<uint64_t>()` (`astra-sim/astra-sim/workload/Workload.cc:399-416`).

2. **System layer (`Sys.cc`):**
   - `front_end_sim_send` (and `_recv`) simply forwards the `uint64_t count` argument to `sim_send`/`sim_recv` (`astra-sim/astra-sim/system/Sys.cc:1455-1600`); no precision adjustment occurs.
   - The request type is `UINT8` (`astra-sim/astra-sim/system/Common.hh:21`), so the network stack interprets each unit in `count` as one byte.

3. **Analytical network backend:**
   - `CongestionUnawareNetworkApi::sim_send` invokes `topology->send(src, dst, count)`, passing the same `count` (`astra-sim/astra-sim/network_frontend/analytical/congestion_unaware/CongestionUnawareNetworkApi.cc:59-76`).
   - In the analytical backend, `ChunkSize` is defined as “chunk size in **Bytes**” (`astra-sim/extern/network_backend/analytical/include/astra-network-analytical/common/Type.h:18`). The topology delay calculations therefore assume byte units.
   - Collective algorithms (e.g., ring all-reduce) inject chunks using `front_end_sim_send(..., msg_size, UINT8, …)` where `msg_size` derives from the original `comm_size` (`astra-sim/astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.cc:220-244`).

**Conclusion:** Every layer of AstraSim—workload ingestion, system layer, and network backend—treats `comm_size` as bytes. The simulator does not apply dtype scaling on its own.

---

## 3. Implication

Because STG emits element counts while AstraSim expects byte counts, all communication in STG-generated ET files is underestimated by a factor of the bytes per element (e.g., ×2 for fp16, ×4 for fp32, ×1 for fp8). DeepFlow’s exporter already multiplies by precision, so DeepFlow and AstraSim agree. To reconcile STG with DeepFlow, you need to scale STG’s `comm_size` fields by the appropriate dtype size before AstraSim consumes them (e.g., within `tools/wrapper.py` or via a post-processing step).

---

## 3. Compute/Memory Nodes Are Also In Elements

STG applies the same element-count convention to tensor sizes used by compute and remote-memory ops:

- `_insert_comp` sets `node.tensor_size = Tensor.eval_size(tensor.y_shape)` (`symbolic_tensor_graph/symbolic_tensor_graph/graph/convert_chakra.py:77-96`), so every COMP node’s `tensor_size` records raw element counts.
- AstraSim roofline mode consumes that value directly: `tensor_size = node->tensor_size<uint64_t>()` and operational intensity `= num_ops / tensor_size` (`astra-sim/astra-sim/workload/Workload.cc:250-285`). No dtype scaling is applied in the simulator.
- Remote-memory replay forwards `node->tensor_size()` into the analytical remote-memory backend, which computes latency as `tensor_size / remote_mem_bw` assuming bytes (`astra-sim/astra-sim/workload/Workload.cc:232-244`, `astra-sim/extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.cc:73-109`).

**Result:** Without a post pass, AstraSim implicitly treats every STG tensor element as 1 byte for both compute roofline calculations and remote-memory timing. Byte-accurate comparisons require scaling `tensor_size` along with `comm_size`.

---

## 4. GEMM FLOPs Are Counted Differently

- STG’s matmul ops use the `Einsum` handler, which multiplies the output dimensions by the reduced dimensions once (`symbolic_tensor_graph/symbolic_tensor_graph/ops/einsum.py:40-68`). A single Multiply-Accumulate therefore counts as one operation.
- DeepFlow doubles GEMM MACs: `_compute_gemm_flops_bytes` returns `forward_flops = 2 * M * K * N` (`time_calculation_LLM.py:1012-1045`), matching how GPU peak TFLOPs are specified.
- AstraSim’s `peak-perf` value is derived from the DeepFlow hardware config using that same 2× convention (`astrasim_lib/config_generation.py:162-209`).

**Consequence:** Without correction, STG’s GEMM nodes report half the FLOPs that the hardware peak assumes, leading to inflated roofline utilization. The wrapper now doubles `num_ops` for STG COMP nodes with `op_type == "M"` to align the FLOP accounting.

### TL;DR
- STG: `comm_size = element_count`, `tensor_size = element_count`, GEMM `num_ops = MAC count`
- AstraSim/DeepFlow assume bytes for sizes and MAC×2 for GEMM FLOPs
- Fix: scale STG comm & tensor sizes by bytes/elem and double GEMM `num_ops` before replay
