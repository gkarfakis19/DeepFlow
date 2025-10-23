import math
import os
import json
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List
import simulate_LLM
from LLM_excution import ExecutionMode, LLMExecutionDispatcher
from simulate_LLM import Graph
import LLM_util
from time_calculation import TimeCalculation
# algByte = False  # algorithmic ops false
# proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True

debug = True

showing_ms = False # Show time in ms if True    show time in us if False
if showing_ms:
    m=1e3
    second = "ms"
else:
    m=1e6
    second = "us"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}

    
class ParallelismMode(Enum):
    TENSOR = "tensor"
    TENSOR_SEQUENCE = "tensor_sequence"
    CONTEXT = "context"
    TENSOR_CONTEXT_HYBRID = "tensor_context_hybrid"
    SINGLE = "single"

# TODO: verify all 4 of these
SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT = 10
SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT = 20
GELU_FORWARD_FLOPS_PER_ELEMENT = 10
GELU_BACKWARD_FLOPS_PER_ELEMENT = 20
LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT = 7
LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT = 14
LAYER_NORM_FORWARD_MEM_ACCESSES = 2
LAYER_NORM_BACKWARD_MEM_ACCESSES = 4
SOFTMAX_FORWARD_FLOPS_PER_ELEMENT = 7 # exponentiation 4FLOPS, subtract max 1FLOPS, dropout 2FLOPS
SOFTMAX_FORWARD_MEM_ACCESSES = 7

# Map each parallelism mode to operation-level collective specs used across the
# metadata pipeline. Each spec records the collective kind, the participant
# scope (tp/cp/seq/etc.), the interconnect label, and an identifying suffix.
COMM_RULE_DEFAULT_KEY = "__default__"
COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, str]]]]
] = {
    ParallelismMode.TENSOR: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': 'all_reduce', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_reduce'},
            'backward': {'kind': 'all_reduce', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_reduce'},
        },
    },
    ParallelismMode.TENSOR_SEQUENCE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'layernorm2': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'MHA': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
        'MLP': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
    },
    ParallelismMode.CONTEXT: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'},
        },
        'attention': {'forward': None, 'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'}},
        'output_proj': {'forward': None, 'backward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'}},
        'qkv_proj': {'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'}, 'backward': None},
    },
    ParallelismMode.TENSOR_CONTEXT_HYBRID: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'layernorm2': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'MLP': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
        'attention': {'forward': None, 'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'}},
        'output_proj': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'},
        },
        'qkv_proj': {
            'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
    },
    ParallelismMode.SINGLE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
    },
}
class GemmType(Enum):
    ATTENTION_SCORE = "attention_score"
    ATTENTION_OUTPUT = "attention_output"
    QKV = "qkv"
    OUT_PROJ = "out_proj"
    FFN1 = "ffn1"
    FFN2 = "ffn2"
    LINEAR_SOFTMAX = "linear_softmax"

# === TODO(LLM_COMPARE_ROOFLINE_INTEGRATION) ==================================
# The llm-compare branch emitted FLOP / tensor-byte metadata and flattened
# pipeline graphs before exporting ET files. When we are ready to reintegrate
# that functionality, pull the helpers below (captured verbatim from commit
# fbcca79) and thread the resulting metadata through the call sites that build
# compute/communication nodes:
#
# class PipelineGraphFlattener:
#     """Expand pipeline transformer nodes into explicit tensor-parallel subgraphs."""
#     ...
#     def _clone(self, obj: Any) -> Any:
#         if isinstance(obj, simulate_LLM.Node):
#             ...
#             cloned = simulate_LLM.Node(
#                 obj.name,
#                 self._next_op_id(),
#                 obj.hw_id,
#                 obj.duration,
#                 fwd=obj.fwd,
#                 is_kv_cache=getattr(obj, "is_kv_cache", False),
#                 flops=getattr(obj, "flops", 0),
#                 tensor_bytes=getattr(obj, "tensor_bytes", 0),
#             )
#             ...
#     def _expand_transformer_node(...):
#         ...
#         comp_node = simulate_LLM.Node(
#             f"{gemm_name}_{direction}",
#             self._next_op_id(),
#             self._hw_id_for_rank(stage_id, tp_rank),
#             duration,
#             fwd=direction == "fwd",
#             flops=gemm_entry.get("flops", 0),
#             tensor_bytes=gemm_entry.get("tensor_bytes", 0),
#         )
#
# That code also introduced ExecutionResult / TransformerTimings dataclasses and
# recorded GEMM-level FLOPs and tensor bytes inside compute_all_gemm_and_node_times.
# Keep those snippets handy when we revisit roofline support.
# === END TODO(LLM_COMPARE_ROOFLINE_INTEGRATION) ===============================


class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
# Mode parameter
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self._generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        self.persist_astrasim_artifacts = _env_flag("DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS")
        self.execution_mode = execution_mode
        inference_cfg = getattr(hw_config, "inference_config", None)
        if inference_cfg and getattr(inference_cfg, "kvcache_precision", None) is not None:
            self.kv_cache_precision = inference_cfg.kvcache_precision
        else:
            self.kv_cache_precision = self.precision
        self.kv_cache_type = getattr(inference_cfg, "kvcache_type", "hbm_only") if inference_cfg else "hbm_only"
        self.kv_cache_fetch_overlap = bool(getattr(inference_cfg, "kvcache_fetch_overlap", False)) if inference_cfg else False
        
        self.all_reduce = self.model.all_reduce # when the reduce happens in data parallelism options: "the end"  "every layer"
        self.model_type = self.model.model_type
        self.tied_embeddings = getattr(self.model, "tied_embeddings", True)
        self.memory_capacity_exceeded = False
        self.memory_capacity_violation_gb = 0.0
        annotate_env = os.environ.get("DEEPFLOW_ANNOTATE_COMPUTE")
        self.annotate_compute = True if annotate_env is None else _env_flag("DEEPFLOW_ANNOTATE_COMPUTE")
        self.dtype_size = int(self.precision) if self.precision else 2
        if self.dtype_size <= 0:
            self.dtype_size = 2
        self.pipeline_graph: Optional[Graph] = None
        self.pipeline_root: Optional[Any] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None
        self.transformer_graph: Optional[Graph] = None
        self.transformer_forward_root: Optional[Any] = None
        self.transformer_backward_root: Optional[Any] = None
        self.transformer_analytical_time_forward: Optional[float] = None
        self.transformer_analytical_time_backward: Optional[float] = None
        self.transformer_astrasim_time_forward: Optional[float] = None
        self.transformer_astrasim_time_backward: Optional[float] = None
        self.transformer_astrasim_per_rank_forward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward: Optional[List[float]] = None
        self.pipeline_astrasim_time: Optional[float] = None
        self.pipeline_astrasim_per_rank: Optional[List[float]] = None

    def _sequence_parallel_degree(self) -> int:
        """Return tensor-parallel degree used for sequence-parallel collectives.
            return 1 if no sequence parallelism is used.
            return tp if tp-sp is True and no context parallelism is used
            return cp if context parallelism is used
        """
        if self.tp_sp and self.cp == 1: #tensor parallelism only
            return self.tp
        elif self.cp > 1: #context parallelism or cp-tp hybrid parallelism
            return self.cp
        else: 
            return 1

    def get_parallelism_mode(self):
        if self.tp_sp and self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR_SEQUENCE
        elif self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR
        elif self.cp > 1 and self.tp == 1:
            return ParallelismMode.CONTEXT
        elif self.cp > 1 and self.tp > 1:
            return ParallelismMode.TENSOR_CONTEXT_HYBRID
        else:
            return ParallelismMode.SINGLE

    def get_kv_size_bytes(self) -> int:
        """Return the total size in bytes of the KV cache."""
        total_elements = 2 * self.seq_len * self.microB * self.hidden_dim / self.num_heads * self.kv_heads
        return total_elements * self.precision

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astra":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astra", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode in (
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
            ExecutionMode.FULL_ASTRASIM_FLATTENED,
        ):
            return 'full'
        # Hybrid still uses the analytical pipeline, so keep the hybrid policy.
        return 'hybrid'

    @staticmethod
    def _expand_gemm_descriptor(gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        if len(gemm) == 3:
            return 1, gemm[0], gemm[1], gemm[2]
        if len(gemm) == 4:
            return gemm[0], gemm[1], gemm[2], gemm[3]
        raise ValueError(f"Unsupported GEMM descriptor length: {len(gemm)}")

    @classmethod
    def _effective_dims(cls, gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        batch, m, k, n = cls._expand_gemm_descriptor(gemm) 
        return batch, batch * m, k, n
    def _ffn1_output_dim(self, ffn_dim: int) -> int:
        return 2 * ffn_dim if self.model_type == "llama" else ffn_dim
    # def sequence
    def get_tensor_reduction_time(self, total_bytes: int, kind: str, name: str, participants: Optional[int] = None) -> float:
        """Return collective time for tensor-parallel reductions.

        `size_bytes` is expected to be the per-rank payload. Convert to total bytes so the
        network model sees the same aggregate volume it assumes internally.
        """
        if not total_bytes:
            return 0.0

        if not participants:
            participants = int(self.tp)

        reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return reduction_time
    def flash_attention_kernel_forward(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return time for flash attention kernel."""
        #TODO gqa support
        batch_size = self.microB
        seq_len = self.seq_len
        hidden_dim = self.hidden_dim
        num_heads = self.num_heads
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        d = hidden_dim // num_heads #gemm shape for one head is (seq_len, d) x (d, seq_len)
        Bc = self.attention_tile_size
        Br = min(Bc, d)
        Tr = math.ceil(shard_seq / Br) #number of row tiles
        Tc =  math.ceil(shard_seq / Bc) #number of column tiles
        print(f"Flash Attention: Br={Br}, Tr={Tr}, d={d}, Bc={Bc}, Tc={Tc}")
        #TODO load 3 tiled gemm inputs and 1 output from HBM to SRAM
        
        #TODO disable HBM access
        attention_forward_reduction_time = 0
        attention_size_f = 0
        
        attn_score_time = self.getGEMMTime(Br, d, Bc, "attention_score_f")[0] * Tc * Tr #attention score gemm time for one head
        number_flops =  Br * Bc * d * Tc * Tr * batch_size * num_heads
        print(f"Flash Attention: attention score gemm flops={number_flops/1e9} GFLOPS")
         
        # Softmax time
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  # exponentiation + subtract max + dropout
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        attn_output_time = self.getGEMMTime(Br, Bc, d, "attention_output")[0] * Tc * Tr #attention output gemm time for one head
        attn_score_time *= batch_size * num_heads / max(1, self.tp)
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        attn_output_time *= batch_size * num_heads / max(1, self.tp)
        
        attention_forward_gemm_time = attn_score_time + attn_scale_softmax_time + attn_output_time
        
        attention_forward_time = attention_forward_gemm_time + attention_forward_reduction_time
        attention_size_f = 0
        
        return attention_forward_time, attention_forward_gemm_time, attention_forward_reduction_time, attention_size_f
    
    
    def flash_attention_kernel_backward(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return time for flash attention kernel."""
        #FIXME not finished
        #TODO gqa support
        batch_size = self.microB
        seq_len = self.seq_len
        hidden_dim = self.hidden_dim
        num_heads = self.num_heads
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        d = hidden_dim // num_heads #gemm shape for one head is (seq_len, d) x (d, seq_len)
        Bc = self.attention_tile_size
        Br = min(Bc, d)
        Tr = math.ceil(shard_seq / Br) #number of row tiles
        Tc =  math.ceil(shard_seq / Bc) #number of column tiles
        print(f"Flash Attention: Br={Br}, Tr={Tr}, d={d}, Bc={Bc}, Tc={Tc}")
        #TODO load 3 tiled gemm inputs and 1 output from HBM to SRAM
        
        #TODO disable HBM access
        attention_forward_reduction_time = 0
        attention_size_f = 0
        
        attn_score_time = self.getGEMMTime(Br, d, Bc, "attention_score_f")[0] * Tc * Tr #attention score gemm time for one head
        number_flops =  Br * Bc * d * Tc * Tr * batch_size * num_heads
        print(f"Flash Attention: attention score gemm flops={number_flops/1e9} GFLOPS")
         
        # Softmax time
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  # exponentiation + subtract max + dropout
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        attn_output_time = self.getGEMMTime(Br, Bc, d, "attention_output")[0] * Tc * Tr #attention output gemm time for one head
        attn_score_time *= batch_size * num_heads / max(1, self.tp)
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        attn_output_time *= batch_size * num_heads / max(1, self.tp)
        
        attention_forward_gemm_time = attn_score_time + attn_scale_softmax_time + attn_output_time
        
        attention_forward_time = attention_forward_gemm_time + attention_forward_reduction_time
        attention_size_f = 0
        
    
        return attention_forward_time, attention_forward_gemm_time, attention_forward_reduction_time, attention_size_f
        
        
    
    @staticmethod
    def _normalize_gemm_type(gemm_type: Optional[GemmType]) -> Optional[GemmType]:
        if gemm_type is None or isinstance(gemm_type, GemmType):
            return gemm_type
        raise TypeError(f"Unsupported gemm type specifier: {gemm_type!r}")
    
    def parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_forward(gemm, name, gemm_type)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        
    def parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_backward(gemm, name, gemm_type)

        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_backward(gemm, name, gemm_type)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
    def single_gpu_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.getGEMMTime(m, k, n, name)[0] * batch
            number_flops = m * n * k * batch
            print(f"Single GPU Attention GEMM: {name} flops={number_flops/1e9} GFLOPS")
        else :
            gemm_time = self.getGEMMTime(m, k, n, name)[0]
        return gemm_time, 0, 0
    def single_gpu_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            grad_time_act = self.getGEMMTime(m, k, n, name)[0] * batch
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0] * batch
            gemm_time = grad_time_act + grad_time_wt
        else :
            grad_time_act = self.getGEMMTime(m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0]
            gemm_time = grad_time_act + grad_time_wt
        return gemm_time, 0, 0
        
    def _tensor_context_hybrid_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        # size_bytes = 0
        #attention gemm qkv (outputproj ffn2) (ffn1) (linear_softmax)
        participants = 0
        total_bytes = 0
        reduction_time = 0

        shard_m = math.ceil(m / max(1, self.cp))
        shard_k = math.ceil(k / max(1, self.tp))
        shard_n = math.ceil(n / max(1, self.tp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid forward GEMM")
        
        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type == GemmType.QKV:  # column wise

            gemm_time = self.getGEMMTime(shard_m, k, shard_n, name)[0]
            total_bytes = self.get_kv_size_bytes() / self.tp
            kind = "all_gather"
            participants = self.cp
        elif gemm_type == GemmType.OUT_PROJ:
            gemm_time = self.getGEMMTime(shard_m, shard_k, n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * n)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.FFN2:  # row wise
            gemm_time = self.getGEMMTime(shard_m, shard_k, n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * n)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.FFN1:  # column wise
            gemm_time = self.getGEMMTime(shard_m, k, shard_n, name)[0]
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            gemm_time = self.getGEMMTime(shard_m, shard_k, n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * n)
            kind = "all_gather"
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_context_hybrid_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        # size_bytes = 0
        participants = 0
        total_bytes = 0
        reduction_time = 0

        shard_m = math.ceil(m / max(1, self.cp))
        shard_k = math.ceil(k / max(1, self.tp))
        shard_n = math.ceil(n / max(1, self.tp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid backward GEMM")
        
        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0] * batch / max(1, self.tp)
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0] * batch / max(1, self.tp)
            total_bytes = self.precision * k * n * batch * 2 / self.tp # weight gradient of K V need to be reduce scattered  *2 account for both attn key and value
            kind = "reduce_scatter"
            participants = self.cp
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0] * batch / max(1, self.tp)
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type == GemmType.QKV:  # column wise
            grad_time_act = self.getGEMMTime(shard_m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, shard_n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * k)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
            total_bytes = self.get_kv_size_bytes() / self.tp
            kind = "all_gather"
            participants = self.cp
        elif gemm_type == GemmType.FFN2:  # row wise
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
        elif gemm_type == GemmType.FFN1:  # column wise
            
            grad_time_act = self.getGEMMTime(shard_m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, shard_n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * k)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
            total_bytes = math.ceil(self.precision * shard_m * shard_k) * self.cp * self.tp # in tp-cp hybrid parallelism, the linear softmax weight is sharded by both tp and cp
            kind = "all_gather"
            participants = self.cp * self.tp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        communication happens after out projection and ffn2 gemm
        """
        tp_mode = self.get_parallelism_mode()
        comm_kind_fwd = "all_reduce" if tp_mode == ParallelismMode.TENSOR else "reduce_scatter"
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        size_bytes = 0
        total_bytes = 0
        reduction_time = 0
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel forward GEMM")
        
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.getGEMMTime(m, k, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            shard_n = math.ceil(n / max(1, self.tp))
            gemm_time = self.getGEMMTime(m, k, shard_n, name)[0]
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            shard_k = math.ceil(k / max(1, self.tp))
            gemm_time = self.getGEMMTime(m, shard_k, n, name)[0]
            size_bytes = math.ceil(self.precision * m * n)
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            shard_k = math.ceil(k / max(1, self.tp * self.cp))
            gemm_time = self.getGEMMTime(m, shard_k, n, name)[0]
            size_bytes = math.ceil(self.precision * m * n)
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
            
        if size_bytes > 0:
            total_bytes = size_bytes #FIXME: we already has the totol bytes for all reduce not bytes per rank
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_fwd, name=name)


        return gemm_time, reduction_time, total_bytes
    
    def _tensor_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        # gemm_type:"row", "column" determines the way gemm is distributed
        # comm_after: whether there is communication after gemm, for example, in attention output projection and ffn2
        """
        in backward pass, for each gemm, we need to calculate the time for gradient w.r.t activation and weight
        for attention gemm, we can simply multiply the time of one gemm by batch size and divide by tp
        the all reduce happens after the qkv projection and ffn1 gemm 
        reduction size is the size of activation
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        seq_degree = self._sequence_parallel_degree()
        act_bytes = 0
        total_bytes = 0
        comm_kind_bwd = "all_reduce" if seq_degree == 1 else "reduce_scatter"
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel backward GEMM")

        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            grad_time_act = self.getGEMMTime(m, n, k, name)[0] * batch / max(1, self.tp)
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            shard_n = math.ceil(n / max(1, self.tp))
            grad_time_act = self.getGEMMTime(m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, m, shard_n, name)[0]
            act_bytes = math.ceil(self.precision * m * k)
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            shard_k = math.ceil(k / max(1, self.tp))
            grad_time_act = self.getGEMMTime(m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, m, n, name)[0]
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            shard_k = math.ceil(k / max(1, self.tp * self.cp))
            grad_time_act = self.getGEMMTime(m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, m, n, name)[0]
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        reduction_time = 0
        if act_bytes > 0:
            total_bytes = act_bytes #total bytes for all reduce
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_bwd, name=name)


        return gemm_time, reduction_time, total_bytes
    def _context_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:

        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        shard_m = math.ceil(m / max(1, self.cp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel forward GEMM")
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0] * batch 
        elif gemm_type == GemmType.QKV:  # qkv gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0]
            total_bytes = self.get_kv_size_bytes()
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN1, GemmType.FFN2):
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0]
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        if gemm_type == GemmType.QKV:
            kind = "all_gather" #FIXME gathering Values can be overlapped with attention gemm
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label=name or "comm",
            )

        return gemm_time, reduction_time, total_bytes

    def _context_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        """
        assuming that in backward pass, the K V need to be gathered again for reducing activation memory
        to apply weight gradient, the gradient for K and V need to be reduce-scattered
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        shard_m = math.ceil(m / max(1, self.cp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel backward GEMM")
        if gemm_type == GemmType.ATTENTION_SCORE:
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0] * batch 
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0] * batch 
            total_bytes = self.precision * k * n * batch * 2 # account for both Q and K
            kind = "reduce_scatter"
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0] * batch 
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0] * batch
        elif gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.FFN2):
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0]
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0]
            total_bytes = self.get_kv_size_bytes()
            kind = "all_gather"
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label=name or "comm",
            )
        return gemm_time, reduction_time, total_bytes
    

                
    def get_embedding_f(self):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_f",
            mem_level=self.num_levels - 1,
        ) + self.O
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        return embedding_time + embedding_transfer_time

    def get_linear_softmax_f(self, gemm):
        """Estimate time for final projection + softmax forward."""
        _, effective_m, k, n = self._effective_dims(gemm)

        gemm_time, reduction_time, size_bytes = self._tensor_parallelism_gemm_forward(gemm, "linear_softmax_f", gemm_type=GemmType.LINEAR_SOFTMAX)

            

        point_flop = effective_m * (3 * n - 1)
        point_mem = self.precision * effective_m * (SOFTMAX_FORWARD_MEM_ACCESSES * n)
        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-f",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (f) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (f) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time
    
    def get_linear_softmax_b(self, gemm):


        _, effective_m, k, n = self._effective_dims(gemm)

        gemm_time, reduction_time, size_bytes = self._tensor_parallelism_gemm_backward(gemm, "linear_softmax_b", gemm_type=GemmType.LINEAR_SOFTMAX)

        point_flop = effective_m * n * 6
        point_mem = self.precision * effective_m * n * 10

        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (b) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (b) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time
    def get_scale_softmax_f(self, gemm):
        """
        Process blocks of keys/values at a time (tiling).
        For each block, apply the online softmax trick to update running max & denominator
        each block after QK^T results in a s*s 2D matrix
        model each block with a gemm of shape (s,1,s)
        multiple by n_heads and batch size
        the total time should be divided by tensor parallelism degree
        """
        batch, effective_m, k, n = self._expand_gemm_descriptor(gemm)
        shard_m = math.ceil(effective_m / max(1, self.cp))

        time = batch * self.getGEMMTime(shard_m, 1, n, "scale_softmax_f")[0] / self.tp

        return time
    
    def get_scale_softmax_b(self, gemm):
        batch, effective_m, _, n = self._effective_dims(gemm)

        shard_m = math.ceil(effective_m / max(1, self.cp))
        elements = shard_m * n / self.tp
        scale_flop = elements * 3
        scale_mem = self.precision * elements * 3

        # Backward softmax uses forward probabilities and gradient accumulation (≈6 ops/elt)
        softmax_flop = effective_m * n * 6
        softmax_mem = self.precision * effective_m * n * 10

        scale_time = self.roofline(
            scale_flop,
            scale_mem,
            name="pointwise-scale-b",
            mem_level=self.num_levels - 1,
        ) + self.O
        softmax_time = self.roofline(
            softmax_flop,
            softmax_mem,
            name="pointwise-softmax-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Scale (b) flop: {:,}, mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("Scale (b) time: {:,}".format(scale_time))
            print(
                "Softmax (b) flop: {:,}, mem: {:,}".format(
                    int(softmax_flop / 1e9), int(softmax_mem / 1e9)
                )
            )
        print("Softmax (b) time: {:,}\n".format(softmax_time))

        return scale_time + softmax_time
    
    def _compute_gemm_flops_bytes(self, gemm_shape: Tuple[int, ...], dtype_size: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Return (forward_flops, backward_flops, input_bytes, output_bytes) for ``gemm_shape``."""
        if dtype_size is None:
            dtype_size = self.dtype_size
        if not gemm_shape:
            return 0, 0, 0, 0
        if len(gemm_shape) == 3:
            M, K, N = gemm_shape
            batch = 1
        elif len(gemm_shape) == 4:
            batch, M, K, N = gemm_shape
        else:
            raise ValueError(f"Unexpected GEMM shape dimensions: {len(gemm_shape)}")

        forward_flops = 2 * batch * M * K * N
        backward_flops = 2 * batch * (2 * M * N * K)  # grad_act + grad_wt
        input_bytes = batch * (M * K + K * N) * dtype_size
        output_bytes = batch * M * N * dtype_size
        return forward_flops, backward_flops, input_bytes, output_bytes
        
    def get_residual_f(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements = batch * m * n

        flops = 2 * elements  # add + bias
        mem = self.precision * elements * 3  # read main, read residual, write out
        time = self.roofline(flops, mem, name="pointwise-residual-f", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (f) time: {:,}\n".format(time))

        return time

    def get_residual_b(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements = batch * m * n

        flops = elements  # dL/dx = dL/dy passthrough
        mem = self.precision * elements * 3  # read grad, read forward residual, write grad
        time = self.roofline(flops, mem, name="pointwise-residual-b", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (b) time: {:,}\n".format(time))

        return time

    def get_gelu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision * elements * hidden * 2  # read, write

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-f", mem_level=self.num_levels - 1) + 2 * self.O

        if self.debug:
            print(
                "GELU (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (f) time: {:,}\n".format(time))

        return time
    def get_gelu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision * elements * hidden * 3  # read grad, read forward, write grad

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-b", mem_level=self.num_levels - 1) + 3 * self.O

        if self.debug:
            print(
                "GELU (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (b) time: {:,}\n".format(time))
        return time

    def get_swiglu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT + gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        writes = gate_hidden  # SwiGLU output
        mem_bytes = self.precision * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-f", mem_level=self.num_levels - 1) + 2 * self.O
        if self.debug:
            print(
                "SwiGLU (f) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (f) time: {:,}\n".format(time))
        return time

    def get_swiglu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT + 2 * gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        reads += gate_hidden  # upstream gradient
        writes = 2 * gate_hidden  # gradients for gate and up projections
        mem_bytes = self.precision * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-b", mem_level=self.num_levels - 1) + 3 * self.O
        if self.debug:
            print(
                "SwiGLU (b) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (b) time: {:,}\n".format(time))
        return time

    def get_layernorm_f(self, batch, seq_len, d_model, comm_after=False):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        elif tp_mode == ParallelismMode.TENSOR_SEQUENCE:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        else:
            elements = batch * seq_len * d_model
        compute_flops = elements * LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision * elements * LAYER_NORM_FORWARD_MEM_ACCESSES
        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-f",
            mem_level=self.num_levels - 1,
        ) + 3 * self.O
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):  # all-gather after layernorm
            per_rank_bytes = self.precision * elements
            total_bytes = int(math.ceil(per_rank_bytes * self.tp))
            reduction_time = self.network_model.collective(
                kind="all_gather",
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label="layernorm_f_all_gather",
            )
        else:
            reduction_time = 0.0
            total_bytes = 0

        return compute_time, reduction_time, total_bytes
    
    

    def get_layernorm_b(self, batch, seq_len, d_model, comm_after=False):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        else:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision * elements * LAYER_NORM_BACKWARD_MEM_ACCESSES

        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):  # communication after layernorm
            per_rank_bytes = self.precision * elements
            total_bytes = int(math.ceil(per_rank_bytes * self.tp))
            reduction_time = self.network_model.collective(
                kind="all_gather",
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label="layernorm_b_all_gather",
            )

        else:
            reduction_time = 0.0
            total_bytes = 0


        return compute_time, reduction_time, total_bytes

    
    def get_embedding_b(self):
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_mem_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_b",
            mem_level=self.num_levels - 1,
        ) + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    

    def get_inter_layer_comm_latency_llm(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.lp > 1:
            w_size = self.precision * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.IBL + self.LLL
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer", mem_level=self.num_levels - 1)
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size

    def get_data_parallel_reduction_sizes(self, d, ffn_dim):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0
            }

        # Calculate sizes only (no timing)
        qkv_size = math.ceil(self.precision * d * 3 * d)
        output_size = math.ceil(self.precision * d * d)
        ffn1_dim = self._ffn1_output_dim(ffn_dim)
        ffn1_size = math.ceil(self.precision * ffn1_dim * d)
        ffn2_size = math.ceil(self.precision * ffn_dim * d)
        total_size = qkv_size + output_size + ffn1_size + ffn2_size  # FFN appears twice

        return {
            'qkv_size': qkv_size,
            'output_size': output_size,
            'ffn_size': ffn1_size + ffn2_size,
            'total_size': total_size
        }

    def get_data_parallel_local_computation(self, d, ffn_dim):
        """Calculate local computation times for apply_grad operations."""
        qkv_local = self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        output_local = self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
        ffn1_dim = self._ffn1_output_dim(ffn_dim)
        ffn_local = (
            self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")
            + self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn2 reduction")
        )

        return {
            'qkv_local': qkv_local,
            'output_local': output_local,
            'ffn_local': ffn_local,
            'total_local': qkv_local + output_local + ffn_local
        }

    def get_data_parallel_reduction_llm(self, d, ffn_dim): 
        # If no data parallelism, still apply gradients locally but no cross-device reduction
        if not getattr(self, "dp", 1) or self.dp <= 1:
            apply_grad_time = 0.0
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
            ffn1_dim = self._ffn1_output_dim(ffn_dim)
            apply_grad_time += self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")
            apply_grad_time += self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn2 reduction")
            if self.debug:
                print(f"(dp=1) apply_grad_time: {apply_grad_time}")
            return apply_grad_time



        reduction_time = 0.0
        apply_grad_time = 0.0

        total_bytes = math.ceil(self.precision * d * 3 * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="qkv_proj reduction",
        )
        apply_grad_time += self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")

        total_bytes = math.ceil(self.precision * d * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="output_proj reduction",
        )
        apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")

        ffn1_dim = self._ffn1_output_dim(ffn_dim)
        total_bytes = math.ceil(self.precision * ffn1_dim * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="ffn1 reduction",
        )
        apply_grad_time += self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")

        total_bytes = math.ceil(self.precision * ffn_dim * d)
        reduction_time += self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="ffn2 reduction",
        )
        apply_grad_time += self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn2 reduction")

        if self.debug:
            print(f"apply_grad_time: {apply_grad_time}")

        return reduction_time + apply_grad_time
    
    
    # TODO TODO:
    # we need a significant refactor here. The comm sizes are ingested in a weird way and never used. Instead we use old precomputed sizes.
    # FIX at some point!
    def compute_all_gemm_and_node_times(self, batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, ffn_dim):
        """Compute latency and metadata for all transformer sub-operations."""

        annotate = self.annotate_compute
        dtype_size = self.dtype_size

        def _time(val: float) -> float:
            return val if annotate else 0.0

        gemm_shapes = LLM_util.process_gemm_shapes(
            self, batch_size, seq_len, hidden_dim, num_heads, kv_heads, ffn_dim, vocab_size
        )
        if self.debug:
            print(
                "generating gemm shapes for transformer batch size:",
                batch_size,
                "seq_len:",
                seq_len,
                "hidden_dim:",
                hidden_dim,
                "num_heads:",
                num_heads,
                "kv_heads:",
                kv_heads,
                "ffn_dim:",
                ffn_dim,
            )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        gemm_linear = gemm_shapes["linear"]

        gemm_results: Dict[str, Dict[str, Any]] = {}
        transformer_results: Dict[str, Dict[str, Any]] = {}

        # QKV projection
        qkv_proj_gemm_f, qkv_proj_reduction_f, qkv_proj_size_f = self.parallelism_gemm_forward(
            gemm_qkv_proj, "qkv_projection_f", gemm_type=GemmType.QKV
        )
        qkv_proj_gemm_b, qkv_proj_reduction_b, qkv_proj_size_b = self.parallelism_gemm_backward(
            gemm_qkv_proj, "qkv_projection_b", gemm_type=GemmType.QKV
        )
        qkv_proj_f_raw = qkv_proj_gemm_f + qkv_proj_reduction_f
        qkv_proj_b_raw = qkv_proj_gemm_b + qkv_proj_reduction_b
        qkv_f_flops, qkv_b_flops, qkv_in_bytes, qkv_out_bytes = self._compute_gemm_flops_bytes(gemm_qkv_proj)

        gemm_results['qkv_proj'] = {
            'forward': qkv_proj_f_raw,
            'backward': qkv_proj_b_raw,
            'forward_gemm': qkv_proj_gemm_f,
            'forward_reduction': qkv_proj_reduction_f,
            'backward_gemm': qkv_proj_gemm_b,
            'backward_reduction': qkv_proj_reduction_b,
            'flops': qkv_f_flops,
            'flops_backward': qkv_b_flops,
            'input_bytes': qkv_in_bytes,
            'output_bytes': qkv_out_bytes,
            'input_bytes_backward': qkv_in_bytes,
            'output_bytes_backward': qkv_out_bytes,
            'comm_size_forward': qkv_proj_size_f,
            'comm_size_backward': qkv_proj_size_b,
        }
        transformer_results['qkv_proj'] = {
            'forward': _time(qkv_proj_f_raw),
            'backward': _time(qkv_proj_b_raw),
            'forward_gemm': _time(qkv_proj_gemm_f),
            'forward_reduction': _time(qkv_proj_reduction_f),
            'backward_gemm': _time(qkv_proj_gemm_b),
            'backward_reduction': _time(qkv_proj_reduction_b),
            'comm_size_forward': qkv_proj_size_f,
            'comm_size_backward': qkv_proj_size_b,
            'flops': qkv_f_flops,
            'flops_backward': qkv_b_flops,
            'input_bytes': qkv_in_bytes,
            'output_bytes': qkv_out_bytes,
            'input_bytes_backward': qkv_in_bytes,
            'output_bytes_backward': qkv_out_bytes,
        }

        attention_size_f_raw = 0
        attention_size_b_raw = 0
        attention_reduction_f_raw = 0.0
        attention_reduction_b_raw = 0.0
        attention_gemm_f_raw = 0.0
        attention_gemm_b_raw = 0.0
        attention_f_raw = 0.0
        attention_b_raw = 0.0

        if not self.flash_attention:
            attn_score_gemm_f, attn_score_reduction_f, attn_score_size_f = self.parallelism_gemm_forward(
                gemm_attention_score, "attention_score_f", gemm_type=GemmType.ATTENTION_SCORE
            )
            attn_score_gemm_b, attn_score_reduction_b, attn_score_size_b = self.parallelism_gemm_backward(
                gemm_attention_score, "attention_score_b", gemm_type=GemmType.ATTENTION_SCORE
            )
            attention_score_f_raw = attn_score_gemm_f + attn_score_reduction_f
            attention_score_b_raw = attn_score_gemm_b + attn_score_reduction_b
            attn_score_f_flops, attn_score_b_flops, attn_score_in_bytes, attn_score_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_score)

            gemm_results['attention_score'] = {
                'forward': attention_score_f_raw,
                'backward': attention_score_b_raw,
                'forward_gemm': attn_score_gemm_f,
                'forward_reduction': attn_score_reduction_f,
                'backward_gemm': attn_score_gemm_b,
                'backward_reduction': attn_score_reduction_b,
                'flops': attn_score_f_flops,
                'flops_backward': attn_score_b_flops,
                'input_bytes': attn_score_in_bytes,
                'output_bytes': attn_score_out_bytes,
                'input_bytes_backward': attn_score_in_bytes,
                'output_bytes_backward': attn_score_out_bytes,
                'comm_size_forward': attn_score_size_f,
                'comm_size_backward': attn_score_size_b,
            }
            transformer_results['attention_score'] = {
                'forward': _time(attention_score_f_raw),
                'backward': _time(attention_score_b_raw),
                'forward_gemm': _time(attn_score_gemm_f),
                'forward_reduction': _time(attn_score_reduction_f),
                'backward_gemm': _time(attn_score_gemm_b),
                'backward_reduction': _time(attn_score_reduction_b),
                'comm_size_forward': attn_score_size_f,
                'comm_size_backward': attn_score_size_b,
                'flops': attn_score_f_flops,
                'flops_backward': attn_score_b_flops,
                'input_bytes': attn_score_in_bytes,
                'output_bytes': attn_score_out_bytes,
                'input_bytes_backward': attn_score_in_bytes,
                'output_bytes_backward': attn_score_out_bytes,
            }

            attn_out_gemm_f, attn_out_reduction_f, attn_out_size_f = self.parallelism_gemm_forward(
                gemm_attention_output, "attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT
            )
            attn_out_gemm_b, attn_out_reduction_b, attn_out_size_b = self.parallelism_gemm_backward(
                gemm_attention_output, "attention_output_b", gemm_type=GemmType.ATTENTION_OUTPUT
            )
            attention_output_f_raw = attn_out_gemm_f + attn_out_reduction_f
            attention_output_b_raw = attn_out_gemm_b + attn_out_reduction_b
            attn_out_f_flops, attn_out_b_flops, attn_out_in_bytes, attn_out_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_output)

            gemm_results['attention_output'] = {
                'forward': attention_output_f_raw,
                'backward': attention_output_b_raw,
                'forward_gemm': attn_out_gemm_f,
                'forward_reduction': attn_out_reduction_f,
                'backward_gemm': attn_out_gemm_b,
                'backward_reduction': attn_out_reduction_b,
                'flops': attn_out_f_flops,
                'flops_backward': attn_out_b_flops,
                'input_bytes': attn_out_in_bytes,
                'output_bytes': attn_out_out_bytes,
                'input_bytes_backward': attn_out_in_bytes,
                'output_bytes_backward': attn_out_out_bytes,
                'comm_size_forward': attn_out_size_f,
                'comm_size_backward': attn_out_size_b,
            }
            transformer_results['attention_output'] = {
                'forward': _time(attention_output_f_raw),
                'backward': _time(attention_output_b_raw),
                'forward_gemm': _time(attn_out_gemm_f),
                'forward_reduction': _time(attn_out_reduction_f),
                'backward_gemm': _time(attn_out_gemm_b),
                'backward_reduction': _time(attn_out_reduction_b),
                'comm_size_forward': attn_out_size_f,
                'comm_size_backward': attn_out_size_b,
                'flops': attn_out_f_flops,
                'flops_backward': attn_out_b_flops,
                'input_bytes': attn_out_in_bytes,
                'output_bytes': attn_out_out_bytes,
                'input_bytes_backward': attn_out_in_bytes,
                'output_bytes_backward': attn_out_out_bytes,
            }

            attention_scale_softmax_f_raw = self.get_scale_softmax_f(gemm_attention_score)
            attention_scale_softmax_b_raw = self.get_scale_softmax_b(gemm_attention_score)
            softmax_elems = batch_size * num_heads * seq_len * seq_len
            softmax_f_flops = 6 * softmax_elems
            softmax_b_flops = 7 * softmax_elems
            softmax_f_bytes_total = dtype_size * softmax_elems * 11
            softmax_b_bytes_total = dtype_size * softmax_elems * 13
            gemm_results['attention_scale_softmax'] = {
                'forward': attention_scale_softmax_f_raw,
                'backward': attention_scale_softmax_b_raw,
                'flops': softmax_f_flops,
                'flops_backward': softmax_b_flops,
                'input_bytes': softmax_f_bytes_total // 2,
                'output_bytes': softmax_f_bytes_total // 2,
                'input_bytes_backward': softmax_b_bytes_total // 2,
                'output_bytes_backward': softmax_b_bytes_total // 2,
                'comm_size_forward': 0,
                'comm_size_backward': 0,
            }
            transformer_results['attention_scale_softmax'] = {
                'forward': _time(attention_scale_softmax_f_raw),
                'backward': _time(attention_scale_softmax_b_raw),
                'forward_gemm': _time(attention_scale_softmax_f_raw),
                'forward_reduction': 0.0,
                'backward_gemm': _time(attention_scale_softmax_b_raw),
                'backward_reduction': 0.0,
                'comm_size_forward': 0,
                'comm_size_backward': 0,
                'flops': softmax_f_flops,
                'flops_backward': softmax_b_flops,
                'input_bytes': softmax_f_bytes_total // 2,
                'output_bytes': softmax_f_bytes_total // 2,
                'input_bytes_backward': softmax_b_bytes_total // 2,
                'output_bytes_backward': softmax_b_bytes_total // 2,
            }

            attention_f_raw = attention_score_f_raw + attention_scale_softmax_f_raw + attention_output_f_raw
            attention_b_raw = attention_score_b_raw + attention_scale_softmax_b_raw + attention_output_b_raw
            attention_gemm_f_raw = attn_score_gemm_f + attn_out_gemm_f + attention_scale_softmax_f_raw
            attention_gemm_b_raw = attn_score_gemm_b + attn_out_gemm_b + attention_scale_softmax_b_raw
            attention_reduction_f_raw = attn_score_reduction_f + attn_out_reduction_f
            attention_reduction_b_raw = attn_score_reduction_b + attn_out_reduction_b
            attention_size_f_raw = attn_score_size_f + attn_out_size_f
            attention_size_b_raw = attn_score_size_b + attn_out_size_b

            attention_flops = (
                attn_score_f_flops + attn_out_f_flops + softmax_f_flops
            )
            attention_flops_bwd = (
                attn_score_b_flops + attn_out_b_flops + softmax_b_flops
            )
            attention_bytes_f = (
                attn_score_in_bytes + attn_score_out_bytes +
                attn_out_in_bytes + attn_out_out_bytes +
                softmax_f_bytes_total
            )
            attention_bytes_b = (
                attn_score_in_bytes + attn_score_out_bytes +
                attn_out_in_bytes + attn_out_out_bytes +
                softmax_b_bytes_total
            )
        else:
            attention_f_raw, attention_gemm_f_raw, attention_reduction_f_raw, attention_size_f_raw = self.flash_attention_kernel_forward(
                attn_score_gemm=gemm_attention_score, attn_out_gemm=gemm_attention_output
            )
            attention_b_raw = 2 * attention_f_raw
            attention_gemm_b_raw = 2 * attention_gemm_f_raw
            attention_reduction_b_raw = 2 * attention_reduction_f_raw
            attention_size_b_raw = 2 * attention_size_f_raw
            # FLOP/byte estimates for flash attention (approximate)
            attn_score_f_flops, attn_score_b_flops, attn_score_in_bytes, attn_score_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_score)
            attn_out_f_flops, attn_out_b_flops, attn_out_in_bytes, attn_out_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_output)
            attention_flops = attn_score_f_flops + attn_out_f_flops
            attention_flops_bwd = attn_score_b_flops + attn_out_b_flops
            attention_bytes_f = attn_score_in_bytes + attn_score_out_bytes + attn_out_in_bytes + attn_out_out_bytes
            attention_bytes_b = attention_bytes_f * 2

        transformer_results['attention'] = {
            'forward': _time(attention_f_raw),
            'backward': _time(attention_b_raw),
            'forward_gemm': _time(attention_gemm_f_raw),
            'forward_reduction': _time(attention_reduction_f_raw),
            'backward_gemm': _time(attention_gemm_b_raw),
            'backward_reduction': _time(attention_reduction_b_raw),
            'comm_size_forward': attention_size_f_raw,
            'comm_size_backward': attention_size_b_raw,
            'flops': attention_flops,
            'flops_backward': attention_flops_bwd,
            'input_bytes': attention_bytes_f,
            'output_bytes': 0,
            'input_bytes_backward': attention_bytes_b,
            'output_bytes_backward': 0,
        }

        # Output projection
        out_proj_gemm_f, out_proj_reduction_f, out_proj_size_f = self.parallelism_gemm_forward(
            gemm_output_proj, "output_projection_f", gemm_type=GemmType.OUT_PROJ
        )
        out_proj_gemm_b, out_proj_reduction_b, out_proj_size_b = self.parallelism_gemm_backward(
            gemm_output_proj, "output_projection_b", gemm_type=GemmType.OUT_PROJ
        )
        out_proj_f_raw = out_proj_gemm_f + out_proj_reduction_f
        out_proj_b_raw = out_proj_gemm_b + out_proj_reduction_b
        out_proj_f_flops, out_proj_b_flops, out_proj_in_bytes, out_proj_out_bytes = self._compute_gemm_flops_bytes(gemm_output_proj)

        gemm_results['output_proj'] = {
            'forward': out_proj_f_raw,
            'backward': out_proj_b_raw,
            'forward_gemm': out_proj_gemm_f,
            'forward_reduction': out_proj_reduction_f,
            'backward_gemm': out_proj_gemm_b,
            'backward_reduction': out_proj_reduction_b,
            'flops': out_proj_f_flops,
            'flops_backward': out_proj_b_flops,
            'input_bytes': out_proj_in_bytes,
            'output_bytes': out_proj_out_bytes,
            'input_bytes_backward': out_proj_in_bytes,
            'output_bytes_backward': out_proj_out_bytes,
            'comm_size_forward': out_proj_size_f,
            'comm_size_backward': out_proj_size_b,
        }
        transformer_results['output_proj'] = {
            'forward': _time(out_proj_f_raw),
            'backward': _time(out_proj_b_raw),
            'forward_gemm': _time(out_proj_gemm_f),
            'forward_reduction': _time(out_proj_reduction_f),
            'backward_gemm': _time(out_proj_gemm_b),
            'backward_reduction': _time(out_proj_reduction_b),
            'comm_size_forward': out_proj_size_f,
            'comm_size_backward': out_proj_size_b,
            'flops': out_proj_f_flops,
            'flops_backward': out_proj_b_flops,
            'input_bytes': out_proj_in_bytes,
            'output_bytes': out_proj_out_bytes,
            'input_bytes_backward': out_proj_in_bytes,
            'output_bytes_backward': out_proj_out_bytes,
        }

        # FFN layers
        ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f = self.parallelism_gemm_forward(
            gemm_ffn1, "ffn_f", gemm_type=GemmType.FFN1
        )
        ffn1_gemm_b, ffn1_reduction_b, ffn1_size_b = self.parallelism_gemm_backward(
            gemm_ffn1, "ffn_b", gemm_type=GemmType.FFN1
        )
        ffn1_f_raw = ffn1_gemm_f + ffn1_reduction_f
        ffn1_b_raw = ffn1_gemm_b + ffn1_reduction_b
        ffn1_f_flops, ffn1_b_flops, ffn1_in_bytes, ffn1_out_bytes = self._compute_gemm_flops_bytes(gemm_ffn1)

        gemm_results['ffn1'] = {
            'forward': ffn1_f_raw,
            'backward': ffn1_b_raw,
            'forward_gemm': ffn1_gemm_f,
            'forward_reduction': ffn1_reduction_f,
            'backward_gemm': ffn1_gemm_b,
            'backward_reduction': ffn1_reduction_b,
            'flops': ffn1_f_flops,
            'flops_backward': ffn1_b_flops,
            'input_bytes': ffn1_in_bytes,
            'output_bytes': ffn1_out_bytes,
            'input_bytes_backward': ffn1_in_bytes,
            'output_bytes_backward': ffn1_out_bytes,
            'comm_size_forward': ffn1_size_f,
            'comm_size_backward': ffn1_size_b,
        }
        transformer_results['ffn1'] = {
            'forward': _time(ffn1_f_raw),
            'backward': _time(ffn1_b_raw),
            'forward_gemm': _time(ffn1_gemm_f),
            'forward_reduction': _time(ffn1_reduction_f),
            'backward_gemm': _time(ffn1_gemm_b),
            'backward_reduction': _time(ffn1_reduction_b),
            'comm_size_forward': ffn1_size_f,
            'comm_size_backward': ffn1_size_b,
            'flops': ffn1_f_flops,
            'flops_backward': ffn1_b_flops,
            'input_bytes': ffn1_in_bytes,
            'output_bytes': ffn1_out_bytes,
            'input_bytes_backward': ffn1_in_bytes,
            'output_bytes_backward': ffn1_out_bytes,
        }

        ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f = self.parallelism_gemm_forward(
            gemm_ffn2, "ffn2_f", gemm_type=GemmType.FFN2
        )
        ffn2_gemm_b, ffn2_reduction_b, ffn2_size_b = self.parallelism_gemm_backward(
            gemm_ffn2, "ffn2_b", gemm_type=GemmType.FFN2
        )
        ffn2_f_raw = ffn2_gemm_f + ffn2_reduction_f
        ffn2_b_raw = ffn2_gemm_b + ffn2_reduction_b
        ffn2_f_flops, ffn2_b_flops, ffn2_in_bytes, ffn2_out_bytes = self._compute_gemm_flops_bytes(gemm_ffn2)

        gemm_results['ffn2'] = {
            'forward': ffn2_f_raw,
            'backward': ffn2_b_raw,
            'forward_gemm': ffn2_gemm_f,
            'forward_reduction': ffn2_reduction_f,
            'backward_gemm': ffn2_gemm_b,
            'backward_reduction': ffn2_reduction_b,
            'flops': ffn2_f_flops,
            'flops_backward': ffn2_b_flops,
            'input_bytes': ffn2_in_bytes,
            'output_bytes': ffn2_out_bytes,
            'input_bytes_backward': ffn2_in_bytes,
            'output_bytes_backward': ffn2_out_bytes,
            'comm_size_forward': ffn2_size_f,
            'comm_size_backward': ffn2_size_b,
        }
        transformer_results['ffn2'] = {
            'forward': _time(ffn2_f_raw),
            'backward': _time(ffn2_b_raw),
            'forward_gemm': _time(ffn2_gemm_f),
            'forward_reduction': _time(ffn2_reduction_f),
            'backward_gemm': _time(ffn2_gemm_b),
            'backward_reduction': _time(ffn2_reduction_b),
            'comm_size_forward': ffn2_size_f,
            'comm_size_backward': ffn2_size_b,
            'flops': ffn2_f_flops,
            'flops_backward': ffn2_b_flops,
            'input_bytes': ffn2_in_bytes,
            'output_bytes': ffn2_out_bytes,
            'input_bytes_backward': ffn2_in_bytes,
            'output_bytes_backward': ffn2_out_bytes,
        }

        # Embedding metadata
        embedding_f_raw = self.get_embedding_f()
        embedding_b_raw = self.get_embedding_b()
        emb_elems = seq_len * batch_size * hidden_dim
        emb_flops = emb_elems
        emb_flops_bwd = emb_elems
        emb_bytes_total = 2 * seq_len * batch_size * hidden_dim * dtype_size
        gemm_results['embedding'] = {
            'forward': embedding_f_raw,
            'backward': embedding_b_raw,
            'flops': emb_flops,
            'flops_backward': emb_flops_bwd,
            'input_bytes': emb_bytes_total // 2,
            'output_bytes': emb_bytes_total // 2,
            'input_bytes_backward': emb_bytes_total // 2,
            'output_bytes_backward': emb_bytes_total // 2,
            'comm_size_forward': 0,
            'comm_size_backward': 0,
        }
        transformer_results['embedding'] = {
            'forward': _time(embedding_f_raw),
            'backward': _time(embedding_b_raw),
            'flops': emb_flops,
            'flops_backward': emb_flops_bwd,
            'input_bytes': emb_bytes_total // 2,
            'output_bytes': emb_bytes_total // 2,
            'input_bytes_backward': emb_bytes_total // 2,
            'output_bytes_backward': emb_bytes_total // 2,
        }

        # Residual + layernorm metadata (layernorm1/refers output proj)
        residual1_f_raw = self.get_residual_f(gemm_output_proj)
        residual1_b_raw = self.get_residual_b(gemm_output_proj)
        residual_elems = batch_size * seq_len * hidden_dim
        residual_f_flops = 2 * residual_elems
        residual_b_flops = residual_elems
        residual_bytes_total = dtype_size * residual_elems * 3
        gemm_results['residual1'] = {
            'forward': residual1_f_raw,
            'backward': residual1_b_raw,
            'flops': residual_f_flops,
            'flops_backward': residual_b_flops,
            'input_bytes': residual_bytes_total // 2,
            'output_bytes': residual_bytes_total // 2,
            'input_bytes_backward': residual_bytes_total // 2,
            'output_bytes_backward': residual_bytes_total // 2,
            'comm_size_forward': 0,
            'comm_size_backward': 0,
        }

        layernorm1_f_raw, layernorm1_reduction_raw, ln1_comm_bytes_f = self.get_layernorm_f(
            batch=batch_size, seq_len=seq_len, d_model=hidden_dim
        )
        layernorm1_b_raw, layernorm1_reduction_b_raw, ln1_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size, seq_len=seq_len, d_model=hidden_dim
        )
        layernorm_elems = batch_size * seq_len
        layernorm_f_flops = layernorm_elems * hidden_dim * 11
        layernorm_b_flops = layernorm_elems * hidden_dim * 8
        layernorm_f_bytes_total = dtype_size * layernorm_elems * hidden_dim * 10
        layernorm_b_bytes_total = dtype_size * layernorm_elems * hidden_dim * 12
        gemm_results['layernorm1'] = {
            'forward': layernorm1_f_raw,
            'backward': layernorm1_b_raw,
            'flops': layernorm_f_flops,
            'flops_backward': layernorm_b_flops,
            'input_bytes': layernorm_f_bytes_total // 2,
            'output_bytes': layernorm_f_bytes_total // 2,
            'input_bytes_backward': layernorm_b_bytes_total // 2,
            'output_bytes_backward': layernorm_b_bytes_total // 2,
            'comm_size_forward': ln1_comm_bytes_f,
            'comm_size_backward': ln1_comm_bytes_b,
        }

        residual2_f_raw = self.get_residual_f(gemm_ffn2)
        residual2_b_raw = self.get_residual_b(gemm_ffn2)
        gemm_results['residual2'] = {
            'forward': residual2_f_raw,
            'backward': residual2_b_raw,
            'flops': residual_f_flops,
            'flops_backward': residual_b_flops,
            'input_bytes': residual_bytes_total // 2,
            'output_bytes': residual_bytes_total // 2,
            'input_bytes_backward': residual_bytes_total // 2,
            'output_bytes_backward': residual_bytes_total // 2,
            'comm_size_forward': 0,
            'comm_size_backward': 0,
        }

        layernorm2_f_raw, layernorm2_reduction_raw, ln2_comm_bytes_f = self.get_layernorm_f(
            batch=batch_size, seq_len=seq_len, d_model=hidden_dim
        )
        layernorm2_b_raw, layernorm2_reduction_b_raw, ln2_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size, seq_len=seq_len, d_model=hidden_dim
        )
        gemm_results['layernorm2'] = {
            'forward': layernorm2_f_raw,
            'backward': layernorm2_b_raw,
            'flops': layernorm_f_flops,
            'flops_backward': layernorm_b_flops,
            'input_bytes': layernorm_f_bytes_total // 2,
            'output_bytes': layernorm_f_bytes_total // 2,
            'input_bytes_backward': layernorm_b_bytes_total // 2,
            'output_bytes_backward': layernorm_b_bytes_total // 2,
            'comm_size_forward': ln2_comm_bytes_f,
            'comm_size_backward': ln2_comm_bytes_b,
        }

        if self.model_type == "llama":
            act_f_raw = self.get_swiglu_f(tensor_shape=gemm_ffn1)
            act_b_raw = self.get_swiglu_b(tensor_shape=gemm_ffn1)
            act_forward_flops = self.seq_len * batch_size * hidden_dim * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT
            act_backward_flops = self.seq_len * batch_size * hidden_dim * SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT
        else:
            act_f_raw = self.get_gelu_f(tensor_shape=gemm_ffn1)
            act_b_raw = self.get_gelu_b(tensor_shape=gemm_ffn1)
            act_forward_flops = self.seq_len * batch_size * hidden_dim * GELU_FORWARD_FLOPS_PER_ELEMENT
            act_backward_flops = self.seq_len * batch_size * hidden_dim * GELU_BACKWARD_FLOPS_PER_ELEMENT
        act_bytes_total = dtype_size * batch_size * seq_len * hidden_dim * 2
        gemm_results['gelu'] = {
            'forward': act_f_raw,
            'backward': act_b_raw,
            'flops': act_forward_flops,
            'flops_backward': act_backward_flops,
            'input_bytes': act_bytes_total // 2,
            'output_bytes': act_bytes_total // 2,
            'input_bytes_backward': act_bytes_total // 2,
            'output_bytes_backward': act_bytes_total // 2,
            'comm_size_forward': 0,
            'comm_size_backward': 0,
        }

        linear_softmax_f_raw = self.get_linear_softmax_f(gemm=gemm_linear)
        linear_softmax_b_raw = self.get_linear_softmax_b(gemm=gemm_linear)
        linear_softmax_elems = batch_size * seq_len * vocab_size
        linear_softmax_f_flops = 5 * linear_softmax_elems
        linear_softmax_b_flops = 6 * linear_softmax_elems
        linear_softmax_f_bytes_total = dtype_size * linear_softmax_elems * 8
        linear_softmax_b_bytes_total = dtype_size * linear_softmax_elems * 10
        gemm_results['linear_softmax'] = {
            'forward': linear_softmax_f_raw,
            'backward': linear_softmax_b_raw,
            'flops': linear_softmax_f_flops,
            'flops_backward': linear_softmax_b_flops,
            'input_bytes': linear_softmax_f_bytes_total // 2,
            'output_bytes': linear_softmax_f_bytes_total // 2,
            'input_bytes_backward': linear_softmax_b_bytes_total // 2,
            'output_bytes_backward': linear_softmax_b_bytes_total // 2,
            'comm_size_forward': 0,
            'comm_size_backward': 0,
        }
        transformer_results['linear_softmax'] = {
            'forward': _time(linear_softmax_f_raw),
            'backward': _time(linear_softmax_b_raw),
            'flops': linear_softmax_f_flops,
            'flops_backward': linear_softmax_b_flops,
            'input_bytes': linear_softmax_f_bytes_total // 2,
            'output_bytes': linear_softmax_f_bytes_total // 2,
            'input_bytes_backward': linear_softmax_b_bytes_total // 2,
            'output_bytes_backward': linear_softmax_b_bytes_total // 2,
        }

        # Combine per-op metadata
        mha_flops_f = (
            gemm_results['qkv_proj']['flops'] + attention_flops + gemm_results['output_proj']['flops']
        )
        mha_flops_b = (
            gemm_results['qkv_proj']['flops_backward'] + attention_flops_bwd + gemm_results['output_proj']['flops_backward']
        )
        mha_bytes_f = (
            gemm_results['qkv_proj']['input_bytes'] + gemm_results['qkv_proj']['output_bytes'] +
            attention_bytes_f + gemm_results['output_proj']['input_bytes'] + gemm_results['output_proj']['output_bytes']
        )
        mha_bytes_b = (
            gemm_results['qkv_proj']['input_bytes_backward'] + gemm_results['qkv_proj']['output_bytes_backward'] +
            attention_bytes_b + gemm_results['output_proj']['input_bytes_backward'] + gemm_results['output_proj']['output_bytes_backward']
        )

        ffn_time_f_raw = ffn1_f_raw + ffn2_f_raw + act_f_raw
        ffn_time_b_raw = ffn1_b_raw + ffn2_b_raw + act_b_raw
        ffn_flops_f = gemm_results['ffn1']['flops'] + gemm_results['ffn2']['flops'] + gemm_results['gelu']['flops']
        ffn_flops_b = gemm_results['ffn1']['flops_backward'] + gemm_results['ffn2']['flops_backward'] + gemm_results['gelu']['flops_backward']
        ffn_bytes_f = (
            gemm_results['ffn1']['input_bytes'] + gemm_results['ffn1']['output_bytes'] +
            gemm_results['ffn2']['input_bytes'] + gemm_results['ffn2']['output_bytes']
        )
        ffn_bytes_b = (
            gemm_results['ffn1']['input_bytes_backward'] + gemm_results['ffn1']['output_bytes_backward'] +
            gemm_results['ffn2']['input_bytes_backward'] + gemm_results['ffn2']['output_bytes_backward']
        )

        transformer_results['MHA'] = {
            'forward': _time(attention_f_raw + qkv_proj_f_raw + out_proj_f_raw),
            'backward': _time(attention_b_raw + qkv_proj_b_raw + out_proj_b_raw),
            'forward_gemm': _time(attention_gemm_f_raw + qkv_proj_gemm_f + out_proj_gemm_f),
            'forward_reduction': _time(attention_reduction_f_raw + qkv_proj_reduction_f + out_proj_reduction_f),
            'backward_gemm': _time(attention_gemm_b_raw + qkv_proj_gemm_b + out_proj_gemm_b),
            'backward_reduction': _time(attention_reduction_b_raw + qkv_proj_reduction_b + out_proj_reduction_b),
            'comm_size_forward': attention_size_f_raw + qkv_proj_size_f + out_proj_size_f,
            'comm_size_backward': attention_size_b_raw + qkv_proj_size_b + out_proj_size_b,
            'flops': mha_flops_f,
            'flops_backward': mha_flops_b,
            'input_bytes': mha_bytes_f,
            'output_bytes': 0,
            'input_bytes_backward': mha_bytes_b,
            'output_bytes_backward': 0,
        }

        transformer_results['MLP'] = {
            'forward': _time(ffn_time_f_raw),
            'backward': _time(ffn_time_b_raw),
            'forward_gemm': _time(ffn1_gemm_f + ffn2_gemm_f + act_f_raw),
            'forward_reduction': _time(ffn1_reduction_f + ffn2_reduction_f),
            'backward_gemm': _time(ffn1_gemm_b + ffn2_gemm_b + act_b_raw),
            'backward_reduction': _time(ffn1_reduction_b + ffn2_reduction_b),
            'comm_size_forward': ffn1_size_f + ffn2_size_f,
            'comm_size_backward': ffn1_size_b + ffn2_size_b,
            'flops': ffn_flops_f,
            'flops_backward': ffn_flops_b,
            'input_bytes': ffn_bytes_f,
            'output_bytes': 0,
            'input_bytes_backward': ffn_bytes_b,
            'output_bytes_backward': 0,
        }

        layernorm1_forward_total = layernorm1_f_raw + residual1_f_raw + layernorm1_reduction_raw
        layernorm1_backward_total = layernorm1_b_raw + residual1_b_raw + layernorm1_reduction_b_raw
        transformer_results['layernorm1'] = {
            'forward': _time(layernorm1_forward_total),
            'backward': _time(layernorm1_backward_total),
            'forward_compute': _time(layernorm1_f_raw + residual1_f_raw),
            'forward_reduction': _time(layernorm1_reduction_raw),
            'backward_compute': _time(layernorm1_b_raw + residual1_b_raw),
            'backward_reduction': _time(layernorm1_reduction_b_raw),
            'comm_size_forward': ln1_comm_bytes_f,
            'comm_size_backward': ln1_comm_bytes_b,
            'flops': gemm_results['layernorm1']['flops'] + gemm_results['residual1']['flops'],
            'flops_backward': gemm_results['layernorm1']['flops_backward'] + gemm_results['residual1']['flops_backward'],
            'input_bytes': gemm_results['layernorm1']['input_bytes'] + gemm_results['residual1']['input_bytes'],
            'output_bytes': gemm_results['layernorm1']['output_bytes'] + gemm_results['residual1']['output_bytes'],
            'input_bytes_backward': gemm_results['layernorm1']['input_bytes_backward'] + gemm_results['residual1']['input_bytes_backward'],
            'output_bytes_backward': gemm_results['layernorm1']['output_bytes_backward'] + gemm_results['residual1']['output_bytes_backward'],
        }

        layernorm2_forward_total = layernorm2_f_raw + residual2_f_raw + layernorm2_reduction_raw
        layernorm2_backward_total = layernorm2_b_raw + residual2_b_raw + layernorm2_reduction_b_raw
        transformer_results['layernorm2'] = {
            'forward': _time(layernorm2_forward_total),
            'backward': _time(layernorm2_backward_total),
            'forward_compute': _time(layernorm2_f_raw + residual2_f_raw),
            'forward_reduction': _time(layernorm2_reduction_raw),
            'backward_compute': _time(layernorm2_b_raw + residual2_b_raw),
            'backward_reduction': _time(layernorm2_reduction_b_raw),
            'comm_size_forward': ln2_comm_bytes_f,
            'comm_size_backward': ln2_comm_bytes_b,
            'flops': gemm_results['layernorm2']['flops'] + gemm_results['residual2']['flops'],
            'flops_backward': gemm_results['layernorm2']['flops_backward'] + gemm_results['residual2']['flops_backward'],
            'input_bytes': gemm_results['layernorm2']['input_bytes'] + gemm_results['residual2']['input_bytes'],
            'output_bytes': gemm_results['layernorm2']['output_bytes'] + gemm_results['residual2']['output_bytes'],
            'input_bytes_backward': gemm_results['layernorm2']['input_bytes_backward'] + gemm_results['residual2']['input_bytes_backward'],
            'output_bytes_backward': gemm_results['layernorm2']['output_bytes_backward'] + gemm_results['residual2']['output_bytes_backward'],
        }

        transformer_time_f_raw = (
            transformer_results['MHA']['forward'] + transformer_results['MLP']['forward'] +
            transformer_results['layernorm1']['forward'] + transformer_results['layernorm2']['forward']
        ) if annotate else (
            qkv_proj_f_raw + attention_f_raw + out_proj_f_raw + ffn_time_f_raw + layernorm1_forward_total + layernorm2_forward_total
        )
        transformer_time_b_raw = (
            transformer_results['MHA']['backward'] + transformer_results['MLP']['backward'] +
            transformer_results['layernorm1']['backward'] + transformer_results['layernorm2']['backward']
        ) if annotate else (
            qkv_proj_b_raw + attention_b_raw + out_proj_b_raw + ffn_time_b_raw + layernorm1_backward_total + layernorm2_backward_total
        )

        transformer_f_flops = (
            gemm_results['qkv_proj']['flops'] + attention_flops + gemm_results['output_proj']['flops'] +
            gemm_results['ffn1']['flops'] + gemm_results['ffn2']['flops'] + gemm_results['gelu']['flops'] +
            gemm_results['layernorm1']['flops'] + gemm_results['residual1']['flops'] +
            gemm_results['layernorm2']['flops'] + gemm_results['residual2']['flops']
        )
        transformer_b_flops = (
            gemm_results['qkv_proj']['flops_backward'] + attention_flops_bwd + gemm_results['output_proj']['flops_backward'] +
            gemm_results['ffn1']['flops_backward'] + gemm_results['ffn2']['flops_backward'] + gemm_results['gelu']['flops_backward'] +
            gemm_results['layernorm1']['flops_backward'] + gemm_results['residual1']['flops_backward'] +
            gemm_results['layernorm2']['flops_backward'] + gemm_results['residual2']['flops_backward']
        )
        transformer_f_bytes = (
            gemm_results['qkv_proj']['input_bytes'] + gemm_results['qkv_proj']['output_bytes'] + attention_bytes_f +
            gemm_results['output_proj']['input_bytes'] + gemm_results['output_proj']['output_bytes'] +
            gemm_results['ffn1']['input_bytes'] + gemm_results['ffn1']['output_bytes'] +
            gemm_results['ffn2']['input_bytes'] + gemm_results['ffn2']['output_bytes'] +
            gemm_results['gelu']['input_bytes'] + gemm_results['gelu']['output_bytes'] +
            gemm_results['layernorm1']['input_bytes'] + gemm_results['layernorm1']['output_bytes'] +
            gemm_results['residual1']['input_bytes'] + gemm_results['residual1']['output_bytes'] +
            gemm_results['layernorm2']['input_bytes'] + gemm_results['layernorm2']['output_bytes'] +
            gemm_results['residual2']['input_bytes'] + gemm_results['residual2']['output_bytes']
        )
        transformer_b_bytes = (
            gemm_results['qkv_proj']['input_bytes_backward'] + gemm_results['qkv_proj']['output_bytes_backward'] + attention_bytes_b +
            gemm_results['output_proj']['input_bytes_backward'] + gemm_results['output_proj']['output_bytes_backward'] +
            gemm_results['ffn1']['input_bytes_backward'] + gemm_results['ffn1']['output_bytes_backward'] +
            gemm_results['ffn2']['input_bytes_backward'] + gemm_results['ffn2']['output_bytes_backward'] +
            gemm_results['gelu']['input_bytes_backward'] + gemm_results['gelu']['output_bytes_backward'] +
            gemm_results['layernorm1']['input_bytes_backward'] + gemm_results['layernorm1']['output_bytes_backward'] +
            gemm_results['residual1']['input_bytes_backward'] + gemm_results['residual1']['output_bytes_backward'] +
            gemm_results['layernorm2']['input_bytes_backward'] + gemm_results['layernorm2']['output_bytes_backward'] +
            gemm_results['residual2']['input_bytes_backward'] + gemm_results['residual2']['output_bytes_backward']
        )

        transformer_results['transformer'] = {
            'forward': _time(transformer_time_f_raw),
            'backward': _time(transformer_time_b_raw),
            'flops': transformer_f_flops,
            'flops_backward': transformer_b_flops,
            'input_bytes': transformer_f_bytes,
            'output_bytes': 0,
            'input_bytes_backward': transformer_b_bytes,
            'output_bytes_backward': 0,
        }

        node_breakdown = {
            'transformer_time_f': _time(transformer_time_f_raw),
            'transformer_time_b': _time(transformer_time_b_raw),
            'embedding_f': _time(embedding_f_raw),
            'embedding_b': _time(embedding_b_raw),
            'linear_softmax_f': _time(linear_softmax_f_raw),
            'linear_softmax_b': _time(linear_softmax_b_raw),
            'transformer_f_flops': transformer_f_flops,
            'transformer_b_flops': transformer_b_flops,
            'transformer_f_bytes': transformer_f_bytes,
            'transformer_b_bytes': transformer_b_bytes,
            'embedding_f_flops': gemm_results['embedding']['flops'],
            'embedding_b_flops': gemm_results['embedding']['flops_backward'],
            'embedding_f_bytes': gemm_results['embedding']['input_bytes'] + gemm_results['embedding']['output_bytes'],
            'embedding_b_bytes': gemm_results['embedding']['input_bytes_backward'] + gemm_results['embedding']['output_bytes_backward'],
            'linear_softmax_f_flops': linear_softmax_f_flops,
            'linear_softmax_b_flops': linear_softmax_b_flops,
            'linear_softmax_f_bytes': linear_softmax_f_bytes_total,
            'linear_softmax_b_bytes': linear_softmax_b_bytes_total,
            'kv_cache_fetch': 0.0,
            'kv_cache_store': 0.0,
        }

        results_path = os.path.join(self.output_dir, "transformer_results.txt")
        try:
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": transformer_results,
                        "gemm_results": gemm_results,
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )
        except OSError as exc:
            if self.debug:
                print(f"[WARN] Unable to write transformer results to {results_path}: {exc}")

        return transformer_results, node_breakdown
    def _effective_transformer_batch(self) -> int:
        if self.lp > 1:
            return self.microB
        if self.dp > 1:
            return self.miniB
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes: Dict[str, int],
        local_comp: Dict[str, float], 
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': 'pipeline',
                'participants': 2,
                'interconnect_type': 'lp',
                'local_comp_time': 0
            }
        }

    def _populate_transformer_comm_metadata(
        self,
        entry: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
        # gemm_spec: Tuple[int, ...],
        comm_bytes_fwd: int,
        comm_bytes_bwd: int,
    ) -> None:
        """Attach tensor-parallel collectives for a GEMM to metadata and entry."""


        if not self.tp or self.tp <= 1 and self.cp <= 1:
            return



        def add_comm(direction: str, suffix: str, kind: str, size_bytes: float, participants: int, interconnect: str) -> None:
            if participants <= 1:
                return
            bytes_int = int(math.ceil(size_bytes))
            if bytes_int <= 0:
                return
            key = f"{entry['name']}_{direction}_{suffix}"
            # ensure uniqueness when multiple collectives share the same suffix
            unique_key = key
            counter = 1
            while unique_key in metadata:
                counter += 1
                unique_key = f"{key}_{counter}"

            metadata[unique_key] = {
                'size': bytes_int,
                'type': kind,
                'participants': int(participants),
                'interconnect_type': interconnect,
            }
            entry[direction]['comm_keys'].append(unique_key)

        parallelism_mode = self.get_parallelism_mode()
        rules_by_mode = COMMUNICATION_RULES.get(parallelism_mode)
        if not rules_by_mode:
            return

        spec = rules_by_mode.get(entry['name'])
        if spec is None:
            spec = rules_by_mode.get(COMM_RULE_DEFAULT_KEY)
        if not spec:
            return


        participants_lookup = {
            'tp': getattr(self, 'tp', 0),
            'cp': getattr(self, 'cp', 0),
            'dp': getattr(self, 'dp', 0),
            'lp': getattr(self, 'lp', 0),
        }

        for direction in ('forward', 'backward'):
            rule = spec.get(direction)
            if not rule:
                continue
            size_bytes = comm_bytes_fwd if direction == 'forward' else comm_bytes_bwd
            scope = rule.get('participants')
            participants = participants_lookup.get(scope or '', 0)
            if participants <= 1:
                continue
            kind = rule['kind']
            suffix = rule.get('suffix', kind)
            interconnect = rule.get('interconnect', scope or 'tp')
            add_comm(direction, suffix, kind, size_bytes, participants, interconnect)

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'tp': (self.IBTP, self.LLTP),
            'cp': (self.IBTP, self.LLTP),
        }

    def _prepare_execution_graphs(
        self,
        *,
        node_breakdown: Dict[str, float],
        transformer_results: Dict[str, Dict[str, Any]],
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        ffn_dim: int,
        vocab_size: int,
        include_pipeline_backward: bool,
        include_transformer_backward: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None, # optional override, decode only.
    ) -> Tuple[Graph, Any, Optional[Graph], Optional[Any], Optional[Any], Dict[str, Tuple[float, float]]]:
        """Build pipeline/transformer graphs shared across training and inference."""

        if not include_pipeline_backward and not include_transformer_backward:
            # Forward-only inference: skip training all-reduce bookkeeping.
            reduction_sizes = {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0,
            }
            local_comp = {
                'qkv_local': 0.0,
                'output_local': 0.0,
                'ffn_local': 0.0,
                'total_local': 0.0,
            }
        else:
            reduction_sizes = self.get_data_parallel_reduction_sizes(hidden_dim, ffn_dim)
            local_comp = self.get_data_parallel_local_computation(hidden_dim, ffn_dim)

        embedding_size = math.ceil(self.precision * vocab_size * hidden_dim) + math.ceil(self.precision * seq_len * hidden_dim)
        softmax_size = math.ceil(self.precision * hidden_dim * vocab_size)
        # Below, we fix pipeline comm sizes for decode
        attn_shape = (gemm_shapes or {}).get("attention_score")
        pipeline_seq_len = max(1, int(attn_shape[1])) if attn_shape and len(attn_shape) > 1 else seq_len

        cross_layer_bytes = self.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, pipeline_seq_len)[1]

        comm_metadata = self._build_comm_metadata(
            reduction_sizes=reduction_sizes,
            local_comp=local_comp, #when doing all_reduce between dp nodes, there is local computation on each node to apply gradients
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
        )

        transformer_operation_entries: List[Dict[str, Any]] = []
        transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode in (ParallelismMode.CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            for key in ("layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2", "MLP"):
                try:
                    spec = transformer_results[key]
                except KeyError:
                    print(f"Key {key} not found in transformer_results")
                    print(transformer_results)
                    exit()
                fwd_time = spec['forward_gemm'] if 'forward_gemm' in spec else spec['forward_compute'] if 'forward_compute' in spec else spec['forward']
                bwd_time = spec['backward_gemm'] if 'backward_gemm' in spec else spec['backward_compute'] if 'backward_compute' in spec else spec['backward']
                fwd_red = spec.get('forward_reduction', 0.0)
                bwd_red = spec.get('backward_reduction', 0.0)
                comm_bytes_fwd = spec.get('comm_size_forward', 0)
                comm_bytes_bwd = spec.get('comm_size_backward', 0)

                entry = {
                    "name": key,
                    "forward": {
                        "duration": fwd_time,
                        "reduction": fwd_red,
                        "comm_keys": [],
                    },
                    "backward": {
                        "duration": bwd_time,
                        "reduction": bwd_red,
                        "comm_keys": [],
                    },
                }
                entry["forward"]["flops"] = spec.get("flops", 0)
                entry["forward"]["bytes"] = spec.get("input_bytes", 0) + spec.get("output_bytes", 0)
                entry["backward"]["flops"] = spec.get("flops_backward", 0)
                entry["backward"]["bytes"] = spec.get("input_bytes_backward", spec.get("input_bytes", 0)) + spec.get("output_bytes_backward", spec.get("output_bytes", 0))

                self._populate_transformer_comm_metadata(
                    entry=entry,
                    metadata=transformer_comm_metadata,
                    comm_bytes_fwd=comm_bytes_fwd,
                    comm_bytes_bwd=comm_bytes_bwd,
                )

                transformer_operation_entries.append(entry)
            
            
            
            
        elif parallelism_mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.SINGLE):
            for key in ("layernorm1", "MHA", "layernorm2", "MLP"):
                spec = transformer_results[key]
                fwd_time = spec['forward']
                bwd_time = spec['backward']
                fwd_red = spec.get('forward_reduction', 0.0)
                bwd_red = spec.get('backward_reduction', 0.0)
                comm_bytes_fwd = spec.get('comm_size_forward', 0)
                comm_bytes_bwd = spec.get('comm_size_backward', 0)

                entry = {
                    "name": key,
                    "forward": {
                        "duration": fwd_time,
                        "reduction": fwd_red,
                        "comm_keys": [],
                    },
                    "backward": {
                        "duration": bwd_time,
                        "reduction": bwd_red,
                        "comm_keys": [],
                    },
                }
                entry["forward"]["flops"] = spec.get("flops", 0)
                entry["forward"]["bytes"] = spec.get("input_bytes", 0) + spec.get("output_bytes", 0)
                entry["backward"]["flops"] = spec.get("flops_backward", 0)
                entry["backward"]["bytes"] = spec.get("input_bytes_backward", spec.get("input_bytes", 0)) + spec.get("output_bytes_backward", spec.get("output_bytes", 0))

                self._populate_transformer_comm_metadata(
                    entry=entry,
                    metadata=transformer_comm_metadata,
                    comm_bytes_fwd=comm_bytes_fwd,
                    comm_bytes_bwd=comm_bytes_bwd,
                )

                transformer_operation_entries.append(entry)

        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        transformer_graph: Optional[Graph] = None
        transformer_forward_root: Optional[Any] = None
        transformer_backward_root: Optional[Any] = None

        transformer_comp_times = {
            "transformer": {
                "gemms": transformer_operation_entries,
            }
        }

        transformer_graph = simulate_LLM.Graph(
            mode="transformer",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            cp=self.cp,
            comp_times=transformer_comp_times,
            comm_metadata=transformer_comm_metadata,
            misc_metadata={},
        )
        transformer_forward_root = transformer_graph.construct_transformer_graph(direction="forward")
        if include_transformer_backward:
            transformer_backward_root = transformer_graph.construct_transformer_graph(direction="backward")

        comp_times = {
            "embedding_f": node_breakdown.get('embedding_f', 0.0),
            "embedding_b": node_breakdown.get('embedding_b', 0.0) if include_pipeline_backward else 0.0,
            "embedding_f_flops": node_breakdown.get('embedding_f_flops', 0),
            "embedding_f_bytes": node_breakdown.get('embedding_f_bytes', 0),
            "embedding_b_flops": node_breakdown.get('embedding_b_flops', 0) if include_pipeline_backward else 0,
            "embedding_b_bytes": node_breakdown.get('embedding_b_bytes', 0) if include_pipeline_backward else 0,
            "linear_softmax_f": node_breakdown.get('linear_softmax_f', 0.0),
            "linear_softmax_b": node_breakdown.get('linear_softmax_b', 0.0) if include_pipeline_backward else 0.0,
            "linear_softmax_f_flops": node_breakdown.get('linear_softmax_f_flops', 0),
            "linear_softmax_f_bytes": node_breakdown.get('linear_softmax_f_bytes', 0),
            "linear_softmax_b_flops": node_breakdown.get('linear_softmax_b_flops', 0) if include_pipeline_backward else 0,
            "linear_softmax_b_bytes": node_breakdown.get('linear_softmax_b_bytes', 0) if include_pipeline_backward else 0,
            "transformer_f": node_breakdown.get('transformer_time_f', 0.0),
            "transformer_b": node_breakdown.get('transformer_time_b', 0.0) if include_pipeline_backward else 0.0,
            "transformer_f_flops": node_breakdown.get('transformer_f_flops', 0),
            "transformer_f_bytes": node_breakdown.get('transformer_f_bytes', 0),
            "transformer_b_flops": node_breakdown.get('transformer_b_flops', 0) if include_pipeline_backward else 0,
            "transformer_b_bytes": node_breakdown.get('transformer_b_bytes', 0) if include_pipeline_backward else 0,
            "kv_cache_fetch": node_breakdown.get('kv_cache_fetch', 0.0),
            "kv_cache_store": node_breakdown.get('kv_cache_store', 0.0),
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        misc_metadata = {
            "num_batch": self.mb,
            "num_layer": self.num_layers,
            "all_reduce": getattr(self, "all_reduce", "the end"),
        }
        if getattr(self, "kv_cache_fetch_overlap", False):
            misc_metadata["kv_cache_fetch_overlap"] = True

        pipeline_graph_obj = simulate_LLM.Graph(
            mode="pipeline",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            cp=self.cp,
            comp_times=comp_times,
            comm_metadata=comm_metadata,
            misc_metadata=misc_metadata,
        )
        graph_root = pipeline_graph_obj.construct_fwd_bwd_graph(include_backward=include_pipeline_backward)
        interconnect_params = self._build_interconnect_params()

        return (
            pipeline_graph_obj,
            graph_root,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            interconnect_params,
        )

    def calc_time_llm(self):
        """Calculate time for LLM model."""
        # Extract model parameters
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim
        
        attention_type = self.attention_type
        kv_heads = self.kv_heads
        
        # Adjust types and calculate node latencies
        self.readjust_type()

        # Get structured transformer results and node breakdown in one efficient call

        transformer_results, node_breakdown = self.compute_all_gemm_and_node_times(batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, ffn_dim )
        #get the memory for one transformer layer per gpu in tp-sp node, actvation memory is for per micro-batch, weight, gradient and optimizer memory is consatnt 
        transformer_mem_layer, transformer_act_layer,transformer_act_layer_inf, transformer_static_layer, gradient_mem_layer, optimizer_mem_layer, weight_memory_layer = (
            LLM_util.get_transformer_mem_layer( #return memory per layer per gpu in one lp group
                d = self.dp,
                t = self.tp,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                seq_len=seq_len/self.cp, #in cp, seq_len is divided by cp
                ffn_dim=ffn_dim,
                n_heads=num_heads,
                precision=self.precision,
                model_type=self.model_type,
        )
    )
        memory_data = {
            'activation_mem_per_layer': transformer_act_layer,
            'activation_mem_per_layer_inference': transformer_act_layer_inf, # prefill max activation memory per layer per gpu
            'weight_mem_per_layer': weight_memory_layer,
            'gradient_mem_per_layer': gradient_mem_layer,
            'optimizer_mem_per_layer': optimizer_mem_layer,
            'static_mem_per_layer': transformer_static_layer,
            'total_mem_per_layer': transformer_mem_layer,
        }

    

        if self.debug:
            print("self.tp:", self.tp)
            print("Calculating LLM time...")
            print(
                "simulating parallelism with dp = {}, lp = {}, total data batch = {}, "
                "for each dp node, data batch = {}, for each pipeline stage, data batch = {}".format(
                    self.dp, self.lp, self.batch_size, self.miniB, self.microB
                )
            )
            print("total number of workers: {}".format(self.num_workers))
            print("number of workers for each data parallelism batch: {}".format(self.num_workers_dp))
            print("number of workers for each pipeline stage: {}".format(self.num_workers_lp))
        

        (
            pipeline_graph_obj,
            graph_root,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_results=transformer_results,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            # num_heads=num_heads,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
        )

        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.transformer_analytical_time_forward = node_breakdown['transformer_time_f']
        self.transformer_analytical_time_backward = node_breakdown['transformer_time_b']

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params
        forward_root, backward_root = self.pipeline_graph.extract_forward_graph(graph_root)
        self.pipeline_graph.save_graph(
            forward_root,
            filename="llm_pipeline_forward_graph",
        )
        _, peak_mem_inf = self._simulate_with_memory(forward_root, memory_data, mode="inference")
        # _, peak_mem = self._simulate_with_memory(graph_root, memory_data, mode="training")
        mode = self.execution_mode
        
        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
        )
        mode = self.execution_mode
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc

        time_fw_bw = result.total_time

        pipeline_root = result.graph_root
        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = dispatcher.interconnect_params
        


        # debug helper. If set, print analytical transformer time and actual transformer time
        if self._generate_graphs:
            print(f"Analytical transformer forward time: {self.transformer_analytical_time_forward:.4f}s")
            print(f"Analytical transformer backward time: {self.transformer_analytical_time_backward:.4f}s")
            if self.transformer_astrasim_time_forward is not None and self.transformer_astrasim_time_backward is not None:
                print(f"Actual transformer forward time: {self.transformer_astrasim_time_forward:.4f}s")
                print(f"Actual transformer backward time: {self.transformer_astrasim_time_backward:.4f}s")

        self.tot_time = time_fw_bw

        output_file = os.path.join(self.output_dir, "LLM_time_results.txt")
        
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            # f.write("Forward Time: {0:.8f} {1}\n".format(time_fw * m, second))
            # f.write("Backward Time: {0:.8f} {1}\n".format(time_bw * m, second))
            f.write("Forward + Backward Time: {0:.8f} {1}\n".format(time_fw_bw * m, second))

            # f.write("Total Time: {0:.8f}\n".format(TC.getTime()))

        return time_fw_bw
        
    def _simulate_with_memory( 
        self,
        graph_root: Any,
        memory_data: Dict[str, Any],
        mode: str = "training" #training or inference
    ) -> Tuple[float, float]:
        """Run memory-aware simulation and report duration plus peak usage."""

        time_with_memory, peak_mem = self.pipeline_graph.simulate_memory(
            graph_root,
            memory_data,
            mode,
            self.output_dir,
            
        )
        self.memory_peak_gb = peak_mem

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None:
            hardware_mem_bytes = getattr(
                getattr(self.hw_config, "tech_config", object()),
                "DRAM",
                object(),
            )
            hardware_mem_bytes = getattr(hardware_mem_bytes, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / (1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - peak_mem
            self.memory_headroom_gb = mem_delta
            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                f"Simulation mode: {mode}",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.6f} GiB",
                f"Simulated peak memory usage(per gpu): {peak_mem:.6f} GiB",
            ]
            if mem_delta < 0:
                info_lines.append(f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.6f} GiB")
                self.memory_capacity_exceeded = True
                self.memory_capacity_violation_gb = max(self.memory_capacity_violation_gb, abs(mem_delta))
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.6f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None

        return time_with_memory, peak_mem

    def get_time(self):
        return self.tot_time

    def memory_capacity_warning(self) -> Optional[str]:
        violation = max(0.0, getattr(self, "memory_capacity_violation_gb", 0.0))
        if violation > 0.0:
            return (
                f"[WARN] Peak memory exceeds capacity by {violation:.2f} GiB. "
                "Please change parallelism settings for realistic results."
            )
        return None
