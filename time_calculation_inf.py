"""LLM inference prefill time-calculation entry points."""

import math
import os
from typing import Any, Dict, List, Optional, Tuple
from time_calculation_LLM import LLMExecutionDispatcher, TimeCalculationLLM, GemmType, GELU_FORWARD_FLOPS_PER_ELEMENT, SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT
from simulate_inf import DecodeSample, InferenceConfig, InferenceEngine
import LLM_util

class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        self._raw_model_config = model_config

    def _build_decode_transformer_results(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Construct transformer metadata for a single decode step."""

        annotate = self.annotate_compute
        dtype_size = self.dtype_size

        def _time(val: float) -> float:
            return val

        head_dim = self.hidden_dim // self.num_heads

        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        kv_cache_fetch_time = self.roofline(
            0,
            token_bytes * total_seq_len,
            name="kv_cache_fetch",
        ) + self.O
        kv_cache_store_time = self.roofline(
            0,
            token_bytes,
            name="kv_cache_store",
        ) + self.O

        ffn_dim = self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim
        gemm_shapes = gemm_shapes or LLM_util.process_decode_gemm_shapes(
            batch_size=batch_size,
            current_seq_len=total_seq_len,
            d_model=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            ffn_dim=ffn_dim,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
        )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]

        gemm_results: Dict[str, Dict[str, Any]] = {}

        qkv_proj_gemm, qkv_proj_reduction, qkv_proj_size = self._tensor_parallelism_gemm_forward(
            gemm_qkv_proj, "decode_qkv_proj_f", gemm_type=GemmType.QKV
        )
        qkv_proj_forward_raw = qkv_proj_gemm + qkv_proj_reduction
        qkv_f_flops, _, qkv_in_bytes, qkv_out_bytes = self._compute_gemm_flops_bytes(gemm_qkv_proj)
        gemm_results['qkv_proj'] = {
            'forward': qkv_proj_forward_raw,
            'forward_gemm': qkv_proj_gemm,
            'forward_reduction': qkv_proj_reduction,
            'flops': qkv_f_flops,
            'input_bytes': qkv_in_bytes,
            'output_bytes': qkv_out_bytes,
            'comm_size_forward': qkv_proj_size,
        }

        attn_score_gemm, attn_score_reduction, attn_score_size = self._tensor_parallelism_gemm_forward(
            gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE
        )
        attention_score_raw = attn_score_gemm + attn_score_reduction
        attn_score_f_flops, _, attn_score_in_bytes, attn_score_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_score)
        gemm_results['attention_score'] = {
            'forward': attention_score_raw,
            'forward_gemm': attn_score_gemm,
            'forward_reduction': attn_score_reduction,
            'flops': attn_score_f_flops,
            'input_bytes': attn_score_in_bytes,
            'output_bytes': attn_score_out_bytes,
            'comm_size_forward': attn_score_size,
        }

        attn_out_gemm, attn_out_reduction, attn_out_size = self._tensor_parallelism_gemm_forward(
            gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT
        )
        attention_output_raw = attn_out_gemm + attn_out_reduction
        attn_out_f_flops, _, attn_out_in_bytes, attn_out_out_bytes = self._compute_gemm_flops_bytes(gemm_attention_output)
        gemm_results['attention_output'] = {
            'forward': attention_output_raw,
            'forward_gemm': attn_out_gemm,
            'forward_reduction': attn_out_reduction,
            'flops': attn_out_f_flops,
            'input_bytes': attn_out_in_bytes,
            'output_bytes': attn_out_out_bytes,
            'comm_size_forward': attn_out_size,
        }

        out_proj_gemm, _, out_proj_size = self._tensor_parallelism_gemm_forward(
            gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ
        )
        out_proj_reduction = (
            self.get_tensor_reduction_time(out_proj_size, "all_reduce", "decode_output_projection")
            if out_proj_size
            else 0.0
        )
        out_proj_forward_raw = out_proj_gemm + out_proj_reduction
        out_proj_f_flops, _, out_proj_in_bytes, out_proj_out_bytes = self._compute_gemm_flops_bytes(gemm_output_proj)
        gemm_results['output_proj'] = {
            'forward': out_proj_forward_raw,
            'forward_gemm': out_proj_gemm,
            'forward_reduction': out_proj_reduction,
            'flops': out_proj_f_flops,
            'input_bytes': out_proj_in_bytes,
            'output_bytes': out_proj_out_bytes,
            'comm_size_forward': out_proj_size,
        }

        ffn1_gemm, ffn1_reduction, ffn1_size = self._tensor_parallelism_gemm_forward(
            gemm_ffn1, "decode_ffn1_f", gemm_type=GemmType.FFN1
        )
        ffn1_forward_raw = ffn1_gemm + ffn1_reduction
        ffn1_f_flops, _, ffn1_in_bytes, ffn1_out_bytes = self._compute_gemm_flops_bytes(gemm_ffn1)
        gemm_results['ffn1'] = {
            'forward': ffn1_forward_raw,
            'forward_gemm': ffn1_gemm,
            'forward_reduction': ffn1_reduction,
            'flops': ffn1_f_flops,
            'input_bytes': ffn1_in_bytes,
            'output_bytes': ffn1_out_bytes,
            'comm_size_forward': ffn1_size,
        }

        ffn2_gemm, _, ffn2_size = self._tensor_parallelism_gemm_forward(
            gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2
        )
        ffn2_reduction = (
            self.get_tensor_reduction_time(ffn2_size, "all_reduce", "decode_ffn2")
            if ffn2_size
            else 0.0
        )
        ffn2_forward_raw = ffn2_gemm + ffn2_reduction
        ffn2_f_flops, _, ffn2_in_bytes, ffn2_out_bytes = self._compute_gemm_flops_bytes(gemm_ffn2)
        gemm_results['ffn2'] = {
            'forward': ffn2_forward_raw,
            'forward_gemm': ffn2_gemm,
            'forward_reduction': ffn2_reduction,
            'flops': ffn2_f_flops,
            'input_bytes': ffn2_in_bytes,
            'output_bytes': ffn2_out_bytes,
            'comm_size_forward': ffn2_size,
        }

        output_seq_len = 1
        output_proj_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.hidden_dim,
        )
        residual1_f = self.get_residual_f(output_proj_shape)
        layernorm1_f, layernorm1_reduction, layernorm1_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )

        ffn2_shape = (
            batch_size,
            output_seq_len,
            ffn_dim,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f, layernorm2_reduction, layernorm2_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f_raw = self.get_linear_softmax_f(linear_shape)
        linear_softmax_flops = 5 * batch_size * output_seq_len * self.vocab_size
        linear_softmax_bytes_total = dtype_size * batch_size * output_seq_len * self.vocab_size * 8

        if self.model_type == "llama":
            act_f_raw = self.get_swiglu_f(gemm_ffn1)
            act_flops = batch_size * output_seq_len * ffn_dim * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT
        else:
            act_f_raw = self.get_gelu_f(gemm_ffn1)
            act_flops = batch_size * output_seq_len * ffn_dim * GELU_FORWARD_FLOPS_PER_ELEMENT

        attention_scale_softmax_f_raw = self.get_scale_softmax_f(gemm_attention_score)
        softmax_elems = batch_size * self.num_heads * total_seq_len * total_seq_len
        softmax_f_flops = 6 * softmax_elems
        softmax_f_bytes_total = dtype_size * softmax_elems * 11
        if self.zero_internal_softmax:
            attention_scale_softmax_f_raw = 0.0
            softmax_f_flops = 0
            softmax_f_bytes_total = 0

        residual_elems = batch_size * output_seq_len * self.hidden_dim
        residual_bytes_total = dtype_size * residual_elems * 3
        residual_flops = 2 * residual_elems

        layernorm_elems = batch_size * output_seq_len
        layernorm_f_flops = layernorm_elems * self.hidden_dim * 11
        layernorm_f_bytes_total = dtype_size * layernorm_elems * self.hidden_dim * 10

        gemm_results['embedding'] = {
            'forward': 0.0,
            'flops': 0,
            'input_bytes': 0,
            'output_bytes': 0,
        }
        gemm_results['attention_scale_softmax'] = {
            'forward': attention_scale_softmax_f_raw,
            'flops': softmax_f_flops,
            'input_bytes': softmax_f_bytes_total // 2,
            'output_bytes': softmax_f_bytes_total // 2,
        }
        gemm_results['residual1'] = {
            'forward': residual1_f,
            'flops': residual_flops,
            'input_bytes': residual_bytes_total // 2,
            'output_bytes': residual_bytes_total // 2,
        }
        gemm_results['residual2'] = {
            'forward': residual2_f,
            'flops': residual_flops,
            'input_bytes': residual_bytes_total // 2,
            'output_bytes': residual_bytes_total // 2,
        }
        gemm_results['layernorm1'] = {
            'forward': layernorm1_f,
            'flops': layernorm_f_flops,
            'input_bytes': layernorm_f_bytes_total // 2,
            'output_bytes': layernorm_f_bytes_total // 2,
        }
        gemm_results['layernorm2'] = {
            'forward': layernorm2_f,
            'flops': layernorm_f_flops,
            'input_bytes': layernorm_f_bytes_total // 2,
            'output_bytes': layernorm_f_bytes_total // 2,
        }
        gemm_results['gelu'] = {
            'forward': act_f_raw,
            'flops': act_flops,
            'input_bytes': dtype_size * batch_size * output_seq_len * ffn_dim,
            'output_bytes': dtype_size * batch_size * output_seq_len * ffn_dim,
        }
        gemm_results['linear_softmax'] = {
            'forward': linear_softmax_f_raw,
            'flops': linear_softmax_flops,
            'input_bytes': linear_softmax_bytes_total // 2,
            'output_bytes': linear_softmax_bytes_total // 2,
        }

        attention_forward_raw = attention_score_raw + attention_scale_softmax_f_raw + attention_output_raw
        attention_gemm_raw = attn_score_gemm + attn_out_gemm + attention_scale_softmax_f_raw
        attention_reduction_raw = attn_score_reduction + attn_out_reduction
        attention_comm_bytes = attn_score_size + attn_out_size
        mha_forward_raw = qkv_proj_forward_raw + attention_forward_raw + out_proj_forward_raw
        ffn_forward_raw = ffn1_forward_raw + act_f_raw + ffn2_forward_raw
        layernorm1_forward_raw = residual1_f + layernorm1_f
        layernorm2_forward_raw = residual2_f + layernorm2_f

        transformer_forward_raw = (
            mha_forward_raw
            + ffn_forward_raw
            + layernorm1_forward_raw
            + layernorm1_reduction
            + layernorm2_forward_raw
            + layernorm2_reduction
        )

        transformer_results = {
            "qkv_proj": {
                "forward": _time(qkv_proj_forward_raw),
                "backward": 0.0,
                "forward_gemm": _time(qkv_proj_gemm),
                "forward_reduction": _time(qkv_proj_reduction),
                "backward_gemm": 0.0,
                "backward_reduction": 0.0,
                "comm_size_forward": qkv_proj_size,
                "comm_size_backward": 0,
                "flops": qkv_f_flops,
                "flops_backward": 0,
                "input_bytes": qkv_in_bytes,
                "output_bytes": qkv_out_bytes,
                "input_bytes_backward": 0,
                "output_bytes_backward": 0,
            },
            "attention": {
                "forward": _time(attention_forward_raw),
                "backward": 0.0,
                "forward_gemm": _time(attention_gemm_raw),
                "forward_reduction": _time(attention_reduction_raw),
                "backward_gemm": 0.0,
                "backward_reduction": 0.0,
                "comm_size_forward": attention_comm_bytes,
                "comm_size_backward": 0,
                "flops": attn_score_f_flops + attn_out_f_flops + softmax_f_flops,
                "flops_backward": 0,
                "input_bytes": attn_score_in_bytes + attn_score_out_bytes + attn_out_in_bytes + attn_out_out_bytes + softmax_f_bytes_total,
                "output_bytes": 0,
                "input_bytes_backward": 0,
                "output_bytes_backward": 0,
            },
            "output_proj": {
                "forward": _time(out_proj_forward_raw),
                "backward": 0.0,
                "forward_gemm": _time(out_proj_gemm),
                "forward_reduction": _time(out_proj_reduction),
                "backward_gemm": 0.0,
                "backward_reduction": 0.0,
                "comm_size_forward": out_proj_size,
                "comm_size_backward": 0,
                "flops": out_proj_f_flops,
                "flops_backward": 0,
                "input_bytes": out_proj_in_bytes,
                "output_bytes": out_proj_out_bytes,
                "input_bytes_backward": 0,
                "output_bytes_backward": 0,
            },
        }
        transformer_results["MHA"] = {
            "forward": _time(mha_forward_raw),
            "backward": 0.0,
            "forward_reduction": _time(qkv_proj_reduction + attention_reduction_raw + out_proj_reduction),
            "backward_reduction": 0.0,
            "comm_size_forward": qkv_proj_size + attention_comm_bytes + out_proj_size,
            "comm_size_backward": 0,
            "flops": transformer_results["attention"]["flops"] + transformer_results["qkv_proj"]["flops"] + transformer_results["output_proj"]["flops"],
            "flops_backward": 0,
            "input_bytes": transformer_results["attention"]["input_bytes"] + transformer_results["qkv_proj"]["input_bytes"] + transformer_results["qkv_proj"]["output_bytes"] + transformer_results["output_proj"]["input_bytes"] + transformer_results["output_proj"]["output_bytes"],
            "output_bytes": 0,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["ffn1"] = {
            "forward": _time(ffn1_forward_raw),
            "backward": 0.0,
            "forward_gemm": _time(ffn1_gemm),
            "forward_reduction": _time(ffn1_reduction),
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
            "comm_size_forward": ffn1_size,
            "comm_size_backward": 0,
            "flops": ffn1_f_flops,
            "flops_backward": 0,
            "input_bytes": ffn1_in_bytes,
            "output_bytes": ffn1_out_bytes,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["ffn2"] = {
            "forward": _time(ffn2_forward_raw),
            "backward": 0.0,
            "forward_gemm": _time(ffn2_gemm),
            "forward_reduction": _time(ffn2_reduction),
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
            "comm_size_forward": ffn2_size,
            "comm_size_backward": 0,
            "flops": ffn2_f_flops,
            "flops_backward": 0,
            "input_bytes": ffn2_in_bytes,
            "output_bytes": ffn2_out_bytes,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["MLP"] = {
            "forward": _time(ffn_forward_raw),
            "backward": 0.0,
            "forward_reduction": _time(ffn1_reduction + ffn2_reduction),
            "backward_reduction": 0.0,
            "comm_size_forward": ffn1_size + ffn2_size,
            "comm_size_backward": 0,
            "flops": transformer_results["ffn1"]["flops"] + transformer_results["ffn2"]["flops"] + act_flops,
            "flops_backward": 0,
            "input_bytes": ffn1_in_bytes + ffn1_out_bytes + ffn2_in_bytes + ffn2_out_bytes,
            "output_bytes": 0,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["layernorm1"] = {
            "forward": _time(layernorm1_forward_raw + layernorm1_reduction),
            "backward": 0.0,
            "forward_compute": _time(layernorm1_f + residual1_f),
            "forward_reduction": _time(layernorm1_reduction),
            "backward_compute": 0.0,
            "backward_reduction": 0.0,
            "comm_size_forward": layernorm1_bytes,
            "comm_size_backward": 0,
            "flops": layernorm_f_flops + residual_flops,
            "flops_backward": 0,
            "input_bytes": layernorm_f_bytes_total // 2 + residual_bytes_total // 2,
            "output_bytes": layernorm_f_bytes_total // 2 + residual_bytes_total // 2,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["layernorm2"] = {
            "forward": _time(layernorm2_forward_raw + layernorm2_reduction),
            "backward": 0.0,
            "forward_compute": _time(layernorm2_f + residual2_f),
            "forward_reduction": _time(layernorm2_reduction),
            "backward_compute": 0.0,
            "backward_reduction": 0.0,
            "comm_size_forward": layernorm2_bytes,
            "comm_size_backward": 0,
            "flops": layernorm_f_flops + residual_flops,
            "flops_backward": 0,
            "input_bytes": layernorm_f_bytes_total // 2 + residual_bytes_total // 2,
            "output_bytes": layernorm_f_bytes_total // 2 + residual_bytes_total // 2,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["linear_softmax"] = {
            "forward": _time(linear_softmax_f_raw),
            "backward": 0.0,
            "flops": linear_softmax_flops,
            "flops_backward": 0,
            "input_bytes": linear_softmax_bytes_total // 2,
            "output_bytes": linear_softmax_bytes_total // 2,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }
        transformer_results["transformer"] = {
            "forward": _time(transformer_forward_raw),
            "backward": 0.0,
            "flops": (
                transformer_results["MHA"]["flops"] + transformer_results["MLP"]["flops"] +
                transformer_results["layernorm1"]["flops"] + transformer_results["layernorm2"]["flops"]
            ),
            "flops_backward": 0,
            "input_bytes": (
                transformer_results["MHA"]["input_bytes"] + transformer_results["MLP"]["input_bytes"] +
                transformer_results["layernorm1"]["input_bytes"] + transformer_results["layernorm2"]["input_bytes"]
            ),
            "output_bytes": 0,
            "input_bytes_backward": 0,
            "output_bytes_backward": 0,
        }

        node_breakdown = {
            "transformer_time_f": _time(transformer_forward_raw),
            "transformer_time_b": 0.0,
            "linear_softmax_f": _time(linear_softmax_f_raw),
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
            "transformer_f_flops": transformer_results["transformer"]["flops"],
            "transformer_b_flops": 0,
            "transformer_f_bytes": transformer_results["transformer"]["input_bytes"],
            "transformer_b_bytes": 0,
            "linear_softmax_f_flops": linear_softmax_flops,
            "linear_softmax_b_flops": 0,
            "linear_softmax_f_bytes": linear_softmax_bytes_total,
            "linear_softmax_b_bytes": 0,
            "embedding_f_flops": 0,
            "embedding_b_flops": 0,
            "embedding_f_bytes": 0,
            "embedding_b_bytes": 0,
            "kv_cache_fetch": kv_cache_fetch_time,
            "kv_cache_store": kv_cache_store_time,
        }

        return transformer_results, node_breakdown
    def prepare_decode_graphs(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        ffn_dim = self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim
        transformer_results, node_breakdown = self._build_decode_transformer_results(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            gemm_shapes=gemm_shapes,
        )
        return self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_results=transformer_results,
            batch_size=batch_size,
            seq_len=total_seq_len,
            hidden_dim=self.hidden_dim,
            ffn_dim=ffn_dim,
            vocab_size=self.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
            gemm_shapes=gemm_shapes,
        )

    def calc_time(self) -> float:
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        decode_len = self.model.decode_len
        prefill_len = self.seq_len - decode_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim

        if prefill_len == 0:
            print("Skipping prefill")
            return 0.0
        elif prefill_len < 0:
            raise ValueError(f"Prefill length is negative. Prefill len = seq_len ({self.seq_len}) - decode_len ({decode_len})")

        self.readjust_type()

        transformer_results, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            prefill_len,
            num_heads,
            self.kv_heads,
            ffn_dim,
        )

        head_dim = hidden_dim // num_heads
        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        prefill_store_time = self.roofline(
            0,
            token_bytes * prefill_len,
            name="kv_cache_store_prefill",
        ) + self.O

        node_breakdown["kv_cache_store"] = prefill_store_time
        node_breakdown["kv_cache_fetch"] = 0.0

        (
            pipeline_graph,
            pipeline_root,
            transformer_graph,
            transformer_forward_root,
            _,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_results=transformer_results,
            batch_size=batch_size,
            seq_len=prefill_len,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
        )

        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = None
        self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
        self.transformer_analytical_time_backward = None

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
            raise NotImplementedError(
                f"{exc}. Selected execution mode '{mode.value}'."
            ) from exc

        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = result.graph_root
        self.pipeline_interconnect = dispatcher.interconnect_params

        total_time = result.total_time

        return total_time

    def calc_decode_time(self) -> Tuple[float, List[DecodeSample]]:
        """
        Calculate autoregressive decode phase execution time.

        Calculate autoregressive decode phase execution time using sample-based approach.

        Returns:
            float: Total decode phase execution time
        """
        # Get inference sampling configuration
        sample_every = self.model.inference_sample_every
        if sample_every == -1:
            sample_every = 2**31 - 1

        decode_len = self.model.decode_len
        if decode_len == 0:
            print("Skipping decode")
            return 0.0, []

        # Create inference configuration from model parameters
        inference_config = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len - decode_len,
            decode_len=decode_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            ffn_dim=self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            tp_sp=self.tp_sp,
            sample_every=sample_every,
            kv_cache_fetch_overlap=self.kv_cache_fetch_overlap,
        )


        # Create inference engine with proper hardware and model configs
        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self._raw_model_config,
            time_calc_cls=TimeCalculationLLMInference,
        )

        # Build decode phase using sample-based approach with real DeepFlow integration
        decode_time, decode_samples = inference_engine._build_decode_graph()
        return decode_time, decode_samples

    def calc_total_inference_time(self) -> dict:
        """
        Calculate complete inference time including prefill + decode phases.

        Returns:
            dict: Breakdown of inference timing components
        """
        # Calculate prefill time (existing functionality)
        prefill_time = self.calc_time()

        # Calculate decode time (new functionality)
        decode_time, decode_samples = self.calc_decode_time()
        total_time = prefill_time + decode_time

        time_to_first_token = prefill_time
        if decode_samples:
            time_to_first_token += decode_samples[0].execution_time

        head_dim = self.hidden_dim // self.num_heads
        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=self._effective_transformer_batch(),
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        prefill_len = self.seq_len - self.model.decode_len
        decode_len = self.model.decode_len
        num_layers = self.num_layers

        if prefill_len < 0:
            raise ValueError(f"Prefill length is negative. Prefill len = seq_len ({self.seq_len}) - decode_len ({decode_len})")

        prefill_store_bytes = token_bytes * prefill_len * num_layers
        decode_store_bytes = token_bytes * decode_len * num_layers
        decode_fetch_bytes = token_bytes * num_layers * (
            decode_len * prefill_len + decode_len * (decode_len + 1) // 2
        )

        def _to_gib(byte_val: int) -> float:
            return byte_val / (1024 ** 3)

        if decode_samples:
            # do NOT use effective_transformer_batch here
            decode_rates = self._decode_token_rates(decode_samples, decode_len, decode_time, self.batch_size)
        else:
            decode_rates = None

        print(
            f"[prefill] time: {prefill_time:.6f}s, "
            f"[decode] time: {decode_time:.6f}s, "
            f"[total] time: {total_time:.6f}s"
        )
        print(
            f"[kv-cache] prefill_store={_to_gib(prefill_store_bytes):.2f} GiB, "
            f"decode_store={_to_gib(decode_store_bytes):.2f} GiB, "
            f"decode_fetch={_to_gib(decode_fetch_bytes):.2f} GiB"
        )


        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time,
            "time_to_first_token": time_to_first_token,
            "kv_cache_prefill_store_bytes": prefill_store_bytes,
            "kv_cache_decode_store_bytes": decode_store_bytes,
            "kv_cache_decode_fetch_bytes": decode_fetch_bytes,
            "decode_tokens_per_s": decode_rates,
        }

    @staticmethod
    def _decode_token_rates(
        samples: List[DecodeSample],
        decode_len: int,
        total_decode_time: float,
        batch_size: int,
    ) -> Dict[str, float]:
        if decode_len <= 0:
            return {}

        def token_time_at(step: int) -> float:
            if not samples:
                return 0.0
            if step <= samples[0].step_id:
                return samples[0].execution_time
            for idx in range(1, len(samples)):
                prev = samples[idx - 1]
                curr = samples[idx]
                if step <= curr.step_id:
                    gap = curr.step_id - prev.step_id
                    if gap <= 0:
                        return curr.execution_time
                    ratio = (step - prev.step_id) / gap
                    return prev.execution_time + ratio * (curr.execution_time - prev.execution_time)
            return samples[-1].execution_time

        def safe_rate(token_time: float) -> float:
            if token_time <= 0.0:
                return 0.0
            return (1.0 / token_time) * batch_size

        last_step = max(decode_len - 1, 0)
        mid_step = decode_len // 2

        start_rate = safe_rate(token_time_at(0))
        mid_rate = safe_rate(token_time_at(mid_step))
        end_rate = safe_rate(token_time_at(last_step))

        overall_rate = 0.0
        if total_decode_time > 0.0:
            overall_rate = decode_len / total_decode_time

        return {
            "start": start_rate,
            "midpoint": mid_rate,
            "end": end_rate,
            "midpoint_step": mid_step,
            "overall": overall_rate,
        }



__all__ = ["TimeCalculationLLMInference"]
