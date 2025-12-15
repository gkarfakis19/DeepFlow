"""LLM inference prefill time-calculation entry points."""

import math
import os
from typing import Dict, List, Optional, Tuple, Mapping, Set
from time_calculation_LLM import (
    LLMExecutionDispatcher,
    TimeCalculationLLM,
    GemmType,
    COMMUNICATION_RULES,
    COMM_RULE_DEFAULT_KEY,
)
from simulate_inf import DecodeSample, InferenceConfig, InferenceEngine
import LLM_util
import json
from timing_model import CommSpec, DirectionTiming, OperationTiming, OperationGroup

def convert_prefix(value: float) -> float:
    """Assign SI unit prefixes to numerical values."""
    if value > 1:
        prefixes = ["", "k", "M", "G"]
        index = min(int(math.log10(value) // 3), len(prefixes) - 1)
        scaled_value = value / (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"
    else:
        prefixes = ["", "m", "µ", "n"]
        index = min(int(-(math.log10(value) // 3)), len(prefixes) - 1)
        scaled_value = value * (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"

class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        if self.zero_stage >= 3 and self.dp > 1:
            raise ValueError("ZeRO-3 data parallelism is not supported for inference runs (dp_zero_stage must be <3 or dp=1).")
        self._raw_model_config = model_config

    def _build_decode_transformer_results(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[Dict[str, OperationTiming], Dict[str, float]]:
        """Construct transformer timings + node breakdown for a single decode step."""

        head_dim = self.hidden_dim // self.num_heads

        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        seq_degree = self._sequence_parallel_degree()
        comm_kind = "all_reduce" if seq_degree == 1 else "reduce_scatter"

        intermediate_size = self.intermediate_size
        gemm_shapes = gemm_shapes or LLM_util.process_decode_gemm_shapes(
            self,
            batch_size=batch_size,
            current_seq_len=total_seq_len,
            d_model=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            intermediate_size=intermediate_size,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
        )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        # FlashAttention is not used during single-token (incremental) decoding.


        output_seq_len = 1

        transformer_timings: Dict[str, OperationTiming] = {}

        def _make_forward(
            op_name: str,
            compute_time: float,
            comm_time: float,
            comm_bytes: float,
            *,
            flops: float = 0.0,
            memory: Optional[Mapping[str, float]] = None,
        ) -> DirectionTiming:
            bytes_int = int(math.ceil(float(comm_bytes or 0.0)))
            return DirectionTiming(
                compute_time=compute_time,
                comm_time=comm_time,
                comm_bytes=bytes_int,
                flops=flops,
                memory_accesses=dict(memory) if memory else {},
            )

        # QKV projection
        qkv_proj_time, qkv_proj_reduction, qkv_proj_size, qkv_proj_flops, qkv_proj_mem = self.parallelism_gemm_forward(
            gemm_qkv_proj, "decode_qkv_proj_f", gemm_type=GemmType.QKV
        )
        transformer_timings["qkv_proj"] = OperationTiming(
            "qkv_proj",
            forward=_make_forward(
                "qkv_proj",
                compute_time=qkv_proj_time,
                comm_time=qkv_proj_reduction,
                comm_bytes=qkv_proj_size,
                flops=qkv_proj_flops,
                memory=self._mem_levels(qkv_proj_mem),
            ),
            backward=None,
        )

        # Attention components
        attention_score_time, attention_score_reduction, attention_score_size, attention_score_flops, attention_score_mem = self.parallelism_gemm_forward(
            gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE
        )
        attention_output_time, attention_output_reduction, attention_output_size, attention_output_flops, attention_output_mem = self.parallelism_gemm_forward(
            gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT
        )
        attention_scale_softmax_f = self.get_scale_softmax_f(gemm_attention_score)

        attention_reduction = attention_score_reduction + attention_output_reduction
        attention_comm_bytes = attention_score_size + attention_output_size
        attention_forward_compute = attention_score_time + attention_scale_softmax_f + attention_output_time
        attention_forward_time = attention_forward_compute + attention_reduction
        attention_flops = (attention_score_flops or 0.0) + (attention_output_flops or 0.0)
        attention_mem = self._combine_mem(attention_score_mem, attention_output_mem)

        transformer_timings["attention"] = OperationTiming(
            "attention",
            forward=_make_forward(
                "attention",
                compute_time=attention_forward_compute,
                comm_time=attention_reduction,
                comm_bytes=attention_comm_bytes,
                flops=attention_flops,
                memory=attention_mem,
            ),
            backward=None,
        )

        attention_scale_softmax_op = OperationTiming(
            "attention_scale_softmax",
            forward=_make_forward(
                "attention_scale_softmax",
                compute_time=attention_scale_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
            ),
            backward=None,
        )
        transformer_timings["attention_scale_softmax"] = attention_scale_softmax_op

        # Output projection
        out_proj_time, _, out_proj_size, out_proj_flops, out_proj_mem = self.parallelism_gemm_forward(
            gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ
        )
        out_proj_reduction = (
            self.get_tensor_reduction_time(out_proj_size, comm_kind, "decode_output_projection")
            if out_proj_size
            else 0.0
        )
        transformer_timings["output_proj"] = OperationTiming(
            "output_proj",
            forward=_make_forward(
                "output_proj",
                compute_time=out_proj_time,
                comm_time=out_proj_reduction,
                comm_bytes=out_proj_size,
                flops=out_proj_flops,
                memory=self._mem_levels(out_proj_mem),
            ),
            backward=None,
        )

        # FFN layers
        ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.parallelism_gemm_forward(
            gemm_ffn1, "decode_ffn1_f", gemm_type=GemmType.FFN1
        )
        ffn2_time, _, ffn2_size, ffn2_flops, ffn2_mem = self.parallelism_gemm_forward(
            gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2
        )
        ffn2_reduction = (
            self.get_tensor_reduction_time(ffn2_size, comm_kind, "decode_ffn2") if ffn2_size else 0.0
        )
        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=_make_forward(
                "ffn1",
                compute_time=ffn1_time,
                comm_time=ffn1_reduction,
                comm_bytes=ffn1_size,
                flops=ffn1_flops,
                memory=self._mem_levels(ffn1_mem),
            ),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=_make_forward(
                "ffn2",
                compute_time=ffn2_time,
                comm_time=ffn2_reduction,
                comm_bytes=ffn2_size,
                flops=ffn2_flops,
                memory=self._mem_levels(ffn2_mem),
            ),
            backward=None,
        )

        # GELU/SwiGLU activation
        if self.model_type == "llama":
            act_f = self.get_swiglu_f(gemm_ffn1)
        else:
            act_f = self.get_gelu_f(gemm_ffn1)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=_make_forward("gelu", compute_time=act_f, comm_time=0.0, comm_bytes=0.0),
            backward=None,
        )

        # Layer norms
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
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=_make_forward(
                "layernorm1",
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction,
                comm_bytes=layernorm1_bytes,
            ),
            backward=None,
        )

        ffn2_shape = (
            batch_size,
            output_seq_len,
            intermediate_size,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f, layernorm2_reduction, layernorm2_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=_make_forward(
                "layernorm2",
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction,
                comm_bytes=layernorm2_bytes,
            ),
            backward=None,
        )

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(linear_shape)
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=_make_forward(
                "linear_softmax",
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
                memory=self._mem_levels(linear_softmax_mem),
            ),
            backward=None,
        )

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        # Match exact floating-point operation order from original code
        qkv_proj_forward = qkv_proj_time + qkv_proj_reduction
        attention_forward = attention_score_time + attention_scale_softmax_f + attention_output_time + attention_reduction
        out_proj_forward = out_proj_time + out_proj_reduction
        mha_forward = qkv_proj_forward + attention_forward + out_proj_forward

        ffn1_forward = ffn1_time + ffn1_reduction
        ffn2_forward = ffn2_time + ffn2_reduction
        mlp_forward = ffn1_forward + act_f + ffn2_forward

        layernorm1_forward = residual1_f + layernorm1_f
        layernorm2_forward = residual2_f + layernorm2_f

        transformer_forward = (
            mha_forward
            + mlp_forward
            + layernorm1_forward
            + layernorm1_reduction
            + layernorm2_forward
            + layernorm2_reduction
        )

        node_breakdown = {
            "transformer_time_f": transformer_forward,
            "transformer_time_b": 0.0,
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
        }

        return transformer_timings, node_breakdown


    def prepare_decode_graphs(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        intermediate_size = self.intermediate_size
        transformer_timings, node_breakdown = self._build_decode_transformer_results(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            gemm_shapes=gemm_shapes,
        )

        output_act_bytes = gemm_shapes["qkv_proj"][0] * gemm_shapes["qkv_proj"][1] * self.precision_bytes
        energy = self.calc_energy(transformer_timings, output_act_bytes)

        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "decode_transformer_results.txt")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )

        return self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            batch_size=batch_size,
            seq_len=1,
            hidden_dim=self.hidden_dim,
            intermediate_size=intermediate_size,
            vocab_size=self.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
            gemm_shapes=gemm_shapes,
        ), energy
    
    def calc_energy(self, transformer_timings: Dict[str, OperationTiming], cross_layer_comm) -> float:
        """
        Calculate energy consumption based on transformer results.
        """
        total_flops = 0.0
        total_hbm_bytes = 0.0
        inter_comm_bytes = 0.0  # data parallelism?

        aggregate_groups = {
            "MLP": ("ffn1", "gelu", "ffn2"),
        }
        solo_ops = ("layernorm1", "layernorm2", "embedding", "linear_softmax", "qkv_proj", "attention", "output_proj")

        for members in aggregate_groups.values():
            for name in members:
                op = transformer_timings.get(name)
                if op is None:
                    continue
                total_flops += op.forward.flops
                total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
                inter_comm_bytes += op.forward.comm_bytes

        for name in solo_ops:
            op = transformer_timings.get(name)
            if op is None:
                continue
            total_flops += op.forward.flops
            total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
            inter_comm_bytes += op.forward.comm_bytes

        total_comm_bytes = self.num_layers * inter_comm_bytes + (self.lp - 1) * cross_layer_comm

        energy_per_flop = self.core.nominal_energy_per_flop
        energy_hbm_byte = self.DRAM.dynamic_energy_per_bit * 8
        # TODO: honor per-dimension interconnect energies; currently assumes all comms use dimension 0.
        energy_comm_byte = (self.network.energies_per_bit[0] if self.network.energies_per_bit else 0.0) * 8

        total_energy = (total_flops * energy_per_flop) + \
            (total_hbm_bytes * energy_hbm_byte) + \
                (total_comm_bytes * energy_comm_byte)   
        
        return total_energy



    def calc_time(self) -> float:
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        decode_len = self.model.decode_len
        prefill_len = self.seq_len - decode_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads
        if prefill_len == 0:
            print("Skipping prefill")
            return 0.0
        elif prefill_len < 0:
            raise ValueError(f"Prefill length is negative. Prefill len = seq_len ({self.seq_len}) - decode_len ({decode_len})")

        self.readjust_type()
        num_SMs = self.hw_config.tech_config.core.num_bundles
        transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            prefill_len,
            num_heads,
            kv_heads,
            intermediate_size,
            num_SMs,
        )

        output_act_bytes = batch_size * self.seq_len * hidden_dim * self.precision_bytes

        total_energy = self.calc_energy(transformer_timings, output_act_bytes)

        head_dim = hidden_dim // num_heads
        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )

        (
            pipeline_graph,
            pipeline_root,
            _,
            _,
            transformer_graph,
            transformer_forward_root,
            _,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            batch_size=batch_size,
            seq_len=prefill_len,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
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

        return total_time, total_energy

    def calc_decode_time(self) -> Tuple[float, List[DecodeSample]]:
        """
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
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            use_moe=self.use_moe,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            tp_sp=self.tp_sp,
            sample_every=sample_every,
        )


        # Create inference engine with proper hardware and model configs
        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self._raw_model_config,
            time_calc_cls=TimeCalculationLLMInference,
        )

        # Build decode phase using sample-based approach with real DeepFlow integration
        # decode_time, decode_energy, decode_samples = inference_engine._build_decode_graph()
        return inference_engine._build_decode_graph()

    def calc_total_inference_time(self) -> dict:
        """
        Calculate complete inference time including prefill + decode phases.

        Returns:
            dict: Breakdown of inference timing components
        """
        # Calculate prefill time (existing functionality)
        prefill_time, prefill_energy = self.calc_time()

        # Calculate decode time (new functionality)
        decode_time, decode_energy, decode_samples = self.calc_decode_time()
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
            decode_len * (prefill_len + self.seq_len) // 2
        )

        def _to_gib(byte_val: int) -> str:
            gib_val = byte_val / (1024 ** 3)
            if gib_val > 1024:
                tib_val = gib_val / 1024
                return f"{tib_val:.1f} TiB"
            return f"{gib_val:.1f} GiB"

        if decode_samples:
            # do NOT use effective_transformer_batch here
            decode_rates = self._decode_token_rates(decode_samples, decode_len, decode_time, self.batch_size)
        else:
            decode_rates = None

        print(
            f"[prefill] time: {prefill_time:.4f}s, "
            f"[decode] time: {decode_time:.4f}s, "
            f"[total] time: {total_time:.4f}s"
        )
        print(
            f"[kv-cache] prefill_store={_to_gib(prefill_store_bytes)}, "
            f"decode_store={_to_gib(decode_store_bytes)}, "
            f"decode_fetch={_to_gib(decode_fetch_bytes)}"
        )
        
        total_energy = prefill_energy + decode_energy
        print(
            f"[prefill] energy: {convert_prefix(prefill_energy)}J, energy/tok: {convert_prefix(prefill_energy / (self._effective_transformer_batch() * prefill_len))}J",
            f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy / (self._effective_transformer_batch() * decode_len))}J",
            f"[total] energy: {convert_prefix(total_energy)}J, energy/tok: {convert_prefix(total_energy / (self._effective_transformer_batch() * (prefill_len + decode_len)))}J",
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
            return 1.0 / token_time

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
