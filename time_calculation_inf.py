"""LLM inference prefill time-calculation entry points."""

import os
from typing import Dict, List, Optional, Tuple
from time_calculation_LLM import LLMExecutionDispatcher, TimeCalculationLLM
from simulate_inf import DecodeSample, InferenceConfig, InferenceEngine
import LLM_util


class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        self._raw_model_config = model_config

    def _decode_node_breakdown(
        self,
        *,
        gemm_results: Dict[str, Dict[str, float]],
        batch_size: int,
        total_seq_len: int,
    ) -> Dict[str, float]:
        head_dim = self.hidden_dim // self.num_heads
        attention_score_shape = (
            batch_size * self.num_heads,
            1,
            head_dim,
            total_seq_len,
        )
        attention_scale_softmax_f = self.get_scale_softmax_f(attention_score_shape)

        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=head_dim,
            precision_bytes=self.kv_cache_precision,
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

        output_seq_len = 1

        output_proj_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.hidden_dim,
        )
        residual1_f = self.get_residual_f(output_proj_shape)
        layernorm1_f = self.get_layernorm_f(output_proj_shape)

        ffn_dim = self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim
        ffn2_shape = (
            batch_size,
            output_seq_len,
            ffn_dim,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f = self.get_layernorm_f(ffn2_shape)

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f = self.get_linear_softmax_f(linear_shape)

        def _inject(name: str, forward: float, backward: float = 0.0) -> None:
            gemm_results.setdefault(
                name,
                {
                    "forward": forward,
                    "backward": backward,
                    "forward_gemm": forward,
                    "forward_reduction": 0.0,
                    "backward_gemm": backward,
                    "backward_reduction": 0.0,
                },
            )

        _inject("attention_scale_softmax", attention_scale_softmax_f)
        _inject("residual1", residual1_f)
        _inject("layernorm1", layernorm1_f)
        _inject("residual2", residual2_f)
        _inject("layernorm2", layernorm2_f)
        _inject("kv_cache_fetch", kv_cache_fetch_time)
        _inject("kv_cache_store", kv_cache_store_time)

        core_time = (
            gemm_results["qkv_proj"]["forward"]
            + gemm_results["attention_score"]["forward"]
            + attention_scale_softmax_f
            + gemm_results["attention_output"]["forward"]
            + gemm_results["output_proj"]["forward"]
            + residual1_f
            + layernorm1_f
            + gemm_results["ffn1"]["forward"]
            + gemm_results["ffn2"]["forward"]
            + residual2_f
            + layernorm2_f
        )

        layer_time = core_time
        kv_fetch_node_time = kv_cache_fetch_time

        return {
            "transformer_time_f": layer_time,
            "transformer_time_b": 0.0,
            "linear_softmax_f": linear_softmax_f,
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
            "kv_cache_fetch": kv_fetch_node_time,
            "kv_cache_store": kv_cache_store_time,
        }

    def prepare_decode_graphs(
        self,
        *,
        gemm_results: Dict[str, Dict[str, float]],
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        ffn_dim = self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim
        return self._prepare_execution_graphs(
            node_breakdown=self._decode_node_breakdown(
                gemm_results=gemm_results,
                batch_size=batch_size,
                total_seq_len=total_seq_len,
            ),
            gemm_results=gemm_results,
            batch_size=batch_size,
            seq_len=total_seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
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

        gemm_results, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            prefill_len,
            num_heads,
            ffn_dim,
        )

        head_dim = hidden_dim // num_heads
        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            precision_bytes=self.kv_cache_precision,
        )
        prefill_store_time = self.roofline(
            0,
            token_bytes * prefill_len,
            name="kv_cache_store_prefill",
        ) + self.O

        node_breakdown["kv_cache_store"] = prefill_store_time
        node_breakdown["kv_cache_fetch"] = 0.0

        gemm_results["kv_cache_store"] = {
            "forward": prefill_store_time,
            "backward": 0.0,
            "forward_gemm": prefill_store_time,
            "forward_reduction": 0.0,
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
        }
        gemm_results["kv_cache_fetch"] = {
            "forward": 0.0,
            "backward": 0.0,
            "forward_gemm": 0.0,
            "forward_reduction": 0.0,
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
        }

        (
            pipeline_graph,
            pipeline_root,
            transformer_graph,
            transformer_forward_root,
            _,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            gemm_results=gemm_results,
            batch_size=batch_size,
            seq_len=prefill_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
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
            ffn_dim=self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1,   
            kp2=self.kp2,
            tp_mode=self.t,
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
            num_heads=self.num_heads,
            head_dim=head_dim,
            precision_bytes=self.kv_cache_precision,
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
