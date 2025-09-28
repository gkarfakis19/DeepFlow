"""
LLM Inference Simulation - Extended Graph Generation for Prefill + Decode

This module extends DeepFlow's LLM training simulation to support full inference
workflows including both prefill and autoregressive decode phases.
"""

import math
import os
import shutil
from typing import Any, Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass

from simulate_LLM import Graph
import LLM_util
from time_calculation_LLM import LLMExecutionDispatcher

@dataclass
class InferenceConfig:
    """Configuration for inference simulation parameters."""
    batch_size: int
    seq_len: int  # prefill sequence length
    decode_len: int  # number of decode steps
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    vocab_size: int
    num_layers: int
    dp: int = 1  # data parallel (acts as replica count during inference)
    lp: int = 1  # layer parallel
    kp1: int = 1  # tensor parallel dim 1
    kp2: int = 1  # tensor parallel dim 2
    tp_mode: str = "row_col"
    # Decode sampling configuration
    sample_every: int = 32  # Sample every N decode steps
    kv_cache_fetch_overlap: bool = False
@dataclass
class DecodeSample:
    """Represents a sampled decode step with execution results."""
    step_id: int
    current_seq_len: int
    execution_time: float
    graph_root: Any
    kv_cache_tokens: int


class DecodeGraph(Graph):
    """
    Graph builder for autoregressive decode phase.

    Extends the base Graph class to handle step-by-step token generation
    with evolving sequence lengths and KV-cache considerations.
    """

    def __init__(
        self,
        config: InferenceConfig,
        hw_config,
        model_config,
        time_calc_cls: Callable[..., Any],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.time_calc_cls = time_calc_cls
        self._decode_sample_root: Optional[str] = None

    def build_decode_graph(self) -> Tuple[float, List[DecodeSample]]:
        """
        Build decode phase using sample-based approach for efficiency.

        Instead of simulating every decode step, we sample at regular intervals
        and integrate between sample points using linear interpolation.

        Returns:
            Tuple of (total_decode_time, list_of_decode_samples)
        """
        # Determine decode steps we actually simulate
        sample_points = self._generate_sample_points()

        # Execute graphs at sample points
        decode_samples = []
        for step_id in sample_points:
            generated_tokens = step_id + 1
            total_seq_len = self.config.seq_len + generated_tokens

            gemm_shapes = LLM_util.process_decode_gemm_shapes(
                batch_size=self.config.batch_size,
                current_seq_len=total_seq_len,
                d_model=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                vocab_size=self.config.vocab_size,
                option="multiply_batch_into_m"
            )

            sample_time = self._execute_decode_step(
                step_id=step_id,
                total_seq_len=total_seq_len,
                gemm_shapes=gemm_shapes,
            )

            decode_samples.append(
                DecodeSample(
                    step_id=step_id,
                    current_seq_len=total_seq_len,
                    execution_time=sample_time,
                    graph_root=None,
                    kv_cache_tokens=total_seq_len,
                )
            )

        total_decode_time = self._integrate_decode_samples(decode_samples)

        return total_decode_time, decode_samples

    def _generate_sample_points(self) -> List[int]:
        """Generate decode step sample points based on sampling configuration."""
        sample_points = []

        # Always sample step 0 (first decode step)
        sample_points.append(0)

        # Sample at regular intervals
        for step_id in range(self.config.sample_every - 1, self.config.decode_len, self.config.sample_every):
            sample_points.append(step_id)

        # Always sample the last step if configured
        last_step = self.config.decode_len - 1
        if last_step not in sample_points:
            sample_points.append(last_step)

        return sorted(sample_points)

    def _execute_decode_step(
        self,
        *,
        step_id: int,
        total_seq_len: int,
        gemm_shapes: Dict[str, Tuple[int, ...]],
    ) -> float:
        """Execute decode step using appropriate DeepFlow execution mode."""

        if not self.hw_config or not self.model_config:
            raise RuntimeError("Hardware config and model config are required for decode step execution.")
        if not self.time_calc_cls:
            raise RuntimeError("time_calc_cls must be provided for decode execution.")

        temp_time_calc = self.time_calc_cls(
            hw_config=self.hw_config,
            model_config=self.model_config,
            mode="LLM",
            output_dir="./output_graph/",
        )

        def measure_gemm(name: str, shape: Tuple[int, ...]) -> Dict[str, float]:
            if len(shape) == 3:
                gemm_args = shape
            elif len(shape) == 4:
                batch, seq_len, k_dim, n_dim = shape
                gemm_args = (batch * seq_len, k_dim, n_dim)
            else:
                raise ValueError(f"Invalid GEMM shape for {name}: {shape}")

            gemm_time, reduction_time = temp_time_calc._distributed_gemm_forward(
                gemm_args,
                f"decode_{name}_f",
            )

            return {
                "forward": gemm_time + reduction_time,
                "backward": 0.0,
                "forward_gemm": gemm_time,
                "forward_reduction": reduction_time,
                "backward_gemm": 0.0,
                "backward_reduction": 0.0,
            }

        gemm_results = {
            name: measure_gemm(name, gemm_shapes[name])
            for name in (
                "qkv_proj",
                "attention_score",
                "attention_output",
                "output_proj",
                "ffn1",
                "ffn2",
            )
        }

        base_dir = temp_time_calc.output_dir.rstrip(os.sep)
        sample_dir = os.path.join(base_dir, "decode_samples", f"step_{step_id:04d}")
        root_dir = os.path.dirname(sample_dir)
        if self._decode_sample_root is None:
            self._decode_sample_root = root_dir
            if self._should_cleanup_decode_samples() and os.path.isdir(root_dir):
                shutil.rmtree(root_dir, ignore_errors=True)
        os.makedirs(sample_dir, exist_ok=True)
        prev_output_dir = temp_time_calc.output_dir
        temp_time_calc.output_dir = sample_dir

        (
            pipeline_graph,
            pipeline_root,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            interconnect_params,
        ) = temp_time_calc.prepare_decode_graphs(
            gemm_results=gemm_results,
            batch_size=self.config.batch_size,
            total_seq_len=total_seq_len,
            gemm_shapes=gemm_shapes,
        )

        dispatcher = LLMExecutionDispatcher(
            time_calc=temp_time_calc,
            pipeline_graph=pipeline_graph,
            pipeline_root=pipeline_root,
            interconnect_params=interconnect_params,
            transformer_graph=transformer_graph,
            transformer_forward_root=transformer_forward_root,
            transformer_backward_root=transformer_backward_root,
        )

        result = dispatcher.run(temp_time_calc.execution_mode)
        temp_time_calc.output_dir = prev_output_dir
        print(
            f"[decode] sample step {step_id:04d}: seq_len={total_seq_len}, "
            f"time={result.total_time:.6f}s, artifacts={sample_dir}"
        )
        return result.total_time

    def _integrate_decode_samples(self, samples: List[DecodeSample]) -> float:
        """
        Integrate execution times between sample points using linear interpolation.

        Since attention cost grows linearly, we can use trapezoid rule integration
        to get accurate total time from sparse samples.
        """
        if not samples:
            raise ValueError("No decode samples available for integration")

        total_time = 0.0

        if len(samples) == 1:
            only_sample = samples[0]
            total_time += only_sample.execution_time
            print(
                f"[decode] integration single sample {only_sample.step_id:04d}: "
                f"time={only_sample.execution_time:.6f}s"
            )
        else:
            seed_sample = samples[0]
            print(
                f"[decode] integration seed step {seed_sample.step_id:04d}: "
                "skipping explicit contribution to avoid double counting"
            )

            # Each segment below applies the trapezoidal rule between consecutive
            # sample points: width = step_gap and height = average runtime. This
            # already accounts for the seed sample's contribution via the first
            # segment (0.5 * step_gap * seed_time), so we avoid adding an extra
            # standalone 0.5 * seed_time term that previously double counted it.
            for idx in range(1, len(samples)):
                prev_sample = samples[idx - 1]
                sample = samples[idx]
                step_gap = sample.step_id - prev_sample.step_id
                if step_gap <= 0:
                    raise ValueError("Sample points must be strictly increasing")

                midpoint = 0.5 * (prev_sample.execution_time + sample.execution_time)
                segment_time = midpoint * step_gap
                total_time += segment_time
                print(
                    f"[decode] integration segment {prev_sample.step_id:04d}->{sample.step_id:04d}: "
                    f"width={step_gap}, midpoint={midpoint:.6f}s, contribution={segment_time:.6f}s"
                )

        last_sample = samples[-1]
        remaining_steps = self.config.decode_len - (last_sample.step_id + 1)
        if remaining_steps > 0:
            tail_time = remaining_steps * last_sample.execution_time
            total_time += tail_time
            print(
                f"[decode] integration tail from {last_sample.step_id:04d} covering {remaining_steps} steps: "
                f"contribution={tail_time:.6f}s"
            )

        print(
            f"[decode] total interpolated decode time: {total_time:.6f}s "
            f"from {len(samples)} samples"
        )

        return total_time

    def _should_cleanup_decode_samples(self) -> bool:
        flag = os.environ.get("DEEPFLOW_VISUALIZE_GRAPHS")
        if flag is None:
            return False
        return flag.strip().lower() not in {"", "0", "false", "no"}



class InferenceEngine:
    """
    Main orchestrator for complete LLM inference simulation.

    Manages the transition from prefill to decode phases and coordinates
    the overall inference execution using existing DeepFlow infrastructure.
    """

    def __init__(
        self,
        config: InferenceConfig,
        hw_config=None,
        model_config=None,
        *,
        time_calc_cls: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.prefill_graph: Optional[Graph] = None
        self.decode_graph: Optional[DecodeGraph] = None
        self.time_calc_cls = time_calc_cls


    def _build_decode_graph(self) -> Tuple[float, List[DecodeSample]]:
        """
        Build decode phase using sample-based approach with proper DeepFlow integration.

        Returns:
            Tuple of (total_decode_time, decode_samples)
        """
        if self.time_calc_cls is None:
            raise RuntimeError("InferenceEngine requires time_calc_cls for decode graph building.")

        self.decode_graph = DecodeGraph(
            config=self.config,
            mode="inference",
            dp=self.config.dp,
            lp=self.config.lp,
            kp1=self.config.kp1,
            kp2=self.config.kp2,
            tp_mode=self.config.tp_mode,
            comp_times={},
            comm_metadata={},
            misc_metadata={"kv_cache_fetch_overlap": self.config.kv_cache_fetch_overlap},
            hw_config=self.hw_config,
            model_config=self.model_config,
            time_calc_cls=self.time_calc_cls,
        )

        return self.decode_graph.build_decode_graph()
