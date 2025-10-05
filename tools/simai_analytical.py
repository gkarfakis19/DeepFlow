"""SimAI analytical workload generation helpers.

This module owns the non-trivial glue required to translate DeepFlow's
flattened pipeline graphs into the text-based workload schema consumed by
`SimAI_analytical`.  The wrapper keeps a very small surface area and defers the
heavy lifting to the classes below so the conversion logic can evolve without
ballooning ``tools/wrapper.py``.

The implementation is intentionally defensive: most of the data that SimAI
expects is *not* represented explicitly in the DeepFlow graph objects.  The
conversion code therefore performs a significant amount of bookkeeping and
validates the assumptions it makes along the way.  When an unsupported
configuration is detected (for example, metadata missing from the flattened
graph), a ``SimAIConversionError`` is raised with a descriptive message so the
wrapper can surface a clear diagnostic to the user instead of silently
producing nonsense.

At the time of writing the focus is on producing a *workable* baseline that can
be extended in follow-up changes.  The initial version intentionally limits the
scope to dense transformer training workloads and only records the aggregate
forward compute time alongside best-effort communication statistics.  The
weight-gradient and input-gradient columns in the SimAI workload are populated
with zeroes for now – this matches the current fidelity of the data we can
recover from the DeepFlow graphs without building a full backward pass model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import yaml

import config as df_config  # type: ignore
from time_calculation_LLM import PipelineGraphFlattener, TimeCalculationLLM  # type: ignore

import simulate_LLM  # type: ignore


class SimAIConversionError(RuntimeError):
    """Raised when we cannot construct a valid SimAI analytical workload."""


@dataclass(frozen=True)
class SimAIParallelConfig:
    """Parallelism metadata required by the SimAI workload header."""

    tensor_parallel: int
    expert_parallel: int
    pipeline_parallel: int
    virtual_pipeline: int
    data_parallel: int
    micro_batches: int
    pp_comm_bytes: int

    @property
    def all_gpus(self) -> int:
        """Return the total accelerator count implied by the parallelism."""

        return (
            max(1, self.tensor_parallel)
            * max(1, self.expert_parallel)
            * max(1, self.pipeline_parallel)
            * max(1, self.data_parallel)
        )

    @property
    def gradient_accumulation(self) -> int:
        """Expose a heuristic gradient-accumulation count for SimAI.

        SimAI's header exposes ``ga`` which corresponds to the number of
        micro-batches accumulated per data-parallel replica.  DeepFlow tracks
        the same concept via ``mb`` inside the scheduling parameters, so we use
        that value directly.
        """

        return max(1, self.micro_batches)


@dataclass(frozen=True)
class SimAIRecord:
    """Single row in the SimAI workload text file."""

    name: str
    dependency: int
    fwd_time_us: int
    fwd_comm: str
    fwd_bytes: int
    ig_time_us: int
    ig_comm: str
    ig_bytes: int
    wg_time_us: int
    wg_comm: str
    wg_bytes: int
    wg_update_time_us: int

    def to_line(self) -> str:
        """Render the record into the whitespace separated SimAI format."""

        fields: Sequence[object] = (
            self.name,
            self.dependency,
            self.fwd_time_us,
            self.fwd_comm,
            self.fwd_bytes,
            self.ig_time_us,
            self.ig_comm,
            self.ig_bytes,
            self.wg_time_us,
            self.wg_comm,
            self.wg_bytes,
            self.wg_update_time_us,
        )
        return "\t".join(str(field) for field in fields)


def _iter_graph(root: object) -> Iterator[object]:
    """Depth-first iteration over all reachable graph objects."""

    stack: List[object]
    if isinstance(root, (list, tuple)):
        stack = list(root)
    else:
        stack = [root]

    visited: set[int] = set()
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        yield obj
        for child in getattr(obj, "children", []):
            stack.append(child)


def _seconds_to_microseconds(value: float) -> int:
    """Convert a floating-point duration in seconds to rounded microseconds."""

    if not math.isfinite(value):
        return 0
    return int(round(value * 1_000_000))


_COMM_BASE_LABELS = {
    None: "NONE",
    "none": "NONE",
    "": "NONE",
    "all_reduce": "ALLREDUCE",
    "allreduce": "ALLREDUCE",
    "all_gather": "ALLGATHER",
    "allgather": "ALLGATHER",
    "reduce_scatter": "REDUCESCATTER",
    "reducescatter": "REDUCESCATTER",
    "all_to_all": "ALLTOALL",
    "alltoall": "ALLTOALL",
    "all_reduce_all_to_all": "ALLREDUCEALLTOALL",
}

_COMM_SUFFIXES = {
    None: "",
    "": "",
    "dp": "",
    "lp": "",
    "pipeline": "",
    "kp1": "_TP",
    "kp2": "_TP",
    "tp": "_TP",
    "ep": "_EP",
    "dp_ep": "_DP_EP",
}


@dataclass
class _LayerAccumulator:
    """Helper structure storing intermediate per-layer aggregates."""

    forward_compute_s: float = 0.0
    backward_compute_s: float = 0.0
    forward_comm_bytes: float = 0.0
    forward_comm_label: Optional[str] = None
    input_grad_comm_bytes: float = 0.0
    input_grad_comm_label: Optional[str] = None
    weight_grad_comm_bytes: float = 0.0
    weight_grad_comm_label: Optional[str] = None
    forward_tp_ranks: Set[int] = field(default_factory=set)
    backward_tp_ranks: Set[int] = field(default_factory=set)
    forward_comm_tp_ranks: Set[int] = field(default_factory=set)
    input_grad_comm_tp_ranks: Set[int] = field(default_factory=set)
    weight_grad_comm_tp_ranks: Set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _RecordKey:
    """Dictionary key describing a unique SimAI workload record."""

    stage_id: int
    micro_batch: int
    layer_index: Optional[int]
    alias: str


_RANK_SENTINEL = -1


class SimAIWorkloadBuilder:
    """Translate DeepFlow's flattened pipeline into SimAI records."""

    def __init__(
        self,
        *,
        flattened_root: object,
        parallel_spec: SimAIParallelConfig,
    ) -> None:
        self._root = flattened_root
        self._parallel = parallel_spec

    def build_records(self) -> List[SimAIRecord]:
        """Collect per-layer metrics and emit SimAI workload records.

        Returns:
            Ordered list of ``SimAIRecord`` instances describing the flattened
            pipeline.  Each record maps to one logical transformer layer for a
            specific pipeline stage and micro-batch combination.
        """

        accumulators: Dict[_RecordKey, _LayerAccumulator] = defaultdict(_LayerAccumulator)

        for obj in _iter_graph(self._root):
            key = _make_record_key(obj)
            if key is None:
                continue
            acc = accumulators[key]
            tp_rank = getattr(obj, "tp_rank", None)

            if isinstance(obj, simulate_LLM.Node):
                direction = getattr(obj, "direction", "forward" if obj.fwd else "backward")
                duration = float(getattr(obj, "duration", 0.0) or 0.0)
                if direction == "forward":
                    acc.forward_compute_s += duration
                    _register_rank(acc.forward_tp_ranks, tp_rank)
                else:
                    # We currently do not distinguish between weight-gradient and
                    # input-gradient compute – treat everything as part of the
                    # backward slice and expose it via the weight-grad column.
                    acc.backward_compute_s += duration
                    _register_rank(acc.backward_tp_ranks, tp_rank)
            elif isinstance(obj, simulate_LLM.Edge):
                comm_type = getattr(obj, "comm_type", None)
                if comm_type in {None, "pipeline"}:
                    continue
                direction = getattr(obj, "direction", "forward")
                label = _COMM_BASE_LABELS.get(str(comm_type).lower(), "NONE")
                suffix = _COMM_SUFFIXES.get(getattr(obj, "comm_interconnect_type", None), "")
                label = f"{label}{suffix}" if label != "NONE" else label
                size_bytes = float(getattr(obj, "comm_size_bytes", 0) or 0)
                if direction == "forward":
                    acc.forward_comm_bytes += size_bytes
                    acc.forward_comm_label = _merge_comm_label(acc.forward_comm_label, label)
                    _register_rank(acc.forward_comm_tp_ranks, tp_rank)
                elif direction == "input_grad":
                    acc.input_grad_comm_bytes += size_bytes
                    acc.input_grad_comm_label = _merge_comm_label(acc.input_grad_comm_label, label)
                    _register_rank(acc.input_grad_comm_tp_ranks, tp_rank)
                else:
                    acc.weight_grad_comm_bytes += size_bytes
                    acc.weight_grad_comm_label = _merge_comm_label(acc.weight_grad_comm_label, label)
                    _register_rank(acc.weight_grad_comm_tp_ranks, tp_rank)

        if not accumulators:
            raise SimAIConversionError(
                "Flattened DeepFlow graph did not expose any transformer layer metadata."
            )

        records: List[SimAIRecord] = []
        for key in sorted(accumulators.keys(), key=_record_sort_key):
            acc = accumulators[key]
            forward_div = _rank_count(acc.forward_tp_ranks)
            backward_div = _rank_count(acc.backward_tp_ranks)
            fwd_comm_div = _rank_count(acc.forward_comm_tp_ranks)
            ig_comm_div = _rank_count(acc.input_grad_comm_tp_ranks)
            wg_comm_div = _rank_count(acc.weight_grad_comm_tp_ranks)

            layer_fragment = (
                f"layer{key.layer_index:03d}" if key.layer_index is not None else key.alias
            )
            layer_name = f"stage{key.stage_id:02d}_{layer_fragment}_mb{key.micro_batch:02d}"
            fwd_bytes = _normalize_comm_bytes(acc.forward_comm_bytes, fwd_comm_div)
            ig_bytes = _normalize_comm_bytes(acc.input_grad_comm_bytes, ig_comm_div)
            wg_bytes = _normalize_comm_bytes(acc.weight_grad_comm_bytes, wg_comm_div)
            records.append(
                SimAIRecord(
                    name=layer_name,
                    dependency=-1,
                    fwd_time_us=_seconds_to_microseconds(acc.forward_compute_s / forward_div if acc.forward_compute_s else 0.0),
                    fwd_comm=_normalize_comm_label(acc.forward_comm_label, fwd_bytes),
                    fwd_bytes=fwd_bytes,
                    ig_time_us=0,
                    ig_comm=_normalize_comm_label(acc.input_grad_comm_label, ig_bytes),
                    ig_bytes=ig_bytes,
                    wg_time_us=_seconds_to_microseconds(acc.backward_compute_s / backward_div if acc.backward_compute_s else 0.0),
                    wg_comm=_normalize_comm_label(acc.weight_grad_comm_label, wg_bytes),
                    wg_bytes=wg_bytes,
                    wg_update_time_us=100,
                )
            )
        return records


def _merge_comm_label(existing: Optional[str], new_label: str) -> Optional[str]:
    """Keep track of the label used for aggregated communication edges."""

    if not new_label or new_label == "NONE":
        return existing
    if existing is None or existing == new_label:
        return new_label
    # Mixed collectives – fall back to NONE and let the caller know by returning
    # a neutral label.  The workload will still record the aggregate byte count.
    return "NONE"


def _normalize_comm_label(label: Optional[str], size_bytes: int) -> str:
    """Return a final SimAI label for the provided collective."""

    if not label or size_bytes <= 0:
        return "NONE"
    return label


def _normalize_comm_bytes(size_bytes: float, divisor: int) -> int:
    if size_bytes <= 0:
        return 0
    return int(round(size_bytes / max(1, divisor)))


def _register_rank(container: Set[int], rank: Optional[int]) -> None:
    if rank is None:
        container.add(_RANK_SENTINEL)
    else:
        container.add(int(rank))


def _rank_count(ranks: Set[int]) -> int:
    if not ranks:
        return 1
    actual = {rk for rk in ranks if rk >= 0}
    if actual:
        return max(1, len(actual))
    return 1


def _sanitize_alias(raw: str) -> str:
    if not raw:
        return "extra"
    sanitized = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(raw))
    sanitized = sanitized.strip('_') or 'extra'
    return sanitized[:40]


def _make_record_key(obj: object) -> Optional[_RecordKey]:
    stage_id = getattr(obj, "stage_id", getattr(obj, "hw_id", None))
    if stage_id is None:
        return None
    micro_batch = getattr(obj, "micro_batch_index", 0) or 0
    layer_index = getattr(obj, "layer_index", None)

    if layer_index is not None:
        alias = f"layer{int(layer_index):03d}"
        numeric_layer = int(layer_index)
    else:
        alias_source = getattr(obj, "name", None)
        if not alias_source:
            alias_source = f"op{getattr(obj, 'op_id', 0)}"
        alias = _sanitize_alias(alias_source)
        numeric_layer = None

    return _RecordKey(
        stage_id=int(stage_id),
        micro_batch=int(micro_batch),
        layer_index=numeric_layer,
        alias=alias,
    )


def _record_sort_key(key: _RecordKey) -> Tuple[int, float, int, str]:
    layer_sort = key.layer_index if key.layer_index is not None else math.inf
    return (key.micro_batch, layer_sort, key.stage_id, key.alias)


@dataclass
class SimAIAnalyticalArtifacts:
    """Container describing the outputs produced for SimAI analytical runs."""

    workload_path: Path
    busbw_path: Path
    records: List[SimAIRecord]
    parallel_config: SimAIParallelConfig


class SimAIAnalyticalRunner:
    """High-level orchestrator used by ``tools/wrapper.py``."""

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.simai_root = repo_root / "SimAI"

    def generate_artifacts(
        self,
        *,
        hardware_config: Path,
        model_config: Path,
        output_dir: Path,
    ) -> SimAIAnalyticalArtifacts:
        """Build the workload and bus-bandwidth files from the provided configs."""

        hw_cfg = df_config.parse_config(str(hardware_config), config_type="hardware")
        model_cfg = df_config.parse_config(str(model_config), config_type="LLM")

        tmp_dir = output_dir / "deepflow_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        time_calc = TimeCalculationLLM(hw_cfg, model_cfg, "LLM", output_dir=str(tmp_dir))

        batch_size = time_calc._effective_transformer_batch()
        seq_len = time_calc.seq_len
        hidden_dim = time_calc.hidden_dim
        num_heads = time_calc.num_heads
        ffn_mult = time_calc.ffn_mult
        ffn_dim = time_calc.hidden_dim * ffn_mult if ffn_mult else time_calc.ffn_dim
        vocab_size = time_calc.vocab_size

        gemm_results, node_breakdown = time_calc.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            seq_len,
            num_heads,
            ffn_dim,
        )

        (
            pipeline_graph,
            pipeline_root,
            transformer_graph,
            _tf_fwd,
            _tf_bwd,
            _interconnect,
        ) = time_calc._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            gemm_results=gemm_results,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
        )

        flattener = PipelineGraphFlattener(pipeline_graph=pipeline_graph, transformer_graph=transformer_graph)
        flattened_root = flattener.build(pipeline_root)

        parallel_spec = _derive_parallel_spec(time_calc, pipeline_graph)
        builder = SimAIWorkloadBuilder(flattened_root=flattened_root, parallel_spec=parallel_spec)
        records = builder.build_records()

        workload_path = output_dir / "deepflow_generated_workload.txt"
        self._write_workload(workload_path, parallel_spec, records)

        busbw_path = output_dir / "deepflow_busbw.yaml"
        self._write_busbw_file(time_calc, busbw_path)

        try:
            if tmp_dir.exists():
                for entry in tmp_dir.iterdir():
                    if entry.is_file():
                        entry.unlink(missing_ok=True)
        finally:
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

        return SimAIAnalyticalArtifacts(
            workload_path=workload_path,
            busbw_path=busbw_path,
            records=records,
            parallel_config=parallel_spec,
        )

    def _write_workload(
        self,
        path: Path,
        parallel_spec: SimAIParallelConfig,
        records: Sequence[SimAIRecord],
    ) -> None:
        """Persist the workload in the textual format consumed by SimAI."""

        header = (
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            f"model_parallel_NPU_group: {parallel_spec.tensor_parallel} "
            f"ep: {parallel_spec.expert_parallel} "
            f"pp: {parallel_spec.pipeline_parallel} "
            f"vpp: {parallel_spec.virtual_pipeline} "
            f"ga: {parallel_spec.gradient_accumulation} "
            f"all_gpus: {parallel_spec.all_gpus} "
            "checkpoints: 0 checkpoint_initiates: 0 "
            f"pp_comm {parallel_spec.pp_comm_bytes}"
        )

        with path.open("w", encoding="utf-8") as handle:
            handle.write(header)
            handle.write("\n")
            handle.write(str(len(records)))
            handle.write("\n")
            for record in records:
                handle.write(record.to_line())
                handle.write("\n")

    def _write_busbw_file(self, time_calc: TimeCalculationLLM, path: Path) -> None:
        """Emit a best-effort bus-bandwidth description.

        The DeepFlow hardware model exposes injection bandwidths per collective
        dimension (``IBD`` for data-parallel, ``IBK1``/``IBK2`` for tensor
        parallel).  SimAI expects the values in gigabytes per second.  When a
        dimension is inactive or the input bandwidth is zero, ``null`` is used
        so SimAI falls back to its internal defaults.
        """

        def _gb_per_sec(value: float) -> Optional[float]:
            if value is None:
                return None
            if not math.isfinite(value) or value <= 0:
                return None
            return value / 1e9

        busbw = {
            "generated_by": "deepflow-wrapper",
            "TP": {
                "allreduce": _gb_per_sec(time_calc.IBK1),
                "allgather": _gb_per_sec(time_calc.IBK1),
                "reducescatter": _gb_per_sec(time_calc.IBK1),
                "alltoall": _gb_per_sec(time_calc.IBK2),
            },
            "DP": {
                "allreduce": _gb_per_sec(time_calc.IBD),
                "allgather": _gb_per_sec(time_calc.IBD),
                "reducescatter": _gb_per_sec(time_calc.IBD),
                "alltoall": _gb_per_sec(time_calc.IBD),
            },
            "EP": {
                "allreduce": None,
                "allgather": None,
                "reducescatter": None,
                "alltoall": None,
            },
        }

        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(busbw, handle, sort_keys=True)


def _derive_parallel_spec(time_calc: TimeCalculationLLM, pipeline_graph: simulate_LLM.Graph) -> SimAIParallelConfig:
    """Construct the SimAI parallelism description from DeepFlow state."""

    dp = max(1, getattr(time_calc, "dp", 1))
    lp = max(1, getattr(time_calc, "lp", 1))
    kp1 = max(1, getattr(time_calc, "kp1", 1))
    kp2 = max(1, getattr(time_calc, "kp2", 1))
    ep = max(1, getattr(getattr(time_calc, "model", None), "expert_parallel_degree", 1))
    mb = max(1, getattr(time_calc, "mb", 1))
    tp_degree = max(1, kp1 * kp2)

    pp_comm = 0
    if isinstance(pipeline_graph.comm_metadata, dict):
        cross_layer = pipeline_graph.comm_metadata.get("cross_layer")
        if isinstance(cross_layer, dict):
            try:
                pp_comm = int(cross_layer.get("size", 0) or 0)
            except Exception:
                pp_comm = 0

    num_layers = getattr(time_calc, "num_layers", None)
    if num_layers is None:
        num_layers = getattr(getattr(time_calc, "model", None), "num_layers", None)
    if num_layers is None:
        num_layers = getattr(pipeline_graph, "num_layer", 1)

    return SimAIParallelConfig(
        tensor_parallel=tp_degree,
        expert_parallel=ep,
        pipeline_parallel=lp,
        virtual_pipeline=max(1, int(num_layers)),
        data_parallel=dp,
        micro_batches=mb,
        pp_comm_bytes=pp_comm,
    )


__all__ = [
    "SimAIAnalyticalRunner",
    "SimAIAnalyticalArtifacts",
    "SimAIConversionError",
]
