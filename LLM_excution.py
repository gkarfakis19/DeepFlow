import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Set

import simulate_LLM
from astrasim_lib.executor import run_astra_simulation_only_onepath
from simulate_LLM import Graph


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}





class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"
    
    
@dataclass
class ExecutionResult:
    total_time: float
    graph_root: Any
    mode: ExecutionMode


@dataclass
class TransformerTimings:
    forward: float
    backward: float
    
class PipelineGraphFlattener:
    """Expand pipeline transformer nodes into explicit tensor-parallel subgraphs."""

    def __init__(
        self,
        pipeline_graph: Graph,
        transformer_graph: Graph,
    ) -> None:
        if transformer_graph is None:
            raise ValueError("Transformer graph is required for flattening")

        transformer_cfg = getattr(transformer_graph, "transformer_cfg", None) or {}
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM template is missing")

        self.pipeline_graph = pipeline_graph
        self.transformer_graph = transformer_graph
        self._gemm_entries = list(gemm_entries)
        par_degree = transformer_graph.tp * transformer_graph.cp
        self._par_degree = max(1, int(par_degree))

        self._clone_cache: Dict[int, Any] = {}
        self._op_id_counter: int = 0

    def build(self, root: Any) -> Any:
        """Return a flattened clone of the provided pipeline root."""

        if root is None:
            raise ValueError("Pipeline root is required for flattening")
        return self._clone(root)

    def _clone(self, obj: Any) -> Any:
        if obj is None:
            return None

        obj_id = id(obj)
        if obj_id in self._clone_cache:
            return self._clone_cache[obj_id]

        if isinstance(obj, simulate_LLM.Node):
            if obj.name in {"transformer", "transformer_b"}:
                expanded = self._expand_transformer_node(obj)
                self._clone_cache[obj_id] = expanded
                return expanded

            if "linear_softmax" in obj.name:
                # for linear softmax we need to carefully look at hw id to choose.
                cloned = simulate_LLM.Node(
                    obj.name,
                    self._next_op_id(),
                    self._hw_id_for_rank(obj.hw_id, 0),
                    obj.duration,
                    fwd=obj.fwd,
                    is_kv_cache=getattr(obj, "is_kv_cache", False),
                    flops=getattr(obj, "flops", 0),
                    tensor_bytes=getattr(obj, "tensor_bytes", 0),
                )
            else:
                cloned = simulate_LLM.Node(
                    obj.name,
                    self._next_op_id(),
                    obj.hw_id,
                    obj.duration,
                    fwd=obj.fwd,
                    is_kv_cache=getattr(obj, "is_kv_cache", False),
                    flops=getattr(obj, "flops", 0),
                    tensor_bytes=getattr(obj, "tensor_bytes", 0),
                )

            self._clone_cache[obj_id] = cloned
            self._copy_metadata(obj, cloned)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned, child_clone)
            return cloned

        if isinstance(obj, simulate_LLM.Edge):
            cloned_edge = simulate_LLM.Edge(
                obj.name,
                self._next_op_id(),
                obj.duration,
                is_all_reduce=getattr(obj, "is_all_reduce", False),
                comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                comm_type=getattr(obj, "comm_type", None),
                participants=getattr(obj, "participants", 1),
                comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
            )
            self._clone_cache[obj_id] = cloned_edge
            self._copy_metadata(obj, cloned_edge)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_edge, child_clone)
            return cloned_edge

        if isinstance(obj, simulate_LLM.Data_batch):
            cloned_batch = simulate_LLM.Data_batch(obj.name, obj.batch_id, obj.duration)
            self._clone_cache[obj_id] = cloned_batch
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_batch, child_clone)
            return cloned_batch

        if isinstance(obj, simulate_LLM.Gradient):
            cloned_grad = simulate_LLM.Gradient(obj.name, self._next_op_id(), obj.hw_id, obj.duration)
            self._clone_cache[obj_id] = cloned_grad
            self._copy_metadata(obj, cloned_grad)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_grad, child_clone)
            return cloned_grad

        raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

    def _expand_transformer_node(self, node: simulate_LLM.Node) -> Tuple[Any, ...]:
        node_id = id(node)
        if node_id in self._clone_cache:
            cached_entry = self._clone_cache[node_id]
            if isinstance(cached_entry, (list, tuple)):
                return tuple(cached_entry)

        stage_id = getattr(node, "stage_id", node.hw_id)
        micro_batch = getattr(node, "micro_batch_index", None)
        layer_index = getattr(node, "layer_index", None)
        direction = getattr(node, "direction", "forward" if node.fwd else "backward")

        rank_heads: List[Any] = []
        rank_tails: List[Any] = []
        

        for tp_rank in range(self._par_degree):
            previous: Optional[Any] = None
            head: Optional[Any] = None
            hw_id = self._hw_id_for_rank(stage_id, tp_rank)

            gemm_iterable = self._gemm_entries
            if direction == "backward":
                gemm_iterable = list(reversed(self._gemm_entries))

            for gemm_idx, entry in enumerate(gemm_iterable):
                entry_name = entry.get("name", f"g{gemm_idx}")
                cfg = entry.get(direction, {})
                duration = cfg.get("duration")
                if duration is None:
                    raise ValueError(
                        f"Missing duration for transformer entry '{entry_name}' in direction '{direction}'"
                    )

                gemm_node = simulate_LLM.Node(
                    name=self._format_gemm_name(entry_name, direction, micro_batch, layer_index, tp_rank),
                    op_id=self._next_op_id(),
                    hw_id=hw_id,
                    duration=duration,
                    fwd=(direction == "forward"),
                    flops=cfg.get("flops", 0),
                    tensor_bytes=cfg.get("bytes", 0),
                )
                gemm_node.stage_id = stage_id
                gemm_node.tp_rank = tp_rank
                gemm_node.micro_batch_index = micro_batch
                gemm_node.layer_index = layer_index
                gemm_node.direction = direction

                if previous is not None:
                    previous.add_child(gemm_node)
                previous = gemm_node
                if head is None:
                    head = gemm_node

                for comm_key in cfg.get("comm_keys", []):
                    comm_edge = self._create_transformer_comm_edge(
                        comm_key,
                        hw_id,
                        stage_id,
                        micro_batch,
                        layer_index,
                        direction,
                        tp_rank,
                    )
                    previous.add_child(comm_edge)
                    previous = comm_edge

            if head is None:
                raise ValueError("Transformer expansion produced no GEMM nodes")

            rank_heads.append(head)
            rank_tails.append(previous or head)

        dp_children: List[Any] = []
        other_children: List[Any] = []

        for child in getattr(node, "children", []):
            comm_type = getattr(child, "comm_interconnect_type", None)
            if comm_type == "dp":
                dp_children.append(child)
            else:
                other_children.append(child)

        # Keep the main trunk pointing to the per-rank compute tails.
        downstream_parents: List[Any] = list(rank_tails)

        # Attach DP collectives as side branches from the compute tails, without
        # reparenting the trunk. This preserves the true cross-layer pipeline
        # edge between compute nodes for ET conversion.
        for child in dp_children:
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(rank_tails[0], child_clone) # only attach to the first tail for DP collectives

        # Non-DP edges (e.g., cross_layer) stay on the trunk so (parent, target)
        # compute → compute pipeline edges remain visible.
        for child in other_children:
            # Special-case marked cross_layer edges (set in original graph):
            # create one per TP rank and wire tail[r] -> cross_layer_r -> next_head[r].
            is_pipeline_edge = False
            if isinstance(child, simulate_LLM.Edge):
                comm_type = getattr(child, "comm_type", None)
                if comm_type == "pipeline":
                    is_pipeline_edge = True
            if is_pipeline_edge:
                # Determine per-rank byte size (ceil split)
                try:
                    total_bytes = int(getattr(child, "comm_size_bytes", 0))
                except Exception:
                    total_bytes = 0
                per_rank_bytes = int(math.ceil(float(total_bytes) / float(max(1, self._par_degree))))

                # Clone the original targets of this pipeline edge
                target_clones: List[Any] = []
                for tgt in getattr(child, "children", []):
                    tgt_clone = self._clone(tgt)
                    if tgt_clone is None:
                        continue
                    target_clones.append(tgt_clone)
                if not target_clones:
                    # No downstream target; skip safely
                    continue

                # For each TP rank, create its own pipeline edge and connect
                for r, tail in enumerate(rank_tails):
                    # Create rank-specific pipeline edge
                    edge_obj = simulate_LLM.Edge(
                        name=f"{getattr(child, 'name', '')}_rank{r}",
                        op_id=self._next_op_id(),
                        duration=0,
                        is_all_reduce=False,
                        comm_size_bytes=per_rank_bytes,
                        comm_type="pipeline",
                        participants=2,
                        comm_interconnect_type="lp",
                    )
                    edge_obj.is_cross_layer = True
                    tail.add_child(edge_obj)
                    # Also anchor to the compute node (two parents) for mapping clarity
                    last_compute = rank_heads[r]
                    # Find the nearest compute ancestor for this rank: walk back from tail if needed
                    compute_anchor = None
                    cur = tail
                    visited_ids = set()
                    while cur is not None and id(cur) not in visited_ids:
                        visited_ids.add(id(cur))
                        if isinstance(cur, simulate_LLM.Node):
                            compute_anchor = cur
                            break
                        parents = getattr(cur, "parents", [])
                        cur = parents[-1] if parents else None
                    if compute_anchor is not None and compute_anchor is not edge_obj:
                        compute_anchor.add_child(edge_obj)

                    # Connect to each cloned target, aligning ranks where possible
                    for tgt_clone in target_clones:
                        if isinstance(tgt_clone, (list, tuple)):
                            # Map by identity index when available
                            idx = r % len(tgt_clone)
                            edge_obj.add_child(tgt_clone[idx])
                        else:
                            edge_obj.add_child(tgt_clone)
                continue

            # Default path for non-pipeline children
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(downstream_parents, child_clone)

        heads_tuple = tuple(rank_heads)
        self._clone_cache[node_id] = heads_tuple
        return heads_tuple

    def _create_transformer_comm_edge(
        self,
        comm_key: str,
        hw_id: int,
        stage_id: int,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        direction: str,
        tp_rank: int,
    ) -> simulate_LLM.Edge:
        comm_info = self.transformer_graph.comm_metadata.get(comm_key, {})
        is_all_reduce = comm_info.get("type") == "all_reduce"

        comm_edge = self.transformer_graph.create_comm_edge(
            name=comm_key,
            op_id=self._next_op_id(),
            comm_key=comm_key,
            is_all_reduce=is_all_reduce,
            local_hw_id=hw_id,
        )
        comm_edge.stage_id = stage_id
        comm_edge.micro_batch_index = micro_batch
        comm_edge.layer_index = layer_index
        comm_edge.direction = direction
        comm_edge.tp_rank = tp_rank
        return comm_edge

    def _copy_metadata(self, source: Any, target: Any) -> None:
        for attr in (
            "micro_batch_index",
            "layer_index",
            "direction",
            "stage_id",
            "tp_rank",
            "is_kv_cache",
            "flops",
            "tensor_bytes",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _attach(self, parent: Any, child: Any) -> None:
        if parent is None or child is None:
            return

        if isinstance(parent, (list, tuple)):
            for item in parent:
                self._attach(item, child)
            return

        if isinstance(child, (list, tuple)):
            for item in child:
                self._attach(parent, item)
            return

        parent.add_child(child)

    def _format_gemm_name(
        self,
        base_name: str,
        direction: str,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        tp_rank: int,
    ) -> str:
        return f"{base_name}_{direction}_mb{micro_batch}_l{layer_index}_rank{tp_rank}"

    def _next_op_id(self) -> int:
        self._op_id_counter += 1
        return self._op_id_counter

    def _hw_id_for_rank(self, stage_id: int, tp_rank: int) -> int:
        stage_int = int(stage_id) if stage_id is not None else 0
        return stage_int * self._par_degree + tp_rank
    
    
class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,
        pipeline_graph: Graph,
        pipeline_root: Any,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_graph: Optional[Graph] = None,
        transformer_forward_root: Optional[Any] = None,
        transformer_backward_root: Optional[Any] = None,
    ) -> None:
        self.time_calc = time_calc
        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.interconnect_params = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.flattened_root: Optional[Any] = None

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            return self._run_full_astrasim_flattened()
        raise ValueError(f"Unsupported execution mode: {mode}")

    def _run_pipeline_with_analytical_comm(self, declared_mode: ExecutionMode) -> ExecutionResult:
        if declared_mode == ExecutionMode.HYBRID:
            filename = "/hybrid_graph"
            timed_root = self.pipeline_root
        else: # must be "ANALYTICAL"
            filename = "/analytical_graph"
            timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
                self.pipeline_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            
        generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        if generate_graphs:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                filename,
            )

        # Persist timed root for any downstream consumer
        self.pipeline_root = timed_root
        total_time = self.pipeline_graph.simulate(timed_root)
        return ExecutionResult(total_time=total_time, graph_root=timed_root, mode=declared_mode)

    def _run_hybrid(self) -> ExecutionResult:
        generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        if generate_graphs:
            self.transformer_graph.save_graph(
                self.transformer_forward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer",
            )
        transformer_time = self._run_transformer_astrasim(ExecutionMode.HYBRID)
        if generate_graphs:
            self.transformer_graph.save_graph(
                self.transformer_forward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer",
            )

        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)
        return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID)

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        transformer_time = self._run_transformer_astrasim(ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)
        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)

        dp_count = getattr(self.time_calc, "dp", 1) or 1
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for AstraSim execution")

        # Use hierarchical artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
        }
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        if _env_flag("DEEPFLOW_VISUALIZE_GRAPHS") and self.pipeline_root is not None:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                "/pipeline_graph_hierarchical",
            )

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            self.pipeline_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )
        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        return ExecutionResult(total_time=max_sec, graph_root=self.pipeline_root, mode=ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for flattening")
        if not self.transformer_graph:
            raise RuntimeError("Transformer graph metadata is required for flattening")

        flattener = PipelineGraphFlattener(
            pipeline_graph=self.pipeline_graph,
            transformer_graph=self.transformer_graph,
        )
        flattened_root = flattener.build(self.pipeline_root)
        if flattened_root is None:
            raise RuntimeError("Pipeline flattening produced an empty graph")

        self.time_calc.flattened_pipeline_root = flattened_root
        if _env_flag("DEEPFLOW_VISUALIZE_GRAPHS") and self.pipeline_root is not None:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                "/pipeline_graph_pre_flatten",
            )
            self.pipeline_graph.save_graph(
                flattened_root,
                self.time_calc.output_dir,
                "/pipeline_graph_post_flatten",
            )
        self.pipeline_root = flattened_root
        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_flattened")
        # dot = visualize_graph(flattened_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="png", cleanup=True)
        # except Exception as exc:  # pragma: no cover - visualization best-effort
        #     print(f"[WARN] Failed to render flattened pipeline graph: {exc}")

        unique_hw_ids = self._collect_hw_ids(flattened_root)
        if not unique_hw_ids:
            raise RuntimeError("Flattened pipeline graph exposes no compute nodes with hardware IDs")

        # Use flattened artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
        }
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            flattened_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings")

        expected_rank_count = effective_dp * len(unique_hw_ids)
        # Special case: If expected rank count is 1, then 2 is fine, but we prune the extra result
        # this is done, since astrasim backend only supports >1 ranks, so we generate extra fake result for that case.
        if expected_rank_count == 1:
            if len(per_rank_sec) > 2:
                raise RuntimeError(
                    "AstraSim rank count mismatch for flattened execution: "
                    f"expected {expected_rank_count}, got {len(per_rank_sec)}"
                )
            per_rank_sec = per_rank_sec[:1]

        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration")

        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        self.time_calc.flattened_astrasim_per_rank = per_rank_sec
        self.time_calc.flattened_astrasim_total = max_sec

        return ExecutionResult(
            total_time=max_sec,
            graph_root=flattened_root,
            mode=ExecutionMode.FULL_ASTRASIM_FLATTENED,
        )

    def _collect_hw_ids(self, root: Any) -> Set[int]:
        visited: Set[int] = set()
        hw_ids: Set[int] = set()

        def enqueue_children(obj: Any) -> None:
            for child in getattr(obj, "children", []):
                stack.append(child)

        stack: List[Any]
        if isinstance(root, (list, tuple)):
            stack = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if isinstance(obj, simulate_LLM.Node):
                hw_id = getattr(obj, "hw_id", None)
                if hw_id is not None and hw_id >= 0:
                    hw_ids.add(int(hw_id))

            enqueue_children(obj)

        return hw_ids



    def _run_transformer_astrasim(self, mode: ExecutionMode) -> Optional[TransformerTimings]:
        del mode  # mode currently unused but kept for signature consistency

        # Use hierarchical artifact directory when persisting artifacts for transformer simulation
        artifact_dir = self.time_calc.output_dir
        artifact_dir_fwd = artifact_dir
        artifact_dir_bwd = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")
            artifact_dir_fwd = os.path.join(artifact_dir, "fwd")
            artifact_dir_bwd = os.path.join(artifact_dir, "bwd")
            os.makedirs(artifact_dir_fwd, exist_ok=True)
            os.makedirs(artifact_dir_bwd, exist_ok=True)

        fwd_per_rank = None
        bwd_per_rank = None
        fwd_max = 0
        bwd_max = 0
        if self.transformer_forward_root:
            fwd_per_rank, fwd_max = run_astra_simulation_only_onepath(
                self.transformer_forward_root,
                self.time_calc,
                artifact_dir_fwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            )
        if self.transformer_backward_root:
            bwd_per_rank, bwd_max = run_astra_simulation_only_onepath(
                self.transformer_backward_root,
                self.time_calc,
                artifact_dir_bwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            )

        self.time_calc.transformer_astrasim_per_rank_forward = fwd_per_rank
        self.time_calc.transformer_astrasim_per_rank_backward = bwd_per_rank
        self.time_calc.transformer_astrasim_time_forward = fwd_max
        self.time_calc.transformer_astrasim_time_backward = bwd_max

        if fwd_max < 0 or bwd_max < 0:
            raise RuntimeError("AstraSim transformer execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max)

    def _apply_transformer_time(self, timings: TransformerTimings) -> None:
        if timings.forward < 0 or timings.backward < 0:
            raise ValueError("AstraSim transformer times must be positive")

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if "transformer_f" in comp_times:
                comp_times["transformer_f"] = timings.forward
            if "transformer_b" in comp_times:
                comp_times["transformer_b"] = timings.backward

        visited: Set[int] = set()
        roots: List[Any]
        if isinstance(self.pipeline_root, (list, tuple)):
            roots = list(self.pipeline_root)
        else:
            roots = [self.pipeline_root]

        for root in roots:
            self._assign_transformer_durations(root, visited, timings.forward, timings.backward)

    def _assign_transformer_durations(self, node: Any, visited: Set[int], forward_value: float, backward_value: float) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, simulate_LLM.Node):
            if node.name == "transformer":
                node.duration = forward_value
            elif node.name == "transformer_b":
                node.duration = backward_value

        for child in getattr(node, "children", []):
            self._assign_transformer_durations(child, visited, forward_value, backward_value)
