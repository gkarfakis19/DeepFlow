import math
from heapq import heappush, heappop
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from graphviz import Digraph
import os

import graphviz_async
debug = False
BYTES_PER_GIB = 1024 ** 3

class Node:
    def __init__(self, name, op_id, hw_id, duration, fwd=True, *, is_kv_cache=False, flops=0, tensor_bytes=0):
        self.name = name
        self.op_id = op_id
        self.hw_id = hw_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.memory = 0  # memory usage
        self.fwd = fwd  # forward or backward
        self.scheduled = False
        self.is_kv_cache = is_kv_cache
        self.flops = flops  # FLOPs for roofline mode (optional, 0 if not provided)
        self.tensor_bytes = tensor_bytes  # Tensor size in bytes for roofline mode (optional, 0 if not provided)

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self):
        return f"Node({self.name},op={self.op_id},hw={self.hw_id},{round(self.duration, 2)})"
    
class Data_batch:
    def __init__(self, name, batch_id, duration):
        self.name = name
        self.batch_id = batch_id
        self.duration = duration
        self.done = False
        self.finish_time = -1
        self.parents = []
        self.children = []
        self.scheduled = False
        # self.memory = 0  # memory usage

    def add_child(self, obj):
        self.children.append(obj)
        obj.parents.append(self)

    def remove_self_from_children(self):
        for child in self.children:
            child.parents.remove(self)
        self.children = []

class Edge:
  def __init__(self, name, op_id, duration, is_all_reduce=False, comm_size_bytes=0, comm_type=None, participants=1, comm_interconnect_type=None):
    self.name = name
    self.op_id = op_id
    self.duration = duration
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.is_all_reduce = is_all_reduce
    self.scheduled = False
    self.comm_size_bytes = comm_size_bytes
    self.comm_type = comm_type
    self.participants = participants
    self.comm_interconnect_type = comm_interconnect_type

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)

  def __repr__(self):
      return f"Edge({self.name},op={self.op_id},{round(self.duration, 2)})"
      
class Gradient:
  def __init__(self, name, op_id, hw_id, duration):
    self.name = name
    self.op_id = op_id
    self.hw_id = hw_id
    self.duration = duration
  # "dp", "lp", "kp1", "kp2"
    self.done = False
    self.finish_time = -1
    self.parents = [] 
    self.children = []
    self.scheduled = False

  def add_child(self, obj):
      self.children.append(obj)
      obj.parents.append(self)

class Graph:

    def __init__(
        self,
        mode: str,
        dp: int,
        lp: int,
        tp: int,
        cp: int,
        comp_times: Dict[str, Any],
        comm_metadata: Dict[str, Any],
        misc_metadata: Dict[str, Any],
    ) -> None:
        self.mode = mode
        self.dp = int(dp)
        self.lp = int(lp)
        self.tp = int(tp)
        self.cp = int(cp)
        self.comp_times = comp_times or {}
        self.comm_metadata = comm_metadata or {}
        self.misc_metadata = misc_metadata or {}

        self.num_batch = self.misc_metadata.get("num_batch", 0)
        self.num_layer = self.misc_metadata.get("num_layer", 0)
        self.layer_per_device = max(1, math.ceil(self.num_layer / self.lp)) if self.lp else self.num_layer
        self.all_reduce = self.misc_metadata.get("all_reduce", "the end")

        self.transformer_cfg = self.comp_times.get("transformer", {})
        self.T_grad_transformer = self.comp_times.get("grad_transformer", 0.0)

    def _time(self, key: str, default: float = 0.0) -> float:
        value = self.comp_times.get(key)
        return float(value) if value is not None else default

    def _metadata(self, key: str, default_flops: float = 0.0, default_bytes: float = 0.0) -> tuple:
        """Extract FLOP and tensor byte metadata for a computation key.

        Args:
            key: Base key in comp_times (e.g., 'transformer_f')
            default_flops: Default FLOP count if not found
            default_bytes: Default tensor bytes if not found

        Returns:
            (flops, tensor_bytes) tuple
        """
        flops = self.comp_times.get(f"{key}_flops", default_flops)
        bytes_val = self.comp_times.get(f"{key}_bytes", default_bytes)
        return (float(flops) if flops is not None else default_flops,
                float(bytes_val) if bytes_val is not None else default_bytes)

    def create_comm_edge(self, name, op_id, comm_key, is_all_reduce=False, local_hw_id=None):
        """Create a communication edge with optional local computation node.

        Args:
            name: Edge name
            op_id: Operation ID
            comm_key: Key into self.comm_metadata dict

        Returns:
            The communication edge (connection point for graph building)
        """
        comm_data = self.comm_metadata[comm_key]

        # Create communication edge with metadata
        # print(f"Creating edge with name: {name}, size: {comm_data['size']}, type: {comm_data['type']}")
        comm_edge = Edge(
            name=name,
            op_id=op_id,
            duration=0,  # Will be filled in second pass
            comm_size_bytes=comm_data['size'],
            comm_type=comm_data['type'],
            participants=comm_data['participants'],
            comm_interconnect_type=comm_data['interconnect_type'],
            is_all_reduce=is_all_reduce
        )

        # If there's local computation time, create local node after edge
        local_comp_time = comm_data.get('local_comp_time', 0)
        if local_comp_time > 0:
            # TODO TODO TODO
            # We want to ignore this in the future. Skipping it directly so that we match astrasim for now.
            # FIX THIS (FIGURE OUT IF WE WANT TO SKIP OR NOT)
            local_comp_time = 0

        return comm_edge

    def extract_forward_graph(self, root: Any) -> Tuple[Any, Any]:
        """Detach backward branches and return separate forward/backward clones."""

        if root is None:
            return None, None

        clone_cache: Dict[str, Dict[int, Any]] = {"forward": {}, "backward": {}}
        creation_order: Dict[str, List[Any]] = {"forward": [], "backward": []}
        visited: Set[int] = set()

        def is_backward(node: Any) -> bool:
            direction = getattr(node, "direction", None)
            if direction is not None and str(direction).lower() == "backward":
                return True
            if hasattr(node, "fwd"):
                return bool(getattr(node, "fwd") is False)
            return False

        def classify(node: Any) -> str:
            return "backward" if is_backward(node) else "forward"

        def copy_optional_attrs(source: Any, target: Any) -> None:
            for attr in (
                "stage_id",
                "micro_batch_index",
                "layer_index",
                "direction",
                "tp_rank",
                "flatten_placeholder",
                "comm_interconnect_type",
                "comm_type",
                "comm_size_bytes",
                "participants",
                "is_cross_layer",
                "local_hw_id",
            ):
                if hasattr(source, attr):
                    setattr(target, attr, getattr(source, attr))

        def ensure_clone(obj: Any, cls: str) -> Any:
            obj_id = id(obj)
            cache = clone_cache[cls]
            cached = cache.get(obj_id)
            if cached is not None:
                return cached

            if isinstance(obj, Node):
                clone = Node(obj.name, obj.op_id, obj.hw_id, obj.duration, fwd=obj.fwd)
                clone.memory = getattr(obj, "memory", 0)
            elif isinstance(obj, Edge):
                clone = Edge(
                    obj.name,
                    obj.op_id,
                    obj.duration,
                    is_all_reduce=getattr(obj, "is_all_reduce", False),
                    comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                    comm_type=getattr(obj, "comm_type", None),
                    participants=getattr(obj, "participants", 1),
                    comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
                )
            elif isinstance(obj, Data_batch):
                clone = Data_batch(obj.name, obj.batch_id, obj.duration)
            elif isinstance(obj, Gradient):
                clone = Gradient(obj.name, obj.op_id, obj.hw_id, obj.duration)
            else:
                raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

            copy_optional_attrs(obj, clone)
            cache[obj_id] = clone
            creation_order[cls].append(clone)
            return clone

        def traverse(obj: Any) -> None:
            if obj is None:
                return

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    traverse(item)
                return

            obj_id = id(obj)
            cls = classify(obj)
            current_clone = ensure_clone(obj, cls)

            for child in getattr(obj, "children", []):
                child_cls = classify(child)
                child_clone = ensure_clone(child, child_cls)
                if cls == child_cls:
                    if child_clone not in current_clone.children:
                        current_clone.add_child(child_clone)

            if obj_id in visited:
                return
            visited.add(obj_id)

            for child in getattr(obj, "children", []):
                traverse(child)

        traverse(root)

        def clone_structure(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, (list, tuple)):
                cloned_items: List[Any] = []
                for item in obj:
                    cloned = clone_structure(item)
                    if cloned is not None:
                        cloned_items.append(cloned)
                if not cloned_items:
                    return None
                return tuple(cloned_items) if isinstance(obj, tuple) else cloned_items

            cls = classify(obj)
            if cls != "forward":
                return None
            return ensure_clone(obj, cls)

        forward_root = clone_structure(root)
        if forward_root is None:
            forward_root = ensure_clone(root, classify(root))

        candidates: List[Any] = []
        seen: Set[int] = set()
        for clone in creation_order["backward"]:
            clone_id = id(clone)
            if clone_id in seen:
                continue
            seen.add(clone_id)
            backward_parents = [p for p in getattr(clone, "parents", []) if classify(p) == "backward"]
            if backward_parents:
                continue
            candidates.append(clone)

        if not candidates:
            backward_root: Optional[Any] = None
        elif len(candidates) == 1:
            backward_root = candidates[0]
        else:
            backward_root = Data_batch("backward_root", -1, 0.0)
            for candidate in candidates:
                backward_root.add_child(candidate)

        return forward_root, backward_root

    def extract_backward_graph(self, root: Any) -> Any:
        """Return a backward-only clone of the provided graph root."""

        if root is None:
            return None

        def is_backward(node):
            direction = getattr(node, "direction", None)
            if direction is not None:
                return str(direction).lower() == "backward"
            if hasattr(node, "fwd"):
                return bool(getattr(node, "fwd") is False)
            for child in getattr(node, "children", []):
                if getattr(child, "direction", None) == "backward" or getattr(child, "fwd", True) is False:
                    return True
            return False


        visited: Set[int] = set()
        all_objects: List[Any] = []

        def collect(obj: Any) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
                return

            all_objects.append(obj)
            for child in getattr(obj, "children", []):
                collect(child)

        collect(root)

        backward_objects = [obj for obj in all_objects if is_backward(obj)]
        if not backward_objects:
            return None

        included_ids = {id(obj) for obj in backward_objects}

        def should_include(obj: Any) -> bool:
            if is_backward(obj):
                return True
            if getattr(obj, "comm_type", None):
                return True
            name = getattr(obj, "name", "")
            if isinstance(obj, Node) and "local_comp" in name:
                return True
            return False

        queue = deque(backward_objects)
        while queue:
            current = queue.popleft()
            for child in getattr(current, "children", []):
                child_id = id(child)
                if child_id in included_ids:
                    continue
                if should_include(child):
                    included_ids.add(child_id)
                    backward_objects.append(child)
                    queue.append(child)

        backward_ids = included_ids
        clone_cache: Dict[int, Any] = {}

        def ensure_clone(obj: Any) -> Any:
            obj_id = id(obj)
            cached = clone_cache.get(obj_id)
            if cached is not None:
                return cached

            if isinstance(obj, Node):
                clone = Node(obj.name, obj.op_id, obj.hw_id, obj.duration, fwd=obj.fwd)
                clone.memory = getattr(obj, "memory", 0)
            elif isinstance(obj, Edge):
                clone = Edge(
                    obj.name,
                    obj.op_id,
                    obj.duration,
                    is_all_reduce=getattr(obj, "is_all_reduce", False),
                    comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                    comm_type=getattr(obj, "comm_type", None),
                    participants=getattr(obj, "participants", 1),
                    comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
                )
            elif isinstance(obj, Data_batch):
                clone = Data_batch(obj.name, obj.batch_id, obj.duration)
            elif isinstance(obj, Gradient):
                clone = Gradient(obj.name, obj.op_id, obj.hw_id, obj.duration)
            else:
                raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

            for attr in (
                "stage_id",
                "micro_batch_index",
                "layer_index",
                "direction",
                "tp_rank",
                "flatten_placeholder",
                "comm_interconnect_type",
                "comm_type",
                "comm_size_bytes",
                "participants",
                "is_cross_layer",
                "local_hw_id",
            ):
                if hasattr(obj, attr):
                    setattr(clone, attr, getattr(obj, attr))

            clone_cache[obj_id] = clone
            return clone

        for obj in backward_objects:
            ensure_clone(obj)

        for obj in backward_objects:
            clone = clone_cache[id(obj)]
            for child in getattr(obj, "children", []):
                if id(child) not in backward_ids:
                    continue
                child_clone = ensure_clone(child)
                if child_clone not in clone.children:
                    clone.add_child(child_clone)

        root_candidates: List[Any] = []
        for obj in backward_objects:
            parents = [p for p in getattr(obj, "parents", []) if id(p) in backward_ids]
            if parents:
                continue
            root_candidates.append(clone_cache[id(obj)])

        if not root_candidates:
            return None
        # if len(root_candidates) == 1:
        #     return root_candidates[0]

        # backward_root = Data_batch("backward_root", -1, 0.0)
        # for candidate in root_candidates:
            # backward_root.add_child(candidate)
        return root_candidates[0]

    @staticmethod
    def _reset_execution_state(root: Any) -> None:
        """Clear scheduling metadata so the graph can be re-simulated."""
        if root is None:
            return

        visited: Set[int] = set()
        if isinstance(root, (list, tuple, set)):
            stack: List[Any] = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if hasattr(obj, "done"):
                setattr(obj, "done", False)
            if hasattr(obj, "scheduled"):
                setattr(obj, "scheduled", False)
            if hasattr(obj, "finish_time"):
                setattr(obj, "finish_time", -1)

            stack.extend(getattr(obj, "children", []))

    def convert_comm_sizes_to_times(self, roots, network_model, interconnect_params):
        """
        Args:
            network_model: NetworkModel instance for collective timing
            interconnect_params: Dict with bandwidth/latency for each type
                                {'dp': (ib, ll), 'lp': (ib, ll), 'kp1': (ib, ll), 'kp2': (ib, ll)}
        """
        def traverse_and_convert(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return
            visited.add(id(node))

            # Process children (edges and nodes)
            for child in node.children:
                # If it's an edge with communication size, convert to time
                if hasattr(child, 'comm_size_bytes') and child.comm_size_bytes > 0:
                    # Get the appropriate bandwidth/latency for this interconnect type
                    interconnect_type = child.comm_interconnect_type
                    if interconnect_type and interconnect_type in interconnect_params:
                        ib, ll = interconnect_params[interconnect_type]
                    else:
                        raise ValueError(f"Invalid interconnect type: {interconnect_type}") 

                    child.duration = network_model.collective(
                        kind=child.comm_type,
                        size_bytes=child.comm_size_bytes,
                        participants=child.participants,
                        ib=ib,
                        ll=ll,
                        local_bytes=0.0,
                        local_ops=0.0,
                        debug_label=f"{child.name}_conversion"
                    )
                    # print(f"Converted {child.name} size {child.comm_size_bytes} bytes to duration {child.duration:.6f} sec using {interconnect_type} (ib={ib}, ll={ll})")

                # Recursively process this child
                traverse_and_convert(child, visited)

        traverse_and_convert(roots)
        return roots

    def construct_fwd_bwd_graph(self, include_backward: bool = True):
        embedding_node = []
        data_batch_node = []
        softmax_node = []
        transformer_nodes = [[] for _ in range(self.num_batch)]  # transformer compute nodes per layer
        layer_entry_nodes = [[] for _ in range(self.num_batch)]  # lists of entry nodes per layer
        layer_exit_nodes = [[] for _ in range(self.num_batch)]   # transformer nodes per layer
        fetch_nodes = [[] for _ in range(self.num_batch)]        # kv-fetch nodes per layer

        embedding_b_time = self._time("embedding_b")
        linear_softmax_b_time = self._time("linear_softmax_b")
        transformer_b_time = self._time("transformer_b")
        linear_softmax_f_time = self._time("linear_softmax_f")
        linear_softmax_b_time = self._time("linear_softmax_b")
        transformer_f_time = self._time("transformer_f")
        transformer_b_time = self._time("transformer_b")
        embedding_f_time = self._time("embedding_f")
        embedding_b_time = self._time("embedding_b")
        cross_layer_time = self._time("cross_layer_f")
        kv_cache_fetch_time = self._time("kv_cache_fetch")
        kv_cache_store_time = self._time("kv_cache_store")
        fetch_overlap_enabled = bool(self.misc_metadata.get("kv_cache_fetch_overlap", False))

        # Extract metadata for nodes
        embedding_f_flops, embedding_f_bytes = self._metadata("embedding_f")
        embedding_b_flops, embedding_b_bytes = self._metadata("embedding_b")
        linear_softmax_f_flops, linear_softmax_f_bytes = self._metadata("linear_softmax_f")
        linear_softmax_b_flops, linear_softmax_b_bytes = self._metadata("linear_softmax_b")
        transformer_f_flops, transformer_f_bytes = self._metadata("transformer_f")
        transformer_b_flops, transformer_b_bytes = self._metadata("transformer_b")


        if include_backward:
            embedding_node_b = [[] for _ in range(self.num_batch)]
            softmax_node_b = [[] for _ in range(self.num_batch)]
            R_edge = [[] for _ in range(self.num_batch)]
            G_edge = [[] for _ in range(self.num_batch)]

            transformer_nodes_b = [[] for _ in range(self.num_batch)]  #
            for b in range(self.num_batch):
                transformer_nodes_b[b] = [[] for _ in range(self.num_layer)]

        op_id = 0  # operation ID, used to distinguish nodes and edges
        batch_id = 0  # batch ID, used to distinguish data batches
        
        data0 = Data_batch("data0", batch_id, 0)
        data_batch_node.append(data0)
        
        for i in range(1, self.num_batch):#create data batch node
            
            data_batch_node.append(Data_batch(f"data{i}", i, 0))
            data_batch_node[i-1].add_child(data_batch_node[i])

        for b in range(self.num_batch): #connect each data batch node with corresponding nodes
            linear_softmax = Node(f"linear_softmax{b}", op_id, self.lp-1, linear_softmax_f_time,
                                  flops=linear_softmax_f_flops, tensor_bytes=linear_softmax_f_bytes)
            op_id += 1
            softmax_node.append(linear_softmax)
            emb = Node(f"embedding{b}", op_id, 0, embedding_f_time,
                       flops=embedding_f_flops, tensor_bytes=embedding_f_bytes)      # hw_id = 0
            op_id += 1
            embedding_node.append(emb)
            data_batch_node[b].add_child(embedding_node[b])

            transformer_nodes[b] = []
            layer_entry_nodes[b] = []
            layer_exit_nodes[b] = []
            fetch_nodes[b] = []

            for l in range(self.num_layer):
                hw_id = min(l // self.layer_per_device , self.lp - 1)#assign hw_id for transformer
                transformer_node = Node("transformer", op_id, hw_id, transformer_f_time,
                                        flops=transformer_f_flops, tensor_bytes=transformer_f_bytes)
                transformer_node.micro_batch_index = b
                transformer_node.layer_index = l
                transformer_node.direction = "forward"
                transformer_node.stage_id = hw_id
                op_id += 1

                fetch_node = None
                if kv_cache_fetch_time > 0:
                    fetch_node = Node(
                        f"kv_cache_fetch_b{b}_l{l}",
                        op_id,
                        hw_id,
                        kv_cache_fetch_time,
                        fwd=True,
                        is_kv_cache=True,
                    )
                    fetch_node.micro_batch_index = b
                    fetch_node.layer_index = l
                    fetch_node.direction = "forward"
                    fetch_node.stage_id = hw_id
                    op_id += 1

                store_node = None
                if kv_cache_store_time > 0:
                    store_node = Node(
                        f"kv_cache_store_b{b}_l{l}",
                        op_id,
                        hw_id,
                        kv_cache_store_time,
                        fwd=True,
                        is_kv_cache=True,
                    )
                    store_node.micro_batch_index = b
                    store_node.layer_index = l
                    store_node.direction = "forward"
                    store_node.stage_id = hw_id
                    op_id += 1
                    transformer_node.add_child(store_node)

                if fetch_overlap_enabled and fetch_node is not None:
                    entry_nodes = [fetch_node, transformer_node]
                elif fetch_node is not None:
                    fetch_node.add_child(transformer_node)
                    entry_nodes = [fetch_node]
                else:
                    entry_nodes = [transformer_node]

                transformer_nodes[b].append(transformer_node)
                layer_entry_nodes[b].append(entry_nodes)
                layer_exit_nodes[b].append(transformer_node)
                fetch_nodes[b].append(fetch_node)

            for l in range(1, self.num_layer):

                prev_node = transformer_nodes[b][l-1]        # previous layer compute node
                curr_node  = transformer_nodes[b][l]            # current layer compute node
                prev_exit = layer_exit_nodes[b][l-1]
                curr_entries = layer_entry_nodes[b][l]
                prev_fetch = fetch_nodes[b][l-1]


                if prev_node.hw_id == curr_node.hw_id:
                    edge = Edge("cross_layer", op_id, 0, comm_type="pipeline")  # on same GPU
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')  # on different GPU
                op_id += 1

                prev_exit.add_child(edge)
                for entry in curr_entries:
                    edge.add_child(entry)
                if fetch_overlap_enabled and prev_fetch is not None:
                    prev_fetch.add_child(edge)

            first_entries = layer_entry_nodes[b][0]   # first layer entry nodes list
            primary_entry = first_entries[0]
            if primary_entry.hw_id == embedding_node[b].hw_id:
                edge = Edge("Emb_node0", op_id, 0, comm_type="pipeline")
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            embedding_node[b].add_child(edge) #connect embedding node and first transformer layer
            for entry in first_entries:
                edge.add_child(entry)


            last_exit = layer_exit_nodes[b][-1]  # last layer transformer node
            if last_exit.hw_id == softmax_node[b].hw_id:
                node_Softmax = Edge("node_Softmax", op_id, 0, comm_type="pipeline")  # same GPU
            else:
                node_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            last_exit.add_child(node_Softmax) #connect last layer and softmax layer
            node_Softmax.add_child(softmax_node[b])
            last_fetch = fetch_nodes[b][-1]
            if fetch_overlap_enabled and last_fetch is not None:
                last_fetch.add_child(node_Softmax)

            #add dependency edges
        for b in range(self.num_batch - 1):
            gpu_index = 0
            last_transformer_layer = []
            first_transformer_layer = []
            first_transformer_layer.append(0)
            for l in range(self.num_layer-1):

                if transformer_nodes[b][l].hw_id != transformer_nodes[b][l+1].hw_id: # check if on different GPUs
                    last_transformer_layer.append(l) # record last layer on each GPU
                    first_transformer_layer.append(l+1) # record first layer on each GPU
                    gpu_index += 1
                    if transformer_nodes[b][l].hw_id == 0: #if first pipeline stage
                        layer_exit_nodes[b][l].add_child(embedding_node[b+1]) # dependency: finish stage before next batch embedding starts
                    else:
                        next_entries = layer_entry_nodes[b+1][first_transformer_layer[gpu_index-1]]
                        for entry in next_entries:
                            layer_exit_nodes[b][l].add_child(entry)
                        # fetch nodes already connected to cross-layer edges above when overlap is enabled

            next_entries = layer_entry_nodes[b+1][first_transformer_layer[-1]]
            for entry in next_entries:
                softmax_node[b].add_child(entry)

        for db_node in data_batch_node:
            db_node.remove_self_from_children()

        if not include_backward:
            return embedding_node[0]

        for b in reversed(range(self.num_batch)): #connect each data batch node with corresponding nodes
            emb_b = Node("embedding_b", op_id, 0, embedding_b_time, fwd=False,
                         flops=embedding_b_flops, tensor_bytes=embedding_b_bytes)      # hw_id = 0
            op_id += 1
            embedding_node_b[b] = emb_b
            linear_softmax_b = Node("linear_softmax_b", op_id, self.lp-1, linear_softmax_b_time, fwd=False,
                                    flops=linear_softmax_b_flops, tensor_bytes=linear_softmax_b_bytes)
            op_id += 1
            softmax_node_b[b] = linear_softmax_b
            softmax_node[b].add_child(linear_softmax_b)


            for l in reversed(range(self.num_layer)):

                hw_id = min(l // self.layer_per_device , self.lp - 1)
                transformer_node_b = Node("transformer_b", op_id, hw_id, transformer_b_time, fwd=False,
                                          flops=transformer_b_flops, tensor_bytes=transformer_b_bytes)
                transformer_node_b.micro_batch_index = b
                transformer_node_b.layer_index = l
                transformer_node_b.direction = "backward"
                transformer_node_b.stage_id = hw_id
                op_id += 1
                transformer_nodes_b[b][l] = transformer_node_b

            for l in reversed(range(1, self.num_layer)):
                curr_node = transformer_nodes_b[b][l]         # current layer's qkv_proj.
                next_ffn2  = transformer_nodes_b[b][l-1]            # next layer's layernorm.
                if curr_node.hw_id == next_ffn2.hw_id:
                    edge = Edge("cross_layer", op_id, 0, comm_type="pipeline")  
                else:
                    edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
                op_id += 1

                curr_node.add_child(edge); edge.add_child(next_ffn2)

            qkv_0_b = transformer_nodes_b[b][0]     # first layer's qkv_proj
            if qkv_0_b.hw_id == emb_b.hw_id:
                edge = Edge("Emb_node0", op_id, 0, comm_type="pipeline")
            else:
                edge = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            qkv_0_b.add_child(edge)
            edge.add_child(emb_b)


            prev_layer_norm2 = transformer_nodes_b[b][self.num_layer-1] # last layer's layernorm2
            if prev_layer_norm2.hw_id == softmax_node_b[b].hw_id:
                layernorm_Softmax = Edge("layernorm2_Softmax", op_id, 0, comm_type="pipeline")  # same GPU
            else:
                layernorm_Softmax = self.create_comm_edge('cross_layer', op_id, 'cross_layer')
            op_id += 1
            softmax_node_b[b].add_child(layernorm_Softmax)
            layernorm_Softmax.add_child(prev_layer_norm2)
                

            # all-reduce
            if self.dp > 1:
                R_edge[b].append(self.create_comm_edge("embedding", op_id, "embedding", is_all_reduce=True))
                op_id += 1
                if self.all_reduce == "the end":
                    R_edge[b].append(
                        self.create_comm_edge(
                            "transformer",
                            op_id,
                            "transformer",
                            is_all_reduce=True,
                            local_hw_id=transformer_nodes_b[b][-1].hw_id,
                        )
                    )
                    op_id += 1

                    R_edge[b].append(self.create_comm_edge("softmax", op_id, "softmax", is_all_reduce=True))
                    op_id += 1
                # Attach All-Reduce Edges
                    softmax_node_b[b].add_child(R_edge[b][-1])
                    embedding_node_b[b].add_child(R_edge[b][0])
                    transformer_nodes_b[b][0].add_child(R_edge[b][1])

                elif self.all_reduce == "every layer":
                    for i in range(0, self.num_layer):
                        R_edge[b].append(
                            self.create_comm_edge(
                                "transformer",
                                op_id,
                                "transformer",
                                is_all_reduce=True,
                                local_hw_id=transformer_nodes_b[b][i].hw_id,
                            )
                        )
                        op_id += 1
                        # G_edge[b].append(Gradient("Grad_transformer", op_id, transformer_nodes_b[b][-1].hw_id, self.T_grad_transformer))
                        # op_id += 1
                        # R_edge[b][-1].add_child(G_edge[b][-1])

                    R_edge[b].append(self.create_comm_edge("softmax", op_id, "softmax", is_all_reduce=True))
                    op_id += 1
                # Attach All-Reduce Edges
                    softmax_node_b[b].add_child(R_edge[b][-1])
                    embedding_node_b[b].add_child(R_edge[b][0])
                    for i in range(0, self.num_layer):
                        transformer_nodes_b[b][i].add_child(R_edge[b][i + 1])
                else:
                    sys.exit("Invalid all_reduce option")
                    
                
                
        last_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs
        first_transformer_layer = [-1] * self.lp  # Initialize with -1 for all GPUs

        # first_transformer_layer.append(0)
        gpu_index = self.lp - 1
        for l in range(self.num_layer - 1, 0, -1):
            
            if transformer_nodes_b[0][l].hw_id != transformer_nodes_b[0][l-1].hw_id:  # Check if on different GPU
                # print("Layer ", l, " is on GPU ", transformer_nodes_b[0][l].hw_id)
                first_transformer_layer[gpu_index-1] = l-1  # Record first layer on each GPU
                last_transformer_layer[gpu_index] = l  # Record last layer on each GPU
                gpu_index -= 1
        # for id in range(self.lp):
            # print("GPU ", id, " first layer ", first_transformer_layer[id], " last layer ", last_transformer_layer[id])



        for b in range(self.num_batch-1, 0, -1):
            gpu_index = self.lp - 1
            

            for l in range(self.num_layer - 1, 0, -1):

                if transformer_nodes_b[b][l].hw_id != transformer_nodes_b[b][l-1].hw_id:  # Check if on different GPUs
                    # last_transformer_layer.append(l)  # Record last layer on each GPU
                    # first_transformer_layer.append(l-1)  # Record first layer on each GPU
                    
                    if transformer_nodes_b[b][l].hw_id == self.lp - 1:
                        transformer_nodes_b[b][l].add_child(softmax_node_b[b-1])  # Add dependency edge
                    else:
                        transformer_nodes_b[b][l].add_child(transformer_nodes_b[b-1][first_transformer_layer[gpu_index]])  # Add dependency edge
                    gpu_index -= 1
            # Ensure embedding_node_b[b] is connected to the correct transformer node
            # if first_transformer_layer:
            embedding_node_b[b].add_child(transformer_nodes_b[b-1][first_transformer_layer[0]])  # Add dependency edge

        return embedding_node[0]

    def construct_transformer_graph(self, direction: str = "both"):
        transformer_cfg = self.transformer_cfg
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM times not provided")

        tp_degree = self.tp

        root = Data_batch("transformer_root", 0, 0)
        op_id = 0

        for rank in range(tp_degree):
            previous = root

            if direction in {"forward", "both"}:
                for idx, entry in enumerate(gemm_entries):
                    entry_name = entry.get("name", f"g{idx}")
                    forward_cfg = entry.get("forward", {})
                    fwd_duration = forward_cfg.get("duration")
                    if fwd_duration is None:
                        raise ValueError("Transformer GEMM entry missing forward duration")

                    fwd_flops = forward_cfg.get("flops", 0)
                    fwd_bytes = forward_cfg.get("bytes", 0)

                    node = Node(
                        name=f"{entry_name}_fwd_rank{rank}",
                        op_id=op_id,
                        hw_id=rank,
                        duration=fwd_duration,
                        fwd=True,
                        flops=fwd_flops,
                        tensor_bytes=fwd_bytes,
                    )
                    op_id += 1
                    previous.add_child(node)
                    previous = node
                    comm_key = forward_cfg.get("comm_keys",None)
                    if comm_key:
                        if len(comm_key) > 1:
                            print(f"*CONSTRUCT INFO: Multiple comm keys for {entry_name}")
                            print(f"*CONSTRUCT INFO: Comm keys: {comm_key}")
                            raise ValueError(f"Multiple comm keys for {entry_name}")

                        comm_key = comm_key[0]
                        if comm_key not in self.comm_metadata:
                            raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                        comm_type = self.comm_metadata[comm_key]['type']
                        comm_edge = self.create_comm_edge(
                            name=comm_key,
                            op_id=op_id,
                            comm_key=comm_key,
                            is_all_reduce=(comm_type == 'all_reduce'),
                            local_hw_id=rank,
                        )
                        op_id += 1
                        previous.add_child(comm_edge)
                        previous = comm_edge

            if direction in {"backward", "both"}:
                for idx, entry in enumerate(reversed(gemm_entries)):
                    entry_name = entry.get("name", f"g{idx}")
                    backward_cfg = entry.get("backward", {})
                    bwd_duration = backward_cfg.get("duration")
                    if bwd_duration is None:
                        raise ValueError("Transformer GEMM entry missing backward duration")

                    bwd_flops = backward_cfg.get("flops", 0)
                    bwd_bytes = backward_cfg.get("bytes", 0)

                    node = Node(
                        name=f"{entry_name}_bwd_rank{rank}",
                        op_id=op_id,
                        hw_id=rank,
                        duration=bwd_duration,
                        fwd=False,
                        flops=bwd_flops,
                        tensor_bytes=bwd_bytes,
                    )
                    op_id += 1
                    previous.add_child(node)
                    previous = node

                    for comm_idx, comm_key in enumerate(backward_cfg.get("comm_keys", [])):
                        if comm_key not in self.comm_metadata:
                            raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
                        comm_type = self.comm_metadata[comm_key]['type']
                        comm_edge = self.create_comm_edge(
                            name=comm_key,
                            op_id=op_id,
                            comm_key=comm_key,
                            is_all_reduce=(comm_type == 'all_reduce'),
                            local_hw_id=rank,
                        )
                        op_id += 1
                        previous.add_child(comm_edge)
                        previous = comm_edge

        return root
        
    def simulate(self, root):
        time = 0
        counter = 0
        event_queue = []
        ready_list = []

        self._reset_execution_state(root)

        ready_list.append(root)
        root.scheduled = True
        ###find number of devices needed
        base_devices = max(1, int(self.lp) if self.lp else 1)
        max_hw_id = -1
        visited_nodes: Set[int] = set()
        stack = list(root if isinstance(root, (list, tuple)) else [root])
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            hw_id = getattr(node, "hw_id", None)
            if hw_id is not None:
                try:
                    hw_val = int(hw_id)
                except (TypeError, ValueError):
                    hw_val = None
                if hw_val is not None and hw_val >= 0:
                    max_hw_id = max(max_hw_id, hw_val)

            stack.extend(getattr(node, "children", []))

        if max_hw_id >= 0:
            base_devices = max(base_devices, max_hw_id + 1)

        GPU_list = [True for _ in range(base_devices)]
        data_list = [False for i in range(0, self.num_batch)]

        heappush(event_queue, (root.duration, counter, root))
        if debug:
            print("{} enqueued at time {} batch id {}".format(root.name, 0, root.batch_id))
        ready_list.remove(root)
        counter = counter + 1

        while len(event_queue) > 0:
            time, _, event = heappop(event_queue)
            event.done = True
            event.scheduled = False
            event.finish_time = time
            if debug:
                print("Event {} finished at time {}".format(event.name, time))

            for child in event.children:

                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                # if is_ready == True:
                if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                    ready_list.append(child)
                    if debug:
                        print("child {}  ready at time {} ".format(child.name, time))

            if isinstance(event, Node):
                if not getattr(event, "is_kv_cache", False):
                    GPU_list[event.hw_id] = True

                



            for event in ready_list[:]:
                enqueued = False
                if isinstance(event, Data_batch):
                    # if GPU_list[event.hw_id] == True:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{} enqueued at time".format(event.name,  time))
                    counter = counter + 1
                    data_list[event.batch_id] = True #data batch sent to gpu
                    # GPU_list[event.hw_id] = False
                    ready_list.remove(event)

                elif isinstance(event, Node): 
                    if getattr(event, "is_kv_cache", False):
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} (kv_cache) enqueued at time {}".format(event.name, event.op_id, time))
                        counter = counter + 1
                        ready_list.remove(event)
                    else:
                        if GPU_list[event.hw_id] == True:
                            new_time = time + event.duration
                            heappush(event_queue, (new_time, counter, event))
                            event.scheduled = True
                            enqueued = True
                            if debug:
                                print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                            counter = counter + 1
                            GPU_list[event.hw_id] = False
                            ready_list.remove(event)
                elif isinstance(event, Edge): 
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                    enqueued = True
                    counter = counter + 1
                    ready_list.remove(event)
                elif isinstance(event, Gradient):
                    # print("Gradient event")
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                    counter = counter + 1
                    ready_list.remove(event)

        return time
    
    def simulate_memory(self, root, memory_data, mode = "training", output_folder = "output_graph/", filename="memory_graph"):
        time = 0
        counter = 0
        event_queue = []
        ready_list = []

        self._reset_execution_state(root)

        
        ready_list.append(root)
        root.scheduled = True
        ###find number of devices needed
        base_devices = max(1, int(self.lp) if self.lp else 1)
        max_hw_id = -1
        visited_nodes: Set[int] = set()
        stack = list(root if isinstance(root, (list, tuple)) else [root])
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            hw_id = getattr(node, "hw_id", None)
            if hw_id is not None:
                try:
                    hw_val = int(hw_id)
                except (TypeError, ValueError):
                    hw_val = None
                if hw_val is not None and hw_val >= 0:
                    max_hw_id = max(max_hw_id, hw_val)

            stack.extend(getattr(node, "children", []))

        if max_hw_id >= 0:
            base_devices = max(base_devices, max_hw_id + 1)

        class _MemorySnapshot:
            def __init__(self, num_devices: int, output_dir: Optional[str], file_basename: str) -> None:
                self.static: List[float] = [0.0 for _ in range(num_devices)]
                self.current: List[float] = [0.0 for _ in range(num_devices)]
                self.peak: List[float] = [0.0 for _ in range(num_devices)]
                # self.activation_allocations: memory_data["activation_allocations"]
                self._log_files: List[Optional[Any]] = [None for _ in range(num_devices)]
                self._log_paths: List[Optional[str]] = [None for _ in range(num_devices)]
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    for gpu_idx in range(num_devices):
                        log_path = os.path.join(output_dir, f"{file_basename}_gpu_{gpu_idx}_memory_log.txt")
                        self._log_paths[gpu_idx] = log_path
                        log_file = open(log_path, "w", encoding="utf-8")
                        log_file.write(
                            "timestamp_s | action               | delta_gib   | current_gib | peak_gib    | details\n"
                        )
                        self._log_files[gpu_idx] = log_file

            def add_static(self, gpu_id: int, size_bytes: float, timestamp: float = 0.0) -> None:
                if size_bytes <= 0:
                    return
                self.static[gpu_id] += size_bytes
                self.current[gpu_id] += size_bytes
                self._update_peak(gpu_id)
                self._record_change(
                    gpu_id,
                    "static_allocate",
                    size_bytes,
                    timestamp,
                    details="static_mem",
                )

            def allocate_activation(self, gpu_id: int, node: Any, timestamp: float) -> None:
                if mode != "training":
                    size_bytes = memory_data.get("activation_mem_per_layer_inference", 0)
                else:
                    size_bytes = memory_data.get("activation_mem_per_layer", 0) 
                node_identifier = getattr(node, "op_id", None)
                if node_identifier is None:
                    node_identifier = id(node)

                if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
                    return
                self.current[gpu_id] += size_bytes
                details = "node={} op_id={} fwd={}".format(
                    getattr(node, "name", "unknown"),
                    getattr(node, "op_id", "N/A"),
                    getattr(node, "fwd", "N/A"),
                )
                self._record_change(gpu_id, "allocate_activation", size_bytes, timestamp, details=details)
                self._update_peak(gpu_id)

            def release_activation(self, gpu_id: int, node: Any, timestamp: float) -> None:
                size_bytes = memory_data.get("activation_mem_per_layer", 0)
                node_identifier = getattr(node, "op_id", None)
                if node_identifier is None:
                    node_identifier = id(node)
                
                if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
                    return
                self.current[gpu_id] = max(0.0, self.current[gpu_id] - size_bytes)
                details = "node={} op_id={} fwd={}".format(
                    getattr(node, "name", "unknown"),
                    getattr(node, "op_id", "N/A"),
                    getattr(node, "fwd", "N/A"),
                )
                self._record_change(gpu_id, "release_activation", -size_bytes, timestamp, details=details)

            def summary(self) -> List[Dict[str, float]]:
                summary: List[Dict[str, float]] = []
                for idx in range(len(self.current)):
                    static_gib = self.static[idx] / BYTES_PER_GIB
                    current_gib = self.current[idx] / BYTES_PER_GIB
                    peak_gib = self.peak[idx] / BYTES_PER_GIB
                    summary.append(
                        {
                            "gpu_id": idx,
                            "static_gib": static_gib,
                            "current_gib": current_gib,
                            "peak_gib": peak_gib,
                        }
                    )
                return summary

            def _update_peak(self, gpu_id: int) -> None:
                if self.current[gpu_id] > self.peak[gpu_id]:
                    self.peak[gpu_id] = self.current[gpu_id]

            def _record_change(
                self,
                gpu_id: int,
                action: str,
                delta_bytes: float,
                timestamp: float,
                *,
                details: str = "",
            ) -> None:
                if not self._log_files or gpu_id < 0 or gpu_id >= len(self._log_files):
                    return
                log_file = self._log_files[gpu_id]
                if log_file is None:
                    return

                delta_gib = delta_bytes / BYTES_PER_GIB
                current_gib = self.current[gpu_id] / BYTES_PER_GIB
                peak_gib = self.peak[gpu_id] / BYTES_PER_GIB
                line = "{:.6f} | {:<20} | {:>+.6f} | {:>+.6f} | {:>+.6f}".format(
                    timestamp,
                    action,
                    delta_gib,
                    current_gib,
                    peak_gib,
                )
                if details:
                    line = f"{line} | {details}"
                log_file.write(line + "\n")
                log_file.flush()

            def close(self) -> None:
                if not self._log_files:
                    return
                for handle in self._log_files:
                    if handle:
                        handle.close()

        def _is_transformer_block(node: Any) -> bool:
            """Return True when node represents a transformer compute block."""
            if not isinstance(node, Node):
                return False
            name = getattr(node, "name", "")
            if not isinstance(name, str):
                return False
            return "transformer" in name.lower()

        def _count_transformer_layers_per_device(root_obj: Any, num_devices: int) -> List[int]:
            layer_sets: List[Set[Any]] = [set() for _ in range(num_devices)]
            stack_local = list(root_obj if isinstance(root_obj, (list, tuple)) else [root_obj])
            visited_local: Set[int] = set()
            while stack_local:
                current = stack_local.pop()
                current_id = id(current)
                if current_id in visited_local:
                    continue
                visited_local.add(current_id)

                if isinstance(current, Node):
                    hw_id = getattr(current, "hw_id", None)
                    layer_idx = getattr(current, "layer_index", None)
                    if (
                        layer_idx is not None
                        and isinstance(hw_id, int)
                        and 0 <= hw_id < num_devices
                        and _is_transformer_block(current)
                        and getattr(current, "fwd", True)
                    ):
                        layer_sets[hw_id].add(layer_idx)

                stack_local.extend(getattr(current, "children", []))
            return [len(layer_set) for layer_set in layer_sets]

        memory_output_dir: Optional[str] = None
        if output_folder:
            memory_output_dir = os.path.join(output_folder, "memory-summary")

        transformer_layers_per_device = _count_transformer_layers_per_device(root, base_devices)
        static_mem_per_layer = memory_data.get("static_mem_per_layer", 0)
        weight_mem_per_layer = memory_data.get("weight_mem_per_layer", 0)
        total_tracked_layers = sum(transformer_layers_per_device)
        if total_tracked_layers == 0:
            transformer_layers_per_device = [1 for _ in range(base_devices)]

        GPU_list = [True for _ in range(base_devices)]
        data_list = [False for _ in range(0, self.num_batch)]
        memory_snapshot = _MemorySnapshot(base_devices, memory_output_dir, filename)
        
        for idx in range(base_devices):
            
            if mode == "inference":
                static_bytes = weight_mem_per_layer * transformer_layers_per_device[idx] if idx < len(transformer_layers_per_device) else weight_mem_per_layer
                memory_snapshot.add_static(idx, static_bytes, timestamp=0.0)
            elif mode == "training":
                static_bytes = static_mem_per_layer * transformer_layers_per_device[idx] if idx < len(transformer_layers_per_device) else static_mem_per_layer
                memory_snapshot.add_static(idx, static_bytes, timestamp=0.0)
            else:
                raise ValueError(f"Invalid mode '{mode}' for memory simulation")
            
        self.memory_monitor_snapshot = memory_snapshot

        heappush(event_queue, (root.duration, counter, root))
        if debug:
            print("{} enqueued at time {} batch id {}".format(root.name, 0, root.batch_id))
        ready_list.remove(root)
        counter = counter + 1

        while len(event_queue) > 0:
            time, _, event = heappop(event_queue)
            event.done = True
            event.scheduled = False
            event.finish_time = time
            if debug:
                print("Event {} finished at time {}".format(event.name, time))

            for child in event.children:

                is_ready = True
                max_time = -1
                for parent in child.parents:
                    if parent.done == False:
                        is_ready = False
                    else:
                        max_time = max(max_time, parent.finish_time)
                # if is_ready == True:
                if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                    ready_list.append(child)
                    if debug:
                        print("child {}  ready at time {} ".format(child.name, time))

            if isinstance(event, Node):
                if not getattr(event, "is_kv_cache", False):
                    GPU_list[event.hw_id] = True
                is_transformer_block = _is_transformer_block(event) #TODO: embedding and softmax layers
                if mode == "training":
                    if not event.fwd:
                        if is_transformer_block:
                            memory_snapshot.release_activation(event.hw_id, event, time)
                    elif event.fwd:
                        if is_transformer_block:
                            memory_snapshot.allocate_activation(event.hw_id, event, time)
                elif mode == "inference":
                    if event.fwd:
                        if is_transformer_block:
                            memory_snapshot.allocate_activation(event.hw_id, event, time)
                            memory_snapshot.release_activation(event.hw_id, event, time)

                
            for event in ready_list[:]:
                enqueued = False
                if isinstance(event, Data_batch):
                    # if GPU_list[event.hw_id] == True:
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{} enqueued at time".format(event.name,  time))
                    counter = counter + 1
                    data_list[event.batch_id] = True #data batch sent to gpu
                    # GPU_list[event.hw_id] = False
                    ready_list.remove(event)

                elif isinstance(event, Node): 
                    if getattr(event, "is_kv_cache", False):
                        new_time = time + event.duration
                        heappush(event_queue, (new_time, counter, event))
                        event.scheduled = True
                        enqueued = True
                        if debug:
                            print("{}.{} (kv_cache) enqueued at time {}".format(event.name, event.op_id, time))
                        counter = counter + 1
                        ready_list.remove(event)
                    else:
                        if GPU_list[event.hw_id] == True:
                            new_time = time + event.duration
                            heappush(event_queue, (new_time, counter, event))
                            event.scheduled = True
                            enqueued = True
                            if debug:
                                print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                            counter = counter + 1
                            GPU_list[event.hw_id] = False
                            ready_list.remove(event)
                elif isinstance(event, Edge): 
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                    enqueued = True
                    counter = counter + 1
                    ready_list.remove(event)
                elif isinstance(event, Gradient):
                    # print("Gradient event")
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    enqueued = True
                    if debug:
                        print("{}.{} enqueued at time {} at device {}".format(event.name, event.op_id, time, event.hw_id))
                    counter = counter + 1
                    ready_list.remove(event)

        summary = memory_snapshot.summary()
        memory_snapshot.close()
        self.memory_monitor_summary = summary
        peak_mem = max(entry["peak_gib"] for entry in summary) if summary else 0.0
        return time, peak_mem

    def save_graph(self, roots, output_folder = "output_graph/", filename="graph"):
        os.makedirs(output_folder, exist_ok=True)

        printstr = " | Graph saved to    %s%s.png" % (output_folder, filename)
        def _render_graph() -> None:
            dot_fw = visualize_graph(roots, filename=output_folder + filename)
            dot_fw.render(output_folder + filename, format="png", cleanup=True)

        graphviz_async.submit(f"{filename}.png", _render_graph, print_message=printstr)

# dedeepyo : 27-May-25 : Print DFS traversal of the graph.
def print_graph(root_nodes, visited=None):
    if visited is None:
        visited = set()

    for node in root_nodes:
        if node in visited:
            continue
        visited.add(node)

        node_type = "Node" if isinstance(node, Node) else "Edge" if isinstance(node, Edge) else "Data_batch" if isinstance(node, Data_batch) else "Gradient" if isinstance(node, Gradient) else "Unknown"
        print(f"{node_type}: {node.name}, op_id: {node.op_id}, hw_id: {node.hw_id}, duration: {node.duration}")

        for child in node.children:
            child_type = "Node" if isinstance(child, Node) else "Edge" if isinstance(child, Edge) else "Data_batch" if isinstance(child, Data_batch) else "Gradient" if isinstance(child, Gradient) else "Unknown"
            print(f"  --> {child_type}: {child.name}, op_id: {child.op_id}")

        # Recursively print the children
        print_graph(node.children, visited)

def total_duration(root, visited=None):
    if visited is None:
        visited = set()
    if root in visited:
        return 0
    visited.add(root)

    # Only add duration if it's a Node (not a Stage)
    duration = root.compute_time if isinstance(root, Node) else 0

    for child in root.children:
        duration += total_duration(child, visited)

    return duration

def visualize_graph(root, filename="graph", visited=None, dot=None):
    if visited is None:
        visited = set()
    if dot is None:
        dot = Digraph(comment='Computation Graph')

    if root in visited:
        return dot
    visited.add(root)
    def _format_duration(value: float) -> str:
        ms = value * 1e3
        if abs(ms) > 1000:
            return f"{value:.2f}s"
        return f"{ms:.2f}ms"

    if isinstance(root, Node):
        root_type = "Node"
        if getattr(root, "is_kv_cache", False):
            color = "mediumorchid"
        elif root.fwd:
            color = "lightblue"
        else:
            color = "lightcoral"
    elif isinstance(root, Data_batch):
        root_type = "Data_batch"
        color = "gray"
    elif isinstance(root, Edge):
        root_type = "Edge"
        if root.is_all_reduce == True:
            color = "green"
        else:
            color = "yellow"
    elif isinstance(root, Gradient):
        root_type = "Gradient"
        color = "white"

    node_id = str(id(root))
    if isinstance(root, Data_batch):
            label = f"{root.name}\n( batch_id={root.batch_id}, dur={_format_duration(root.duration)})"
    elif isinstance(root, Node):
            label = (
                f"{root.name}\n(op_id={root.op_id}, hw_id={root.hw_id}, "
                f"dur={_format_duration(root.duration)})"
            )
    elif isinstance(root, Edge):
            label = f"{root.name}\n(op_id={root.op_id}, dur={_format_duration(root.duration)})"
    elif isinstance(root, Gradient):
            label = (
                f"{root.name}\n(op_id={root.op_id}, hw_id={root.hw_id}, "
                f"dur={_format_duration(root.duration)})"
            )
    # color = "lightblue" if isinstance(root, Node) else "gray" if isinstance(root, Data_batch) else "lightgreen"

    dot.node(node_id, label=label, style='filled', fillcolor=color, shape='box')

    for child in root.children:
        child_id = str(id(child))
        if isinstance(child, Data_batch):
            child_label = f"{child.name}\n( batch_id={child.batch_id}, dur={_format_duration(child.duration)})"
        elif isinstance(child, Edge):
            child_label = f"{child.name}\n(op_id={child.op_id}, dur={_format_duration(child.duration)})"
        elif isinstance(child, Node):
            child_label = (
                f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, "
                f"dur={_format_duration(child.duration)})"
            )
        elif isinstance(child, Gradient):
            child_label = (
                f"{child.name}\n(op_id={child.op_id}, hw_id={child.hw_id}, "
                f"dur={_format_duration(child.duration)})"
            )

        if isinstance(child, Node) and getattr(child, "is_kv_cache", False):
            child_color = "mediumorchid"
        elif isinstance(child, Node) and child.fwd:
            child_color = "lightblue"
        elif isinstance(child, Node) and not child.fwd:
            child_color = "lightcoral"
        elif isinstance(child, Data_batch):
            child_color = "gray"
        elif isinstance(child, Edge) and child.is_all_reduce:
            child_color = "green"
        else:
            child_color = "white"

        dot.node(child_id, label=child_label, style='filled', fillcolor=child_color, shape='box')
        dot.edge(node_id, child_id)
        visualize_graph(child, filename, visited, dot)

    return dot


        
