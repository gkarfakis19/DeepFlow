"""Visualization helpers for STG-generated Chakra ET bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set

IMPLICIT_DATA_EDGES = False

from graphviz import Digraph

from .et_utils import chakra_decode, chakra_open, pb


class ParsedET:
    def __init__(self) -> None:
        self.nodes: Dict[int, pb.Node] = {}
        self.outputs: Dict[str, int] = {}
        self.consumers: Dict[str, Set[int]] = {}


def _parse_et(path: Path) -> ParsedET:
    parsed = ParsedET()
    fh = chakra_open(str(path))
    try:
        meta = pb.GlobalMetadata()
        chakra_decode(fh, meta)
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            parsed.nodes[int(node.id)] = node
            for attr in node.attr:
                if attr.name == "outputs":
                    which = attr.WhichOneof("value")
                    if which in ("str_list", "string_list"):
                        values = (
                            list(attr.str_list.values)
                            if which == "str_list"
                            else list(attr.string_list.values)
                        )
                        for i in range(0, len(values), 2):
                            tensor_name = values[i]
                            parsed.outputs[tensor_name] = int(node.id)
                elif attr.name == "inputs":
                    which = attr.WhichOneof("value")
                    if which in ("str_list", "string_list"):
                        values = (
                            list(attr.str_list.values)
                            if which == "str_list"
                            else list(attr.string_list.values)
                        )
                        for i in range(0, len(values), 2):
                            tensor_name = values[i]
                            parsed.consumers.setdefault(tensor_name, set()).add(
                                int(node.id)
                            )
    finally:
        fh.close()
    return parsed


def _graph_for_et(parsed: ParsedET, name: str) -> Digraph:
    dot = Digraph(name=name)
    dot.graph_attr.update({"rankdir": "TB", "fontsize": "10"})
    type_color = {
        pb.COMP_NODE: "lightblue",
        pb.COMM_COLL_NODE: "palegreen",
        pb.COMM_SEND_NODE: "khaki",
        pb.COMM_RECV_NODE: "lightsalmon",
    }
    for node_id, node in parsed.nodes.items():
        node_type = pb.NodeType.Name(int(node.type))
        label_lines = [node.name or f"node_{node_id}", f"id={node_id}", node_type]
        if node.duration_micros:
            label_lines.append(f"dur={int(node.duration_micros)}us")
        num_ops = None
        tensor_sz = None
        for attr in node.attr:
            if attr.name == "num_ops":
                which = attr.WhichOneof("value")
                if which == "uint64_val":
                    num_ops = attr.uint64_val
                elif which == "int64_val":
                    num_ops = attr.int64_val
            elif attr.name == "tensor_size":
                which = attr.WhichOneof("value")
                if which == "uint64_val":
                    tensor_sz = attr.uint64_val
                elif which == "int64_val":
                    tensor_sz = attr.int64_val
        if num_ops is not None:
            label_lines.append(f"ops={num_ops}")
        if tensor_sz is not None:
            label_lines.append(f"tensor={tensor_sz}")
        color = type_color.get(node.type, "white")
        dot.node(
            str(node_id),
            "\n".join(label_lines),
            style="filled",
            fillcolor=color,
            shape="box",
            fontsize="9",
        )
    seen_edges: Set[tuple[str, str, str]] = set()
    if IMPLICIT_DATA_EDGES:
        for tensor, producer in parsed.outputs.items():
            for consumer in parsed.consumers.get(tensor, []):
                edge = (str(producer), str(consumer), "solid")
                if edge not in seen_edges:
                    dot.edge(edge[0], edge[1])
                    seen_edges.add(edge)
    for node_id, node in parsed.nodes.items():
        for dep in node.data_deps:
            edge = (str(dep), str(node_id), "solid")
            if edge not in seen_edges:
                dot.edge(edge[0], edge[1])
                seen_edges.add(edge)
        for dep in node.ctrl_deps:
            edge = (str(dep), str(node_id), "dashed")
            if edge not in seen_edges:
                dot.edge(edge[0], edge[1], style="dashed")
                seen_edges.add(edge)
    return dot


def render_stg_bundle(et_paths: Iterable[str], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: List[Path] = []
    for idx, et_path in enumerate(sorted(et_paths)):
        parsed = _parse_et(Path(et_path))
        dot = _graph_for_et(parsed, name=f"stg_et_{idx}")
        stem = output_dir / (Path(et_path).stem + "_stg")
        out_file = dot.render(str(stem), format="png", cleanup=True)
        rendered.append(Path(out_file))
    return rendered
