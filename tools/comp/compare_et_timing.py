#!/usr/bin/env python3
"""
Compare DeepFlow ET files against AstraSim roofline expectations.

Legacy mode:
    python compare_et_timing.py <et_file> [bandwidth_gbps] [peak_perf_tflops]
        - Generates <et_file>.comp.csv summarising annotated vs roofline time.

Comparison mode:
    python compare_et_timing.py --deepflow <annotated.et> --ablation <roofline.et> [--stg <stg.et>] [--output <csv>]
        - Always writes individual *.comp.csv files for the supplied DeepFlow ETs.
        - When an STG ET is provided, additionally emits a combined CSV aligning
          DeepFlow, DeepFlow (ablation) and STG compute nodes. Only STG nodes with
          op_type in {M, E, A} are considered.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Add Chakra paths
repo_root = Path(__file__).resolve().parents[1]
chakra_pb_dir = repo_root / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"
chakra_utils_dir = repo_root / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src" / "third_party" / "utils"
sys.path.insert(0, str(chakra_pb_dir))
sys.path.insert(0, str(chakra_utils_dir))

import et_def_pb2 as pb  # type: ignore  # noqa: E402
from protolib import decodeMessage as chakra_decode  # type: ignore  # noqa: E402
from protolib import openFileRd as chakra_open  # type: ignore  # noqa: E402


DEFAULT_BANDWIDTH_GBPS = 2132.451262
DEFAULT_PEAK_TFLOPS = 311.86944

ALLOWED_STG_OP_TYPES = {"M", "E", "A"}

ALLOWED_DEEPFLOW_BASES = {
    "attention_output_backward",
    "attention_output_forward",
    "attention_score_backward",
    "attention_score_forward",
    "embedding0",
    "embedding_b",
    "ffn1_backward",
    "ffn1_forward",
    "ffn2_backward",
    "ffn2_forward",
    "output_proj_backward",
    "output_proj_forward",
    "pt_attention_scale_softmax_backward",
    "pt_attention_scale_softmax_forward",
    "pt_layernorm1_backward",
    "pt_layernorm1_forward",
    "pt_layernorm2_backward",
    "pt_layernorm2_forward",
    "pt_residual1_backward",
    "pt_residual1_forward",
    "pt_residual2_backward",
    "pt_residual2_forward",
    "qkv_proj_backward",
    "qkv_proj_forward",
}

STG_MAPPING: Dict[str, List[str]] = {
    "embedding0": ["mb0.in_emb.y"],
    "embedding_b": ["mb0.in_emb.dw", "mb0.in_emb.dx"],
    "ffn1_backward": [
        "mb0.transformer.{layer}.ffn.dwgate",
        "mb0.transformer.{layer}.ffn.dx000",
    ],
    "ffn1_forward": ["mb0.transformer.{layer}.ffn.xgate"],
    "ffn2_backward": [
        "mb0.transformer.{layer}.ffn.dwdown",
        "mb0.transformer.{layer}.ffn.dxgate",
    ],
    "ffn2_forward": ["mb0.transformer.{layer}.ffn.xdown1"],
    "output_proj_backward": [
        "mb0.transformer.{layer}.mha.dwo",
        "mb0.transformer.{layer}.mha.dattn",
    ],
    "output_proj_forward": ["mb0.transformer.{layer}.mha.o1"],
    "pt_layernorm1_backward": [
        "mb0.transformer.{layer}.post_attn_norm.dy",
        "mb0.transformer.{layer}.post_attn_norm.dx",
    ],
    "pt_layernorm1_forward": ["mb0.transformer.{layer}.post_attn_norm.y"],
    "pt_layernorm2_backward": [
        "mb0.transformer.{layer}.input_norm.dy",
        "mb0.transformer.{layer}.input_norm.dx",
    ],
    "pt_layernorm2_forward": ["mb0.transformer.{layer}.input_norm.y"],
    "pt_residual1_forward": ["mb0.transformer.{layer}.ffn_res.y"],
    "pt_residual2_forward": ["mb0.transformer.{layer}.mha_res.y"],
    "qkv_proj_backward": [
        "mb0.transformer.{layer}.mha.dwqkv",
        "mb0.transformer.{layer}.mha.dx1",
    ],
    "qkv_proj_forward": ["mb0.transformer.{layer}.mha.qkv"],
}


@dataclass
class NodeInfo:
    node_id: int
    name: str
    num_ops: int
    tensor_size: int
    roofline_ns: int
    annotated_us: float
    op_type: Optional[str] = None
    layer: Optional[int] = None
    base: Optional[str] = None
    full_base: Optional[str] = None


# ==============================================================================
# ASTRASIM ROOFLINE IMPLEMENTATION (C++)
# ==============================================================================

def roofline_time_ns(num_ops: int, tensor_size_bytes: int, bandwidth_gbps: float, peak_perf_tflops: float) -> int:
    """
    Calculate roofline time in nanoseconds using AstraSim's algorithm.
    """
    if tensor_size_bytes == 0:
        return 0

    operational_intensity = float(num_ops) / float(tensor_size_bytes)

    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
    peak_perf_flops = peak_perf_tflops * 1e12

    achieved_perf_flops = min(bandwidth_bytes_per_sec * operational_intensity, peak_perf_flops)
    elapsed_time_sec = float(num_ops) / achieved_perf_flops
    return int(elapsed_time_sec * 1e9)


def _attr_value(attr) -> Optional[str]:
    which = attr.WhichOneof("value")
    if not which:
        return None
    value = getattr(attr, which)
    if isinstance(value, bytes):
        return value.decode()
    return value


def load_nodes_basic(et_path: Path, bandwidth_gbps: float, peak_perf_tflops: float) -> List[NodeInfo]:
    nodes: List[NodeInfo] = []
    fh = chakra_open(str(et_path))
    meta = pb.GlobalMetadata()
    chakra_decode(fh, meta)

    while True:
        node = pb.Node()
        if not chakra_decode(fh, node):
            break
        if node.type != pb.COMP_NODE:
            continue

        num_ops = 0
        tensor_size = 0
        op_type: Optional[str] = None
        for attr in node.attr:
            value = _attr_value(attr)
            if attr.name == "num_ops" and value is not None:
                num_ops = int(value)
            elif attr.name == "tensor_size" and value is not None:
                tensor_size = int(value)
            elif attr.name == "op_type" and value is not None:
                op_type = str(value)

        annotated_us = float(node.duration_micros or 0)
        roofline_ns = roofline_time_ns(num_ops, tensor_size, bandwidth_gbps, peak_perf_tflops)

        nodes.append(
            NodeInfo(
                node_id=int(node.id),
                name=node.name,
                num_ops=num_ops,
                tensor_size=tensor_size,
                roofline_ns=roofline_ns,
                annotated_us=annotated_us,
                op_type=op_type,
            )
        )

    fh.close()
    return nodes


def extract_deepflow_layer(name: str) -> Optional[int]:
    match = re.search(r"_l(\d+)", name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def extract_deepflow_base(name: str) -> str:
    base = name
    if "_mb0" in base:
        base = base.split("_mb0")[0]
    elif "_rank" in base:
        base = base.split("_rank")[0]
    base = re.sub(r"_[0-9]+$", "", base)
    return base


def extract_stg_metadata(name: str) -> tuple[Optional[int], str, str]:
    core = name.split("@")[0]
    layer: Optional[int] = None
    remainder = core
    if core.startswith("mb0.transformer."):
        parts = core.split(".")
        if len(parts) >= 3:
            try:
                layer = int(parts[2])
            except ValueError:
                layer = None
            remainder = ".".join(parts[3:])
    return layer, remainder, core


def prepare_deepflow_nodes(nodes: Iterable[NodeInfo]) -> List[NodeInfo]:
    prepared: List[NodeInfo] = []
    for node in nodes:
        base = extract_deepflow_base(node.name)
        if ALLOWED_DEEPFLOW_BASES and base not in ALLOWED_DEEPFLOW_BASES:
            continue
        node.base = base
        node.layer = extract_deepflow_layer(node.name)
        prepared.append(node)
    return prepared


STG_SKIP_SUBSTRINGS = ("_sharded_",)


def prepare_stg_nodes(nodes: Iterable[NodeInfo]) -> List[NodeInfo]:
    prepared: List[NodeInfo] = []
    for node in nodes:
        if node.op_type not in ALLOWED_STG_OP_TYPES:
            continue
        if not node.name.startswith("mb0."):
            continue
        if any(substr in node.name for substr in STG_SKIP_SUBSTRINGS):
            continue
        layer, remainder, full_base = extract_stg_metadata(node.name)
        node.layer = layer
        node.base = remainder
        node.full_base = full_base
        prepared.append(node)
    return prepared


def build_stg_pool(nodes: Iterable[NodeInfo]) -> Dict[str, List[NodeInfo]]:
    pool: Dict[str, List[NodeInfo]] = defaultdict(list)
    for node in nodes:
        key = node.full_base or node.base or node.name
        pool[key].append(node)
    return pool


def match_stg_nodes(df_node: NodeInfo, stg_pool: Dict[str, List[NodeInfo]]) -> List[NodeInfo]:
    patterns = STG_MAPPING.get(df_node.base or "", [])
    matched: List[NodeInfo] = []
    for pattern in patterns:
        key = pattern
        if "{layer}" in pattern:
            if df_node.layer is None:
                continue
            key = pattern.format(layer=df_node.layer)
        nodes = stg_pool.get(key)
        if nodes:
            matched_node = nodes.pop(0)
            matched.append(matched_node)
            if not nodes:
                stg_pool.pop(key, None)
    return matched


def create_combined_rows(
    deepflow_nodes: List[NodeInfo],
    ablation_map: Dict[str, NodeInfo],
    stg_pool: Dict[str, List[NodeInfo]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    df_roof_cum_ns = 0.0
    df_annot_cum_ns = 0.0
    stg_roof_cum_ns = 0.0

    for node in deepflow_nodes:
        ablation_node = ablation_map.get(node.name)
        roofline_ns = ablation_node.roofline_ns if ablation_node else node.roofline_ns
        df_roof_cum_ns += roofline_ns
        annotated_ns = node.annotated_us * 1000.0
        df_annot_cum_ns += annotated_ns

        rows.append(
            {
                "deepflow_op": node.name,
                "stg_op": "",
                "deepflow_num_ops": node.num_ops,
                "deepflow_tensor_bytes": node.tensor_size,
                "deepflow_roofline_us": roofline_ns / 1000.0,
                "deepflow_annotated_us": node.annotated_us,
                "stg_num_ops": "",
                "stg_tensor_bytes": "",
                "stg_roofline_us": "",
                "total_deepflow_roofline_us": df_roof_cum_ns / 1000.0,
                "total_deepflow_annotated_us": df_annot_cum_ns / 1000.0,
                "total_stg_roofline_us": stg_roof_cum_ns / 1000.0,
            }
        )

        for stg_node in match_stg_nodes(node, stg_pool):
            stg_roof_cum_ns += stg_node.roofline_ns
            rows.append(
                {
                    "deepflow_op": "",
                    "stg_op": stg_node.name,
                    "deepflow_num_ops": "",
                    "deepflow_tensor_bytes": "",
                    "deepflow_roofline_us": "",
                    "deepflow_annotated_us": "",
                    "stg_num_ops": stg_node.num_ops,
                    "stg_tensor_bytes": stg_node.tensor_size,
                    "stg_roofline_us": stg_node.roofline_ns / 1000.0,
                    "total_deepflow_roofline_us": df_roof_cum_ns / 1000.0,
                    "total_deepflow_annotated_us": df_annot_cum_ns / 1000.0,
                    "total_stg_roofline_us": stg_roof_cum_ns / 1000.0,
                }
            )

    unused_stg_keys = [key for key, remaining in stg_pool.items() if remaining]
    if unused_stg_keys:
        print(
            "[WARN] Unmapped STG nodes present (showing up to 5): "
            + ", ".join(unused_stg_keys[:5])
        )

    return rows


def write_combined_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "deepflow_op",
        "stg_op",
        "deepflow_num_ops",
        "deepflow_tensor_bytes",
        "deepflow_roofline_us",
        "deepflow_annotated_us",
        "stg_num_ops",
        "stg_tensor_bytes",
        "stg_roofline_us",
        "total_deepflow_roofline_us",
        "total_deepflow_annotated_us",
        "total_stg_roofline_us",
    ]
    print(f"Writing combined CSV: {output_csv}")
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_basic_csv(nodes: List[NodeInfo], output_csv: Path) -> None:
    total_annot_ns = 0.0
    total_roof_ns = 0.0
    rows: List[Dict[str, object]] = []

    for node in nodes:
        total_annot_ns += node.annotated_us * 1000.0
        total_roof_ns += node.roofline_ns
        rows.append(
            {
                "id": node.node_id,
                "name": node.name,
                "annotated_us": node.annotated_us,
                "roofline_us": node.roofline_ns / 1000.0,
                "total_annotated_us": total_annot_ns / 1000.0,
                "total_roofline_us": total_roof_ns / 1000.0,
                "num_ops": node.num_ops,
                "tensor_size": node.tensor_size,
            }
        )

    if output_csv.exists():
        print(f"Wiping existing CSV: {output_csv}")
        output_csv.unlink()

    print(f"Writing CSV: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "name",
                "annotated_us",
                "roofline_us",
                "total_annotated_us",
                "total_roofline_us",
                "num_ops",
                "tensor_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_annot_sec = total_annot_ns / 1e9
    total_roof_sec = total_roof_ns / 1e9

    print("Summary:")
    print(f"  Total compute nodes: {len(rows)}")
    print(f"  Total annotated time: {total_annot_sec:.2f} s ({total_annot_ns/1e6:.2f} ms)")
    print(f"  Total roofline time:  {total_roof_sec:.2f} s ({total_roof_ns/1e6:.2f} ms)")
    diff = total_roof_sec - total_annot_sec
    ratio = total_roof_sec / total_annot_sec if total_annot_sec > 0 else 0
    print(f"  Difference: {diff:.2f} s")
    print(f"  Ratio (roofline/annotated): {ratio:.3f}x")
    print(f"CSV written to: {output_csv}")


def run_single_mode(et_path: Path, bandwidth: float, peak: float) -> None:
    print(f"Reading ET file: {et_path}")
    print(f"Roofline parameters: bandwidth={bandwidth:.2f} GB/s, peak={peak:.2f} TFLOPS\n")
    nodes = load_nodes_basic(et_path, bandwidth, peak)
    output_csv = et_path.with_suffix(".comp.csv")
    write_basic_csv(nodes, output_csv)


def run_compare_mode(
    deepflow_path: Path,
    ablation_path: Path,
    stg_path: Optional[Path],
    bandwidth: float,
    peak: float,
    combined_output: Optional[Path],
) -> None:
    print(f"[Compare] DeepFlow annotated ET: {deepflow_path}")
    print(f"[Compare] DeepFlow ablation ET:  {ablation_path}")
    if stg_path:
        print(f"[Compare] STG ET:                {stg_path}")
    print(f"[Compare] Roofline parameters: bandwidth={bandwidth:.2f} GB/s, peak={peak:.2f} TFLOPS\n")

    deepflow_nodes_raw = load_nodes_basic(deepflow_path, bandwidth, peak)
    ablation_nodes_raw = load_nodes_basic(ablation_path, bandwidth, peak)
    deepflow_csv = deepflow_path.with_suffix(".comp.csv")
    ablation_csv = ablation_path.with_suffix(".comp.csv")

    write_basic_csv(deepflow_nodes_raw, deepflow_csv)
    write_basic_csv(ablation_nodes_raw, ablation_csv)

    if not stg_path:
        print("No STG ET supplied; combined CSV not generated.")
        return

    stg_nodes_raw = load_nodes_basic(stg_path, bandwidth, peak)
    deepflow_nodes = prepare_deepflow_nodes(deepflow_nodes_raw)
    ablation_nodes = prepare_deepflow_nodes(ablation_nodes_raw)
    ablation_map = {node.name: node for node in ablation_nodes}
    stg_nodes = prepare_stg_nodes(stg_nodes_raw)
    stg_csv = stg_path.with_suffix(".comp.csv")
    write_basic_csv(stg_nodes, stg_csv)
    stg_pool = build_stg_pool(stg_nodes)

    rows = create_combined_rows(deepflow_nodes, ablation_map, stg_pool)
    output_csv = combined_output or deepflow_path.with_name(f"{deepflow_path.stem}__deepflow_stg_compare.csv")
    write_combined_csv(rows, output_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DeepFlow ET timing vs AstraSim roofline expectations.")
    parser.add_argument("et", nargs="?", help="Single ET file for legacy summary output.")
    parser.add_argument("--deepflow", type=Path, help="DeepFlow ET with annotated timings.")
    parser.add_argument("--ablation", type=Path, help="DeepFlow ablation ET (roofline mode).")
    parser.add_argument("--stg", type=Path, help="Optional STG ET for tri-source comparison.")
    parser.add_argument("--output", type=Path, help="Optional output path for combined CSV.")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_GBPS, help="Memory bandwidth in GB/s (default: A100 80GB).")
    parser.add_argument("--peak", type=float, default=DEFAULT_PEAK_TFLOPS, help="Peak performance in TFLOPS (default: A100 80GB).")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.deepflow and args.ablation:
        run_compare_mode(
            deepflow_path=args.deepflow,
            ablation_path=args.ablation,
            stg_path=args.stg,
            bandwidth=args.bandwidth,
            peak=args.peak,
            combined_output=args.output,
        )
        return

    if args.et:
        run_single_mode(Path(args.et), args.bandwidth, args.peak)
        return

    parser.error("Either provide a single ET path or --deepflow/--ablation pair (with optional --stg).")


if __name__ == "__main__":
    main()
