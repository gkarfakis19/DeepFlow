#!/usr/bin/env python3
"""
Compare DeepFlow annotated times vs roofline-calculated times for an ET file.

Assumes single-node execution where all compute nodes execute sequentially (simple addition).
"""

import sys
import csv
from pathlib import Path

# Add Chakra paths
repo_root = Path(__file__).resolve().parents[1]
chakra_pb_dir = repo_root / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"
chakra_utils_dir = repo_root / "astra-sim" / "extern" / "graph_frontend" / "chakra" / "src" / "third_party" / "utils"
sys.path.insert(0, str(chakra_pb_dir))
sys.path.insert(0, str(chakra_utils_dir))

import et_def_pb2 as pb
from protolib import decodeMessage as chakra_decode, openFileRd as chakra_open


# ==============================================================================
# ASTRASIM ROOFLINE IMPLEMENTATION (C++)
# From: astra-sim/astra-sim/workload/Workload.cc (issue_comp function)
# ==============================================================================
# void Workload::issue_comp(shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
#     if (!this->sys->roofline_enabled) {
#         throw std::runtime_error(
#             "Roofline model is not enabled for non-replay comp");
#     }
#
#     if (node->is_cpu_op()) {
#         throw std::runtime_error("Roofline is only available for GPU nodes");
#         return;
#     }
#
#     WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
#     wlhd->node_id = node->id();
#
#     double num_ops = static_cast<double>(node->num_ops<uint64_t>());
#     double tensor_size = static_cast<double>(node->tensor_size<uint64_t>());
#
#     // if tensor_size is 0 during roofline mode, this is an invalid node
#     if (tensor_size == 0) {
#         skip_invalid(node);
#         return;
#     }
#
#     double operational_intensity = num_ops / tensor_size;
#     double perf = sys->roofline->get_perf(operational_intensity);
#     double elapsed_time = static_cast<double>(node->num_ops()) / perf;  // sec
#     uint64_t runtime = static_cast<uint64_t>(elapsed_time * 1e9);  // sec -> ns
#     ...
# }
#
# From: astra-sim/astra-sim/system/Roofline.cc (get_perf function)
# ==============================================================================
# double Roofline::get_perf(double operational_intensity) {
#     return min(bandwidth * operational_intensity, peak_perf);
# }
# ==============================================================================


def roofline_time_ns(num_ops, tensor_size_bytes, bandwidth_gbps, peak_perf_tflops):
    """
    Calculate roofline time in nanoseconds using AstraSim's exact algorithm.

    Args:
        num_ops: Number of operations (FLOPs)
        tensor_size_bytes: Tensor size in bytes
        bandwidth_gbps: Memory bandwidth in GB/s
        peak_perf_tflops: Peak performance in TFLOPS

    Returns:
        Runtime in nanoseconds
    """
    if tensor_size_bytes == 0:
        # AstraSim skips nodes with zero tensor_size
        return 0

    # Operational intensity = FLOPs / bytes
    operational_intensity = float(num_ops) / float(tensor_size_bytes)

    # Convert units to match AstraSim (FLOPS and bytes/s)
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9  # GB/s -> bytes/s
    peak_perf_flops = peak_perf_tflops * 1e12  # TFLOPS -> FLOPS

    # Roofline performance: min(bandwidth * OI, peak_perf)
    # bandwidth is in bytes/s, OI is in FLOPS/byte, so bandwidth * OI = FLOPS
    achieved_perf_flops = min(bandwidth_bytes_per_sec * operational_intensity, peak_perf_flops)

    # Time = ops / perf (in seconds)
    elapsed_time_sec = float(num_ops) / achieved_perf_flops

    # Convert to nanoseconds
    runtime_ns = int(elapsed_time_sec * 1e9)

    return runtime_ns


def compare_et_timing(et_path, bandwidth_gbps, peak_perf_tflops, output_csv):
    """
    Compare annotated vs roofline timing for all compute nodes in an ET file.

    Args:
        et_path: Path to the .et file
        bandwidth_gbps: Memory bandwidth in GB/s
        peak_perf_tflops: Peak performance in TFLOPS
        output_csv: Path to output CSV file
    """

    print("Reading ET file: {}".format(et_path))
    print("Roofline parameters:")
    print("  Bandwidth: {:.2f} GB/s".format(bandwidth_gbps))
    print("  Peak perf: {:.2f} TFLOPS".format(peak_perf_tflops))
    print()

    nodes = []
    total_annotated_ns = 0
    total_roofline_ns = 0

    # Read all compute nodes from ET file using Chakra utilities
    fh = chakra_open(str(et_path))

    # Read metadata
    meta = pb.GlobalMetadata()
    chakra_decode(fh, meta)

    # Read nodes
    while True:
        node = pb.Node()
        if not chakra_decode(fh, node):
            break

        # Only process compute nodes
        if node.type != pb.COMP_NODE:
            continue

        # Extract node info
        node_id = node.id
        node_name = node.name
        dur_us_annotated = node.duration_micros or 0

        # Extract metadata
        num_ops = 0
        tensor_size = 0
        for attr in node.attr:
            if attr.name == 'num_ops':
                num_ops = attr.uint64_val
            elif attr.name == 'tensor_size':
                tensor_size = attr.uint64_val

        # Calculate roofline time
        roofline_ns = roofline_time_ns(num_ops, tensor_size, bandwidth_gbps, peak_perf_tflops)
        roofline_us = roofline_ns / 1000.0

        # Convert annotated time to ns for accumulation
        annotated_ns = dur_us_annotated * 1000

        # Accumulate totals
        total_annotated_ns += annotated_ns
        total_roofline_ns += roofline_ns

        nodes.append({
            'id': node_id,
            'name': node_name,
            'annotated_us': dur_us_annotated,
            'roofline_us': roofline_us,
            'num_ops': num_ops,
            'tensor_size': tensor_size,
            'total_annotated_us': total_annotated_ns / 1000.0,
            'total_roofline_us': total_roofline_ns / 1000.0,
        })

    fh.close()

    # Write CSV
    with open(str(output_csv), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'name', 'annotated_us', 'roofline_us',
            'total_annotated_us', 'total_roofline_us',
            'num_ops', 'tensor_size'
        ])
        writer.writeheader()
        writer.writerows(nodes)

    # Print summary
    total_annotated_sec = total_annotated_ns / 1e9
    total_roofline_sec = total_roofline_ns / 1e9

    print("Summary:")
    print("  Total compute nodes: {}".format(len(nodes)))
    print("  Total annotated time: {:.2f} s ({:.2f} ms)".format(total_annotated_sec, total_annotated_ns/1e6))
    print("  Total roofline time:  {:.2f} s ({:.2f} ms)".format(total_roofline_sec, total_roofline_ns/1e6))
    print("  Difference: {:.2f} s".format(total_roofline_sec - total_annotated_sec))
    print("  Ratio (roofline/annotated): {:.3f}x".format(total_roofline_sec/total_annotated_sec if total_annotated_sec > 0 else 0))
    print()
    print("CSV written to: {}".format(output_csv))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_et_timing.py <et_file> [bandwidth_gbps] [peak_perf_tflops]")
        print()
        print("Default parameters (from a100_80GB config):")
        print("  bandwidth_gbps = 2132.451262")
        print("  peak_perf_tflops = 311.86944")
        sys.exit(1)

    et_file = Path(sys.argv[1])
    bandwidth = float(sys.argv[2]) if len(sys.argv) > 2 else 2132.451262  # GB/s
    peak_perf = float(sys.argv[3]) if len(sys.argv) > 3 else 311.86944  # TFLOPS

    output_csv = et_file.with_suffix('.comp.csv')

    compare_et_timing(et_file, bandwidth, peak_perf, output_csv)
