#!/usr/bin/env python3
"""
Parallelism sweep utility for DeepFlow LLM configurations.

Update the global configuration section below to point at the desired hardware
and model config files and to tailor the parallelism search space. The tool
enumerates every combination, filters those whose total GPU count falls inside
the configured bounds, evaluates runtime with DeepFlow, and plots a scatter
chart of accuracy (by default, 1 / runtime) versus GPU count with horizontal
jitter to avoid overlap.
"""

from __future__ import print_function

import argparse
import ast
import copy
import itertools
import math
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from time_calculation_LLM import TimeCalculationLLM
from LLM_util import process_gemm_shapes

import seaborn as sns
import numpy as np
import pandas as pd
import math
import numpy as np
from matplotlib.patches import Polygon

# -----------------------------------------------------------------------------
# Global configuration knobs (no CLI)
# -----------------------------------------------------------------------------

# Paths to the baseline configuration files
HARDWARE_CONFIG_PATH = "configs/hardware-config/a100_80GB_faraz.yaml"
MODEL_CONFIG_PATH = "configs/model-config/Llama3.1-405B.yaml"

# Parallelism values to sweep (dense grid). Edit to suit your search space.
# Keys should match the entries under the YAML "parallelism" section.
PARALLELISM_SWEEP = {
    "tp": [2**i for i in range(0, 9)],
    "cp": [2**i for i in range(0, 9)],
    "dp": [2**i for i in range(0, 9)],
    "lp": [2**i for i in range(0, 9)],
}

# Optional knobs that still live inside the parallelism section but do not
# affect GPU counts. Leave empty if you do not want to explore them.
OTHER_PARALLELISM_OPTIONS = {
    "tp_sp": [True],
}

# GPU count filter: only evaluate combinations whose TP*CP*DP*LP fall inside
# this inclusive range.
GPU_COUNT_MIN = 64
GPU_COUNT_MAX = 1024

# When True, discard configurations whose tp*cp product is not a square power of two.
ENFORCE_SQUARE_TP_CP = False

# Select the y-axis metric for the scatter plot and report. Accepted values:
#   "runtime"     -> plot raw runtime in seconds (lower is better)
#   "performance" -> plot 1 / runtime (higher is better)
PLOT_METRIC = "runtime"

# Random seed for reproducible jitter in the scatter plot.
PLOT_JITTER_SEED = 1234
# Maximum absolute horizontal jitter (in GPU units).
PLOT_JITTER_WIDTH = 0.175

# Default output artefacts
PLOT_OUTPUT_PATH = "tools/parallelism_sweep.png"
PLOT_MFU_OUTPUT_PATH = "tools/parallelism_sweep_mfu.png"
REPORT_OUTPUT_PATH = "tools/parallelism_sweep.tsv"

# AstraSim cache handling within DeepFlow (mirrors run_perf default options).
ASTRA_CACHE_MODE = "NO_CACHE"  # Options: NO_CACHE, CACHE_READONLY, CACHE_READWRITE

# Plotting behaviour toggles
MEM_AWARE_FILTER = False  # When True, skip memory-violating configurations in plots.

# Maximum number of parallel worker processes (set <= available CPUs - 1). Set to 1 to disable multiprocessing.
MAX_WORKERS = 100


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def add_rgb_ternary_legend(ax,
                           corner_labels=("R = log2(tp+cp)",
                                          "G = log2(lp)",
                                          "B = log2(dp)"),
                           gamma=0.85,
                           inset_xywh=(0.72, 0.58, 0.24, 0.24),  # Axes fraction: x, y, w, h
                           n=120):
    """
    Draw a tiny triangular legend that shows how the RGB mix works.
    - Uses barycentric blending inside an equilateral triangle.
    - 'gamma' should match your color gamma used in the scatter.
    - inset_xywh is (x0,y0,w,h) in axes-relative coords.
    """
    # Inset axis
    iax = ax.inset_axes(inset_xywh, transform=ax.transAxes)
    iax.set_aspect("equal")
    iax.set_axis_off()

    # Equilateral triangle vertices (R, G, B corners)
    A = np.array([0.0, 0.0])                        # R corner
    B = np.array([1.0, 0.0])                        # G corner
    C = np.array([0.5, math.sqrt(3)/2.0])           # B corner

    # Sample interior with barycentric coords r,g,b (r+g+b=1)
    xs, ys, cols = [], [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            r = i / n
            g = j / n
            b = 1.0 - r - g
            p = r * A + g * B + b * C
            # apply same gamma tweak used in your plot colors
            rr = r ** gamma
            gg = g ** gamma
            bb = b ** gamma
            xs.append(p[0]); ys.append(p[1]); cols.append((rr, gg, bb))

    # iax.scatter(xs, ys, s=2, c=cols, edgecolors="none")
    iax.add_patch(Polygon([A, B, C], facecolor="none", edgecolor="black", linewidth=1))

    # Corner labels
    iax.text(A[0]-0.06, A[1]-0.04, corner_labels[0], ha="right", va="top", fontsize=8)
    iax.text(B[0]+0.06, B[1]-0.04, corner_labels[1], ha="left",  va="top", fontsize=8)
    iax.text(C[0],      C[1]+0.06, corner_labels[2], ha="center", va="bottom", fontsize=8)

    # Optional tick marks (25/50/75%) along edges (comment out if you want it cleaner)
    for t in (0.25, 0.5, 0.75):
        # Edge AB (vary r vs g, b=0)
        p = (1-t) * A + t * B
        iax.plot([p[0]], [p[1]], marker="|", color="black", ms=6)
        # Edge BC (vary g vs b, r=0)
        p = (1-t) * B + t * C
        iax.plot([p[0]], [p[1]], marker="_", color="black", ms=6)
        # Edge CA (vary b vs r, g=0)
        p = (1-t) * C + t * A
        iax.plot([p[0]], [p[1]], marker="|", color="black", ms=6)

    # Tiny caption (how channels were normalized)
    iax.text(0.5, -0.18, "Each channel min–max normalized\nthen gamma-adjusted",
             ha="center", va="top", fontsize=7, transform=iax.transAxes)


def read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def determine_model_mode(model_config_path):
    model_dict = read_yaml(model_config_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError("model_param.mode must be defined in {}".format(model_config_path))
    return mode


def cartesian_product(option_map):
    """Yield dictionaries for every combination inside option_map."""
    if not option_map:
        yield {}
        return
    keys = sorted(option_map.keys())
    value_lists = [option_map[key] for key in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def total_gpu_count(parallel_cfg):
    total = 1
    for axis in ("tp", "cp", "dp", "lp"):
        value = int(parallel_cfg.get(axis, 1) or 1)
        total *= max(1, value)
    return total


def tp_cp_product_is_power_of_two_square(tp_value, cp_value):
    """Return True when tp*cp is a square whose root is also a power of two."""
    if tp_value == 1 and cp_value == 1:
        return False # special case
    try:
        tp_int = int(tp_value)
        cp_int = int(cp_value)
    except (TypeError, ValueError):
        return False
    if tp_int <= 0 or cp_int <= 0:
        return False
    product = tp_int * cp_int
    root = math.isqrt(product)
    if root * root != product:
        return False
    return (root & (root - 1)) == 0


def make_temp_hw_config(base_hw_dict, parallel_settings, hw_mutator=None):
    """Return (parsed HW config, YAML string) for the given override."""
    updated = copy.deepcopy(base_hw_dict)
    parallel_block = updated.setdefault("parallelism", {})
    for key, value in parallel_settings.items():
        parallel_block[key] = value

    if hw_mutator:
        hw_mutator(updated)
    try:
        debug_yaml = yaml.safe_dump(updated, default_flow_style=False)
    except Exception:
        debug_yaml = None

    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(updated, tmp_file, default_flow_style=False, sort_keys=False)
        tmp_file.flush()
        tmp_file.close()
        hw_config = config.parse_config(tmp_file.name, config_type="hardware")
        return hw_config, debug_yaml
    finally:
        try:
            tmp_file.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass

def _rgb_from_parallelism(entries, gamma: float = 0.85):
    """
    Build per-point RGBA colors using:
      R = log2(tp+cp), G = log2(lp), B = log2(dp)
    Each channel is min-max normalized over the dataset, then gamma-adjusted.
    """
    tps = np.array([max(1, int(e["parallelism"].get("tp", 1))) for e in entries], dtype=float)
    cps = np.array([max(1, int(e["parallelism"].get("cp", 1))) for e in entries], dtype=float)
    dps = np.array([max(1, int(e["parallelism"].get("dp", 1))) for e in entries], dtype=float)
    lps = np.array([max(1, int(e["parallelism"].get("lp", 1))) for e in entries], dtype=float)

    r_raw = np.log2(tps + cps)
    g_raw = np.log2(lps)
    b_raw = np.log2(dps)

    def _norm(x):
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return np.zeros_like(x)
        y = (x - xmin) / (xmax - xmin)
        if gamma and gamma != 1.0:
            y = np.power(y, gamma)
        return y

    r = _norm(r_raw)
    g = _norm(g_raw)
    b = _norm(b_raw)

    # RGBA with slight transparency to reduce overdraw
    return np.stack([r, g, b, np.full_like(r, 0.9)], axis=1)



def gpu_peak_flops(hw_config) -> float:
    """Return the theoretical peak FLOPs/s for a single GPU."""
    core = hw_config.tech_config.core
    bundles = core.num_bundles or 1
    per_cycle_flops = float(core.nominal_flop_rate_per_mcu)
    mcu_per_bundle = float(core.num_mcu_per_bundle)
    frequency = float(core.operating_frequency or core.nominal_frequency or 0.0)
    util = float(getattr(core, "util", 1.0) or 1.0)
    peak = per_cycle_flops * mcu_per_bundle * float(bundles)
    if frequency > 0:
        peak *= frequency
    return peak * util if peak > 0 else 0.0


def compute_total_flops(calculator: TimeCalculationLLM) -> float:
    """Estimate total FLOPs executed for one global batch (forward + backward)."""
    global_batch = getattr(calculator, "batch_size", None)
    if global_batch is None or global_batch <= 0:
        global_batch = calculator._effective_transformer_batch()
    if not global_batch or global_batch <= 0:
        return float("nan")

    vocab_size = calculator.vocab_size
    hidden_dim = calculator.hidden_dim
    seq_len = calculator.seq_len
    num_heads = calculator.num_heads
    kv_heads = calculator.kv_heads
    intermediate_size = calculator.intermediate_size

    gemm_shapes = process_gemm_shapes(
        calculator,
        global_batch,
        seq_len,
        hidden_dim,
        num_heads,
        kv_heads,
        intermediate_size,
        vocab_size,
    )

    def _gemm_flops(shape) -> float:
        if shape is None:
            return 0.0
        try:
            dims = list(shape)
        except TypeError:
            return 0.0
        if len(dims) == 4:
            b, m, k, n = dims
            return 2.0 * float(b) * float(m) * float(k) * float(n)
        if len(dims) == 3:
            m, k, n = dims
            return 2.0 * float(m) * float(k) * float(n)
        return 0.0

    def _forward_backward(shape) -> float:
        forward = _gemm_flops(shape)
        if forward <= 0.0:
            return 0.0
        backward = 2.0 * forward
        return forward + backward

    per_layer_flops = 0.0
    per_layer_flops += _forward_backward(gemm_shapes.get("qkv_proj"))
    per_layer_flops += _forward_backward(gemm_shapes.get("attention_score"))
    per_layer_flops += _forward_backward(gemm_shapes.get("attention_output"))
    per_layer_flops += _forward_backward(gemm_shapes.get("output_proj"))
    per_layer_flops += _forward_backward(gemm_shapes.get("ffn1"))
    per_layer_flops += _forward_backward(gemm_shapes.get("ffn2"))

    total_flops = per_layer_flops * float(calculator.num_layers)

    total_flops += _forward_backward(gemm_shapes.get("linear"))

    # Embedding forward/backward is comparatively small; keep placeholder for future refinement.

    return float(total_flops)


def evaluate_parallelism(hw_dict, model_config_obj, mode, parallel_settings, hw_mutator=None):
    hw_config, debug_yaml = make_temp_hw_config(hw_dict, parallel_settings, hw_mutator=hw_mutator)
    # if debug_yaml:
    #     print("=== DEBUG HW CONFIG ===")
    #     print(debug_yaml)
    #     try:
    #         with open("debug.yaml", "w") as debug_handle:
    #             debug_handle.write(debug_yaml)
    #     except Exception as exc:
            # print(f"Warning: failed to write debug.yaml: {exc}", file=sys.stderr)
    temp_dir = tempfile.mkdtemp(prefix="parallelism_sweep_")
    try:
        calculator = TimeCalculationLLM(hw_config, model_config_obj, mode, output_dir=temp_dir)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            runtime = calculator.calc_time_llm()
            total_flops = compute_total_flops(calculator)
        performance = (1.0 / runtime) if runtime and runtime > 0.0 else float("nan")
        peak_flops = gpu_peak_flops(hw_config)
        num_gpus = total_gpu_count(parallel_settings)
        achieved_flops = (total_flops / runtime) if runtime > 0 else float("nan")
        denom = peak_flops * num_gpus if peak_flops and num_gpus else float("nan")
        mfu = (achieved_flops / denom) if denom and denom > 0 else float("nan")
        return {
            "runtime": runtime,
            "performance": performance,
            "total_flops": total_flops,
            "peak_flops": peak_flops,
            "mfu": mfu,
            "achieved_flops": achieved_flops,
            "memory_exceeded": getattr(calculator, "memory_capacity_exceeded", False),
            "memory_violation_gb": getattr(calculator, "memory_capacity_violation_gb", 0.0),
            "hw_yaml": debug_yaml,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def set_astrasim_cache_mode(mode_str):
    mapping = {
        "NO_CACHE": "NO_CACHE",
        "CACHE_READONLY": "CACHE_READONLY",
        "CACHE_READWRITE": "CACHE_READWRITE",
    }
    env_value = mapping.get(str(mode_str).strip().upper(), "NO_CACHE")
    os.environ["DEEPFLOW_ASTRA_CACHE_MODE"] = env_value


def write_report(results, path):
    header = [
        "num_gpus",
        "runtime_s",
        "performance_1_over_s",
        "total_flops",
        "achieved_flops_per_s",
        "peak_flops_per_gpu",
        "mfu",
        "memory_exceeded",
        "memory_violation_gb",
        "parallelism",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for entry in results:
            row = [
                str(entry["num_gpus"]),
                "{:.6f}".format(entry["runtime"]),
                (
                    "{:.6f}".format(entry["performance"])
                    if entry["performance"] == entry["performance"]
                    else "nan"
                ),
                "{:.6e}".format(entry["total_flops"]),
                "{:.6e}".format(entry["achieved_flops"]) if entry["achieved_flops"] == entry["achieved_flops"] else "nan",
                "{:.6e}".format(entry["peak_flops"]),
                "{:.6f}".format(entry["mfu"]) if entry["mfu"] == entry["mfu"] else "nan",
                str(entry["memory_exceeded"]),
                "{:.6f}".format(entry["memory_violation_gb"]),
                repr(entry["parallelism"]),
            ]
            handle.write("\t".join(row) + "\n")


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_results_from_report(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No report found at {path}")

    results: List[Dict[str, object]] = []
    with open(path, "r") as handle:
        header = handle.readline()
        if not header:
            return results
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                print(f"Warning: skipping malformed row in report: {line}", file=sys.stderr)
                continue
            try:
                parallelism = ast.literal_eval(parts[9])
            except (SyntaxError, ValueError) as exc:
                print(f"Warning: failed to parse parallelism '{parts[9]}': {exc}", file=sys.stderr)
                continue
            entry = {
                "num_gpus": int(parts[0]),
                "runtime": _parse_float(parts[1]),
                "performance": _parse_float(parts[2]),
                "total_flops": _parse_float(parts[3]),
                "achieved_flops": _parse_float(parts[4]),
                "peak_flops": _parse_float(parts[5]),
                "mfu": _parse_float(parts[6]),
                "memory_exceeded": str(parts[7]).strip().lower() == "true",
                "memory_violation_gb": _parse_float(parts[8]),
                "parallelism": parallelism,
            }
            results.append(entry)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFlow parallelism sweep utility.")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip evaluation and only regenerate plots from the existing TSV report.",
    )
    parser.add_argument(
        "--enforce-square-tp-cp",
        action="store_true",
        help="Require tp*cp to be a square power of two when enumerating configurations.",
    )
    return parser.parse_args()


def jitter_positions(gpu_counts, jitter_width):
    jittered = []
    for count in gpu_counts:
        offset = random.uniform(1 - jitter_width, 1 + jitter_width)
        jittered.append(max(count * offset, 1e-3))
    return jittered


# def plot_results_legacy(results, output_path):
#     if not results:
#         print("No successful configurations to plot.", file=sys.stderr)
#         return

#     random.seed(PLOT_JITTER_SEED)
#     xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
#     metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"
#     ys = [item[metric_key] for item in results]

#     plt.figure(figsize=(10, 6))
#     plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")

#     best = min(results, key=lambda item: item["runtime"])
#     best_x = best["num_gpus"]
#     best_y = best[metric_key]
#     plt.scatter([best_x], [best_y], s=180, marker="*", c="red", label="Best runtime")

#     plt.xlabel("Number of GPUs")
#     if metric_key == "runtime":
#         plt.ylabel("Runtime (s)")
#     else:
#         plt.ylabel("Performance (1 / s)")
#     plt.xscale("log")
#     if metric_key == "runtime":
#         plt.yscale("log")
#     plt.title("Parallelism sweep")
#     plt.grid(alpha=0.3)
#     xticks = sorted(set(item["num_gpus"] for item in results))
#     plt.xticks(xticks, [str(int(x)) for x in xticks])
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print("Saved scatter plot to {}".format(output_path))

def _gpu_exp(n: int) -> float:
    """Return log2(num_gpus). Assumes powers of two; will return floats otherwise."""
    return math.log2(float(n))


def plot_results(results, output_path):
    if not results:
        print("No successful configurations to plot.", file=sys.stderr)
        return

    plot_entries = results
    if MEM_AWARE_FILTER:
        plot_entries = [item for item in results if not item.get("memory_exceeded", False)]
        if not plot_entries:
            print("Warning: all configurations violate memory limits; skipping plot.", file=sys.stderr)
            return

    metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"

    # Build tidy frame (include GPU exponent)
    rows = []
    for i, item in enumerate(plot_entries):
        p = item["parallelism"]
        ng = int(item["num_gpus"])
        rows.append({
            "row_id": i,
            "num_gpus": ng,
            "gpu_exp": _gpu_exp(ng),
            "tp": int(p.get("tp", 1)),
            "cp": int(p.get("cp", 1)),
            "dp": int(p.get("dp", 1)),
            "lp": int(p.get("lp", 1)),
            metric_key: float(item[metric_key]),
        })
    df = pd.DataFrame(rows)

    # Order categories by exponent (gives even spacing like log2)
    order = sorted(df["gpu_exp"].unique())
    df["gpu_exp_cat"] = pd.Categorical(df["gpu_exp"], categories=order, ordered=True)

    # --- Per-point RGB colors: R=log2(tp+cp), G=log2(lp), B=log2(dp) ---
    tps = df["tp"].to_numpy(dtype=float)
    cps = df["cp"].to_numpy(dtype=float)
    dps = df["dp"].to_numpy(dtype=float)
    lps = df["lp"].to_numpy(dtype=float)

    r_raw = np.log2(tps + cps)
    g_raw = np.log2(lps)
    b_raw = np.log2(dps)

    def _norm(x, gamma=0.85):
        x = x.astype(float)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            y = np.zeros_like(x)
        else:
            y = (x - xmin) / (xmax - xmin)
        return np.power(y, gamma) if gamma and gamma != 1.0 else y

    r = _norm(r_raw)
    g = _norm(g_raw)
    b = _norm(b_raw)
    a = np.full_like(r, 0.9)

    # Palette: unique color per row via hue="row_id"
    palette = {rid: (float(r[i]), float(g[i]), float(b[i]), float(a[i]))
               for i, rid in enumerate(df["row_id"])}

    plt.figure(figsize=(10, 6))
    ax = sns.swarmplot(
        data=df,
        x="gpu_exp_cat",
        y=metric_key,
        hue="row_id",
        palette=palette,   # per-point RGBA
        size=3.75,
        linewidth=0,
        dodge=False,
        legend=False
    )
    # add_rgb_ternary_legend(
    #     ax,
    #     corner_labels=("R = log2(tp+cp)", "G = log2(lp)", "B = log2(dp)"),
    #     gamma=0.85,                      # match your _rgb_from_parallelism gamma
    #     inset_xywh=(0.70, 0.50, 0.26, 0.26)  # tweak position/size to taste
    # )


    # Best runtime star — place at the matching exponent category index
    best = min(plot_entries, key=lambda it: it["runtime"])
    best_exp = _gpu_exp(int(best["num_gpus"]))
    best_idx = order.index(best_exp)
    best_y = float(best[metric_key])
    ax.scatter([best_idx], [best_y], s=180, marker="*", c="red", zorder=5, label="Best runtime")

    # Labels & scales
    ax.set_xlabel("Number of GPUs (log2 categories)")
    if metric_key == "runtime":
        ax.set_ylabel("Runtime (s)")
        ax.set_yscale("log")
    else:
        ax.set_ylabel("Performance (1 / s)")

    # Tick labels show the REAL GPU count (2**exp) so axis *looks* log2
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(int(round(2 ** e))) if float(e).is_integer() else f"2^{e:.2f}"
                        for e in order])

    ax.set_title("Parallelism sweep – beeswarm on log2(GPUs) categories")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")

    # Reminder of color encoding
    ax.text(0.99, 0.01,
            "Color: R=log2(tp+cp), G=log2(lp), B=log2(dp)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved swarm plot to {output_path}")


# def plot_results_categorical(results, output_path):
#     if not results:
#         print("No successful configurations to plot.", file=sys.stderr)
#         return

#     metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"

#     # Build a tidy DataFrame for seaborn (categorical x keeps each GPU count together)
#     df_rows = []
#     for item in results:
#         p = item["parallelism"]
#         df_rows.append({
#             "num_gpus": int(item["num_gpus"]),
#             metric_key: float(item[metric_key]),
#             "tp": int(p.get("tp", 1)),
#             "cp": int(p.get("cp", 1)),
#             "dp": int(p.get("dp", 1)),
#             "lp": int(p.get("lp", 1)),
#         })
#     df = pd.DataFrame(df_rows)

#     # Order GPU categories left->right by numeric value
#     order = sorted(df["num_gpus"].unique())
#     df["num_gpus_cat"] = pd.Categorical(df["num_gpus"], categories=order, ordered=True)

#     # --- Beeswarm placement ---
#     plt.figure(figsize=(10, 6))
#     ax = sns.swarmplot(
#         data=df,
#         x="num_gpus_cat",
#         y=metric_key,
#         size=5,
#         color="k",         # temporary (we'll recolor with our RGBs after layout)
#         linewidth=0,
#         alpha=0.0          # invisible; we only want the positions it computes
#     )

#     # Grab the laid-out positions, remove seaborn's collection, and redraw with our colors
#     if not ax.collections:
#         print("Warning: swarmplot produced no collections.", file=sys.stderr)
#         return
#     coll = ax.collections[0]
#     offsets = coll.get_offsets()        # Nx2 array (x_index, y_value) in data coords (x is categorical index)
#     coll.remove()

#     # Compute per-point RGBA colors from parallelism
#     colors = _rgb_from_parallelism(results, gamma=0.85)

#     # Plot the recolored points at the computed positions
#     ax.scatter(
#         offsets[:, 0], offsets[:, 1],
#         s=60,
#         c=colors,
#         edgecolors="none",
#         zorder=3
#     )

#     # Global best marker (use category index for x)
#     best = min(results, key=lambda item: item["runtime"])
#     best_x_idx = order.index(int(best["num_gpus"]))  # categorical index
#     best_y = float(best[metric_key])
#     ax.scatter([best_x_idx], [best_y], s=180, marker="*", c="red", zorder=5, label="Best runtime")

#     # Axes/labels
#     ax.set_xlabel("Number of GPUs")
#     if metric_key == "runtime":
#         ax.set_ylabel("Runtime (s)")
#         ax.set_yscale("log")
#     else:
#         ax.set_ylabel("Performance (1 / s)")

#     # Pretty ticks for categorical axis
#     ax.set_xticks(range(len(order)))
#     ax.set_xticklabels([str(x) for x in order])

#     ax.set_title("Parallelism sweep (beeswarm with log-RGB coloring)")
#     ax.grid(alpha=0.3, axis="y")
#     ax.legend(loc="best")

#     # Small caption to recall color encoding
#     ax.text(0.99, 0.01,
#             "Color channels: R=log2(tp+cp), G=log2(lp), B=log2(dp)",
#             transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print(f"Saved swarm plot to {output_path}")


# def plot_mfu(results, output_path):
#     if not results:
#         return
#     random.seed(PLOT_JITTER_SEED)
#     xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
#     ys = [item["mfu"] for item in results]
#     plt.figure(figsize=(10, 6))
#     plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")
#     valid = [item for item in results if item["mfu"] == item["mfu"]]
#     if valid:
#         best = max(valid, key=lambda item: item["mfu"])
#         plt.scatter([best["num_gpus"]], [best["mfu"]], s=180, marker="*", c="green", label="Max MFU")
#     plt.xlabel("Number of GPUs")
#     plt.ylabel("MFU")
#     plt.xscale("log")
#     plt.ylim(0.0, 1.05)
#     plt.title("Parallelism sweep - MFU")
#     plt.grid(alpha=0.3)
#     xticks = sorted(set(item["num_gpus"] for item in results))
#     plt.xticks(xticks, [str(int(x)) for x in xticks])
#     if valid:
#         plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print("Saved MFU scatter plot to {}".format(output_path))


_GLOBAL_MODEL_CONFIG = None
_GLOBAL_MODE = None
_GLOBAL_HW_DICT = None


def _worker_init(hw_dict, model_config_path, mode):
    global _GLOBAL_MODEL_CONFIG, _GLOBAL_MODE, _GLOBAL_HW_DICT
    _GLOBAL_HW_DICT = hw_dict
    _GLOBAL_MODE = mode
    _GLOBAL_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)


def _worker_task(parallel_items: Tuple[Tuple[str, object], ...]):
    parallel_settings = {k: v for k, v in parallel_items}
    try:
        metrics = evaluate_parallelism(
            _GLOBAL_HW_DICT,
            _GLOBAL_MODEL_CONFIG,
            _GLOBAL_MODE,
            parallel_settings,
        )
        return {
            "status": "ok",
            "parallelism": parallel_settings,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "error",
            "parallelism": parallel_settings,
            "error": str(exc),
        }


def _build_tasks(
    gpu_choices: Iterable[Dict[str, int]],
    other_choices: Iterable[Dict[str, object]],
) -> List[Tuple[Tuple[str, object], ...]]:
    tasks: List[Tuple[Tuple[str, object], ...]] = []
    for gpu_choice in gpu_choices:
        for other_choice in other_choices:
            settings: Dict[str, object] = {}
            settings.update(gpu_choice)
            settings.update(other_choice)
            settings["mb"] = settings.get("lp", 1)
            tasks.append(tuple(sorted(settings.items())))
    return tasks


def main():
    args = parse_args()
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)

    enforce_square = ENFORCE_SQUARE_TP_CP or args.enforce_square_tp_cp

    results: List[Dict[str, object]] = []

    if args.plot_only:
        try:
            results = load_results_from_report(REPORT_OUTPUT_PATH)
        except FileNotFoundError as exc:
            print(str(exc))
            return

        if not results:
            print(f"No entries found in {REPORT_OUTPUT_PATH}; nothing to plot.")
            return

        best = min(results, key=lambda item: item["runtime"])
        print(f"Loaded {len(results)} configuration(s) from {REPORT_OUTPUT_PATH}")
        print("\nBest configuration (lowest runtime):")
        print("  Parallelism: {}".format(best["parallelism"]))
        print("  GPUs: {}".format(best["num_gpus"]))
        print("  Runtime: {:.4f} s".format(best["runtime"]))
        print("  Performance (1/s): {:.4f}".format(best["performance"]))
        print("  Total FLOPs: {:.3e}".format(best["total_flops"]))
        print("  MFU: {:.3f}".format(best["mfu"]))
        if best["memory_exceeded"]:
            print("  Memory capacity exceeded by {:.3f} GB".format(best["memory_violation_gb"]))
    else:
        base_hw_dict = read_yaml(HARDWARE_CONFIG_PATH)
        mode = determine_model_mode(MODEL_CONFIG_PATH)

        gpu_axes = list(PARALLELISM_SWEEP.keys())
        other_axes = list(OTHER_PARALLELISM_OPTIONS.keys())
        gpu_combos = list(cartesian_product(PARALLELISM_SWEEP))
        other_combos = list(cartesian_product(OTHER_PARALLELISM_OPTIONS))
        task_items = _build_tasks(gpu_combos, other_combos)
        print("Enumerating {} parallelism combinations: {}".format(len(task_items), ", ".join(gpu_axes)))

        skipped_out_of_range = 0
        skipped_errors = 0
        error_messages: List[str] = []
        evaluated = 0

        filtered_tasks: List[Tuple[Tuple[str, object], ...]] = []
        skipped_square_constraint = 0
        for items in task_items:
            settings = dict(items)
            num_gpus = total_gpu_count(settings)
            if enforce_square and not tp_cp_product_is_power_of_two_square(settings.get("tp"), settings.get("cp")):
                skipped_square_constraint += 1
                continue
            if GPU_COUNT_MIN <= num_gpus <= GPU_COUNT_MAX:
                filtered_tasks.append(items)
            else:
                skipped_out_of_range += 1

        if not filtered_tasks:
            print("No configurations within GPU count bounds.")
            return
        if enforce_square and skipped_square_constraint:
            print(f"Skipped {skipped_square_constraint} configuration(s) due to tp*cp square constraint.")

        available_cpus = max(1, os.cpu_count() or 1)
        if MAX_WORKERS is None or MAX_WORKERS <= 0:
            worker_limit = max(1, available_cpus - 1)
        else:
            worker_limit = min(MAX_WORKERS, max(1, available_cpus - 1))
        worker_count = max(1, worker_limit)
        print(f"Using {worker_count} worker process(es) (out of {available_cpus} CPUs).")

        if worker_count > 1 and len(filtered_tasks) > 1:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_worker_init,
                initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode),
            ) as executor:
                futures = {executor.submit(_worker_task, items): items for items in filtered_tasks}
                with tqdm(total=len(filtered_tasks), desc="Evaluating", unit="config") as progress:
                    for future in as_completed(futures):
                        progress.update(1)
                        result = future.result()
                        settings = result["parallelism"]
                        num_gpus = total_gpu_count(settings)
                        if result["status"] != "ok":
                            skipped_errors += 1
                            msg = result.get("error") or "unknown error"
                            error_messages.append(f"{settings}: {msg}")
                            continue
                        metrics = result["metrics"]
                        evaluated += 1
                        entry = {
                            "parallelism": settings,
                            "num_gpus": num_gpus,
                            "runtime": metrics["runtime"],
                            "performance": metrics["performance"],
                            "total_flops": metrics["total_flops"],
                            "achieved_flops": metrics["achieved_flops"],
                            "peak_flops": metrics["peak_flops"],
                            "mfu": metrics["mfu"],
                            "memory_exceeded": metrics["memory_exceeded"],
                            "memory_violation_gb": metrics["memory_violation_gb"],
                        }
                        results.append(entry)
        else:
            model_config_obj = config.parse_config(MODEL_CONFIG_PATH, config_type=mode)
            with tqdm(total=len(filtered_tasks), desc="Evaluating", unit="config") as progress:
                for items in filtered_tasks:
                    settings = dict(items)
                    num_gpus = total_gpu_count(settings)
                    progress.update(1)
                    try:
                        metrics = evaluate_parallelism(base_hw_dict, model_config_obj, mode, settings)
                    except Exception as exc:
                        skipped_errors += 1
                        error_messages.append(f"{settings}: {exc}")
                        continue

                    evaluated += 1
                    entry = {
                        "parallelism": settings,
                        "num_gpus": num_gpus,
                        "runtime": metrics["runtime"],
                        "performance": metrics["performance"],
                        "total_flops": metrics["total_flops"],
                        "achieved_flops": metrics["achieved_flops"],
                        "peak_flops": metrics["peak_flops"],
                        "mfu": metrics["mfu"],
                        "memory_exceeded": metrics["memory_exceeded"],
                        "memory_violation_gb": metrics["memory_violation_gb"],
                    }
                    results.append(entry)

        total_skipped = skipped_out_of_range + skipped_errors
        if not results:
            print(
                "No valid configurations evaluated ({} skipped: {} out-of-range, {} errors).".format(
                    total_skipped, skipped_out_of_range, skipped_errors
                )
            )
            if error_messages:
                print("Encountered errors for configurations:")
                for msg in error_messages:
                    print(f"  {msg}")
            return

        best = min(results, key=lambda item: item["runtime"])
        print(
            f"Evaluated {evaluated} configuration(s); skipped {total_skipped} "
            f"(out_of_range={skipped_out_of_range}, errors={skipped_errors})."
        )
        if error_messages:
            print("Some configurations failed:")
            for msg in error_messages:
                print(f"  {msg}")
        print("\nBest configuration (lowest runtime):")
        print("  Parallelism: {}".format(best["parallelism"]))
        print("  GPUs: {}".format(best["num_gpus"]))
        print("  Runtime: {:.4f} s".format(best["runtime"]))
        print("  Performance (1/s): {:.4f}".format(best["performance"]))
        print("  Total FLOPs: {:.3e}".format(best["total_flops"]))
        print("  MFU: {:.3f}".format(best["mfu"]))
        if best["memory_exceeded"]:
            print("  Memory capacity exceeded by {:.3f} GB".format(best["memory_violation_gb"]))

        try:
            write_report(results, REPORT_OUTPUT_PATH)
            print("Wrote detailed report to {}".format(REPORT_OUTPUT_PATH))
        except Exception as exc:
            print("Warning: failed to write report: {}".format(exc), file=sys.stderr)

    if not results:
        return

    plot_results(results, PLOT_OUTPUT_PATH)
    # plot_mfu(results, PLOT_MFU_OUTPUT_PATH)


if __name__ == "__main__":
    main()
