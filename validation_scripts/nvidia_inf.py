import argparse
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .validation_helpers import (
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )

# Data used to be in  https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/performance.html in 2022.
# no longer available now, found via paper. "Performance Modeling and Workload Analysis of Distributed Large Language Model Training and Inference" by J. Kundu et al.

HW_CONFIGS = {
    "A100": "a100_80GB.yaml",
    "H100": "H100_SXM5_80GB.yaml",
}

MODEL_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "Llama 2-13B": "Llama2-13B_inf.yaml",
    "Llama 2-70B": "Llama2-70B_inf.yaml",
}

MODEL_DISPLAY = {
    "Llama 2-7B": "Llama 2-7B",
    "Llama 2-13B": "Llama 2-13B",
    "Llama 2-70B": "Llama 2-70B",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config")
MODEL_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "model-config")
DATA_DIR = Path(__file__).parent / "imec_data"


def _load_data(csv_path: Path) -> pd.DataFrame:
    # Skip the leading "// filepath: ..." line by treating '/' as a comment char.
    df = pd.read_csv(csv_path, comment="/")
    # Normalize dtypes
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df.dropna(subset=["device", "model", "TP", "actual"])


@lru_cache(maxsize=None)
def _load_device_data(device: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{device}_inf.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found for device {device}: {csv_path}")
    return _load_data(csv_path)


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (row["device"], row["model"], row["TP"])


def _build_spec(device: str, model: str, tp: int, idx: int, network_ignored: bool) -> Tuple[ValidationSpec, str, str]:
    label = f"{device} {model} TP={tp}"

    model_overrides = {
        "model_param": {
            "global_batch_size": 1,
            "seq_len": 400,
            "decode_len": 200
        }
    }

    hw_overrides: Dict[str, Dict[str, object]] = {
        "parallelism": {
            "dp": 1,
            "tp": int(tp),
            "tp_sp": False,
            "cp": 1,
            "lp": 1,
            "mb": 1,
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "1e-9" },
                    }
                ]
            },
        }
        hw_overrides.update(override_bw)

    spec = ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model)),
        hardware_config_path=os.path.join(HARDWARE_CONFIG_PATH, HW_CONFIGS.get(device)),
        metadata={"device": device, "model": model, "tp": int(tp)},
        order=idx,
    )
    hw_config_path = spec.hardware_config_path  # type: ignore[arg-type]
    model_config_path = spec.model_config_path  # type: ignore[arg-type]
    return spec, hw_config_path, model_config_path


def _lookup_actual(device: str, model: str, tp: int) -> float:
    data = _load_device_data(device)
    matches = data.loc[
        (data["device"] == device) & (data["model"] == model) & (data["TP"] == int(tp)),
        "actual",
    ]
    if matches.empty:
        raise ValueError(f"No reference inference time for {device} {model} TP={tp}")
    return float(matches.values[0])


def run_single(
    device: str,
    model: str,
    tp: int,
    *,
    network_ignored: bool = True,
    actual_inference_time_s: Optional[float] = None,
    emit_logs: bool = False,
) -> Dict[str, object]:
    spec, hw_config_path, model_config_path = _build_spec(device, model, tp, idx=0, network_ignored=network_ignored)
    validation_results = run_validation_suite(
        [spec],
        base_model_config_path=model_config_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    result = validation_results[0]
    inference_time_s = float(result.metrics.get("inference_time_s", float("nan"))) if result.success else float("nan")
    err_detail = None if result.success else (result.error or "DeepFlow run failed")

    if actual_inference_time_s is None:
        try:
            actual_inference_time_s = _lookup_actual(device, model, tp)
        except Exception:
            actual_inference_time_s = float("nan")

    if math.isnan(inference_time_s) or actual_inference_time_s == 0 or math.isnan(actual_inference_time_s):
        pct_error = float("nan")
    else:
        pct_error = abs(inference_time_s - actual_inference_time_s) / actual_inference_time_s * 100.0

    if emit_logs:
        block_lines = [
            f"\n=== Result (device={device}, model={model}, TP={tp}) ===",
        ]
        if not math.isnan(inference_time_s):
            block_lines.append(f"  DeepFlow Inference Time: {inference_time_s:.2f}s")
            block_lines.append(f"  Actual Inference Time:   {actual_inference_time_s:.2f}s")
            block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
        else:
            block_lines.append(f"  DeepFlow run failed. {err_detail or ''}".rstrip())
            if result.raw_output:
                block_lines.append(result.raw_output.strip())
        print("\n".join(block_lines))

    return {
        "success": result.success,
        "inference_time_s": inference_time_s,
        "actual_inference_time_s": actual_inference_time_s,
        "pct_error": pct_error,
        "error": err_detail,
        "raw_output": result.raw_output,
    }


def build_specs_for_device(
    device: str,
    *,
    network_ignored: bool = True,
    models: Optional[Iterable[str]] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int], float], str, str]:
    data = _load_device_data(device)
    model_filter = set(models) if models is not None else None
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int], float] = {}
    base_model_path: Optional[str] = None
    hw_config_path: Optional[str] = None
    idx = 0
    for device_val, model, tp in _iter_tests(data):
        if model_filter and model not in model_filter:
            continue
        spec, hw_path, model_path = _build_spec(device_val, model, tp, idx, network_ignored)
        specs.append(spec)
        actual_lookup[(model, int(tp))] = _lookup_actual(device_val, model, int(tp))
        base_model_path = base_model_path or model_path
        hw_config_path = hw_config_path or hw_path
        idx += 1

    if not specs:
        raise ValueError(f"No validation specs generated for device={device} (models={models}).")
    return specs, actual_lookup, base_model_path, hw_config_path


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int], float]):
    rows: List[Dict[str, object]] = []
    for res in results:
        metadata = res.spec.metadata or {}
        model = metadata.get("model")
        tp = int(metadata.get("tp")) if "tp" in metadata else None
        inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        actual = actual_lookup.get((model, tp)) if model is not None and tp is not None else float("nan")
        if math.isnan(inf_time) or actual is None or actual == 0 or math.isnan(actual):
            pct_error = float("nan")
        else:
            pct_error = abs(inf_time - actual) / actual * 100.0
        rows.append(
            {
                "device": metadata.get("device"),
                "model": model,
                "tp": tp,
                "inference_time_s": inf_time,
                "actual_inference_time_s": actual,
                "pct_error": pct_error,
                "display_model": MODEL_DISPLAY.get(str(model), str(model)),
                "success": res.success,
                "error": res.error,
            }
        )
    return rows


def plot_device(df: pd.DataFrame, device: str, outdir: Path) -> Path | None:
    sub = df[df["device"] == device].copy()
    if sub.empty:
        return None
    sub.sort_values(["model", "TP"], inplace=True)

    # Hierarchical grouping: model cluster (primary) -> TP subgroup (secondary with two bars: seconds & actual)
    # Enforce explicit primary model order
    desired_order = ["Llama 2-7B", "Llama 2-13B", "Llama 2-70B"]
    unique_models = list(pd.unique(sub["model"]))
    models = [m for m in desired_order if m in unique_models] + [
        m for m in unique_models if m not in desired_order
    ]
    display_models = [MODEL_DISPLAY.get(m, m) for m in models]

    bar_width = 0.28  # width per individual bar
    tp_gap = 0.08  # gap between TP subgroups inside a model cluster
    model_gap = 0.7  # gap between model clusters

    x_seconds = []
    x_actual = []
    seconds_vals = []
    actual_vals = []
    tp_midpoints = []
    tp_labels = []
    model_centers = []
    model_bounds = []

    current_x = 0.0
    for m in models:
        m_rows = sub[sub["model"] == m]
        tps = sorted(m_rows["TP"].unique())
        cluster_start = current_x
        for tp in tps:
            row = m_rows[m_rows["TP"] == tp].iloc[0]
            x_sec = current_x
            x_act = current_x + bar_width
            x_seconds.append(x_sec)
            x_actual.append(x_act)
            seconds_vals.append(row["seconds"])
            actual_vals.append(row["actual"])
            # Midpoint of the TP subgroup (between the two bars)
            tp_mid = x_sec + bar_width * 0.5
            tp_midpoints.append(tp_mid)
            tp_labels.append(f"TP{tp}")
            current_x += 2 * bar_width + tp_gap
        cluster_end = current_x - tp_gap  # last subgroup end (exclude trailing tp_gap)
        model_centers.append((cluster_start + cluster_end) / 2)
        model_bounds.append((cluster_start, cluster_end))
        current_x += model_gap  # gap after cluster

    # Dynamic figure width
    fig_w = max(8.0, 0.12 * len(tp_midpoints) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bars_seconds = ax.bar(
        x_seconds, seconds_vals, bar_width, label="DeepFlow", color="#1f77b4"
    )
    bars_actual = ax.bar(
        x_actual, actual_vals, bar_width, label="Actual", color="#0bbd37"
    )

    # Primary ticks: model names at cluster centers
    ax.set_xticks(model_centers)
    ax.set_xticklabels(display_models, fontsize=11)
    ax.set_ylabel("Inference Latency (s)")
    ax.set_title(
        f"Validation of Inference Latency on Systems of {device} (network ignored)"
    )

    # Secondary TP labels ABOVE bars (centered over subgroup)
    ymin, ymax = ax.get_ylim()
    pad = 0.02 * (ymax - ymin)
    subgroup_max = [max(s, a) for s, a in zip(seconds_vals, actual_vals)]
    new_ymax = max(ymax, max(subgroup_max) + 3 * pad)
    ax.set_ylim(ymin, new_ymax)
    for mid, lbl, v in zip(tp_midpoints, tp_labels, subgroup_max):
        ax.text(mid, v + pad, lbl, ha="center", va="bottom", fontsize=8)

    # Draw light separators between model clusters
    for start, end in model_bounds:
        ax.axvspan(
            start - bar_width * 0.5,
            end - bar_width * 0.5,
            facecolor="#000000",
            alpha=0.02,
        )

    # Custom legend
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.02, y=0.1)

    outpath = outdir / f"inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def run(
    enable_plot: bool = True,
    network_ignored: bool = True,
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    emit_logs: bool = True,
):
    pct_errors = []
    data = _load_device_data(device)

    specs, actual_lookup, base_model_path, hw_config_path = build_specs_for_device(
        device, network_ignored=network_ignored, models=models
    )

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    rows = compute_pct_errors(validation_results, actual_lookup)

    for row in rows:
        pct_error = float(row["pct_error"])
        pct_errors.append(pct_error)
        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "seconds",
        ] = row["inference_time_s"]

        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "pct_error",
        ] = pct_error

        if emit_logs:
            block_lines = [
                f"\n=== Result (device={row['device']}, model={row['model']}, TP={row['tp']}) ===",
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  DeepFlow Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  DeepFlow run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    if enable_plot:
        out_dir = os.path.join(PROJECT_ROOT, "output", "validation")
        os.makedirs(out_dir, exist_ok=True)
        out_dir = Path(out_dir)

        outputs = []
        for device in ["A100", "H100"]:
            out = plot_device(data, device, out_dir)
            if out is not None:
                outputs.append(out)
        # Secondary bar plot matching the training style (combined devices).
        bar_path = _plot_error_bars(
            rows,
            path=out_dir / "inf_errors_bar.png",
            title="Inference validation (combined)",
        )
        outputs.append(bar_path)
        if not outputs:
            print("No plots generated (no matching device rows).")
        else:
            for p in outputs:
                print(f"Saved: {p}")
    valid_errors = [e for e in pct_errors if not math.isnan(e)]
    avg_abs_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    if emit_logs:
        print("Average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error, "rows": rows}


def _plot_error_bars(rows: List[Dict[str, object]], path: Path, title: str) -> Path:
    # Build labels like "<model> xTP<tp>", drop device prefixes.
    labels: List[str] = []
    errors: List[float] = []
    for row in rows:
        model = row.get("display_model") or row.get("model")
        tp = row.get("tp")
        labels.append(f"{model} xTP{tp}")
        errors.append(float(row.get("pct_error", float("nan"))))

    fig_w = max(6.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(range(len(errors)), errors, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    for rect, err in zip(bars, errors):
        if math.isnan(err):
            continue
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    run(enable_plot=True, network_ignored=True)
