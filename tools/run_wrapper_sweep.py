#!/usr/bin/env python3
"""Sweep DeepFlow wrapper across selected (dp, lp, mb) settings and plot runtimes.

This script:
  1. Modifies a copy of the A100 hardware config for each sweep point
  2. Invokes the wrapper harness (DeepFlow, DeepFlow ablation, STG) for every config
  3. Streams results into a CSV as soon as each run completes
  4. Generates a summary line plot comparing AstraSim runtimes across tools

The sweep definition follows the user specification:
  - dp ∈ {1, 4}, lp ∈ {1, 4, 16}
  - Configs with fewer than two devices (dp * lp < 2) are skipped
  - mb is always set equal to lp
"""

from __future__ import annotations

import copy
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import wrapper as wrapper_mod  # type: ignore  # noqa: E402
from astrasim_lib import config_generation as cfggen  # type: ignore  # noqa: E402


BASE_CONFIG_PATH = REPO_ROOT / "configs" / "hardware-config" / "a100_80GB.yaml"
TEMP_CONFIG_DIR = REPO_ROOT / "configs" / "hardware-config" / "__wrapper_sweep_tmp"
OUTPUTS_DIR = wrapper_mod.WRAPPER_OUTPUT_ROOT
CSV_PATH = OUTPUTS_DIR / "wrapper_sweep_results.csv"

MODELS = [
    {
        "path": REPO_ROOT / "configs" / "model-config" / "Llama2-7B.yaml",
        "name": "Llama2-7B",
    },
    {
        "path": REPO_ROOT / "configs" / "model-config" / "Llama3.1-70B.yaml",
        "name": "Llama3.1-70B_8k",
    },
    {
        "path": REPO_ROOT / "configs" / "model-config" / "Llama3.1-70B_32k.yaml",
        "name": "Llama3.1-70B_32k",
    },
    {
        "path": REPO_ROOT / "configs" / "model-config" / "Llama3.1-70B_128k.yaml",
        "name": "Llama3.1-70B_128k",
    },
    {
        "path": REPO_ROOT / "configs" / "model-config" / "Llama3.1-405B.yaml",
        "name": "Llama3.1-405B",
    },
]

DP_VALUES = [1, 4]
LP_VALUES = [1, 2, 4, 8, 16]

MODE_LABELS = {
    "deepflow": "DeepFlow",
    "deepflow_ablation": "DeepFlow Ablation",
    "stg": "STG",
}

CSV_FIELDS = [
    "run_index",
    "dp",
    "lp",
    "mb",
    "mode",
    "duration_seconds",
    "astrasim_total_time",
    "artifact_dir",
    "config_label",
    "model_name",
]


def load_base_config() -> Dict:
    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_temp_dir() -> None:
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def build_config_label(lp: int, mb: int) -> str:
    return f"lp={lp}\nmb={mb}"


def determine_mb_values(lp: int) -> List[int]:
    return [lp]


def generate_sweep_points() -> List[Tuple[int, int, int]]:
    points: List[Tuple[int, int, int]] = []
    for dp in DP_VALUES:
        for lp in LP_VALUES:
            for mb in determine_mb_values(lp):
                points.append((dp, lp, mb))
    return points


def write_temp_config(base_cfg: Dict, dp: int, lp: int, mb: int, model_name: str) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("system_hierarchy", {})
    cfg["system_hierarchy"]["num_devices_per_node"] = int(dp * lp)

    sched = cfg.setdefault("scheduling_param", {})
    sched["dp"] = int(dp)
    sched["lp"] = int(lp)
    sched["mb"] = int(mb)

    temp_path = TEMP_CONFIG_DIR / f"a100_80GB_{model_name}_dp{dp}_lp{lp}_mb{mb}.yaml"
    with temp_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return temp_path


def append_records(csv_path: Path, records: Iterable[Dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    mode = "a" if csv_path.exists() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record)


def execute_wrapper(config_path: Path, model_path: Path) -> List[Dict[str, object]]:
    if hasattr(cfggen, "reset_json_cache"):
        cfggen.reset_json_cache()
    if hasattr(cfggen, "_NET_YAML_CACHE"):
        cfggen._NET_YAML_CACHE.clear()  # type: ignore[attr-defined]

    config = copy.deepcopy(wrapper_mod.GLOBAL_CONFIG)
    config["hardware_config"] = config_path
    config["model_config"] = model_path
    config["generate_visuals"] = False

    results = wrapper_mod.dispatch("all", config)
    return results


def collect_results(
    run_index: int,
    dp: int,
    lp: int,
    mb: int,
    config_label: str,
    wrapper_results: Sequence[Dict[str, object]],
    model_name: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for entry in wrapper_results:
        records.append(
            {
                "run_index": run_index,
                "dp": dp,
                "lp": lp,
                "mb": mb,
                "mode": entry.get("mode"),
                "duration_seconds": entry.get("duration_seconds"),
                "astrasim_total_time": entry.get("astrasim_total_time"),
                "artifact_dir": str(entry.get("artifact_dir")),
                "config_label": config_label,
                "model_name": model_name,
            }
        )
    return records


def load_model_metadata(model_path: Path) -> Dict[str, object]:
    with model_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    model_param = data.get("model_param", {})
    return {
        "batch_size": model_param.get("batch_size"),
        "seq_len": model_param.get("seq_len"),
    }


def plot_results(
    records: Sequence[Dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    if not records:
        print("[WARN] No records collected; skipping plot generation.")
        return

    sorted_records = sorted(records, key=lambda r: (r["run_index"], r["mode"]))
    label_meta: Dict[str, Tuple[int, int]] = {}
    for rec in sorted_records:
        label_meta[rec["config_label"]] = (rec["lp"], rec["mb"])

    if not label_meta:
        print("[WARN] No configuration labels found; cannot plot.")
        return

    config_order = sorted(label_meta.keys(), key=lambda lbl: label_meta[lbl])

    dp_values = sorted({rec["dp"] for rec in sorted_records})
    linestyle_map = {1: "-", 4: ":"}

    x_positions = list(range(len(config_order)))
    plt.figure(figsize=(max(10, len(config_order) * 0.6), 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for mode_index, mode in enumerate(MODE_LABELS.keys()):
        color = color_cycle[mode_index % len(color_cycle)] if color_cycle else None
        for dp in dp_values:
            values: List[float] = []
            for label in config_order:
                value = next(
                    (
                        rec["astrasim_total_time"]
                        for rec in sorted_records
                        if rec["mode"] == mode and rec["config_label"] == label and rec["dp"] == dp
                    ),
                    math.nan,
                )
                values.append(float(value) if value is not None else math.nan)

            if all(math.isnan(v) for v in values):
                continue

            plt.plot(
                x_positions,
                values,
                marker="o",
                linewidth=2.0,
                linestyle=linestyle_map.get(dp, "-."),
                color=color,
                label=f"{MODE_LABELS.get(mode, mode)} (dp={dp})",
            )

    plt.xticks(x_positions, config_order)
    plt.ylabel("AstraSim runtime (s)")
    plt.xlabel("Configuration")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Plot saved to {output_path}")


def main() -> None:
    ensure_temp_dir()
    base_cfg = load_base_config()
    sweep_points = generate_sweep_points()

    all_records: List[Dict[str, object]] = []
    run_index = 0

    for model in MODELS:
        model_path = Path(model["path"])
        model_name = model["name"]
        meta = load_model_metadata(model_path)
        model_records: List[Dict[str, object]] = []

        for dp, lp, mb in sweep_points:
            if dp * lp < 2:
                print(f"[SKIP] {model_name}: dp={dp}, lp={lp} requires at least 2 devices; skipping.")
                continue

            run_index += 1
            config_label = build_config_label(lp, mb)
            print(f"[{model_name}] RUN {run_index}: {config_label.replace(chr(10), ', ')}")

            config_path = write_temp_config(base_cfg, dp, lp, mb, model_name)
            try:
                wrapper_results = execute_wrapper(config_path, model_path)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[ERROR] Wrapper failed for {model_name} ({config_label}): {exc}")
                continue

            records = collect_results(run_index, dp, lp, mb, config_label, wrapper_results, model_name)
            append_records(CSV_PATH, records)
            all_records.extend(records)
            model_records.extend(records)

            for record in records:
                mode = record["mode"]
                runtime = record["astrasim_total_time"]
                print(f"    - {mode}: AstraSim runtime = {runtime}")

        plot_title = f"{model_name} (batch={meta.get('batch_size')}, seq={meta.get('seq_len')})"
        plot_path = OUTPUTS_DIR / f"wrapper_sweep_plot_{model_name}.png"
        plot_results(model_records, plot_path, plot_title)
        print(f"[INFO] Plot saved for {model_name}: {plot_path}")

    print(f"[DONE] Sweep complete. Results CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
