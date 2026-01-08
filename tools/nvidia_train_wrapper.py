"""Run NVIDIA training validation cases through DeepFlow/STG/MLSynth wrappers and plot results."""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import re
import sys
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402
import yaml  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
WRAPPER_OUT_BASE = TOOLS_ROOT / "comp" / "wrapper_outputs" / "nvidia_train"
RESULT_FILENAME = "result.json"
SUMMARY_FILENAME = "result_summary.txt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.comp import wrapper as comp_wrapper  # noqa: E402
from validation_scripts import nvidia_train  # noqa: E402
from validation_scripts import validation_helpers as vh  # noqa: E402


ToolName = str


@dataclass
class ToolResult:
    name: ToolName
    seconds: Optional[float]
    artifact_dir: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class SpecRun:
    label: str
    spec: vh.ValidationSpec
    actual_time: Optional[float]
    tool_results: Dict[ToolName, ToolResult]
    output_root: Path
    error: Optional[str] = None


COLOR_MAP: Dict[str, str] = {
    "actual": "#4c566a",
    "deepflow": "#1f77b4",
    "stg": "#ff7f0e",
    "mlsynth": "#2ca02c",
}


def _slugify(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return cleaned or "spec"


def _tool_result_to_dict(result: ToolResult) -> Dict[str, object]:
    return {
        "name": result.name,
        "seconds": result.seconds,
        "artifact_dir": str(result.artifact_dir) if result.artifact_dir else None,
        "error": result.error,
    }


def _tool_result_from_dict(data: Mapping[str, object]) -> ToolResult:
    return ToolResult(
        name=str(data.get("name")),
        seconds=data.get("seconds") if isinstance(data.get("seconds"), (int, float)) else None,
        artifact_dir=Path(data["artifact_dir"]) if data.get("artifact_dir") else None,
        error=str(data["error"]) if data.get("error") else None,
    )


def _spec_run_to_dict(run: SpecRun) -> Dict[str, object]:
    return {
        "label": run.label,
        "actual_time": run.actual_time,
        "tool_results": {name: _tool_result_to_dict(res) for name, res in run.tool_results.items()},
        "output_root": str(run.output_root),
        "error": run.error,
        "spec_metadata": run.spec.metadata if run.spec else {},
        "order": getattr(run.spec, "order", 0) if run.spec else 0,
    }


def _spec_run_from_dict(data: Mapping[str, object]) -> SpecRun:
    label = str(data.get("label") or "spec")
    spec = vh.ValidationSpec(
        label=label,
        metadata=data.get("spec_metadata") or {},
        order=int(data.get("order") or 0),
    )
    tool_results_raw = data.get("tool_results") or {}
    tool_results: Dict[ToolName, ToolResult] = {}
    if isinstance(tool_results_raw, Mapping):
        for name, result_data in tool_results_raw.items():
            tool_results[name] = _tool_result_from_dict({"name": name, **(result_data or {})})
    output_root_val = data.get("output_root") or "."
    output_root = Path(output_root_val)
    return SpecRun(
        label=label,
        spec=spec,
        actual_time=data.get("actual_time") if isinstance(data.get("actual_time"), (int, float)) else None,
        tool_results=tool_results,
        output_root=output_root,
        error=str(data["error"]) if data.get("error") else None,
    )


def _merge_dicts(base: Dict[str, object], overrides: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not overrides:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    vh._deep_update(merged, overrides)
    return merged


def _write_yaml(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _materialize_configs(spec: vh.ValidationSpec, dest_dir: Path) -> Tuple[Path, Path]:
    if not spec.model_config_path or not spec.hardware_config_path:
        raise ValueError(f"Spec '{spec.label}' is missing config paths.")
    base_model = vh._load_yaml(str(spec.model_config_path))
    base_hw = vh._load_yaml(str(spec.hardware_config_path))
    model_cfg = _merge_dicts(base_model, spec.model_overrides or {})
    hw_cfg = _merge_dicts(base_hw, spec.hardware_overrides or {})

    model_path = dest_dir / "model.yaml"
    hw_path = dest_dir / "hardware.yaml"
    _write_yaml(model_path, model_cfg)
    _write_yaml(hw_path, hw_cfg)
    return model_path, hw_path


def _extract_actual_time(spec: vh.ValidationSpec, actual_lookup: Dict[Tuple, float]) -> Optional[float]:
    meta = spec.metadata or {}
    key = (
        meta.get("model"),
        int(meta.get("batch")),
        int(meta.get("mb")),
        int(meta.get("dp")),
        int(meta.get("tp")),
        int(meta.get("pp")),
        int(meta.get("cp")),
        bool(meta.get("tp_sp")),
        str(meta.get("recomputation")),
    )
    return actual_lookup.get(key)


def _short_label(spec: vh.ValidationSpec) -> str:
    meta = spec.metadata or {}
    model = meta.get("model", "model")
    batch = meta.get("batch", "?")
    mb = meta.get("mb", "?")
    dp = meta.get("dp", "?")
    tp = meta.get("tp", "?")
    pp = meta.get("pp", "?")
    cp = meta.get("cp", "?")
    return f"{model} bs={batch}/mb={mb} dp={dp} tp={tp} pp={pp} cp={cp}"


def _build_base_config(model_cfg: Path, hw_cfg: Path) -> Dict[str, object]:
    config = copy.deepcopy(comp_wrapper.GLOBAL_CONFIG)
    config["model_config"] = model_cfg
    config["hardware_config"] = hw_cfg
    config["generate_visuals"] = False
    config["dry_run"] = False
    # Allow DeepFlow to run its full AstraSim path; isolation short-circuits after forward.
    config["isol_astra"] = False
    if isinstance(config.get("mlsynth"), dict):
        config["mlsynth"]["retain_generator_outputs"] = True
    old_astra_bin = getattr(comp_wrapper, "ASTRASIM_OLD_BINARY", None)
    if old_astra_bin:
        stg_cfg = config.get("stg") if isinstance(config.get("stg"), dict) else {}
        mlsynth_cfg = config.get("mlsynth") if isinstance(config.get("mlsynth"), dict) else {}
        if isinstance(stg_cfg, dict):
            stg_cfg.setdefault("astrasim_binary", old_astra_bin)
            config["stg"] = stg_cfg
        if isinstance(mlsynth_cfg, dict):
            mlsynth_cfg.setdefault("astrasim_binary", old_astra_bin)
            config["mlsynth"] = mlsynth_cfg
    return config


def _apply_isolated_paths(config: Dict[str, object], spec_root: Path, run_id: str) -> None:
    tmp_root = spec_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    stg_cfg: Dict[str, object] = {}
    if isinstance(config.get("stg"), dict):
        stg_cfg = config["stg"]  # type: ignore[assignment]
    config["stg"] = stg_cfg
    stg_cfg.setdefault("work_dir", tmp_root / "stg" / run_id)

    mlsynth_cfg: Dict[str, object] = {}
    if isinstance(config.get("mlsynth"), dict):
        mlsynth_cfg = config["mlsynth"]  # type: ignore[assignment]
    config["mlsynth"] = mlsynth_cfg
    mlsynth_cfg.setdefault("work_dir", tmp_root / "mlsynth" / run_id)

    deepflow_cfg: Dict[str, object] = {}
    if isinstance(config.get("deepflow"), dict):
        deepflow_cfg = config["deepflow"]  # type: ignore[assignment]
    config["deepflow"] = deepflow_cfg
    deepflow_cfg.setdefault("output_root", tmp_root / "deepflow" / run_id)


def _extract_timing(result: Dict[str, object]) -> Optional[float]:
    train = result.get("training_time_s") or result.get("training_time")
    if isinstance(train, (int, float)):
        return float(train)
    total = result.get("astrasim_total_time")
    if isinstance(total, (int, float)):
        return float(total)
    return None


def _with_output_root(root: Path):
    class _Ctx:
        def __enter__(self):
            self.prev = comp_wrapper.WRAPPER_OUTPUT_ROOT
            comp_wrapper.WRAPPER_OUTPUT_ROOT = root
            return self

        def __exit__(self, exc_type, exc, tb):
            comp_wrapper.WRAPPER_OUTPUT_ROOT = self.prev
    return _Ctx()


@contextmanager
def _silence_worker(log_path: Path):
    """Redirect stdout/stderr for a worker into a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file, ExitStack() as stack:
        stack.enter_context(redirect_stdout(log_file))
        stack.enter_context(redirect_stderr(log_file))
        yield


def _run_tool(name: ToolName, config: Dict[str, object]) -> ToolResult:
    handlers = {
        "deepflow": comp_wrapper.run_deepflow_annotated,
        "stg": comp_wrapper.run_stg,
        "mlsynth": comp_wrapper.run_mlsynth,
    }
    if name not in handlers:
        return ToolResult(name=name, seconds=None, error=f"Unknown tool {name}")
    try:
        result = handlers[name](config)
        timing = _extract_timing(result)
        artifact_dir = Path(result["artifact_dir"]) if result.get("artifact_dir") else None
        return ToolResult(name=name, seconds=timing, artifact_dir=artifact_dir)
    except Exception as exc:
        return ToolResult(name=name, seconds=None, error=str(exc))


def run_spec(
    spec: vh.ValidationSpec,
    actual_lookup: Dict[Tuple, float],
    tools: Sequence[ToolName],
    output_root: Path,
) -> SpecRun:
    label = _short_label(spec)
    slug = _slugify(label)
    spec_root = output_root / slug
    config_dir = spec_root / "configs"
    model_path, hw_path = _materialize_configs(spec, config_dir)
    actual_time = _extract_actual_time(spec, actual_lookup)
    base_config = _build_base_config(model_path, hw_path)
    _apply_isolated_paths(base_config, spec_root, slug)

    tool_results: Dict[ToolName, ToolResult] = {}
    with _with_output_root(spec_root):
        for name in tools:
            print(f"[nvidia_train_wrapper] Running {name} for {label}")
            tool_results[name] = _run_tool(name, copy.deepcopy(base_config))

    return SpecRun(
        label=label,
        spec=spec,
        actual_time=actual_time,
        tool_results=tool_results,
        output_root=spec_root,
    )


def _execute_spec(
    spec: vh.ValidationSpec,
    actual_lookup: Dict[Tuple, float],
    tools: Sequence[ToolName],
    output_root: Path,
) -> Dict[str, object]:
    label = _short_label(spec)
    slug = _slugify(label)
    spec_root = output_root / slug
    log_path = spec_root / "worker.log"
    with _silence_worker(log_path):
        try:
            spec_run = run_spec(spec, actual_lookup, tools, output_root)
        except Exception as exc:
            spec_run = SpecRun(
                label=label,
                spec=spec,
                actual_time=None,
                tool_results={},
                output_root=spec_root,
                error=str(exc),
            )
    _persist_spec_run(spec_run)
    return _spec_run_to_dict(spec_run)


def _load_existing_runs(output_root: Path, exclude_models: Optional[Sequence[str]] = None) -> List[SpecRun]:
    runs: List[SpecRun] = []
    exclude_set = set(exclude_models) if exclude_models else set()
    if not output_root.exists():
        return runs
    result_files = sorted(output_root.glob(f"*/{RESULT_FILENAME}"))
    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data.get("spec_metadata") or {}
            model_name = meta.get("model") or ""
            if exclude_set and (model_name in exclude_set or any(ex in str(data.get("label") or "") for ex in exclude_set)):
                continue
            runs.append(_spec_run_from_dict(data))
        except Exception:
            continue
    return sorted(runs, key=lambda run: getattr(run.spec, "order", 0))


def _collect_series(spec_runs: List[SpecRun], names: Sequence[ToolName]) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    for name in names:
        values: List[float] = []
        for run in spec_runs:
            if name == "actual":
                val = run.actual_time
            else:
                val = run.tool_results.get(name).seconds if name in run.tool_results else None
            values.append(float(val) if isinstance(val, (int, float)) else math.nan)
        series[name] = values
    return series


def _write_summary(spec_run: SpecRun) -> None:
    lines = [f"Spec: {spec_run.label}"]
    if spec_run.error:
        lines.append(f"Status: ERROR {spec_run.error}")
    else:
        lines.append(f"Status: success")
    lines.append(f"Actual time: {spec_run.actual_time}")
    for name, result in spec_run.tool_results.items():
        if result.error:
            lines.append(f"{name}: ERROR {result.error}")
        else:
            lines.append(f"{name}: {result.seconds}s -> {result.artifact_dir}")
    summary_path = spec_run.output_root / SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist_spec_run(spec_run: SpecRun) -> Path:
    spec_run.output_root.mkdir(parents=True, exist_ok=True)
    result_path = spec_run.output_root / RESULT_FILENAME
    result_path.write_text(json.dumps(_spec_run_to_dict(spec_run), indent=2), encoding="utf-8")
    _write_summary(spec_run)
    return result_path


def plot_results(spec_runs: List[SpecRun], tools: Sequence[ToolName], path: Path) -> Path:
    if not spec_runs:
        raise ValueError("No spec runs available to plot.")
    labels = [run.label for run in spec_runs]
    tool_list = ["actual", *tools]
    series = _collect_series(spec_runs, tool_list)
    width = 0.15
    x = list(range(len(labels)))

    fig_w = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    for idx, name in enumerate(tool_list):
        offsets = [pos + (idx - (len(tool_list) - 1) / 2) * width for pos in x]
        ax.bar(
            offsets,
            series[name],
            width=width,
            label=name,
            color=COLOR_MAP.get(name, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Training time (s)")
    ax.set_title("NVIDIA training comparison (DeepFlow/STG/MLSynth vs. actual)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run(
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    tp_sp: Optional[bool] = None,
    enable_plot: bool = True,
    show_progress: bool = False,
    emit_logs: bool = True,
    collect_only: bool = False,
    no_gen: bool = False,
    num_configs: Optional[int] = None,
    tools: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    exclude_models = {"GPT 1T"}
    output_root = WRAPPER_OUT_BASE / _slugify(device)
    output_root.mkdir(parents=True, exist_ok=True)

    spec_runs: List[SpecRun] = []
    if collect_only or no_gen:
        spec_runs = _load_existing_runs(output_root, exclude_models=exclude_models)
        actual_lookup: Dict[Tuple, float] = {}
        if not spec_runs and no_gen:
            raise ValueError("no_gen requested but no existing results were found to reuse.")
        if no_gen:
            print("[nvidia_train_wrapper] no_gen set: reusing existing results; no ET generation will be performed.")
    else:
        specs, actual_lookup, _, _ = nvidia_train.build_specs_for_device(
            device,
            models=models,
            tp_sp_only=tp_sp,
            exclude_models=exclude_models,
        )

        if not specs:
            raise ValueError("No specs to run after filtering.")
        if num_configs is not None:
            specs = specs[: max(0, int(num_configs))]

        worker_count = max(1, min(len(specs), os.cpu_count() or 1))
        print(f"Using {worker_count} workers")
        tools = list(tools) if tools else ["deepflow", "stg", "mlsynth"]
        ctx = mp.get_context("spawn")
        spec_runs_data: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            futures = [executor.submit(_execute_spec, spec, actual_lookup, tools, output_root) for spec in specs]
            progress = tqdm(total=len(futures), disable=not show_progress, desc="Specs", leave=False)
            try:
                for fut in as_completed(futures):
                    res = fut.result()
                    spec_runs_data.append(res)
                    progress.update(1)
            finally:
                progress.close()
        spec_runs = [_spec_run_from_dict(data) for data in sorted(spec_runs_data, key=lambda d: d.get("order", 0))]

    plot_path = None
    if enable_plot and spec_runs:
        plot_path = plot_results(spec_runs, ["deepflow", "stg", "mlsynth"], output_root / "train_comparison.png")

    if emit_logs:
        print("[nvidia_train_wrapper] Completed runs:")
        for run_res in spec_runs:
            print(f"  - {run_res.label}")
            print(f"    actual: {run_res.actual_time}")
            if run_res.error:
                print(f"    error: {run_res.error}")
            for name, res in run_res.tool_results.items():
                if res.error:
                    print(f"    {name}: ERROR {res.error}")
                else:
                    print(f"    {name}: {res.seconds}s -> {res.artifact_dir}")
        if plot_path:
            print(f"[nvidia_train_wrapper] Plot saved to {plot_path}")

    return {
        "device": device,
        "spec_runs": spec_runs,
        "plot": plot_path,
    }


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run or collect NVIDIA training comparisons.")
    parser.add_argument("--collect_only", action="store_true", help="Only collect existing results and plot.")
    parser.add_argument("--no_gen", action="store_true", help="Reuse existing ETs/results without regeneration; plot only.")
    parser.add_argument("--num_configs", type=int, default=None, help="Limit number of configs (1=22B only, 2=22B + next, etc.)")
    parser.add_argument("--tools", type=str, default=None, help="Comma-separated tools to run (e.g., 'stg,mlsynth,deepflow'); default runs all.")
    args = parser.parse_args()

    devices = [
        "A100_korthi",
        # "A100_selene",
    ]
    all_runs: List[SpecRun] = []
    for dev in devices:
        print(f"=== Running {dev} training comparison ===")
        tool_list = [t.strip() for t in args.tools.split(",")] if args.tools else None
        result = run(
            device=dev,
            emit_logs=True,
            show_progress=True,
            collect_only=args.collect_only,
            no_gen=args.no_gen,
            num_configs=args.num_configs,
            tools=tool_list,
        )
        all_runs.extend(result.get("spec_runs", []))  # type: ignore

    if all_runs:
        combined_plot = plot_results(all_runs, ["deepflow", "stg", "mlsynth"], WRAPPER_OUT_BASE / "train_combined.png")
        print(f"[nvidia_train_wrapper] Combined plot saved to {combined_plot}")
