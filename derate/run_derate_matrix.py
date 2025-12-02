#!/usr/bin/env python3
import os
import subprocess
import tempfile
import yaml
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DERATE_DIR = os.path.abspath(os.path.dirname(__file__))

# Configurable knobs
MAX_DERATE = float(os.environ.get("DERATE_MAX", "1.10"))
GAUSS_MEAN = (1.0 + MAX_DERATE) / 2.0
GAUSS_STD = float(os.environ.get("DERATE_GAUSS_STD", (MAX_DERATE - GAUSS_MEAN) / 2.0))
GAUSS_SEED = 42

SCENARIOS = [
    {
        "name": "derate_70b_train",
        "script": "derate_70b.sh",
        "derate_file": "derate_70b.yaml",
        "mode": "train",
    },
    {
        "name": "derate_70b_inf",
        "script": "derate_70b_inf.sh",
        "derate_file": "derate_70b_inf.yaml",
        "mode": "inference",
    },
    {
        "name": "derate_70b_longctx_train",
        "script": "derate_70b_longctx.sh",
        "derate_file": "derate_70b_longctx.yaml",
        "mode": "train",
    },
    # {
    #     "name": "derate_70b_longctx_inf",
    #     "script": "derate_70b_longctx_inf.sh",
    #     "derate_file": "derate_70b_longctx_inf.yaml",
    #     "mode": "inference",
    # },
]


def load_derate_map(path: str) -> Dict[int, float]:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    factors = {}
    for k, v in data.items():
        factors[int(k)] = float(v)
    return factors


def write_derate_map(factors: Dict[int, float]) -> str:
    fd, path = tempfile.mkstemp(prefix="derate_", suffix=".yaml", dir=DERATE_DIR)
    os.close(fd)
    with open(path, "w") as fh:
        yaml.safe_dump({int(k): float(v) for k, v in sorted(factors.items())}, fh, default_flow_style=False)
    return path


def variant_maps(base: Dict[int, float]) -> List[Tuple[str, Dict[int, float]]]:
    keys = sorted(base.keys())
    n = len(keys)
    first_key = keys[0]

    all_ones = {k: 1.0 for k in keys}
    all_x = {k: MAX_DERATE for k in keys}
    first_x = {k: (MAX_DERATE if k == first_key else 1.0) for k in keys}

    rng = random.Random(GAUSS_SEED)
    gauss = {}
    for k in keys:
        val = rng.gauss(GAUSS_MEAN, GAUSS_STD)
        val = min(MAX_DERATE, max(1.0, val))
        gauss[k] = val

    return [
        ("baseline_ones", all_ones),
        ("first_max", first_x),
        ("all_max", all_x),
        ("gaussian", gauss),
    ]


def run_shell(script: str, derate_path: str) -> Tuple[str, str, int]:
    env = os.environ.copy()
    env["DERATE_CONFIG_OVERRIDE"] = derate_path
    proc = subprocess.run(
        [os.path.join(DERATE_DIR, script)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout, derate_path, proc.returncode


def parse_training_time() -> float:
    result_path = os.path.join(ROOT, "output", "LLM", "LLM_training_results.txt")
    if not os.path.exists(result_path):
        raise RuntimeError("LLM_training_results.txt not found after run")
    with open(result_path, "r") as fh:
        lines = [line.strip() for line in fh]
    for line in reversed(lines):
        if line.startswith("Total Time:"):
            return float(line.split(":")[1])
    raise RuntimeError("Failed to parse Total Time from LLM_training_results.txt")


def parse_inference_time(log_text: str) -> float:
    for line in reversed(log_text.splitlines()):
        if "LLM inference time:" in line:
            tokens = line.strip().split()
            try:
                return float(tokens[3].rstrip("s"))
            except Exception:
                continue
    raise RuntimeError("Failed to parse inference time from log output")


def clear_training_result():
    result_path = os.path.join(ROOT, "output", "LLM", "LLM_training_results.txt")
    if os.path.exists(result_path):
        os.remove(result_path)


def run_variant(scenario: dict, variant: str, factors: Dict[int, float]) -> float:
    path = write_derate_map(factors)
    try:
        clear_training_result()
        log, _, code = run_shell(scenario["script"], path)
        if code != 0:
            raise RuntimeError(f"{scenario['script']} failed (variant {variant}). Output:\n{log}")
        if scenario["mode"] == "train":
            return parse_training_time()
        return parse_inference_time(log)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def format_table(rows: List[Tuple[str, str, float, float]]) -> str:
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*rows)]
    lines = []
    for row in rows:
        padded = [str(cell).ljust(col_widths[idx]) for idx, cell in enumerate(row)]
        lines.append(" | ".join(padded))
    return "\n".join(lines)


def main():
    summary_rows: List[Tuple[str, str, str, str]] = [("scenario", "variant", "time_s", "x_over_base")]

    for scenario in tqdm(SCENARIOS, desc="scenarios"):
        base_path = os.path.join(DERATE_DIR, scenario["derate_file"])
        base_map = load_derate_map(base_path)
        variants = variant_maps(base_map)
        base_time = None
        for variant_name, factors in tqdm(variants, desc=scenario["name"], leave=False):
            t = run_variant(scenario, variant_name, factors)
            if base_time is None:
                base_time = t
            ratio = t / base_time if base_time else 1.0
            summary_rows.append(
                (
                    scenario["name"],
                    f"{variant_name} (X={MAX_DERATE:.2f})",
                    f"{t:.4f}",
                    f"{ratio:.3f}x",
                )
            )

    print(format_table(summary_rows))


if __name__ == "__main__":
    main()
