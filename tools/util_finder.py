#!/usr/bin/env python3
"""
Utility tuner (core util, HBM bandwidth util) for training validation.

Modeled after overlap_finder.py but searches in two dimensions using a
coordinate-wise golden-section search. Objective is average absolute percent
error across Korthi+Selene training validation suites (nvidia_train.py).

Global knobs (edit below):
  - DEVICES: which training devices to validate (defaults: Korthi + Selene)
  - APPLY_BEST: write the best utils back to all hardware configs via mod_field
  - RANGES/TOL/ITER: search bounds and golden search controls
  - COORD_STEPS: number of coordinate-descent passes (keep small; each eval is slow)
"""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_train  # type: ignore
from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite  # type: ignore
from tools import mod_field

# ======= Global config =======
DEVICES: Sequence[str] = ("A100_korthi", "A100_selene")
APPLY_BEST = False

# Search bounds for utilities (floats in (0,1]). Set to None to skip tuning that axis.
CORE_UTIL_RANGE: Optional[Tuple[float, float]] = None
HBM_UTIL_RANGE: Optional[Tuple[float, float]] = (0.25, 0.55)
GOLD_TOL = 0.02       # stop when interval width < GOLD_TOL
GOLD_MAX_ITER = 6     # cap per-axis golden iterations
COORD_STEPS = 2       # how many coordinate-descent passes (keep small)
# =============================

# Exclude specific heavy/slow validation cases by metadata match.
EXCLUDE_TESTS: Tuple[Dict[str, Any], ...] = (
    # Skip 3k-GPU Selene GPT-1T case (dp=6, tp=8, pp=64, mb=512, batch=3072).
    {"device": "A100_selene", "model": "GPT 1T", "pp": 64},
)


def _merge_dict(target: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, val in updates.items():
        if isinstance(val, Mapping):
            current = target.get(key, {})
            if not isinstance(current, dict):
                current = {}
            target[key] = _merge_dict(dict(current), val)
        else:
            target[key] = val
    return target


def _with_utils(spec: ValidationSpec, core_util: Optional[float], hbm_util: Optional[float]) -> ValidationSpec:
    base_overrides: Dict[str, Any] = {}
    if spec.hardware_overrides and isinstance(spec.hardware_overrides, Mapping):
        base_overrides = _merge_dict({}, spec.hardware_overrides)
    util_overrides: Dict[str, Any] = {"tech_param": {}}
    if core_util is not None:
        util_overrides["tech_param"]["core"] = {"util": float(core_util)}
    if hbm_util is not None:
        util_overrides["tech_param"]["DRAM"] = {"util": float(hbm_util)}
    if not util_overrides["tech_param"]:
        return ValidationSpec(
            label=spec.label,
            model_overrides=spec.model_overrides,
            hardware_overrides=base_overrides,
            metadata=spec.metadata,
            model_config_path=spec.model_config_path,
            hardware_config_path=spec.hardware_config_path,
            order=spec.order,
        )
    base_overrides = _merge_dict(base_overrides, util_overrides)
    return ValidationSpec(
        label=spec.label,
        model_overrides=spec.model_overrides,
        hardware_overrides=base_overrides,
        metadata=spec.metadata,
        model_config_path=spec.model_config_path,
        hardware_config_path=spec.hardware_config_path,
        order=spec.order,
    )


def _avg_pct_error(errors: Iterable[float]) -> float:
    vals = [e for e in errors if not math.isnan(e)]
    if not vals:
        return float("inf")
    return sum(vals) / len(vals)


def _all_hardware_configs() -> List[Path]:
    roots = [
        PROJECT_ROOT / "configs" / "hardware-config",
        PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config",
    ]
    paths: List[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*.yaml")))
    return paths


def _should_exclude(meta: Mapping[str, Any]) -> bool:
    for rule in EXCLUDE_TESTS:
        matches = True
        for key, val in rule.items():
            if str(meta.get(key)) != str(val):
                matches = False
                break
        if matches:
            return True
    return False


def _filter_specs(
    specs: Sequence[ValidationSpec],
    actual_lookup: Mapping[Tuple[str, int, int, int, int, int, int, bool, str], float],
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int, int, int, bool, str], float]]:
    kept: List[ValidationSpec] = []
    kept_lookup: Dict[Tuple[str, int, int, int, int, int, int, bool, str], float] = {}
    for spec in specs:
        meta = spec.metadata or {}
        if _should_exclude(meta):
            continue
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
        if key in actual_lookup:
            kept_lookup[key] = actual_lookup[key]
        kept.append(spec)
    return kept, kept_lookup


@dataclass
class SearchResult:
    best_x: float
    best_y: float
    evaluations: List[Tuple[float, float]]


def golden_section_search(
    fn: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = GOLD_TOL,
    max_iter: int = GOLD_MAX_ITER,
    log_fn: Optional[Callable[[int, int, float, float], None]] = None,
) -> SearchResult:
    phi = (1 + 5 ** 0.5) / 2
    inv_phi = 1 / phi
    inv_phi_sq = inv_phi ** 2

    a, b = lo, hi
    h = b - a
    if h <= tol:
        mid = (a + b) / 2
        val = fn(mid)
        return SearchResult(best_x=mid, best_y=val, evaluations=[(mid, val)])

    n = int(math.ceil(math.log(tol / h) / math.log(inv_phi)))
    c = a + inv_phi_sq * h
    d = a + inv_phi * h
    yc = fn(c)
    yd = fn(d)
    if log_fn:
        log_fn(1, max_iter, c, yc)
        log_fn(2, max_iter, d, yd)
    evals: List[Tuple[float, float]] = [(c, yc), (d, yd)]

    iter_idx = 2
    for _ in range(min(n, max_iter)):
        if yc < yd:
            b = d
            d, yd = c, yc
            h = inv_phi * h
            c = a + inv_phi_sq * h
            yc = fn(c)
            evals.append((c, yc))
        else:
            a = c
            c, yc = d, yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = fn(d)
            evals.append((d, yd))
        iter_idx += 1
        if log_fn:
            log_fn(iter_idx, max_iter, evals[-1][0], evals[-1][1])
        if h < tol:
            break

    best_x, best_y = min(evals, key=lambda t: t[1])
    return SearchResult(best_x=best_x, best_y=best_y, evaluations=evals)


class UtilTuner:
    def __init__(self) -> None:
        self._eval_cache: Dict[Tuple[float, float], float] = {}

    def eval_pair(self, core_util: float, hbm_util: float) -> float:
        key = (round(core_util, 6), round(hbm_util, 6))
        if key in self._eval_cache:
            print(f"[CACHE HIT] core.util={core_util:.4f}, DRAM.util={hbm_util:.4f} -> {self._eval_cache[key]:.3f}")
            return self._eval_cache[key]

        all_errors: List[float] = []
        for device in DEVICES:
            specs, actual_lookup, base_model_path, base_hw_path = nvidia_train.build_specs_for_device(device)
            specs, actual_lookup = _filter_specs(specs, actual_lookup)
            if not specs:
                print(f"[SKIP DEVICE] {device}: no specs after filtering.")
                continue
            specs = [_with_utils(spec, core_util, hbm_util) for spec in specs]
            print(f"[EVAL] device={device}, specs={len(specs)}, core.util={core_util:.4f}, DRAM.util={hbm_util:.4f}")
            with io.StringIO() as buf, redirect_stdout(buf):
                results = run_validation_suite(
                    specs,
                    base_model_config_path=base_model_path,
                    base_hardware_config_path=base_hw_path,
                    result_parser=nvidia_train.parse_training_time,
                    run_perf_path=nvidia_train.RUN_PERF,
                )
            rows = nvidia_train.compute_pct_errors(results, actual_lookup)
            all_errors.extend(float(row["pct_error"]) for row in rows)
            print(f"[RESULTS] device={device}, pct_errors={ [float(row['pct_error']) for row in rows] }")

        avg_error = _avg_pct_error(all_errors)
        self._eval_cache[key] = avg_error
        print(f"[EVAL DONE] core.util={core_util:.4f}, DRAM.util={hbm_util:.4f}, avg_abs_pct_err={avg_error:.3f}")
        return avg_error

    def optimize(self) -> Tuple[Optional[float], Optional[float], float]:
        core_util = None if CORE_UTIL_RANGE is None else (CORE_UTIL_RANGE[0] + CORE_UTIL_RANGE[1]) / 2
        hbm_util = None if HBM_UTIL_RANGE is None else (HBM_UTIL_RANGE[0] + HBM_UTIL_RANGE[1]) / 2

        if CORE_UTIL_RANGE is None and HBM_UTIL_RANGE is None:
            raise ValueError("At least one of CORE_UTIL_RANGE or HBM_UTIL_RANGE must be set.")

        # If only one axis is active, do a single golden search on that axis.
        if CORE_UTIL_RANGE is None and HBM_UTIL_RANGE is not None:
            h_lo, h_hi = HBM_UTIL_RANGE
            print("Tuning only DRAM.util")
            res_hbm = golden_section_search(
                lambda y: self.eval_pair(core_util or 1.0, y),
                lo=h_lo,
                hi=h_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, y, v: print(f"  [{i}/{m}] hbm_util={y:.4f} -> {v:.3f}"),
            )
            best_val = res_hbm.best_y
            return core_util, res_hbm.best_x, best_val

        if HBM_UTIL_RANGE is None and CORE_UTIL_RANGE is not None:
            c_lo, c_hi = CORE_UTIL_RANGE
            print("Tuning only core.util")
            res_core = golden_section_search(
                lambda x: self.eval_pair(x, hbm_util or 1.0),
                lo=c_lo,
                hi=c_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, x, y: print(f"  [{i}/{m}] core_util={x:.4f} -> {y:.3f}"),
            )
            best_val = res_core.best_y
            return res_core.best_x, hbm_util, best_val

        # Both axes active: coordinate descent.
        core_lo, core_hi = CORE_UTIL_RANGE  # type: ignore[assignment]
        hbm_lo, hbm_hi = HBM_UTIL_RANGE    # type: ignore[assignment]
        core = float(core_util)
        hbm = float(hbm_util)
        best_val = self.eval_pair(core, hbm)

        for step in range(COORD_STEPS):
            print(f"\n--- Coordinate step {step + 1}/{COORD_STEPS} ---")
            # Optimize core util with HBM fixed
            print(f"Optimizing core util (hbm_util={hbm:.4f})")
            res_core = golden_section_search(
                lambda x: self.eval_pair(x, hbm),
                lo=core_lo,
                hi=core_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, x, y: print(f"  [{i}/{m}] core_util={x:.4f} -> {y:.3f}"),
            )
            core = res_core.best_x
            best_val = res_core.best_y

            # Optimize HBM util with core fixed
            print(f"Optimizing HBM util (core_util={core:.4f})")
            res_hbm = golden_section_search(
                lambda y: self.eval_pair(core, y),
                lo=hbm_lo,
                hi=hbm_hi,
                tol=GOLD_TOL,
                max_iter=GOLD_MAX_ITER,
                log_fn=lambda i, m, y, v: print(f"  [{i}/{m}] hbm_util={y:.4f} -> {v:.3f}"),
            )
            hbm = res_hbm.best_x
            best_val = min(best_val, res_hbm.best_y)

        best_val = self.eval_pair(core, hbm)
        return core, hbm, best_val


def main() -> None:
    tuner = UtilTuner()
    best_core, best_hbm, best_err = tuner.optimize()
    print("\n=== Best utils ===")
    print(f"  core.util = {best_core if best_core is None else f'{best_core:.4f}'}")
    print(f"  DRAM.util = {best_hbm if best_hbm is None else f'{best_hbm:.4f}'}")
    print(f"  avg abs pct error = {best_err:.3f}")

    if APPLY_BEST:
        core_val = None if best_core is None else round(best_core, 2)
        hbm_val = None if best_hbm is None else round(best_hbm, 2)
        print("\nApplying tuned utils to all hardware configs")
        for cfg_path in _all_hardware_configs():
            if core_val is not None:
                msg1 = mod_field.set_field(cfg_path, "tech_param.core.util", core_val, dry_run=False)
                print(f"  {msg1}")
            if hbm_val is not None:
                msg2 = mod_field.set_field(cfg_path, "tech_param.DRAM.util", hbm_val, dry_run=False)
                print(f"  {msg2}")


if __name__ == "__main__":
    main()
