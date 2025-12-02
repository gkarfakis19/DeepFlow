import os
from typing import Dict, Tuple

import yaml


class DerateConfigError(ValueError):
    pass


def load_derate_config(path: str) -> Dict[int, float]:
    """Load hw_id->derate factor mapping from YAML."""
    if not path:
        raise DerateConfigError("Derate config path is empty")
    if not os.path.exists(path):
        raise DerateConfigError(f"Derate config '{path}' does not exist")
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise DerateConfigError("Derate config is empty")
    if not isinstance(data, dict):
        raise DerateConfigError("Derate config must be a mapping of hw_id:int -> factor:float")

    factors: Dict[int, float] = {}
    for key, raw_value in data.items():
        try:
            hw_id = int(key)
        except (TypeError, ValueError) as exc:
            raise DerateConfigError(f"Derate config key '{key}' is not an integer") from exc
        try:
            factor = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise DerateConfigError(f"Derate factor for hw_id {hw_id} must be numeric") from exc
        if factor <= 0.0:
            raise DerateConfigError(f"Derate factor for hw_id {hw_id} must be > 0")
        factors[hw_id] = factor

    if not factors:
        raise DerateConfigError("Derate config contained no entries")

    max_hw = max(factors.keys())
    expected_keys = set(range(max_hw + 1))
    if set(factors.keys()) != expected_keys:
        missing = sorted(expected_keys - set(factors.keys()))
        extra = sorted(set(factors.keys()) - expected_keys)
        raise DerateConfigError(
            f"Derate config keys must be contiguous from 0..{max_hw}. "
            f"Missing: {missing}; Extra: {extra}"
        )
    return factors
