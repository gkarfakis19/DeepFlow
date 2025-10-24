#!/usr/bin/env python3
"""Convert a Hugging Face transformer config.json into a DeepFlow LLM YAML config."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

import yaml


def _fetch_hf_config(model_id: str, revision: str = "main") -> dict:
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - runtime fetch
        raise SystemExit(f"Failed to fetch config.json for '{model_id}' ({exc.code} {exc.reason}).")
    except urllib.error.URLError as exc:  # pragma: no cover - runtime fetch
        raise SystemExit(f"Failed to reach Hugging Face: {exc.reason}.")

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON returned for '{model_id}': {exc}.")


def _first(cfg: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _infer_model_type(model_type_field: Optional[str]) -> str:
    if not model_type_field:
        return "gpt"
    lowered = model_type_field.lower()
    if "llama" in lowered:
        return "llama"
    if "gpt" in lowered or "opt" in lowered or "mpt" in lowered:
        return "gpt"
    # default catch-all
    return "gpt"


def _build_yaml_config(cfg: dict, args: argparse.Namespace) -> dict:
    hidden_dim = _first(
        cfg,
        "hidden_size",
        "d_model",
        "dim",
    )
    if hidden_dim is None:
        raise SystemExit("Unable to deduce hidden dimension from config.json (missing hidden_size / d_model / dim).")

    num_layers = _first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    if num_layers is None:
        raise SystemExit("Unable to deduce number of layers from config.json.")

    num_heads = _first(cfg, "num_attention_heads", "n_head", "num_heads")
    if num_heads is None:
        raise SystemExit("Unable to deduce number of attention heads from config.json.")

    kv_heads = _first(
        cfg,
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        default=None,
    )

    attention_type = "mha"
    attention_block: Dict[str, Any] = {
        "attention_type": attention_type,
        "num_heads": int(num_heads),
    }
    if kv_heads is not None and int(kv_heads) > 0 and int(kv_heads) != int(num_heads):
        attention_block["attention_type"] = "gqa"
        attention_block["kv_heads"] = int(kv_heads)

    ffn_dim = _first(cfg, "intermediate_size", "mlp_dim", default=None)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = _first(
            cfg,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "sequence_length",
            default=2048,
        )


    tied_embeddings = bool(_first(cfg, "tie_word_embeddings", default=True))

    model_type = _infer_model_type(cfg.get("model_type"))

    vocab_size = _first(cfg, "vocab_size")
    if vocab_size is None:
        raise SystemExit("Unable to deduce vocab_size from config.json.")

    yaml_dict = {
        "model_param": {
            "mode": "LLM",
            "run_type": args.run_type,
            "tied_embeddings": tied_embeddings,
            "model_type": model_type,
            "batch_size": args.batch_size,
            "seq_len": int(seq_len),
            "decode_len": args.decode_len,
            "hidden_dim": int(hidden_dim),
            "attention": attention_block,
            "ffn_dim": int(ffn_dim) if ffn_dim is not None else None,
            "ffn_mult": None,
            "vocab_size": int(vocab_size),
            "num_layers": int(num_layers),
            "n_tokens": None,
            "all_reduce": "every layer"
        },
        "inference_param": {
            "sample_every": -1
        },
    }

    return yaml_dict


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Hugging Face model identifier, e.g. meta-llama/Llama-2-7b-hf")
    parser.add_argument("--revision", default="main", help="Model revision/branch to use (default: main)")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size to encode in the config (default: 1)")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length. Defaults to HF config max position embeddings if not provided")
    parser.add_argument("--decode-len", type=int, default=0, help="Decode sequence length (default: 0)")
    parser.add_argument("--run-type", default="training", help="Config run_type field ('training' or 'inference')")
    parser.add_argument("--output", "-o", default=None, help="Path to write YAML (defaults to stdout)")
    return parser.parse_args(argv)


def _str_to_bool_or_none(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered in {"none", "null", "auto"}:
        return None
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = _fetch_hf_config(args.model_id, revision=args.revision)
    yaml_config = _build_yaml_config(cfg, args)

    yaml_dump = yaml.dump(
        yaml_config,
        default_flow_style=False,
        sort_keys=False,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(yaml_dump)
    else:
        sys.stdout.write(yaml_dump)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
