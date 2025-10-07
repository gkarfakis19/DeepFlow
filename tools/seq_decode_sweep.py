#!/usr/bin/env python3

"""
Quick sweep script for updating LLM_inf.yaml with different seq/decode lengths,
running the DeepFlow inference model, collecting metrics, and plotting results.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "model-config" / "LLM_inf.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOT_PATH = OUTPUT_DIR / "seq_decode_sweep.png"
CSV_PATH = OUTPUT_DIR / "seq_decode_sweep.csv"
RUN_COMMAND = ["uv", "run", "./examples/llm.sh"]


SEQ_SWEEP: Iterable[int] = (
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16536,
)


@dataclass
class SweepResult:
    seq_len: int
    decode_len: int
    prefill_len: int
    ttft_s: float
    avg_decode_throughput: float
    throughput_samples: List[float]
    inference_time_s: float | None = None


def _read_config() -> str:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    return CONFIG_PATH.read_text()


def _extract_lengths(config_text: str) -> tuple[int, int]:
    seq_match = re.search(r"^\s*seq_len:\s*(\d+)", config_text, re.MULTILINE)
    decode_match = re.search(r"^\s*decode_len:\s*(\d+)", config_text, re.MULTILINE)
    if not seq_match or not decode_match:
        raise ValueError("Could not locate seq_len/decode_len in config.")
    return int(seq_match.group(1)), int(decode_match.group(1))


def _write_lengths(config_text: str, seq_len: int, decode_len: int) -> None:
    updated = re.sub(
        r"(^\s*seq_len:\s*)\d+",
        rf"\g<1>{seq_len}",
        config_text,
        count=1,
        flags=re.MULTILINE,
    )
    updated = re.sub(
        r"(^\s*decode_len:\s*)\d+",
        rf"\g<1>{decode_len}",
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    CONFIG_PATH.write_text(updated)


def _run_model() -> str:
    completed = subprocess.run(
        RUN_COMMAND,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(RUN_COMMAND)} failed with code {completed.returncode}\n{output}"
        )
    return output


def _parse_metrics(output: str) -> tuple[float, List[float], float | None]:
    ttft_match = re.search(r"LLM time to first token:\s*([0-9.]+)s", output)
    if not ttft_match:
        raise ValueError("Unable to parse time to first token from output.")
    ttft = float(ttft_match.group(1))

    throughput_match = re.search(
        r"Decode throughput tok/s:\s*start=([0-9.]+),\s*mid\(.*?\)=([0-9.]+),\s*end=([0-9.]+)",
        output,
    )
    if not throughput_match:
        raise ValueError("Unable to parse decode throughput metrics from output.")
    throughputs = [float(throughput_match.group(i)) for i in range(1, 4)]

    inference_match = re.search(r"LLM inference time:\s*([0-9.]+)s", output)
    inference_time = float(inference_match.group(1)) if inference_match else None

    return ttft, throughputs, inference_time


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib is required. Install it with `uv pip install matplotlib`.", file=sys.stderr)
        return 1

    original_config = _read_config()
    seq_default, decode_default = _extract_lengths(original_config)
    prefill_len = seq_default - decode_default
    if prefill_len <= 0:
        raise ValueError("Prefill length must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[SweepResult] = []

    try:
        for seq_len in SEQ_SWEEP:
            decode_len = seq_len - prefill_len
            if decode_len <= 0:
                print(f"Skipping seq_len={seq_len}: derived decode_len={decode_len} invalid.")
                continue

            print(f"\nRunning sweep point seq_len={seq_len}, decode_len={decode_len} (prefill={prefill_len})...")
            _write_lengths(original_config, seq_len, decode_len)

            run_output = _run_model()
            ttft, throughputs, inference_time = _parse_metrics(run_output)
            avg_throughput = statistics.mean(throughputs)

            results.append(
                SweepResult(
                    seq_len=seq_len,
                    decode_len=decode_len,
                    prefill_len=prefill_len,
                    ttft_s=ttft,
                    avg_decode_throughput=avg_throughput,
                    throughput_samples=throughputs,
                    inference_time_s=inference_time,
                )
            )

            print(
                f"  -> TTFT={ttft:.3f}s, avg decode throughput={avg_throughput:.2f} tok/s "
                f"(samples: {', '.join(f'{v:.2f}' for v in throughputs)})"
            )

    finally:
        CONFIG_PATH.write_text(original_config)

    if not results:
        print("No successful sweep points collected.")
        return 1

    # Plot results.
    decode_vals = [r.decode_len for r in results]
    ttft_vals = [r.ttft_s for r in results]
    throughput_vals = [r.avg_decode_throughput for r in results]

    fig, ax_ttft = plt.subplots(figsize=(8, 5))
    ttft_line = ax_ttft.plot(
        decode_vals,
        ttft_vals,
        marker="o",
        color="tab:blue",
        label="Time to first token",
    )[0]
    ax_ttft.set_xlabel("Decode length (tokens)")
    ax_ttft.set_ylabel("Time to first token (s)", color="tab:blue")
    ax_ttft.tick_params(axis="y", labelcolor="tab:blue")

    ax_throughput = ax_ttft.twinx()
    throughput_line = ax_throughput.plot(
        decode_vals,
        throughput_vals,
        marker="s",
        color="tab:orange",
        label="Average decode throughput",
    )[0]
    ax_throughput.set_ylabel("Average decode throughput (tok/s)", color="tab:orange")
    ax_throughput.tick_params(axis="y", labelcolor="tab:orange")

    ax_ttft.set_title(f"LLM decode sweep (prefill={prefill_len} tokens)")
    ax_ttft.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)

    lines = [ttft_line, throughput_line]
    labels = [line.get_label() for line in lines]
    ax_ttft.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    # Save raw results as CSV for quick inspection.
    with CSV_PATH.open("w", encoding="ascii") as csv_file:
        csv_file.write("seq_len,decode_len,prefill_len,ttft_s,avg_decode_throughput,start_tput,mid_tput,end_tput,inference_time_s\n")
        for r in results:
            inference = f"{r.inference_time_s:.6f}" if r.inference_time_s is not None else ""
            csv_file.write(
                f"{r.seq_len},{r.decode_len},{r.prefill_len},{r.ttft_s:.6f},{r.avg_decode_throughput:.6f},"
                f"{r.throughput_samples[0]:.6f},{r.throughput_samples[1]:.6f},{r.throughput_samples[2]:.6f},{inference}\n"
            )

    print(f"\nSaved plot to {PLOT_PATH}")
    print(f"Saved CSV to {CSV_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
