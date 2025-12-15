#!/tools/lm-venv/py3.6-tf-1.3.0-svail/bin/python
import argparse
import math
import os
import sys
import config
import time
import atexit
from astrasim_lib import ensure_chakra_available
import pandas as pd
import yaml
import shutil
import util
from util import log_message, flush_log_queue, extend_log

import graphviz_async
from tile import TiledGEMM, formatBytes
from time_calculation import TimeCalculation
from time_calculation_LLM import TimeCalculationLLM
from time_calculation_inf import TimeCalculationLLMInference

# Cache handling policy for AstraSim integration.
# Options: "NO CACHE", "CACHE READONLY", "CACHE READWRITE"
cache_handling = "CACHE READWRITE"
_CACHE_MODE_MAP = {
    "NO CACHE": "NO_CACHE",
    "CACHE READONLY": "CACHE_READONLY",
    "CACHE READWRITE": "CACHE_READWRITE",
}
os.environ["DEEPFLOW_ASTRA_CACHE_MODE"] = _CACHE_MODE_MAP.get(
    cache_handling.strip().upper(), "CACHE_READWRITE"
)

# Default location for artifacts emitted by run_perf.
DEFAULT_OUTPUT_DIR = "output"

# Global wall-clock timer: report total program runtime at exit
_program_start_time = time.perf_counter()

def _report_total_wall_time() -> None:
    try:
        elapsed = time.perf_counter() - _program_start_time
        print("DeepFlow wall-clock time: {:.2f}s".format(elapsed))
    except Exception:
        # Best-effort only
        pass

atexit.register(flush_log_queue)
atexit.register(_report_total_wall_time)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run performance analysis for LSTM, GEMM, or LLM models.")
    parser.add_argument("--hardware_config", required=True, help="Path to the hardware configuration file.")
    parser.add_argument("--model_config", required=True, help="Path to the model configuration file.")
    return parser.parse_args()

def get_mode_from_config(model_config_path):
    """Read the mode from the model configuration file."""
    with open(model_config_path, "r") as f:
        config_data = yaml.safe_load(f)  # Parse the YAML file
    
    # Access 'mode' under 'model_param'
    model_param = config_data.get("model_param")
    if not model_param or "mode" not in model_param:
        raise ValueError("Error: 'mode' is not specified in the model configuration file under 'model_param'.")
    
    return model_param["mode"]


def _validate_astrasim_dependencies(hw_config) -> None:
    backend = getattr(hw_config, "execution_backend", None)
    model = getattr(backend, "model", "") if backend else ""
    if str(model).lower() != "astra":
        return
    try:
        ensure_chakra_available()
    except RuntimeError as exc:
        raise RuntimeError(
            "Hardware configuration requests the AstraSim execution backend, but the Chakra protobuf dependencies "
            "are not available. Install or build the AstraSim externals before running with execution_backend.model='astra'."
        ) from exc

def _validate_network_topology(hw_config) -> None:
    backend = getattr(hw_config, "execution_backend", None)
    model = getattr(backend, "model", "analytical") if backend else "analytical"
    network_layout = getattr(hw_config, "network_layout", None)

    if str(model).lower() == "analytical" and network_layout:
        for dim in getattr(network_layout, "dimensions", ()):
            topo = str(getattr(dim, "topology_type", "ring")).lower()
            if topo != "ring":
                raise RuntimeError(
                    "Non-ring network topologies are not supported in analytical mode. "
                    "Only execution_backend.model='astra' (requires a valid AstraSim install) supports non-ring networks."
                )


def run_LSTM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode
):
    
    # exp_path = os.path.expandvars(os.path.expanduser(exp_config))
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    _validate_astrasim_dependencies(exp_hw_config)
    _validate_network_topology(exp_hw_config)
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)
    output_file = exp_dir + "/summary_%s.txt" % (
        mode,
    ) 


    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)
    

    tot_time, tot_param = TC.calcTime()
    TC.printSysConfig(exp_hw_config, exp_model_config, output_file)

    with open(output_file, "a+") as f:
        f.write("\n\n==============================================\n")
        f.write("Performance Results\n")
        f.write("==============================================\n")
        f.write("Time: {0:.8f}\n".format(tot_time))
        f.write("Params (Billion): {0:.8f}\n".format(tot_param / 1e9))
    print("Performance Results written to {}".format(output_file))
    # Emit total time for astra_test parsing
    print("Total time: {}".format(tot_time))

def run_GEMM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode
):
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    _validate_astrasim_dependencies(exp_hw_config)
    _validate_network_topology(exp_hw_config)
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)


    TC = TimeCalculation(exp_hw_config, exp_model_config, mode)

    # Forward timing
    forward_time = None
    forward_red = 0.0
    if TC.kp1 == 1 and TC.kp2 == 1:  # no parallelism
        forward_time = TC.getCf(TC.M, TC.K, TC.N)
    elif TC.t == "CR":
        gemm_time, forward_red = TC.getDistGEMM_f_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cf_CR")
        forward_time = gemm_time + forward_red
    elif TC.t == "RC":
        gemm_time, forward_red = TC.getDistGEMM_f_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cf_RC")
        forward_time = gemm_time + forward_red
    else:
        print("Incorrect parallelism type, CR: Column-Row, RC: Row-Column")
        sys.exit(1)

    # Optional backward timing + dp reduction
    backward_time = 0.0
    dp_reduction_time = 0.0
    backward_red = 0.0
    if getattr(TC.model, "backward", False):
        if TC.kp1 == 1 and TC.kp2 == 1:
            grad_act_time, _, _, _ = TC.getGEMMTime(TC.M, TC.N, TC.K, "Cb_act")
            grad_wt_time, _, _, _ = TC.getGEMMTime(TC.K, TC.M, TC.N, "Cb_wt")
            backward_time = grad_act_time + grad_wt_time
        elif TC.t == "CR":
            gemm_time, bg_red = TC.getDistGEMM_b_kp1(TC.M, TC.K, TC.N, TC.kp1, "Cb_CR")
            backward_time = gemm_time + bg_red
            backward_red = bg_red
        elif TC.t == "RC":
            gemm_time, bg_red = TC.getDistGEMM_b_kp2(TC.M, TC.K, TC.N, TC.kp1, TC.kp2, "Cb_RC")
            backward_time = gemm_time + bg_red
            backward_red = bg_red
        # Data-parallel reduction after backward, if applicable
        if TC.dp and TC.dp > 1:
            dp_reduction_time = TC.getDataParallelReduction(
                k=TC.K,
                n=TC.N,
                dim1=TC.kp1,
                dim2=TC.kp2,
                name="GEMM Reduction",
            )
            backward_red += dp_reduction_time

    total_time = forward_time + backward_time + dp_reduction_time

    output_file = exp_dir + "/summary_mode%s_M%s_K%s_N%s.txt" % (mode, TC.M, TC.K, TC.N)
    with open(output_file, "w") as f:
        # Forward/Backward breakdown (no tiling)
        f.write("Forward Compute Time: {}\n".format(forward_time - forward_red))
        f.write("Forward Reduction Time: {}\n".format(forward_red))
        if getattr(TC.model, "backward", False):
            f.write("Backward Compute Time: {}\n".format(backward_time - backward_red))
            f.write("Backward Reduction Time: {}\n".format(backward_red))
            if dp_reduction_time > 0:
                f.write("DP Reduction Time: {}\n".format(dp_reduction_time))
        f.write("Total Time: {}\n".format(total_time))
    print("Performance Results written to {}".format(output_file))
    # Emit lines for astra_test parsing
    print("Total time: {}".format(total_time))
    print("Reduction time: {}".format(forward_red + backward_red))
    print("Reduction FWD time: {}".format(forward_red))
    print("Reduction BWD time: {}".format(backward_red))
    return


def run_LLM(
    exp_hw_config_path,
    exp_model_config_path,
    exp_dir,
    mode):

    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    _validate_astrasim_dependencies(exp_hw_config)
    _validate_network_topology(exp_hw_config)
    exp_model_config = config.parse_config(exp_model_path, config_type=mode)

    llm_run_type = getattr(exp_model_config.model_config, "run_type", "training")
    if str(llm_run_type).lower() == "inference":
        _run_llm_inference(exp_hw_config, exp_model_config, exp_dir, mode)
        return

    _run_llm_training(exp_hw_config, exp_model_config, exp_dir, mode)


def _run_llm_training(exp_hw_config, exp_model_config, exp_dir, mode):
    output_file = os.path.join(exp_dir, "LLM_training_results.txt")
    tc_llm = TimeCalculationLLM(exp_hw_config, exp_model_config, mode, output_dir=exp_dir)
    total_time = tc_llm.calc_time_llm()
    topology_lines = util.network_topology_summary_training(exp_hw_config)

    with open(output_file, "a+") as handle:
        handle.write("\n\n==============================================\n")
        handle.write("Performance Results\n")
        handle.write("==============================================\n")
        handle.write("Execution Mode: {}\n".format(tc_llm.execution_mode.value))
        handle.write("Total Time: {0:.8f}\n".format(total_time))
        handle.write("\n")
        handle.write("For more info, turn on debug flags. See examples/llm_astra_inference_debug_graphviz.sh\n")
        handle.write("\n".join(topology_lines))
        handle.write("\n")

    log_message("Training time for batch: {:.2f}s".format(tc_llm.get_time()), category="results")
    extend_log(topology_lines, category="network")
    log_message("LLM training results written to {}".format(output_file), category="results")
    warning_message = tc_llm.memory_capacity_warning()
    if warning_message:
        log_message(warning_message)


def _run_llm_inference(exp_hw_config, exp_model_config, exp_dir, mode):
    """Run LLM inference simulation including prefill + decode phases."""
    tc_inf = TimeCalculationLLMInference(exp_hw_config, exp_model_config, mode, output_dir=exp_dir)

    # Get total inference time (prefill + decode)
    inference_timing = tc_inf.calc_total_inference_time()
    total_time = inference_timing["total_inference_time"]
    decode_rates = inference_timing.get("decode_tokens_per_s") or {}

    log_message(
        "LLM inference time: {:.2f}s (mode={})".format(
            total_time, tc_inf.execution_mode.value
        ),
        category="results",
    )
    log_message(
        "LLM time to first token: {:.2f}s".format(
            inference_timing["time_to_first_token"],
        ),
        category="results",
    )
    dp_replicas = max(1, getattr(tc_inf, "dp", 1))
    batch_size = getattr(tc_inf, "batch_size", 1)
    if dp_replicas > 1:
        log_message(f"Data parallel replicas: {dp_replicas}", category="results")
    if decode_rates:
        # decode_rates are per-generation rates (tokens per second per generation)
        start_gen_rate = decode_rates.get("start", 0.0)
        mid_gen_rate = decode_rates.get("midpoint", 0.0)
        end_gen_rate = decode_rates.get("end", 0.0)
        mid_step = int(decode_rates.get("midpoint_step", 0.0))
        
        # Print per-generation rates
        log_message(
            "Decode sequences/s: start={:.2f}, mid(token {})={:.2f}, end={:.2f}".format(
                start_gen_rate,
                mid_step,
                mid_gen_rate,
                end_gen_rate,
            ),
            category="results",
        )

        # Print aggregate decode throughput (with batch_size and dp multipliers)
        log_message(
            "Aggregate decode throughput tok/s (batch={}, dp={}): start={:.2f}, mid(token {})={:.2f}, end={:.2f}".format(
                batch_size,
                dp_replicas,
                start_gen_rate * batch_size * dp_replicas,
                mid_step,
                mid_gen_rate * batch_size * dp_replicas,
                end_gen_rate * batch_size * dp_replicas,
            ),
            category="results",
        )

    topology_lines = util.network_topology_summary_inference(exp_hw_config)

    output_path = os.path.join(exp_dir, "LLM_inference_results.txt")
    os.makedirs(exp_dir, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n\n==============================================\n")
        handle.write("LLM Inference Results\n")
        handle.write("==============================================\n")
        handle.write(f"Execution Mode: {tc_inf.execution_mode.value}\n")
        handle.write(f"Inference Time for batch: {total_time:.2f}s\n")
        handle.write(f"Prefill Time: {inference_timing['prefill_time']:.3f}s\n")
        handle.write(f"Decode Time: {inference_timing['decode_time']:.3f}s\n")
        if dp_replicas > 1:
            handle.write(f"Data Parallel Replicas: {dp_replicas}\n")
        handle.write(f"Time to First Token: {inference_timing['time_to_first_token']:.3f}s\n")
        if decode_rates:
            start_gen_rate = decode_rates.get("start", 0.0)
            mid_gen_rate = decode_rates.get("midpoint", 0.0)
            end_gen_rate = decode_rates.get("end", 0.0)
            mid_step = int(decode_rates.get("midpoint_step", 0.0))
            
            handle.write(f"Decode Generations per Second: start={start_gen_rate:.2f}, mid(token {mid_step})={mid_gen_rate:.2f}, end={end_gen_rate:.2f}\n")
            handle.write(f"Aggregate Decode Throughput Tok/s (batch={batch_size}, dp={dp_replicas}): start={start_gen_rate * batch_size * dp_replicas:.2f}, mid(token {mid_step})={mid_gen_rate * batch_size * dp_replicas:.2f}, end={end_gen_rate * batch_size * dp_replicas:.2f}\n")
        handle.write("\n")
        handle.write("For more info, turn on debug flags. See examples/llm_astra_inference_debug_graphviz.sh\n")
        handle.write("\n".join(topology_lines))
        handle.write("\n")

    extend_log(topology_lines, category="network")
    log_message("LLM inference results written to {}".format(output_path), category="results")
    warning_message = tc_inf.memory_capacity_warning()
    if warning_message:
        log_message(warning_message)

if __name__ == "__main__":
    args = parse_arguments()
    # Load configurations
    config_hardware_path = args.hardware_config
    config_model_path = args.model_config
    output_dir = DEFAULT_OUTPUT_DIR

    # Read mode from the model configuration file
    mode = get_mode_from_config(config_model_path)
    exp_dir = os.path.join(output_dir, mode)
    # Check if the directory exists and delete it if it does
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    
    if mode == "LLM":
        run_LLM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    
    elif mode == "LSTM":
        log_message("Using LSTM parameters for computation...")
        run_LSTM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        
        )
    elif mode == "GEMM":
        log_message("Using GEMM parameters for computation...")
        run_GEMM(
            exp_hw_config_path=config_hardware_path,
            exp_model_config_path=config_model_path,
            exp_dir=exp_dir,
            mode=mode,
        )
    else:
        print("Invalid mode selected. Please choose 'LLM', 'LSTM', or 'GEMM'.")
        flush_log_queue()
        sys.exit(1)

    flush_log_queue()
