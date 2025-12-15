import os
import csv
import itertools
import argparse
import logging
from collections import OrderedDict
import re
from validation_scripts.helpers import (
    run_cutlass_profiler,
    CUTLASS_ENV_VAR,
    run_ncu_profiler,
    load_metrics_spec,
    parse_int_list,            
    read_cutlass_results,      
    ensure_cutlass_profiler,   
    get_metric_labels,         
)

from run_perf import import_config

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def parse_llm_shapes_file(filepath):
    """
    Parse llm_shapes.txt into a list of dicts, each containing model name and OrderedDict of operations.
    Each operation maps to a tuple (m,k,n) or (b,m,k,n).
    
    Returns: List[dict] where each dict has 'model' (str) and 'shapes' (OrderedDict[str, tuple])
    """
    models = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected format: "ModelName: OrderedDict([('op1', (dims)), ...])"
            match = re.match(r"^([^:]+):\s*OrderedDict\((.+)\)$", line)
            if not match:
                logger.warning(f"Skipping malformed line: {line}")
                continue
            model_name = match.group(1).strip()
            shapes_str = match.group(2).strip()
            
            # Parse the OrderedDict content: [('key', (tuple)), ...]
            shapes = OrderedDict()
            # Remove outer brackets
            shapes_str = shapes_str.strip('[]')
            # Split by '), (' pattern to separate entries
            entries = re.findall(r"\('([^']+)',\s*\(([^)]+)\)\)", shapes_str)
            for op_name, dims_str in entries:
                dims = tuple(map(int, dims_str.split(',')))
                shapes[op_name] = dims
            
            models.append({'model': model_name, 'shapes': shapes})
            logger.info(f"Parsed model {model_name} with {len(shapes)} operations")
    
    return models


def extract_gemm_shapes(models_data):
    """
    Extract GEMM shapes from parsed models data.
    Returns: List[dict] with keys: 'model', 'op_name', 'm', 'k', 'n', 'batch' (optional)
    """
    gemm_shapes = []
    for model_info in models_data:
        model_name = model_info['model']
        for op_name, dims in model_info['shapes'].items():
            if len(dims) == 3:
                m, k, n = dims
                gemm_shapes.append({
                    'model': model_name,
                    'op_name': op_name,
                    'm': m, 'k': k, 'n': n
                })
            elif len(dims) == 4:
                b, m, k, n = dims
                # For batched GEMMs, we profile with the actual m dimension
                gemm_shapes.append({
                    'model': model_name,
                    'op_name': op_name,
                    'm': m, 'k': k, 'n': n,
                    'batch': b
                })
            else:
                logger.warning(f"Skipping {model_name}:{op_name} with unsupported dims: {dims}")
    
    return gemm_shapes


def _safe_err(ncu_val, pred_val):
    if ncu_val is None or pred_val is None:
        return "", ""
    try:
        abs_err = pred_val - ncu_val
        pct_err = abs_err / ncu_val * 100 if ncu_val != 0 else ""
        return f"{abs_err:<.3f}", f"{pct_err:<.2f}"
    except Exception:
        return "", ""

def main():
    parser = argparse.ArgumentParser(
        description="Search CUTLASS kernels across GEMM and CTA sizes."
    )
    # parser.add_argument("--cutlass-path", required=True, help="Path to CUTLASS profiler binary")
    parser.add_argument("--m-list", default="256,1024,4096")
    parser.add_argument("--n-list", default="256,1024,4096")
    parser.add_argument("--k-list", default="256,1024,4096")
    parser.add_argument("--shapes-file", default="validation_scripts/llm_shapes.txt", 
                        help="Path to file containing LLM GEMM shapes (e.g., llm_shapes.txt)")
    parser.add_argument("--cta-m", default="64,128,256")
    parser.add_argument("--cta-n", default="64,128,256")
    parser.add_argument("--cta-k", default="32,64")
    parser.add_argument("--cutlass-output", default="validation_scripts/kernel_search_results.csv")
    parser.add_argument("--ncu-output", default="validation_scripts/cta_sweep/fastest.csv")
    parser.add_argument("--ncu-metrics", default="validation_scripts/metrics.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--cutlass-path", default=None, help="Path to cutlass_profiler binary; overrides env")
    parser.add_argument("--deepflow-compare", action="store_true",
                        help="Enable DeepFlow model comparison CSV output")
    parser.add_argument("--deepflow-output", default="validation_scripts/deepflow_comparison.csv",
                        help="Output CSV for DeepFlow vs NCU errors")
    parser.add_argument("--hw-config", default="configs/hardware-config/a100_80GB_no_parallelism.yaml",
                        help="Path to A100 hardware config")
    parser.add_argument("--sw-config", default="configs/model-config/GEMM.yaml",
                        help="Path to GEMM software config")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Ensure CUTLASS profiler path (no hardcoded default)
    ensure_cutlass_profiler(args.cutlass_path)  # added
    logger.info(f"{CUTLASS_ENV_VAR}={os.environ.get(CUTLASS_ENV_VAR, '')}")

    # Load NCU metric spec (labels, units) from file or fallback
    if args.ncu_metrics and os.path.isfile(args.ncu_metrics):
        try:
            ncu_metric_spec = load_metrics_spec(args.ncu_metrics)
            logger.info(f"Loaded NCU metrics spec from {args.ncu_metrics} ({len(ncu_metric_spec)} metrics)")
        except Exception as e:
            logger.warning(f"Failed to load metrics spec from {args.ncu_metrics}: {e}.")
            exit(1)
    else:
        logger.info("NCU metrics spec file not found.")
        exit(1)

    # Determine GEMM shapes source
    if args.shapes_file:
        if not os.path.isfile(args.shapes_file):
            logger.error(f"Shapes file not found: {args.shapes_file}")
            exit(1)
        logger.info(f"Loading GEMM shapes from {args.shapes_file}")
        models_data = parse_llm_shapes_file(args.shapes_file)
        gemm_configs = extract_gemm_shapes(models_data)
        logger.info(f"Extracted {len(gemm_configs)} GEMM shapes from {len(models_data)} models")
    else:
        # Use m/n/k lists
        m_vals = parse_int_list(args.m_list)   
        n_vals = parse_int_list(args.n_list)   
        k_vals = parse_int_list(args.k_list)
        gemm_configs = [
            {'m': m, 'k': k, 'n': n, 'model': 'synthetic', 'op_name': f'm{m}_k{k}_n{n}'}
            for m in m_vals for k in k_vals for n in n_vals
        ]
        logger.info(f"Using {len(gemm_configs)} synthetic GEMM shapes from m/k/n lists")

    cta_m_vals = parse_int_list(args.cta_m)
    cta_n_vals = parse_int_list(args.cta_n)
    cta_k_vals = parse_int_list(args.cta_k)

    cta_tuples = list(itertools.product(cta_m_vals, cta_n_vals, cta_k_vals))
    logger.info(f"CTA tuples ({len(cta_tuples)}): {cta_tuples}")

    out_path = args.cutlass_output
    ncu_out_path = args.ncu_output
    ncu_exists = os.path.isfile(ncu_out_path)          
    perform_ncu = not ncu_exists                       
    ncu_write_header = not ncu_exists

    # DeepFlow TimeCalculation initialization
    if args.deepflow_compare:
        try:
            deepflow_tc = import_config(args.hw_config, args.sw_config, "GEMM")
            logger.info("DeepFlow TimeCalculation initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepFlow TimeCalculation: {e}")
            deepflow_tc = None
    else:
        deepflow_tc = None

    # Accumulate NCU results for DeepFlow comparison
    ncu_rows = []  # each: dict(m,k,n, metrics_dict)

    # If NCU output already exists, load it and skip further NCU profiling
    if ncu_exists:
        logger.info(f"{ncu_out_path} exists. Skipping NCU profiling and loading existing metrics.")
        try:
            with open(ncu_out_path, "r", newline="") as f_in:
                reader = csv.reader(f_in)
                header = next(reader, None)
                if header is None:
                    logger.warning(f"{ncu_out_path} is empty. No NCU metrics to load.")
                else:
                    # First 7 columns are fixed: m,k,n,cta_m,cta_k,cta_n,stages
                    metric_labels = header[7:]
                    for row in reader:
                        if len(row) < 7:
                            continue
                        try:
                            m = int(row[0])
                            n = int(row[1])
                            k = int(row[2])
                            metrics = {
                                "cta_m": int(row[3]),
                                "cta_n": int(row[4]),
                                "cta_k": int(row[5]),
                                "stages": row[6],
                            }
                            # Parse remaining metrics as float when possible
                            for lab, val in zip(metric_labels, row[7:]):
                                try:
                                    metrics[lab] = float(val)
                                except ValueError:
                                    metrics[lab] = val
                            # Provide canonical aliases if present
                            if "latency_ms" not in metrics and "latency" in metrics:
                                metrics["latency_ms"] = metrics["latency"]
                            ncu_rows.append({"m": m, "k": k, "n": n, "metrics": metrics})
                        except Exception as e:
                            logger.debug(f"Skipping malformed NCU row: {row} ({e})")
            logger.info(f"Loaded {len(ncu_rows)} NCU metric rows from {ncu_out_path}")
        except Exception as e:
            logger.warning(f"Failed to read existing NCU metrics file {ncu_out_path}: {e}")

    # If CUTLASS results already exist, reuse them; only run NCU if perform_ncu
    if os.path.isfile(out_path): # broken
        logger.info(f"{out_path} exists. Skipping CUTLASS profiling and using existing results.")
        existing = read_cutlass_results(out_path)
        if not existing:
            logger.warning(f"No valid entries found in {out_path}. Nothing to do.")
            # If DeepFlow requested and we have NCU rows, still do comparison
            if deepflow_tc and ncu_rows:
                pass  # fall-through to comparison block below
            else:
                return
        else:
            if perform_ncu:
                with open(ncu_out_path, "a", newline="") as f_ncu:
                    writer_ncu = csv.writer(f_ncu)
                    if ncu_write_header:
                        metric_labels = get_metric_labels(ncu_metric_spec)
                        writer_ncu.writerow(["m", "k", "n", "cta_m", "cta_k", "cta_n", "stages"] + metric_labels)
                    for m, k, n, best_kernel in existing:
                        try:
                            logger.info(f"Running NCU for m={m} k={k} n={n} kernel={best_kernel.name}")
                            ncu_metrics = run_ncu_profiler(
                                m=m, k=k, n=n, kernel_name=best_kernel.name,
                                metric_spec=ncu_metric_spec
                            )
                            ordered_values = [ncu_metrics.get(spec.label, "") for spec in ncu_metric_spec.values()]
                            writer_ncu.writerow([m, k, n, best_kernel.cta_m, best_kernel.cta_k,
                                                 best_kernel.cta_n, best_kernel.stages] + ordered_values)
                            ncu_rows.append({"m": m, "k": k, "n": n, "metrics": ncu_metrics})
                        except Exception as e:
                            logger.warning(f"NCU profiling failed for m={m} k={k} n={n} kernel={best_kernel.name}: {e}")
                logger.info(f"NCU metrics written to {ncu_out_path}")
            else:
                logger.info("NCU profiling skipped (existing NCU metrics file present).")
        # DeepFlow comparison with existing (and/or loaded) NCU rows
        if deepflow_tc and ncu_rows:
            try:
                write_header_df = not os.path.isfile(args.deepflow_output)
                # if deepflow_output already exists, clear it and rewrite
                if not write_header_df:
                    os.remove(args.deepflow_output)
                    write_header_df = True
                with open(args.deepflow_output, "a", newline="") as f_df:
                    w_df = csv.writer(f_df)
                    if write_header_df:
                        w_df.writerow([
                            "m","k","n","cta_m","cta_k","cta_n","l1_m","l1_k","l1_n",
                            "ncu_latency_us","df_latency_us","latency_abs_err","latency_pct_err",
                            "ncu_l2_read","df_l2_read","l2_read_pct_err",
                            "ncu_l2_write","df_l2_write","l2_write_pct_err",
                            "ncu_l2_total","df_l2_total","l2_total_pct_err",
                            "ncu_dram_read","df_dram_read","dram_read_pct_err",
                            "ncu_dram_write","df_dram_write","dram_write_pct_err",
                            "ncu_dram_total","df_dram_total","dram_total_pct_err"
                        ])
                    for row in ncu_rows:
                        m,k,n = row["m"], row["k"], row["n"]
                        metrics = row["metrics"]
                        ncu_latency = metrics.get("latency_ms")
                        ncu_latency *= 1e3  # ms → us
                        ncu_l2_read = metrics.get("l2_read")
                        ncu_l2_write = metrics.get("l2_write")
                        ncu_l2_total = ncu_l2_read + ncu_l2_write
                        ncu_dram_read = metrics.get("dram_read")
                        ncu_dram_write = metrics.get("dram_write")
                        ncu_dram_total = ncu_dram_read + ncu_dram_write
                        ncu_cta_m = metrics.get("cta_m")
                        ncu_cta_n = metrics.get("cta_n")
                        ncu_cta_k = metrics.get("cta_k")
                        try:
                            df_latency, _, df_tile_dims, _, best_rw_access = deepflow_tc.getGEMMTime(m, k, n, name=f"m{m}_k{k}_n{n}")
                        except Exception as e:
                            logger.warning(f"DeepFlow getGEMMTime failed for {m},{k},{n}: {e}")
                            continue
                        # logger.info(f"({m},{k},{n}): {df_tile_dims}")
                        df_latency *= 1e6  # s → us
                        l1_m, l1_k, l1_n = df_tile_dims[1]
                        df_l2_read = best_rw_access.reads[2]
                        df_l2_write = best_rw_access.writes[2]
                        df_l2_total = df_l2_read + df_l2_write
                        df_dram_read = best_rw_access.reads[3]
                        df_dram_write = best_rw_access.writes[3]
                        df_dram_total = df_dram_read + df_dram_write
                        lat_abs, lat_pct = _safe_err(ncu_latency, df_latency)
                        _, l2_read_pct = _safe_err(ncu_l2_read, df_l2_read)
                        _, l2_write_pct = _safe_err(ncu_l2_write, df_l2_write)
                        _, l2_total_pct = _safe_err(ncu_l2_total, df_l2_total)
                        _, dram_read_pct = _safe_err(ncu_dram_read, df_dram_read)
                        _, dram_write_pct = _safe_err(ncu_dram_write, df_dram_write)
                        _, dram_total_pct = _safe_err(ncu_dram_total, df_dram_total)
                        w_df.writerow([
                            m,k,n,ncu_cta_m,ncu_cta_k,ncu_cta_n,l1_m,l1_k,l1_n,
                            f"{ncu_latency:.3f}", f"{df_latency:.3f}", lat_abs, lat_pct,
                            ncu_l2_read, df_l2_read, l2_read_pct,
                            ncu_l2_write, df_l2_write, l2_write_pct,
                            ncu_l2_total, df_l2_total, l2_total_pct,
                            ncu_dram_read, df_dram_read, dram_read_pct,
                            ncu_dram_write, df_dram_write, dram_write_pct,
                            ncu_dram_total, df_dram_total, dram_total_pct
                        ])
                logger.info(f"DeepFlow comparison written to {args.deepflow_output}")
            except Exception as e:
                logger.warning(f"Failed DeepFlow comparison output: {e}")
        return

    # Otherwise (CUTLASS results absent), run CUTLASS search; only run NCU if perform_ncu
    write_header = not os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f_kernel:
        writer_kernel = csv.writer(f_kernel)
        if write_header:
            writer_kernel.writerow([
                "model","op_name","m","k","n","best_kernel_name","cta_m","cta_k","cta_n",
                "stages","inst_m","inst_n","inst_k","best_time_ms",
            ])
        total_jobs = len(gemm_configs)
        job_idx = 0
        for config in gemm_configs:
            m, k, n = config['m'], config['k'], config['n']
            model_name = config.get('model', 'unknown')
            op_name = config.get('op_name', 'unknown')
            job_idx += 1
            logger.info(f"[{job_idx}/{total_jobs}] Profiling {model_name}:{op_name} GEMM m={m} k={k} n={n}")
            best_kernel, best_time = run_cutlass_profiler(m=m, k=k, n=n, cta_tuples=cta_tuples)
            if best_kernel is None:
                logger.warning(f"No kernel found for {model_name}:{op_name} m={m} k={k} n={n}")
                continue
            writer_kernel.writerow([
                model_name,op_name,m,k,n,best_kernel.name,best_kernel.cta_m,best_kernel.cta_k,
                best_kernel.cta_n,best_kernel.stages,best_kernel.inst_m,
                best_kernel.inst_n,best_kernel.inst_k,f"{best_time:.6f}",
            ])
            logger.info(f"Best kernel: {best_kernel.name} CTA=({best_kernel.cta_m},{best_kernel.cta_k},{best_kernel.cta_n}) time={best_time:.4f} ms")
            if perform_ncu:
                # Open (append) NCU file lazily only if needed
                if ncu_write_header:
                    with open(ncu_out_path, "a", newline="") as f_ncu:
                        writer_ncu = csv.writer(f_ncu)
                        metric_labels = get_metric_labels(ncu_metric_spec)
                        writer_ncu.writerow(["m","k","n","cta_m","cta_k","cta_n","stages"] + metric_labels)
                    ncu_write_header = False
                try:
                    ncu_metrics = run_ncu_profiler(
                        m=m, k=k, n=n, kernel_name=best_kernel.name,
                        metric_spec=ncu_metric_spec
                    )
                    ordered_values = [ncu_metrics.get(spec.label, "") for spec in ncu_metric_spec.values()]
                    with open(ncu_out_path, "a", newline="") as f_ncu:
                        writer_ncu = csv.writer(f_ncu)
                        writer_ncu.writerow([m, k, n, best_kernel.cta_m, best_kernel.cta_k,
                                             best_kernel.cta_n, best_kernel.stages] + ordered_values)
                    ncu_rows.append({"m": m, "k": k, "n": n, "metrics": ncu_metrics})
                    logger.info(f"NCU metrics collected for kernel {best_kernel.name}: latency={ncu_metrics.get('latency_ms', 'N/A')}")
                except Exception as e:
                    logger.warning(f"NCU profiling failed for kernel {best_kernel.name}: {e}")

    logger.info(f"Kernel results written to {out_path}")
    if perform_ncu:
        logger.info(f"NCU metrics written to {ncu_out_path}")
    else:
        logger.info(f"Skipped writing NCU metrics (existing file retained: {ncu_out_path})")

    # DeepFlow comparison after new profiling (only if NCU metrics available or loaded)
    if deepflow_tc and ncu_rows:
        try:
            write_header_df = not os.path.isfile(args.deepflow_output)
            with open(args.deepflow_output, "a", newline="") as f_df:
                w_df = csv.writer(f_df)
                if write_header_df:
                    w_df.writerow([
                        "m","k","n","cta_m","cta_k","cta_n","l1_m","l1_k","l1_n",
                        "ncu_latency_ms","df_latency_ms","latency_abs_err","latency_pct_err",
                        "ncu_l2_read","df_l2_read","l2_read_pct_err",
                        "ncu_l2_write","df_l2_write","l2_write_pct_err",
                        "ncu_dram_read","df_dram_read","dram_read_pct_err",
                        "ncu_dram_write","df_dram_write","dram_write_pct_err"
                    ])
                for row in ncu_rows:
                    m,k,n = row["m"], row["k"], row["n"]
                    metrics = row["metrics"]
                    ncu_latency = metrics.get("latency_ms")
                    ncu_l2_read = metrics.get("l2_read")
                    ncu_l2_write = metrics.get("l2_write")
                    ncu_dram_read = metrics.get("dram_read")
                    ncu_dram_write = metrics.get("dram_write")
                    ncu_cta_m = metrics.get("cta_m")
                    ncu_cta_n = metrics.get("cta_n")
                    ncu_cta_k = metrics.get("cta_k")
                    try:
                        df_latency, _, df_tile_dims, _, best_rw_access = deepflow_tc.getGEMMTime(m, k, n)
                    except Exception as e:
                        logger.warning(f"DeepFlow getGEMMTime failed for {m},{k},{n}: {e}")
                        continue
                    l1_m, l1_k, l1_n = df_tile_dims[1]
                    df_l2_read = best_rw_access.reads[2]
                    df_l2_write = best_rw_access.writes[2]
                    df_dram_read = best_rw_access.reads[3]
                    df_dram_write = best_rw_access.writes[3]
                    lat_abs, lat_pct = _safe_err(ncu_latency, df_latency * 1e3)
                    _, l2_read_pct = _safe_err(ncu_l2_read, df_l2_read)
                    _, l2_write_pct = _safe_err(ncu_l2_write, df_l2_write)
                    _, dram_read_pct = _safe_err(ncu_dram_read, df_dram_read)
                    _, dram_write_pct = _safe_err(ncu_dram_write, df_dram_write)
                    w_df.writerow([
                        m,k,n,ncu_cta_m,ncu_cta_k,ncu_cta_n,l1_m,l1_k,l1_n,
                        ncu_latency, df_latency, lat_abs, lat_pct,
                        ncu_l2_read, df_l2_read, l2_read_pct,
                        ncu_l2_write, df_l2_write, l2_write_pct,
                        ncu_dram_read, df_dram_read, dram_read_pct,
                        ncu_dram_write, df_dram_write, dram_write_pct
                    ])
            logger.info(f"DeepFlow comparison written to {args.deepflow_output}")
        except Exception as e:
            logger.warning(f"Failed DeepFlow comparison output: {e}")


if __name__ == "__main__":
    os.environ[CUTLASS_ENV_VAR] = (
        "/u1/ee/nanoproj/users/yaoe888/cutlass/build/tools/profiler/cutlass_profiler"
    )
    main()
