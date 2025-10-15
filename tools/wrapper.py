
"""Wrapper harness for DeepFlow/DeepFlowSim comparisons.

Current implementation focuses on the DeepFlow annotated + AstraSim path.
Other modes are scaffolded but intentionally unimplemented.
Configuration is driven via module-level globals—no CLI parsing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import json
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_ROOT = Path(__file__).resolve().parent
WRAPPER_OUTPUT_ROOT = WRAPPER_ROOT / 'wrapper_outputs'
DEFAULT_STG_PYTHON = Path('../.venv/bin/python3')

# Toggle between 'deepflow', 'deepflow_ablation', 'stg', 'simai_analytical', or 'all'.
RUN_SELECTION = 'all'

GLOBAL_CONFIG: Dict[str, object] = {
    'hardware_config': REPO_ROOT / 'configs/hardware-config/a100_80GB.yaml',
    'model_config': REPO_ROOT / 'configs/model-config/LLM.yaml',
    'dry_run': False,
    'generate_visuals': True,
    'isol_astra': True, # force "deepflow" mode to run AstraSim the exact same way as "stg" mode. Set to True for best comparisons.
    'deepflow': {
        'additional_env': {}
    },
    'deepflow_ablation': {},
    'stg': {
        'python': DEFAULT_STG_PYTHON,
    },
    'simai_analytical': {
        'run_binary': False,
        'gpus_per_server': None,
    },
}


def ensure_path_exists(path: Path, description: str) -> None:
    if not path.exists():
        print("CURRENT PATH: ", os.getcwd())
        raise FileNotFoundError(f"Expected {description} at {path}")


def copy_top_level_files(src: Path, dest: Path) -> List[Path]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for entry in sorted(src.iterdir()):
        if entry.is_file():
            target = dest / entry.name
            shutil.copy2(entry, target)
            copied.append(target)
        else:
            print(f"[Wrapper] Skipping directory {entry} per top-level-only requirement")
    return copied


def snapshot_summary_files(dest_dir: Path) -> List[Path]:
    summaries: List[Path] = []
    out_root = REPO_ROOT / 'output'
    if not out_root.exists():
        return summaries
    for path in sorted(out_root.rglob('summary_*.txt')):
        flattened_name = path.relative_to(out_root).as_posix().replace('/', '__')
        target = dest_dir / flattened_name
        shutil.copy2(path, target)
        summaries.append(target)
    return summaries


def run_deepflow_annotated(config: Dict[str, object]) -> Dict[str, object]:
    hardware_cfg = Path(config['hardware_config'])
    model_cfg = Path(config['model_config'])
    ensure_path_exists(hardware_cfg, 'hardware config')
    ensure_path_exists(model_cfg, 'model config')

    dry_run = bool(config.get('dry_run', False))
    generate_visuals = bool(config.get('generate_visuals', False))

    dest_root = WRAPPER_OUTPUT_ROOT / 'deepflow'
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    isol_astra = bool(config.get('isol_astra', False))

    env = os.environ.copy()
    env['DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS'] = '1'
    if generate_visuals:
        env['DEEPFLOW_PERSIST_ARTIFACT_VIZ'] = '1'
    if dry_run or isol_astra:
        env['DEEPFLOW_ASTRA_SKIP_EXEC'] = '1'
    else:
        env.pop('DEEPFLOW_ASTRA_SKIP_EXEC', None)
    extra_env: Dict[str, object] = {}
    if isinstance(config.get('deepflow'), dict):
        extra_env = config['deepflow'].get('additional_env', {}) or {}
    for key, value in extra_env.items():
        env[str(key)] = str(value)

    cmd = [
        sys.executable,
        str(REPO_ROOT / 'run_perf.py'),
        '--hardware_config', str(hardware_cfg),
        '--model_config', str(model_cfg),
    ]

    start_time = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False, stdout=sys.stdout, stderr=sys.stderr)
    duration = time.perf_counter() - start_time
    if result.returncode != 0:
        raise RuntimeError(f"DeepFlow execution failed with return code {result.returncode}")

    artifact_src = REPO_ROOT / 'output' / 'LLM' / 'astra_flat'
    ensure_path_exists(artifact_src, 'DeepFlow flattened AstraSim artifacts')
    copied_artifacts = copy_top_level_files(artifact_src, dest_root)

    summaries = snapshot_summary_files(dest_root)

    manifest_path: Path | None = None
    per_rank: List[float] = []
    total_time = None
    if not dry_run and isol_astra:
        et_files = sorted(dest_root.glob('llm_graph.*.et'))
        if not et_files:
            raise RuntimeError('DeepFlow isolated AstraSim: no ET files found')
        manifest_path = _build_manifest(et_files, dest_root)
        hw_obj = _load_hw_config_object(hardware_cfg)
        astra_config_dir = dest_root / 'astrasim_configs'
        astra_config_dir.mkdir(parents=True, exist_ok=True)
        reset_json_cache()
        generate_astrasim_configs_from_hw(hw_obj, out_dir=str(astra_config_dir), npus_count=len(et_files), roofline_enabled=False)
        cache_path = astra_config_dir / 'cache.json'
        per_rank, total_time = run_cache_astrasim(
            hw_obj,
            comm='graph',
            npus_count=len(et_files),
            size_bytes=0,
            astra_config_dir=str(astra_config_dir),
            cache_path=str(cache_path),
            manifest_json_path=str(manifest_path),
            workload_prefix=str(dest_root / 'llm_graph'),
            comm_group_json=None,
        )
        print(f"[Wrapper] DeepFlow AstraSim total: {total_time:.6f} s")

    print('[Wrapper] DeepFlow annotated run complete:')
    print(f"  - Duration: {duration:.2f} s")
    print(f"  - Artifacts directory: {dest_root}")
    for path in copied_artifacts:
        print(f"    • {path.name}")
    if summaries:
        for path in summaries:
            print(f"    • summary: {path.name}")
    else:
        print('    • summary: (none found)')

    return {
        'mode': 'deepflow',
        'duration_seconds': duration,
        'artifact_dir': dest_root,
        'summaries': summaries,
        'dry_run': dry_run,
        'manifest': manifest_path,
        'astrasim_total_time': total_time,
        'astrasim_per_rank': per_rank,
    }


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at {path}, got {type(data).__name__}")
    return data


def _compute_ffn_dim(model_param: Dict[str, object], hidden_dim: int) -> int:
    ffn_dim = model_param.get('ffn_dim')
    if ffn_dim is not None:
        return int(ffn_dim)
    ffn_mult = model_param.get('ffn_mult')
    if ffn_mult is None:
        return hidden_dim * 4
    return int(float(ffn_mult) * hidden_dim)



def _ensure_sys_path() -> None:
    repo_entry = str(REPO_ROOT)
    astrasim_entry = str(REPO_ROOT / 'astrasim_lib')
    if repo_entry not in sys.path:
        sys.path.append(repo_entry)
    if astrasim_entry not in sys.path:
        sys.path.append(astrasim_entry)


_ensure_sys_path()
import config as df_config  # type: ignore  # noqa: E402
from astrasim_lib.config_generation import generate_astrasim_configs_from_hw, reset_json_cache  # type: ignore  # noqa: E402
from astrasim_lib.integration import run_cache_astrasim  # type: ignore  # noqa: E402
from astrasim_lib.et_utils import chakra_open, chakra_decode, pb  # type: ignore  # noqa: E402
from simai_analytical import (  # type: ignore  # noqa: E402
    SimAIAnalyticalRunner,
    SimAIConversionError,
)


_HW_CONFIG_CACHE: Dict[Path, object] = {}


def _load_hw_config_object(path: Path):
    resolved = path.resolve()
    cached = _HW_CONFIG_CACHE.get(resolved)
    if cached is None:
        cached = df_config.parse_config(str(resolved), config_type='hardware')
        _HW_CONFIG_CACHE[resolved] = cached
    return cached


def _manifest_op_key(op: List) -> tuple:
    kind = op[0] if op else None
    if kind == 'COMP':
        return (0, int(op[1]) if len(op) > 1 else 0, 0)
    if kind == 'COMM':
        size_val = int(op[2]) if len(op) > 2 else 0
        comm_type = int(op[1]) if len(op) > 1 else -1
        return (1, size_val, comm_type)
    if kind == 'SEND':
        return (2, int(op[1]) if len(op) > 1 else 0, 0)
    if kind == 'RECV':
        return (3, int(op[1]) if len(op) > 1 else 0, 0)
    return (9, 0, 0)


def _build_manifest(et_files: List[Path], dest_dir: Path) -> Path:
    manifest_ranks: Dict[str, List[List]] = {}
    for et_path in sorted(et_files):
        parts = et_path.name.split('.')
        if len(parts) < 3:
            raise RuntimeError(f"Unexpected ET filename format: {et_path}")
        rank = int(parts[-2])
        ops: List[List] = []
        fh = chakra_open(str(et_path))
        try:
            meta = pb.GlobalMetadata()
            chakra_decode(fh, meta)
            while True:
                node = pb.Node()
                if not chakra_decode(fh, node):
                    break
                node_type = int(node.type)
                if node_type == pb.COMP_NODE:
                    ops.append(['COMP', int(node.duration_micros or 0)])
                elif node_type == pb.COMM_COLL_NODE:
                    comm_type = -1
                    comm_size = 0
                    for attr in node.attr:
                        which = attr.WhichOneof('value')
                        if attr.name == 'comm_type' and which:
                            comm_type = int(getattr(attr, which))
                        elif attr.name == 'comm_size' and which:
                            comm_size = int(getattr(attr, which))
                    ops.append(['COMM', comm_type, comm_size, None])
                elif node_type == pb.COMM_SEND_NODE:
                    comm_size = 0
                    for attr in node.attr:
                        which = attr.WhichOneof('value')
                        if attr.name == 'comm_size' and which:
                            comm_size = int(getattr(attr, which))
                    ops.append(['SEND', comm_size])
                elif node_type == pb.COMM_RECV_NODE:
                    comm_size = 0
                    for attr in node.attr:
                        which = attr.WhichOneof('value')
                        if attr.name == 'comm_size' and which:
                            comm_size = int(getattr(attr, which))
                    ops.append(['RECV', comm_size])
        finally:
            fh.close()
        manifest_ranks[str(rank)] = sorted(ops, key=_manifest_op_key)

    manifest_path = dest_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump({'npus': len(et_files), 'ranks': manifest_ranks}, handle, sort_keys=True, separators=(",", ":"))
    return manifest_path


def run_stg(config: Dict[str, object]) -> Dict[str, object]:
    hardware_cfg = Path(config['hardware_config'])
    model_cfg = Path(config['model_config'])
    ensure_path_exists(hardware_cfg, 'hardware config')
    ensure_path_exists(model_cfg, 'model config')

    hw_data = _load_yaml(hardware_cfg)
    model_data = _load_yaml(model_cfg)

    sched = hw_data.get('scheduling_param', {}) or {}
    dp = int(sched.get('dp', 1))
    lp = int(sched.get('lp', 1))
    kp1 = int(sched.get('kp1', 1) or 1)
    kp2 = int(sched.get('kp2', 1) or 1)
    tp = max(1, kp1 * kp2)
    mb = int(sched.get('mb', 1) or 1)

    model_param = model_data.get('model_param', {}) or {}
    batch_size = int(model_param.get('batch_size', 1))
    seq_len = int(model_param.get('seq_len', 1))
    hidden_dim = int(model_param.get('hidden_dim', 1))
    num_layers = int(model_param.get('num_layers', 1))
    vocab_size = int(model_param.get('vocab_size', 32000))
    attention_cfg = model_param.get('attention', {}) or {}
    num_heads = int(attention_cfg.get('num_heads', 1))
    kv_heads = int(attention_cfg.get('kv_heads', num_heads))
    ffn_dim = _compute_ffn_dim(model_param, hidden_dim)

    if dp > num_layers:
        raise ValueError(f"STG: dp ({dp}) cannot exceed num_layers ({num_layers})")

    effective_pp = min(lp, num_layers)
    if effective_pp != lp:
        print(f"[Wrapper] STG pipeline degree {lp} exceeds num_layers {num_layers}; clamping to {effective_pp}")

    stg_cfg = config.get('stg', {}) if isinstance(config.get('stg'), dict) else {}
    python_path = stg_cfg.get('python', DEFAULT_STG_PYTHON)
    python_exe = Path(python_path)
    # ensure_path_exists(python_exe, 'STG Python interpreter') # check fails as path is from stg root not this tool root

    stg_root = REPO_ROOT / 'symbolic_tensor_graph'
    staging_dir = stg_root / 'generated' / 'wrapper_tmp'
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    micro_batch_size = batch_size // (dp * mb) if dp * mb else batch_size
    if micro_batch_size <= 0 or micro_batch_size * dp * mb != batch_size:
        raise ValueError("STG: batch_size must equal dp * micro_batch_size * mb")
        
    cmd = [
        str(python_exe),
        'main.py',
        '--output_dir', str(staging_dir),
        '--output_name', 'workload.%d.et',
        '--dp', str(dp),
        '--tp', str(tp),
        '--pp', str(effective_pp),
        '--sp', '1',
        '--ep', '1',
        '--batch', str(batch_size),
        '--micro_batch', str(micro_batch_size),
        '--seq', str(seq_len),
        '--dmodel', str(hidden_dim),
        '--dff', str(ffn_dim),
        '--head', str(num_heads),
        '--kvhead', str(kv_heads),
        '--num_stacks', str(num_layers),
        '--dvocal', str(vocab_size),
        '--model_type', 'dense',
        '--chakra_schema_version', 'v0.0.4',
    ]
    cmd_string = ' '.join(str(part) for part in cmd)
    print(f"[Wrapper] STG command: {cmd_string}")

    start_time = time.perf_counter()
    result = subprocess.run(cmd, cwd=stg_root, check=False)
    duration = time.perf_counter() - start_time
    if result.returncode != 0:
        raise RuntimeError(f"STG generator failed with return code {result.returncode}")

    et_files = sorted(staging_dir.glob('workload.*.et'))
    if not et_files:
        raise RuntimeError('STG generator produced no ET files')

    dest_root = WRAPPER_OUTPUT_ROOT / 'stg'
    copied_artifacts = copy_top_level_files(staging_dir, dest_root)

    generate_visuals = bool(config.get('generate_visuals', False))
    if generate_visuals:
        vis_targets = [str(p) for p in sorted(dest_root.glob('workload.*.et'))[:10]]
        if vis_targets:
            # try:
            from astrasim_lib.executor import _dump_et_text
            from astrasim_lib import stg_viz
            render_dir = dest_root
            stg_viz.render_stg_bundle(vis_targets, render_dir)
            _dump_et_text(vis_targets)
            # except Exception as exc:
            #     print(f"[Wrapper] STG visualization failed: {exc}")

    dry_run = bool(config.get('dry_run', False))
    manifest_path: Path | None = None
    per_rank: List[float] = []
    total_time = None
    if dry_run:
        print('[Wrapper] STG dry_run set; skipping AstraSim execution.')
    else:
        manifest_path = _build_manifest(sorted(dest_root.glob('workload.*.et')), dest_root)
        hw_obj = _load_hw_config_object(hardware_cfg)
        astra_config_dir = dest_root / 'astrasim_configs'
        astra_config_dir.mkdir(parents=True, exist_ok=True)
        reset_json_cache()
        generate_astrasim_configs_from_hw(hw_obj, out_dir=str(astra_config_dir), npus_count=len(et_files), roofline_enabled=True)
        cache_path = astra_config_dir / 'cache.json'
        comm_group_json = dest_root / 'workload.json'
        if not comm_group_json.exists():
            comm_group_json = None
        per_rank, total_time = run_cache_astrasim(
            hw_obj,
            comm='graph',
            npus_count=len(et_files),
            size_bytes=0,
            astra_config_dir=str(astra_config_dir),
            cache_path=str(cache_path),
            manifest_json_path=str(manifest_path),
            workload_prefix=str(dest_root / 'workload'),
            comm_group_json=str(comm_group_json) if comm_group_json else None,
        )
        print(f"[Wrapper] STG AstraSim total: {total_time:.6f} s")

    try:
        shutil.rmtree(staging_dir)
    except OSError:
        pass

    print(f"[Wrapper] STG generator finished in {duration:.2f} s; artifacts at {dest_root}")
    for path in copied_artifacts:
        print(f"    • {path.name}")

    return {
        'mode': 'stg',
        'duration_seconds': duration,
        'artifact_dir': dest_root,
        'summaries': [],
        'dry_run': dry_run,
        'manifest': manifest_path,
        'astrasim_total_time': total_time,
        'astrasim_per_rank': per_rank,
    }

def run_deepflow_ablation(config: Dict[str, object]) -> Dict[str, object]:
    raise NotImplementedError('DeepFlow ablation workflow not implemented yet.')


def run_simai_analytical(config: Dict[str, object]) -> Dict[str, object]:
    hardware_cfg = Path(config['hardware_config'])
    model_cfg = Path(config['model_config'])
    ensure_path_exists(hardware_cfg, 'hardware config')
    ensure_path_exists(model_cfg, 'model config')

    dest_root = WRAPPER_OUTPUT_ROOT / 'simai_analytical'
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    runner = SimAIAnalyticalRunner(repo_root=REPO_ROOT)

    start_time = time.perf_counter()
    try:
        artifacts = runner.generate_artifacts(
            hardware_config=hardware_cfg,
            model_config=model_cfg,
            output_dir=dest_root,
        )
    except SimAIConversionError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"SimAI conversion failed: {exc}") from exc
    duration = time.perf_counter() - start_time

    workload_rel = artifacts.workload_path.relative_to(dest_root)
    busbw_rel = artifacts.busbw_path.relative_to(dest_root)
    print('[Wrapper] SimAI analytical workload synthesized:')
    print(f"  - Workload: {workload_rel}")
    print(f"  - Bus bandwidth: {busbw_rel}")
    print(f"  - Records: {len(artifacts.records)}")

    simai_cfg = config.get('simai_analytical', {}) if isinstance(config.get('simai_analytical'), dict) else {}
    run_binary = bool(simai_cfg.get('run_binary', False)) and not bool(config.get('dry_run', False))
    gpus_per_server = simai_cfg.get('gpus_per_server')
    if gpus_per_server is None:
        hw_cfg_obj = _load_hw_config_object(hardware_cfg)
        gpus_per_server = getattr(getattr(hw_cfg_obj, 'system_hierarchy', None), 'num_devices_per_node', None)
    binary_path = REPO_ROOT / 'SimAI' / 'bin' / 'SimAI_analytical'
    simai_returncode: Optional[int] = None

    if run_binary:
        ensure_path_exists(binary_path, 'SimAI analytical binary')
        result_dir = dest_root / 'simai_results'
        result_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(binary_path),
            '-w', str(artifacts.workload_path),
            '-g', str(artifacts.parallel_config.all_gpus),
            '-g_p_s', str(int(gpus_per_server) if gpus_per_server else max(1, artifacts.parallel_config.tensor_parallel)),
            '-r', str(result_dir / 'run-'),
            '-busbw', str(artifacts.busbw_path),
        ]
        print(f"[Wrapper] Launching SimAI binary: {' '.join(cmd)}")
        binary_start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT / 'SimAI', check=False)
        duration += time.perf_counter() - binary_start
        simai_returncode = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(f"SimAI analytical binary exited with code {proc.returncode}")

    return {
        'mode': 'simai_analytical',
        'duration_seconds': duration,
        'artifact_dir': dest_root,
        'workload_path': artifacts.workload_path,
        'busbw_path': artifacts.busbw_path,
        'simai_returncode': simai_returncode,
        'ran_binary': run_binary,
        'dry_run': bool(config.get('dry_run', False)),
    }


def dispatch(selection: str, config: Dict[str, object]) -> List[Dict[str, object]]:
    handlers = {
        'deepflow': run_deepflow_annotated,
        'deepflow_ablation': run_deepflow_ablation,
        'stg': run_stg,
        'simai_analytical': run_simai_analytical,
    }

    if selection == 'all':
        results: List[Dict[str, object]] = []
        mode_ls = [
            'deepflow',
            'stg',
            # 'deepflow_ablation',
            # 'simai_analytical',
        ]
        for key in mode_ls:
            results.append(handlers[key](config))
        return results

    if selection not in handlers:
        raise ValueError(f"Unknown RUN_SELECTION '{selection}'")
    return [handlers[selection](config)]


def main() -> None:
    WRAPPER_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = dispatch(RUN_SELECTION, GLOBAL_CONFIG)
    print('[Wrapper] Execution summary:')
    for entry in results:
        line = f"  * {entry['mode']}: {entry['duration_seconds']:.2f}s | artifacts -> {entry['artifact_dir']}"
        total = entry.get('astrasim_total_time')
        if isinstance(total, (int, float)) and total is not None:
            line += f" | astra_total={total:.6f}s"
        print(line)


if __name__ == '__main__':
    main()
