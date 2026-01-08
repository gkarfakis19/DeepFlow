## Current state (high level)

- Ported STG and MLSynth wrapper tooling into `tools/comp/wrapper.py` and `tools/comp/compare_et_timing.py`, using the repo venv (`.venv/bin/python3`) by default.
- Brought over supporting assets:
  - `symbolic_tensor_graph/` and `MLSynth/` trees (from deepflow_compare) for STG/MLSynth generators.
  - `astrasim_lib/stg_viz.py` so STG ET visualizations work when enabled.
- Restored roofline toggles in AstraSim config generation (`astrasim_lib/config_generation.py` now accepts `roofline_enabled` and sets `roofline-enabled` accordingly). Wrapper passes roofline=true for STG/MLSynth/ablation paths.

## What’s still pending

- No ablation-specific metadata reinsertion beyond roofline flagging (full ablation feature work still outstanding).

## Useful paths

- New wrapper & tools: `tools/comp/wrapper.py`, `tools/comp/compare_et_timing.py`.
- Generators: `symbolic_tensor_graph/` (STG), `MLSynth/` (MLSynth).
- AstraSim support & viz: `astrasim_lib/config_generation.py`, `astrasim_lib/stg_viz.py`.
- Config examples in use: `configs/hardware-config/a100_80GB.yaml` (default), `configs/model-config/Llama2-7B.yaml` (default). Older variants retained: `configs/hardware-config/a100_80GB_NEMO.yaml`, `a100_80GB_AA_scale.yaml`.
- Reference old repo with working STG/MLSynth content: `/app/nanocad/projects/ispass_deepflow/deepflow_compare/DeepFlow`.
