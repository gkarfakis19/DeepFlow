[DONE] 1. Debug overlap. Overlap = false is giving me better results than true??
--- 1.1 KV cache fetch and store nodes should be modelled INSIDE the transformer not in the pipeline graph.
[DONE] 2. Debug astrasim flattened vs analytical deepflow giving diff results for num gpus = 1.
--- Microsecond rounding. Unavoidable. Document it in README.
[] 3. Fix network weirdness with collectives during decode.
--- GK. Skipped for now. Will get back to this later. Right now assume astra sim IS NOT compatible with inference.