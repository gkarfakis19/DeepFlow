The hf_to_config.py utility can be used to convert a Hugging Face model config.json into a DeepFlow LLM YAML config.

Usage:
```
python hf_to_config.py <model_id> --revision <revision> --run-type <run_type>  --global-batch-size <global-batch-size> --gradient-accumulation-steps <gradient-accumulation-steps> --seq-len <seq_len> --decode-len <decode_len>  --use-flashattention <bool> --flash-tile-size <int> -o <output_file>
```

Example:
```
python hf_to_config.py NousResearch/Hermes-3-Llama-3.1-405B --run-type inference --global-batch-size 32 --gradient-accumulation-steps 2 --seq-len 65536 --decode-len 1024 --use-flashattention true --flash-tile-size 256 -o Hermes_Llama3.1-405B.yaml
```