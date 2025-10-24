The hf_to_config.py utility can be used to convert a Hugging Face model config.json into a DeepFlow LLM YAML config.

Usage:
```
python hf_to_config.py <model_id> --revision <revision> --batch-size <batch_size> --seq-len <seq_len> --decode-len <decode_len> --run-type <run_type> -o <output_file>
```

Example:
```
python hf_to_config.py NousResearch/Hermes-3-Llama-3.1-405B --run-type inference --batch-size 32 --seq-len 65536 --decode-len 1024 -o Hermes_Llama3.1-405B.yaml

```