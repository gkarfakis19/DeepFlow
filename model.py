class Model_LSTM:
  def __init__(self, exp_config):
      self.batch_size       = exp_config.model_config.batch_size
      self.vocab_size       = exp_config.model_config.vocab_size
      self.num_layers       = exp_config.model_config.num_layers
      self.hidden_dim       = exp_config.model_config.layer_size
      self.projection       = exp_config.model_config.projection
      self.seq_len          = exp_config.model_config.seq_len
      self.num_gates        = exp_config.model_config.num_gates
      self.num_non_linear   = exp_config.model_config.num_non_linear
      self.num_add          = exp_config.model_config.num_add
      self.num_pointwise    = self.num_non_linear + self.num_add
        
class Model_GEMM:
  def __init__(self, exp_config):
      self.M               = exp_config.model_config.M
      self.K               = exp_config.model_config.K
      self.N               = exp_config.model_config.N
      self.backward        = exp_config.model_config.backward
      
class Model_LLM:
  def __init__(self, exp_config):
      self.global_batch_size = exp_config.model_config.global_batch_size
      self.gradient_accumulation_steps = exp_config.model_config.gradient_accumulation_steps
      self.vocab_size       = exp_config.model_config.vocab_size
      self.num_layers      = exp_config.model_config.num_layers
      self.hidden_dim       = exp_config.model_config.hidden_dim
      self.seq_len          = exp_config.model_config.seq_len
      self.decode_len       = exp_config.model_config.decode_len
      self.num_heads        = exp_config.model_config.num_heads
      self.tied_embeddings  = exp_config.model_config.tied_embeddings
      self.model_type       = exp_config.model_config.model_type
      # self.kv_heads       = exp_config.model_config.kv_heads  
      self.intermediate_size          = exp_config.model_config.intermediate_size
      self.n_tokens         = exp_config.model_config.n_tokens
      self.all_reduce       = "every layer"
      self.run_type         = exp_config.model_config.run_type
      self.attention_type  = exp_config.model_config.attention.attention_type
      self.kv_heads        = exp_config.model_config.attention.kv_heads if hasattr(exp_config.model_config.attention, 'kv_heads') else None
      self.use_flashattention = getattr(exp_config.model_config.attention, 'use_flashattention', False)
      self.attention_tile_size = getattr(exp_config.model_config.attention, 'attention_tile_size', None)

      self.moe_num_experts = int(getattr(exp_config.model_config, "num_experts", 1))
      self.moe_top_k = int(getattr(exp_config.model_config, "top_k", 1))
      if self.moe_top_k > self.moe_num_experts and self.moe_num_experts > 1:
          raise ValueError("model_param.top_k cannot exceed model_param.num_experts")
      self.use_moe = self.moe_num_experts > 1
      
      inference_cfg = getattr(exp_config, "inference_config", None)
      if str(self.run_type).lower() == "inference":
          if inference_cfg is None:
              raise ValueError("Inference configuration not found for inference run_type")
          self.inference_sample_every = inference_cfg.sample_every
      else:
          # Training configs do not require inference sampling parameters.
          self.inference_sample_every = -1
      
      
      

      
      
