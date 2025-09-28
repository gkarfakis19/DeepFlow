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
      self.batch_size       = exp_config.model_config.batch_size
      self.vocab_size       = exp_config.model_config.vocab_size
      self.num_layers      = exp_config.model_config.num_layers
      self.hidden_dim       = exp_config.model_config.hidden_dim
      self.seq_len          = exp_config.model_config.seq_len
      self.decode_len       = exp_config.model_config.decode_len
      self.num_heads        = exp_config.model_config.num_heads
      self.ffn_dim          = exp_config.model_config.ffn_dim
      self.ffn_mult        = exp_config.model_config.ffn_mult
      self.n_tokens         = exp_config.model_config.n_tokens
      self.all_reduce       = exp_config.model_config.all_reduce
      self.run_type         = exp_config.model_config.run_type
      inference_cfg = getattr(exp_config, 'inference_config', None)
      if inference_cfg is not None:
          self.inference_sample_every = inference_cfg.sample_every
      else:
          raise ValueError("Inference configuration not found")
      
      
      

      
      
