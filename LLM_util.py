
import math
import os
from collections import OrderedDict

import config

def reshape_gemm_to_3d(arg):
    """
    Reshape a 4-dimensional GEMM [batch_size, M, K, N] into 3 dimensions [M, K, N].

    Parameters:
        arg (list or tuple): A list or tuple containing 4 dimensions [batch_size, M, K, N].

    Returns:
        tuple: A tuple (M, K, N) representing the reshaped GEMM dimensions.
    """
    
    if len(arg) != 4:
        raise ValueError("Input must contain exactly 4 dimensions [batch_size, M, K, N].")
    
    
    batch_size, M, K, N = arg
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
    M *= batch_size  # Multiply batch_size into M
        
    return M, K, N


ATTENTION_GEMM_KEYS = {"attention_score", "attention_output"}


def multihead_decoder_gemm(self, batch_size, seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size, model_type="gpt"):
    """
    Generate GEMM shapes [M, K, N] for a multi-head Transformer decoder block.

    Parameters:
        batch_size (int): batch size (B)
        seq_len (int): sequence length (S)
        d_model (int): hidden size (D)
        num_heads (int): number of attention heads (H)
        intermediate_size (int): first FFN layer output dimension (typically 4 * D)
        vocab_size (int): vocabulary size (V)
        
        for a standard multi-head attention, kv_heads = num_heads


    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads"
    head_dim = d_model // num_heads
    shared_heads = num_heads // kv_heads # how many heads share the same K,V
    gemms = OrderedDict()

    gemms["qkv_proj"] = (batch_size, seq_len, d_model, (2 * kv_heads + num_heads) * head_dim)

    gemms["attention_score"] = (
        batch_size * kv_heads,
        seq_len * shared_heads,
        head_dim,
        seq_len,
    )

    gemms["attention_output"] = (
        batch_size * kv_heads,
        seq_len * shared_heads,
        seq_len,
        head_dim,
    )
    gemms["output_proj"] = (batch_size, seq_len, d_model, d_model)
    if str(model_type).lower() == "llama":
        projected_dim = 2 * intermediate_size
    else:
        projected_dim = intermediate_size
    if self.use_moe:
        # assuming equal load balancing here
        num_experts = max(1, int(getattr(self, "moe_num_experts", 1)))
        top_k = max(1, int(getattr(self, "moe_top_k", 1)))
        gemms["ffn1"] = (batch_size, seq_len * top_k / num_experts, d_model, projected_dim ) #gemm shape per expert assuming each dp group has all experts
        gemms["ffn2"] = (batch_size, seq_len * top_k / num_experts, intermediate_size, d_model)
    else:
        gemms["ffn1"] = (batch_size, seq_len, d_model, projected_dim)
        gemms["ffn2"] = (batch_size, seq_len, intermediate_size, d_model) 
    gemms["linear"] = (batch_size, seq_len, d_model, vocab_size)
    gemms["router"] = (batch_size, seq_len, d_model, self.moe_num_experts) 

    return gemms


def process_gemm_shapes(self, batch_size, seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size):
    """
    Process GEMM shapes, reshape them into 3d.

    Parameters:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        d_model (int): Hidden size.
        num_heads (int): Number of attention heads.
        intermediate_size (int): First FFN layer output dimension.
        vocab_size (int): Vocabulary size.
    """
    # Generate GEMM shapes in 4D
    gemm_shapes_4d = multihead_decoder_gemm(
        self,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type=self.model_type,
    )

    processed = OrderedDict()
    for key, shape in gemm_shapes_4d.items():
        
        if key in ATTENTION_GEMM_KEYS:
            processed[key] = tuple(shape)
        else:
            processed[key] = reshape_gemm_to_3d(shape)

    return processed

def get_transformer_mem_layer( dp, tp, lp, mb, batch_size, hidden_dim, seq_len, intermediate_size, n_heads, precision, zero_stage, flash_attention, full_recomputation, model_type="gpt"):#  https://arxiv.org/pdf/2205.05198. https://shjwudp.github.io/blog/2023/gpt-training-memory-estimation-nemo-training-practice/
    """ memory estimation of transformer layer for single gpu case in inference mode is supported.
    other modes or layers are work in progress."""
    #Activations refer to output activations that need to be stored

    if full_recomputation:
        act_memory_layer = seq_len * batch_size * hidden_dim * (2 ) * (precision.activations / 2) #full recompute:
    elif flash_attention:
        act_memory_layer = seq_len * batch_size * hidden_dim * (34 / tp ) * (precision.activations / 2) #assuming selective recompute
    else:
        act_memory_layer = seq_len * batch_size * hidden_dim * (34 / tp + 5 * n_heads * seq_len/(hidden_dim * tp) ) * (precision.activations / 2) 
    
    act_memory_layer *= lp / mb #pipedream style pipelining activation memory
    act_memory_layer_inf = seq_len * batch_size * intermediate_size / tp * precision.activations  #inference max activation memory, no need to store for backpropagation
    ffn_proj_factor = 3 if str(model_type).lower() == "llama" else 2
    
    transformer_param_layer = 4* hidden_dim * hidden_dim + intermediate_size * ffn_proj_factor * hidden_dim  # weights Wq,Wk,Wv,Wo,ffn

    optimizer_mem = (precision.optimizer_states * 2 + precision.activations) * transformer_param_layer / tp # don't divide by dp for DDP, NeMo ZeRO-1 optimizer style
    tensor_weight_memory_layer = transformer_param_layer * precision.parameters / tp #weight memory
    # master_parameters is set to 0 by default, so this works.
    master_weight_memory_layer = transformer_param_layer * precision.master_parameters / tp
    weight_memory_layer = tensor_weight_memory_layer + master_weight_memory_layer
    
    gradient_mem = transformer_param_layer * precision.gradients / tp  # gradient buffers scaled by precision
    # precision has been replaced with a class that has many different precision types.
    # furthemore, we have added this "master weight" copy for weights that are stored in FP32 optionally.
    # for weight_memory_layer it makes sense to just add them together. But I can see it's only really used in infernece?
    # for training, static_memory_layer needs to be broken up into different equations that use the correct precisions.
    # TODO TODO TODO

    if zero_stage >= 3:
        weight_memory_layer /= dp
    if zero_stage >= 2:
        gradient_mem /= dp
    if zero_stage >= 1:
        optimizer_mem /= dp

    static_memory_layer = optimizer_mem + gradient_mem + weight_memory_layer # optimizer states + gradients + weights
    layer_mem = (act_memory_layer + weight_memory_layer)


    return layer_mem, act_memory_layer, act_memory_layer_inf, static_memory_layer, gradient_mem, optimizer_mem, weight_memory_layer

def get_linear_softmax_mem(batch_size, seq_len, hidden_dim, vocab_size, precision, t):
    # t = 1
    # weights = hidden_dim * vocab_size
    # softmax_act = batch_size * seq_len * vocab_size * precision
    # softmax_wt = (hidden_dim + 1) * vocab_size * precision
    # softmax_point = (2 * batch_size * seq_len * vocab_size + batch_size * seq_len) * precision
    # #NOTE: sigmoid and exp could have been combined
    # #1 sigmoids
    # #1 exp
    # #1 pointwise div
    # softmax_mem = (softmax_act + softmax_wt + softmax_point)
    mem = 4 * seq_len * batch_size * hidden_dim / t *(1+vocab_size/hidden_dim) * (precision.activations / 2) #from https://arxiv.org/pdf/2205.05198
    return mem


def get_embedding_act_mem(batch_size, seq_len, hidden_dim, p, t, precision):
    mem = 4 * seq_len * batch_size * hidden_dim * p / t * (precision.activations / 2)  # from https://arxiv.org/pdf/2205.05198

    return mem
    
def get_embedding_weight_mem(
    vocab_size: int,
    hidden_dim: int,
    precision,
    tied_embeddings: bool = True,
    param_replica_factor: int = 1,  # Should be equal to dp. For ZeRO-3 (future work) set to 1.
) -> int:
    """
    Embedding WEIGHT memory per rank:
      - Input embeddings: (V*H*bytes)/vocab_shards * replica_factor
      (vocab shards is WIP)
      - Output head: same as input if untied, else 0.
    """
    per_matrix = vocab_size * hidden_dim * (precision.parameters + precision.master_parameters) / 1.0
    input_embed = per_matrix * param_replica_factor
    output_head = 0 if tied_embeddings else per_matrix * param_replica_factor
    return input_embed + output_head
    
def get_tot_mem_req(exp_hw_config, exp_model_config, **kwargs):
    # Model Params
    default_global_batch = getattr(
        exp_model_config.model_config,
        "global_batch_size",
        getattr(exp_model_config.model_config, "batch_size"),
    )
    batch_size = int(
        kwargs.get(
            "global_batch_size",
            kwargs.get("batch_size", default_global_batch),
        )
    )
    hidden_dim                   = int(kwargs.get('hidden_dim', exp_model_config.model_config.hidden_dim))
    vocab_size                   = int(kwargs.get('vocab_size', exp_model_config.model_config.vocab_size))
    n_layers                   = int(kwargs.get('num_layer', exp_model_config.model_config.num_layers))
    n_heads                     = int(kwargs.get('num_heads', exp_model_config.model_config.num_heads))
    # projection          = exp_model_config.model_config.projection
    seq_len                   = int(kwargs.get('seq_len', exp_model_config.model_config.seq_len))
    intermediate_size                   = int(kwargs.get('intermediate_size', exp_model_config.model_config.intermediate_size))
    # G                   = exp_model_config.model_config.num_gates
    precision           = exp_hw_config.sw_config.precision

    # MiniBatch
    dp                  = int(kwargs.get('dp', exp_hw_config.sch_config.dp))
    # print("Data Parallelism Degree:", dp)
    dp = 8 #for testing
    miniB               = math.ceil(batch_size / dp)
    mb             = int(kwargs.get('microbatches', exp_hw_config.sch_config.mb))
    microB              = math.ceil(miniB / mb)

    tied_embeddings = exp_model_config.model_config.tied_embeddings

    transformer_mem_layer, transformer_act_layer, transformer_act_layer_inf, transformer_static_layer, gradient_mem_layer, optimizer_mem_layer, weight_memory_layer = (
        get_transformer_mem_layer(
            dp = dp,
            tp = 1,
            zero_stage=1,
            batch_size=microB,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            intermediate_size=intermediate_size,
            n_heads=n_heads,
            precision=precision,
            model_type=exp_model_config.model_config.model_type,
        )
    )
    softmax_mem = get_linear_softmax_mem(
        batch_size=microB,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        precision=precision,
        t=1
    )



    embedding_mem = get_embedding_act_mem(
        batch_size=microB,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        p=1,
        t=1,
        precision=precision
    )

    embedding_weight_mem = get_embedding_weight_mem(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        precision=precision,
        tied_embeddings=tied_embeddings,
        param_replica_factor=dp
    )

    tot_mem = transformer_mem_layer*n_layers + softmax_mem + embedding_mem + embedding_weight_mem

    

    return tot_mem, embedding_mem, transformer_mem_layer*n_layers,transformer_act_layer*n_layers,transformer_static_layer*n_layers, gradient_mem_layer*n_layers, optimizer_mem_layer*n_layers, weight_memory_layer*n_layers, softmax_mem#, projection_mem, wt_mem, act_mem, point_mem


# ====================================================================
# DECODE-SPECIFIC UTILITIES FOR AUTOREGRESSIVE INFERENCE
# ====================================================================

def kv_cache_token_bytes(batch_size, kv_heads, head_dim, precision_bytes):
    """Return total bytes to store K+V for a single new token."""
    return batch_size * kv_heads * head_dim * precision_bytes * 2


def autoregressive_decoder_gemm(self, batch_size, current_seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size, model_type="gpt"):
    """
    Generate GEMM shapes for a single decode step in autoregressive generation.

    Key differences from training/prefill GEMMs:
    - Sequence length is typically 1 (generating one token at a time)
    - Attention over growing KV-cache (current_seq_len)
    - Always uses KV-cache (one-token query, cached keys/values)

    Parameters:
        batch_size (int): Batch size (B)
        current_seq_len (int): Current sequence length including cache (growing: 1, 2, 3, ...)
        d_model (int): Hidden size (D)
        num_heads (int): Number of attention heads (H)
        kv_heads (int): Number of key/value heads (H_kv)
        intermediate_size (int): First FFN layer output dimension (typically 4 * D)
        vocab_size (int): Vocabulary size (V)

    Returns:
        OrderedDict: GEMM shapes [M, K, N] for decode step operations
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads"
    head_dim = d_model // num_heads
    shared_heads = num_heads // kv_heads
    gemms = OrderedDict()

    # Decode-specific GEMM shapes with KV-cache handling (always enabled)

    # QKV Projection: Only the new token (seq_len = 1)
    gemms["qkv_proj"] = (batch_size, 1, d_model, (2 * kv_heads + num_heads) * head_dim)

    # Attention Score: Q(new) @ K(cached+new)
    gemms["attention_score"] = (
        batch_size * kv_heads,
        1 * shared_heads,      # query positions handled per KV head
        head_dim,
        current_seq_len,      # key seq_len (grows with decode steps)
    )

    # Attention Output: attention_weights @ V(cached+new)
    gemms["attention_output"] = (
        batch_size * kv_heads,
        1 * shared_heads,      # output positions handled per KV head
        current_seq_len,      # attention weight dim (grows with decode)
        head_dim,
    )

    # === POST-ATTENTION LAYERS (same regardless of cache) ===

    # Output projection & FFNs only process the new token
    gemms["output_proj"] = (batch_size, 1, d_model, d_model)
    projected_dim = 2 * intermediate_size if str(model_type).lower() == "llama" else intermediate_size
    if self.use_moe:
        # assuming equal load balancing here
        num_experts = max(1, int(getattr(self, "num_experts", getattr(self, "moe_num_experts", 1))))
        top_k = max(1, int(getattr(self, "top_k", getattr(self, "moe_top_k", 1))))
        effective_batch_size = math.ceil(batch_size * top_k / num_experts)

        gemms["ffn1"] = (effective_batch_size , 1, d_model, projected_dim ) #gemm shape per expert
        gemms["ffn2"] = (effective_batch_size , 1, intermediate_size, d_model)
    else:
        gemms["ffn1"] = (batch_size, 1, d_model, projected_dim) 
        gemms["ffn2"] = (batch_size, 1, intermediate_size, d_model)

    return gemms


def process_decode_gemm_shapes(
    self,
    batch_size,
    current_seq_len,
    d_model,
    num_heads,
    kv_heads,
    intermediate_size,
    vocab_size,
    model_type="gpt",
):
    """
    Process decode GEMM shapes and reshape them into 3D.

    Similar to process_gemm_shapes but for decode-specific patterns. Handles grouped
    query attention (GQA) by allowing kv_heads != num_heads.

    Integrates with existing DeepFlow GEMM processing infrastructure.
    """
    # Generate decode GEMM shapes in 4D
    gemm_shapes_4d = autoregressive_decoder_gemm(
        self,
        batch_size=batch_size,
        current_seq_len=current_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type=model_type,
    )

    processed = OrderedDict()
    for key, shape in gemm_shapes_4d.items():
        if key in ATTENTION_GEMM_KEYS:
            # Keep attention GEMMs in 4D for proper computation
            processed[key] = tuple(shape)
        else:
            # Reshape other GEMMs to 3D using existing infrastructure
            processed[key] = reshape_gemm_to_3d(shape)

    return processed






if __name__ == "__main__":

    
    exp_hw_config_path = "configs/hardware-config/a100_80GB.yaml"
    exp_model_config_path = "configs/model-config/LLM.yaml"
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type="LLM")
    mem, embedding_mem, transformer_mem, transformer_act_mem, transformer_static_mem, gradient_mem, optimizer_mem, weight_memory, softmax_mem = get_tot_mem_req(
                exp_hw_config,
                exp_model_config,

            )
    print(f"Total Memory Requirement: {mem/1e9:.2f} GB")
    print(f"Embedding Memory Requirement: {embedding_mem/1e9:.2f} GB")
    print(f"Transformer Memory Requirement: {transformer_mem/1e9:.2f} GB")
    print(f"Transformer Activation Memory Requirement: {transformer_act_mem/1e9:.2f} GB")
    print(f"Transformer Static Memory Requirement(grad+optim+weight): {transformer_static_mem/1e9:.2f} GB")
    print(f"Transformer Gradient Memory Requirement: {gradient_mem/1e9:.2f} GB")
    print(f"Transformer Optimizer Memory Requirement: {optimizer_mem/1e9:.2f} GB")
    print(f"Transformer Weight Memory Requirement: {weight_memory/1e9:.2f} GB")
    print(f"Softmax Memory Requirement: {softmax_mem/1e9:.2f} GB")
