import math
import sys
import os
import config
import threading
from typing import Iterable, List, Optional, Tuple

core=1
DRAM=1
L2=1
shared_mem=1
reg_mem=1
proj = False #Turn off the projection layer

DF_CACHE_DIR = "./cache"

_LOG_MESSAGES: List[Tuple[Optional[str], str]] = []
_LOG_LOCK = threading.Lock()

_SECTION_ORDER = [
    ("network", "TOPOLOGY/NETWORK"),
    ("faults", "FAULTY LINKS"),
    ("results", "RESULTS"),
]

_REPO_ROOT = os.path.abspath(os.environ.get("DEEPFLOW_REPO_ROOT", os.getcwd()))


def log_message(message: str, category: Optional[str] = None) -> None:
    """Append ``message`` to the shared log queue."""
    if message is None:
        return
    text = str(message)
    if not text:
        return
    cat_normalized = str(category).strip().lower() if category else None
    with _LOG_LOCK:
        _LOG_MESSAGES.append((cat_normalized, text))


def extend_log(lines: Iterable[str], category: Optional[str] = None) -> None:
    """Append multiple lines to the log queue."""
    for line in lines:
        log_message(line, category=category)


def drain_log_messages() -> List[Tuple[Optional[str], str]]:
    """Return and clear all queued log messages (category, message)."""
    with _LOG_LOCK:
        if not _LOG_MESSAGES:
            return []
        drained = list(_LOG_MESSAGES)
        _LOG_MESSAGES.clear()
        return drained


def flush_log_queue() -> None:
    """Print and clear the queued log messages grouped by category."""
    entries = drain_log_messages()
    print("\n")
    if not entries:
        return
    categorized = {cat: [] for cat, _ in _SECTION_ORDER}
    uncategorized: List[str] = []
    for category, message in entries:
        if category in categorized:
            categorized[category].append(message)
        else:
            uncategorized.append(message)

    sections_printed = False
    section_border = "=" * 60
    for cat, title in _SECTION_ORDER:
        lines = categorized.get(cat) or []
        if not lines:
            continue
        sections_printed = True
        print(f"{section_border}\n{title}\n{section_border}")
        for line in lines:
            print(line)
    if sections_printed:
        print(section_border)

    for line in uncategorized:
        print(line)


def relpath_display(path: str) -> str:
    """Return ``path`` relative to the repo root when possible."""
    if not path:
        return ""
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, start=_REPO_ROOT)
    except Exception:
        return abs_path
    if rel.startswith(".."):
        return abs_path
    return rel


def _collect_parallelism_values(hw_config):
    sch_config = getattr(hw_config, "sch_config", None)
    if sch_config is None:
        return {}

    values = {}
    for name in getattr(sch_config, "_fields", []):
        values[str(name).lower()] = getattr(sch_config, name)
    return values


def _format_parallelism_terms(dim, parallelism_values):
    terms = []
    for axis in getattr(dim, "parallelisms", ()):
        if axis not in parallelism_values:
            continue
        factor = parallelism_values[axis]
        try:
            factor_value = int(factor)
        except (TypeError, ValueError):
            factor_value = factor
        terms.append(f"{axis} {factor_value}")
    return terms


def network_topology_summary_training(hw_config):
    parallelism_values = _collect_parallelism_values(hw_config)
    dimensions = list(getattr(hw_config.network_layout, "dimensions", ()))
    ordered_axes = ["tp", "cp", "lp", "dp"]
    formatted_terms = []
    for axis in ordered_axes:
        value = parallelism_values.get(axis)
        if value is None:
            continue
        formatted_terms.append(f"{axis}:{value}")
    formatted_parallelisms = ", ".join(formatted_terms) if formatted_terms else "none"
    lines = [
        f"Parallelisms: {formatted_parallelisms}",
        f"Network Topology [dims={len(dimensions)}]",
    ]
    aggregate = 1

    for dim in dimensions:
        terms = _format_parallelism_terms(dim, parallelism_values)
        axis_repr = " × ".join(terms) if terms else "(none)"
        size_value = getattr(dim, "size", 1)
        try:
            size_int = int(size_value)
        except (TypeError, ValueError):
            size_int = None
            size_display = size_value
        else:
            aggregate *= size_int
            size_display = size_int
        lines.append(
            f"  • {dim.id} {dim.label} : {axis_repr} ⇒ size {size_display}"
        )

    lines.append(f"  ⇒  total {aggregate} devices")
    return lines


def network_topology_summary_inference(hw_config):
    parallelism_values = _collect_parallelism_values(hw_config)
    all_dimensions = list(getattr(hw_config.network_layout, "dimensions", ()))
    filtered_dimensions = [
        dim
        for dim in all_dimensions
        if any(axis != "dp" for axis in getattr(dim, "parallelisms", ())) or not dim.parallelisms
    ]
    lines = [f"Network Topology [dims={len(filtered_dimensions)}]"]

    aggregate_per_replica = 1
    for dim in filtered_dimensions:
        terms = _format_parallelism_terms(dim, parallelism_values)
        axis_repr = " × ".join(terms) if terms else "(none)"
        size_value = getattr(dim, "size", 1)
        try:
            size_int = int(size_value)
        except (TypeError, ValueError):
            size_int = None
            size_display = size_value
        else:
            aggregate_per_replica *= size_int
            size_display = size_int
        lines.append(
            f"  • {dim.id} {dim.label} : {axis_repr} ⇒ size {size_display}"
        )

    dp_factor = parallelism_values.get("dp", 1)
    try:
        dp_replicas = max(1, int(dp_factor))
    except (TypeError, ValueError):
        dp_replicas = dp_factor if dp_factor else 1
    if isinstance(aggregate_per_replica, (int, float)) and isinstance(dp_replicas, (int, float)):
        total_aggregate = aggregate_per_replica * dp_replicas
    else:
        total_aggregate = aggregate_per_replica
    lines.append(f"  replicas (dp): {dp_replicas}")
    lines.append(
        f"  => aggregate = {aggregate_per_replica} GPUs per replica ({total_aggregate} total)"
    )
    return lines

def printError(message):
  sys.exit(message)

def getHiddenMem(L, Dim1, Dim2, Dim3, S, precision):
    #Activations refer to output activations that need to be stored
    hidden_act = Dim1 * Dim3 * S * L * precision
    hidden_wt  = (Dim2 + 1) * Dim3 * L * precision
    hidden_point = (Dim1 * Dim3 / 2) * 9 * L * S * precision
    #3 sigmoids
    #2 tanh
    #3 pointwise multiply
    #1 addition
    hidden_mem = (hidden_act + hidden_wt + hidden_point)

    return hidden_mem, hidden_act, hidden_wt, hidden_point

def getSoftmaxMem(B, S, P, V, precision):
     #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_act = B * S * V * precision 
    softmax_wt = (P + 1) * V * precision
    softmax_point = (2 * B * S * V + B * S) * precision
    #NOTE: sigmoid and exp could have been combined
    #1 sigmoids
    #1 exp
    #1 pointwise div
    softmax_mem = (softmax_act + softmax_wt + softmax_point)

    return softmax_mem, softmax_act, softmax_wt, softmax_point

def getProjectionMem(B, S, P, D, precision):
    projection_act = B * S * P * precision
    projection_wt = (D + 1) * P * precision
    projection_point= B * S * P * precision
    projection_mem = (projection_act + projection_wt + projection_point)
  
    return projection_mem, projection_act, projection_wt, projection_point

def getEmbeddingMem(B, S, V, D, precision):
    embedding_act = B * S * D * precision
    embedding_wt = V * D * precision
    embedding_point = 0
    embedding_mem = (embedding_wt + embedding_act + embedding_point)

    return embedding_mem, embedding_act, embedding_wt, embedding_point

def getTotMemReq(exp_hw_config, exp_model_config, **kwargs):
    #Model Params
    B                   = int(kwargs.get('batch_size', exp_model_config.model_config.batch_size))
    D                   = int(kwargs.get('hidden_dim', exp_model_config.model_config.layer_size))
    V                   = int(kwargs.get('vocab_size', exp_model_config.model_config.vocab_size))
    L                   = int(kwargs.get('num_layer', exp_model_config.model_config.num_layers))
    projection          = exp_model_config.model_config.projection 
    S                   = int(kwargs.get('seq_len', exp_model_config.model_config.seq_len))
    G                   = exp_model_config.model_config.num_gates
    precision           = exp_hw_config.sw_config.precision.activations
   
    #MiniBatch
    dp                  = int(kwargs.get('dp', exp_hw_config.sch_config.dp))
    miniB               = math.ceil(B / dp)

    hidden_mem, hidden_act, hidden_wt, hidden_point =  getHiddenMem(L=L, 
                                                       Dim1 = miniB, 
                                                       Dim2 = 2 * D, 
                                                       Dim3 = G * D, 
                                                       S = S, 
                                                       precision = precision)
    softmax_mem, softmax_act, softmax_wt, softmax_point =  getSoftmaxMem(B=miniB,
                                                           S=S, 
                                                           P=(projection if proj else D), 
                                                           V=V, 
                                                           precision = precision)
    if proj:
      projection_mem, projection_act, projection_wt, projection_point =  getProjectionMem(B=miniB, 
                                                                         S=S, 
                                                                         P=projection, 
                                                                         D=D, 
                                                                         precision = precision)
    else:
      projection_mem, projection_act, projection_wt, projection_point = 0, 0, 0 , 0
    
    embedding_mem, embedding_act, embedding_wt, embedding_point =  getEmbeddingMem(B=miniB, 
                                                                   S=S, 
                                                                   V=V, 
                                                                   D=D, 
                                                                   precision = precision)
    
    tot_mem = hidden_mem + softmax_mem + embedding_mem + projection_mem
    
    wt_mem = (hidden_wt + softmax_wt + projection_wt + embedding_wt)
    act_mem = (hidden_act + softmax_act + projection_act + embedding_act)
    point_mem = (hidden_point + softmax_point + projection_point + embedding_point)

    return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem


def getMemUsagePerCore(exp_hw_config, exp_model_config, **kwargs):
    #Model params
    B                   = int(kwargs.get('batch_size', exp_model_config.model_config.batch_size))
    D                   = int(kwargs.get('hidden_dim', exp_model_config.model_config.layer_size))
    V                   = int(kwargs.get('vocab_size', exp_model_config.model_config.vocab_size))
    L                   = int(kwargs.get('num_layer', exp_model_config.model_config.num_layers))
    projection          = exp_model_config.model_config.projection 
    S                   = int(kwargs.get('seq_len', exp_model_config.model_config.seq_len))
    G                   = exp_model_config.model_config.num_gates
    precision           = exp_hw_config.sw_config.precision.activations

    #Parallelism Params
    dp                  = int(kwargs.get('dp', exp_hw_config.sch_config.dp))
    lp                  = int(kwargs.get('lp', exp_hw_config.sch_config.lp))
    
    kp_hidden_dim1      = int(kwargs.get('kp1', exp_hw_config.sch_config.kp_hidden_dim1))
    kp_softmax_dim1     = int(kwargs.get('kp1', exp_hw_config.sch_config.kp_softmax_dim1))
    kp_embedding_dim1   = int(kwargs.get('kp1', exp_hw_config.sch_config.kp_embedding_dim1))
    kp_projection_dim1  = int(kwargs.get('kp1', exp_hw_config.sch_config.kp_projection_dim1))

    kp_hidden_dim2      = int(kwargs.get('kp2', exp_hw_config.sch_config.kp_hidden_dim2))
    kp_softmax_dim2     = int(kwargs.get('kp2', exp_hw_config.sch_config.kp_softmax_dim2))
    kp_embedding_dim2   = int(kwargs.get('kp2', exp_hw_config.sch_config.kp_embedding_dim2))
    kp_projection_dim2  = int(kwargs.get('kp2', exp_hw_config.sch_config.kp_projection_dim2))
    
    kp_hidden_type      = int(kwargs.get('kp_type', exp_hw_config.sch_config.kp_hidden_type)) #1: CR, 2: RC
    kp_softmax_type     = int(kwargs.get('kp_type', exp_hw_config.sch_config.kp_softmax_type)) #1: CR, 2: RC
    kp_embedding_type   = int(kwargs.get('kp_type', exp_hw_config.sch_config.kp_embedding_type)) #1: CR, 2: RC
    kp_projection_type  = int(kwargs.get('kp_type', exp_hw_config.sch_config.kp_projection_type)) #1: CR, 2: RC
    
    #miniBatch
    miniB               = math.ceil(B / dp)

    hlp = lp
    if lp > 2:
      hlp = hlp - 2
    hidden_mem, hidden_act, hidden_wt, hidden_point =  getHiddenMem(L=L/hlp, 
        Dim1 = math.ceil(miniB / (kp_hidden_dim1 if kp_hidden_type == 2 else  1)), 
        Dim2 = math.ceil(2 * D / (1 if kp_hidden_type == 2 else kp_hidden_dim1)),  
        Dim3 = math.ceil(D * G / (kp_hidden_dim2 if kp_hidden_type == 2 else 1)), 
        S = S, 
        precision = precision)

    #activation output from each layer, assuming input ativation are taken 
    #into account in the previous layer
    softmax_mem, softmax_act, softmax_wt, softmax_point =  getSoftmaxMem(
        B=math.ceil(miniB / (kp_softmax_dim1 if kp_softmax_type == 2 else  1)), 
        S=S, 
        P=math.ceil((projection if proj else D)/ (1 if kp_softmax_type == 2 else kp_softmax_dim1)), 
        V=math.ceil(V/(kp_softmax_dim2 if kp_softmax_type == 2 else 1)), 
        precision = precision)

    if proj:
        projection_mem, projection_act, projection_wt, projection_point =  getProjectionMem(
            B=math.ceil(miniB/(kp_projection_dim1 if kp_projection_type == 2 else  1)), 
            S=S, 
            D=math.ceil(D/(1 if kp_projection_type == 2 else kp_projection_dim1)), 
            P=math.ceil(projection/(kp_projection_dim2 if kp_projection_type == 2 else 1)), 
            precision = precision)
    else:
      projection_mem, projection_act, projection_wt, projection_point = 0, 0, 0 , 0
    #embedding_mem = miniB * S * D * precision + V * D / kp_embedding_dim1
    embedding_mem, embedding_act, embedding_wt, embedding_point =  getEmbeddingMem(
        B=math.ceil(miniB/(kp_embedding_dim1 if kp_embedding_type==2 else 1)), 
        S=S, 
        V=math.ceil(V/(1 if kp_embedding_type == 2 else kp_embedding_dim1)), 
        D=math.ceil(D/(kp_embedding_dim2 if kp_hidden_type == 2 else 1)), 
        precision = precision)

    tot_mem = 0

    if lp == 1:
      tot_mem = hidden_mem + softmax_mem + (projection_mem if proj else 0)+ embedding_mem
    elif lp >= 4: 
      tot_mem = max(hidden_mem, embedding_mem, (softmax_mem + projection_mem if proj else 0))
    else:
      NotImplemented
    
    wt_mem = (hidden_wt + softmax_wt + projection_wt + embedding_wt)
    act_mem = (hidden_act + softmax_act + projection_act + embedding_act)
    point_mem = (hidden_point + softmax_point + projection_point + embedding_point)

    return tot_mem, embedding_mem, hidden_mem, softmax_mem, projection_mem, wt_mem, act_mem, point_mem

def getChipArea(exp_config_path, **kwargs):

    exp_path = os.path.expandvars(os.path.expanduser(exp_config_path))
    exp_config = config.parse_config(exp_path)
    
    batch_size = int(kwargs.get('batch_size', exp_config.model_config.batch_size))
    hidden_dim = int(kwargs.get('hidden_dim', exp_config.model_config.layer_size))
    dp = int(kwargs.get('dp', exp_config.sch_config.dp))
    lp = int(kwargs.get('lp', exp_config.sch_config.lp))
    #type:-1 no kp
    #type: 1 col-row
    #type: 2 row-col
    kp_type = int(kwargs.get('kp_type', -1))
    kp1 = int(kwargs.get('kp1', 1))
    kp2 = int(kwargs.get('kp2', 1))
    tot_mem    = getMemUsagePerCore(exp_config, 
                                    batch_size=batch_size, 
                                    hidden_dim=hidden_dim,
                                    dp=dp, 
                                    lp=lp, 
                                    kp_type=kp_type,
                                    kp1=kp1,
                                    kp2=kp2)[0]
    stack_capacity = exp_config.tech_config.DRAM.stack_capacity 
    area_per_stack = exp_config.tech_config.DRAM.area_per_stack
    node_area_budget = exp_config.area_breakdown.node_area_budget 

    mem_area = math.ceil(tot_mem / stack_capacity) * area_per_stack
    #print("Node_Area: {}, Mem_area: {}".format(node_area_budget, mem_area))
    chip_area_budget = node_area_budget - mem_area

    return chip_area_budget

def power2RoundUp(x):
  #round up to a value which is a multiply of power of 2 and an integer number (like 16*3)
  log_power = math.ceil(math.log(x,2))
  power_2   = [2**p for p in range(0, log_power)]
  min_dist  = x
  min_val   = 1
  for i in power_2[::-1]: 
    a = math.ceil(x/i)
    dist = a * i - x
    if (dist < min_dist):
      min_val = a * i
      min_dist = dist
  return min_val

#TODO: move this to topology.py
#this only works if all connections are inter-wafer like V100
def scale_down(ib, dim, name):
  bw = -1
  if dim <= 4:
    bw = ib / 2
  elif dim <= 8:
    bw = ib / 5
  else: #beyond DGX box
    #TODO: modify to account for different network topology
    #assuming a tree beyind DGX box, PCIe is normally 12 GB/s which is half of ib to begin with
    #and then divide by another 2 to account for two parallel traversal over the network 
    #one from 7->8 and one from 15->0
    bw = ib / 4

  print('{} Bandwidth: {}'.format(name, bw/(1024*1024*1024)))
  return bw
