from gemm_calc import GEMMCalc
from hw_component import Core, MemoryHierarchy
from validate_gemm import extract_gemm_shapes, parse_llm_shapes_file
import config
from typing import List, Dict

import os
import csv
from pathlib import Path
import pprint
import matplotlib.pyplot as plt
import numpy as np

STACK_ORDER = [
    'qkv_proj',
    'attention_score',
    'attention_output',
    'output_proj',
    'ffn1',
    'ffn2',
    'linear'
]

def configs():
    """Load hardware and model configurations from YAML files."""
    hw_config_path = "configs/hardware-config/a100_80GB_no_parallelism.yaml"
    model_config_path = "configs/model-config/GEMM.yaml"
    
    # Make paths absolute relative to the repository root
    # base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # hw_config_path = os.path.join(base_dir, hw_config_path)
    # model_config_path = os.path.join(base_dir, model_config_path)
    
    # Load configurations using config module
    hw_config = config.parse_config(hw_config_path, config_type="hardware")
    model_config = config.parse_config(model_config_path, config_type="GEMM")
    
    return hw_config, model_config

def csv_validation_data():
    """Load validation data from ncu_metrics_results.csv."""
    csv_path = Path("validation_scripts") / "ncu_metrics_results.csv"
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {
                'm': int(row['m']),
                'k': int(row['k']),
                'n': int(row['n']),
                'latency_ms': float(row['latency_ms']),
                'clock': float(row['clock_Ghz']) * 1e9,  # Convert GHz to Hz
            }
            data.append(parsed_row)
    return data

def _create_gemm_calc(configs):
    """Helper function to create GEMMCalc instance from configs.
    
    Args:
        configs: Tuple of (hw_config, model_config)
        
    Returns:
        Initialized GEMMCalc instance
    """
    hw_config, model_config = configs
    core = Core(hw_config)
    mem_hierarchy = MemoryHierarchy(hw_config)
    dtype_bytes = hw_config.sw_config.precision.activations
    flashattn_enable = False
    return GEMMCalc(core, mem_hierarchy.memLayer, dtype_bytes, flashattn_enable)

def find_row(
    csv_data: List[Dict],
    m: int,
    k: int,
    n: int
) -> Dict | None:
    """Find a row in CSV data matching given m, k, n dimensions."""
    for row in csv_data:
        if row['m'] == m and row['k'] == k and row['n'] == n:
            return row
    return None

def compare(
    gemm_calc: GEMMCalc,
    csv_validation_data: List[Dict],
    shapes_file: str,
):
    models_data = parse_llm_shapes_file(shapes_file)
    gemm_configs = extract_gemm_shapes(models_data)

    data = {}

    for cfg in gemm_configs:
        if cfg['op_name'] == "router":
            continue  # Skip router operations
        ncu_data = find_row(
            csv_validation_data,
            cfg['m'],
            cfg['k'],
            cfg['n']
        )

        num_heads = cfg.get('batch', 1)

        ncu_time = ncu_data['latency_ms']
        clock = ncu_data['clock']
        derate = clock / gemm_calc.core.operating_freq

        # Run GEMMCalc
        best_gemm, best_time = gemm_calc.run(cfg['m'], cfg['k'], cfg['n'], name=f"{cfg['model']}_{cfg['op_name']}", throughput=gemm_calc.core.getThroughput() * derate)

        best_time = (best_time + 9e-6) * 1e3

        model_dict = data.get(f"{cfg['model']}", {})
        model_dict[f"{cfg['op_name']}"] = {
            'm': cfg['m'],
            'k': cfg['k'],
            'n': cfg['n'],
            'ncu_time_ms': ncu_time * num_heads,
            'gemmcalc_time_ms': best_time * num_heads,
            'err_pct': abs(ncu_time - best_time) / ncu_time * 100,
        }
        data[f"{cfg['model']}"] = model_dict
    return data

def plot_stacked_bar_chart(results: Dict[str, Dict[str, Dict]]):
    """Create a double stacked bar chart comparing NCU and GEMMCalc times.
    
    Args:
        results: Dictionary with structure {'model': {'op_name': {'ncu_time_ms', 'gemmcalc_time_ms', ...}}}
    """
    # Split models into llama2 and llama3.1 groups
    llama2_models = {k: v for k, v in results.items() if 'llama2' in k.lower()}
    llama31_models = {k: v for k, v in results.items() if 'llama3.1' in k.lower() or 'llama3_1' in k.lower()}
    
    # Get all unique operation names across all models
    all_ops = sorted(set(op for ops in results.values() for op in ops.keys()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_ops)))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot llama2 models
    _plot_model_group(ax1, llama2_models, all_ops, colors, 'Llama2 Models (Batch Size 16)')
    
    # Plot llama3.1 models
    _plot_model_group(ax2, llama31_models, all_ops, colors, 'Llama3.1 Models (Batch Size 1)')
    
    # Create shared legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[i], label=op) 
                      for i, op in enumerate(STACK_ORDER)]
    legend_elements.insert(0, plt.Rectangle((0,0),1,1, fc='white', ec='black', 
                                            label='Actual'))
    legend_elements.insert(1, plt.Rectangle((0,0),1,1, fc='white', ec='black', hatch='///',
                                            label='DeepFlowSim'))
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
               ncol=len(legend_elements)//2 + len(legend_elements)%2, frameon=True)
    
    fig.suptitle('aggregated error weighed by time per GEMM', fontsize=16, fontweight='bold', y=0.99)
    
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    output_path = Path("validation_scripts") / "comparison_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def _plot_model_group(ax, model_dict: Dict[str, Dict[str, Dict]], all_ops: List[str], colors, title: str):
    """Helper function to plot a group of models on a given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        model_dict: Dictionary of models and their operations
        all_ops: List of all operation names
        colors: Color map for operations
        title: Title for the subplot
    """
    if not model_dict:
        ax.set_visible(False)
        return
    
    models = list(model_dict.keys())
    num_models = len(models)
    
    # Prepare data structures
    ncu_times = {}
    gemmcalc_times = {}
    
    for model, ops in model_dict.items():
        ncu_times[model] = {op: data['ncu_time_ms'] for op, data in ops.items()}
        gemmcalc_times[model] = {op: data['gemmcalc_time_ms'] for op, data in ops.items()}
    
    x = np.arange(num_models)
    width = 0.25
    gap = 0.05
    
    # Create stacked bars for each operation
    ncu_bottoms = np.zeros(num_models)
    gemmcalc_bottoms = np.zeros(num_models)
    
    for idx, op in enumerate(reversed(STACK_ORDER)):
        ncu_values = [ncu_times[model].get(op, 0) for model in models]
        gemmcalc_values = [gemmcalc_times[model].get(op, 0) for model in models]
        
        ax.bar(x - (width+gap)/2, ncu_values, width, bottom=ncu_bottoms, color=colors[len(all_ops) - idx - 1])
        ax.bar(x + (width+gap)/2, gemmcalc_values, width, bottom=gemmcalc_bottoms, 
               color=colors[len(all_ops) - idx - 1], hatch='///', edgecolor='white', linewidth=0.5)
        
        ncu_bottoms += ncu_values
        gemmcalc_bottoms += gemmcalc_values
    
    # Add overall error annotations with vertical brackets
    for i, model in enumerate(models):
        total_ncu = ncu_bottoms[i]
        total_gemmcalc = gemmcalc_bottoms[i]
        overall_error = abs(total_ncu - total_gemmcalc) / total_ncu * 100 if total_ncu > 0 else 0
        
        # Draw vertical bracket showing difference
        min_height = min(total_ncu, total_gemmcalc)
        max_height = max(total_ncu, total_gemmcalc)
        
        # Bracket position (to the right of the bars)
        bracket_x = x[i]
        bracket_width = 0.02
        
        # Draw vertical line
        ax.plot([bracket_x, bracket_x], [min_height, max_height], 
                'k-', linewidth=1.5, solid_capstyle='butt')
        # Draw top horizontal line
        ax.plot([bracket_x - bracket_width/2, bracket_x + bracket_width/2], [max_height, max_height], 
                'k-', linewidth=1.5)
        # Draw bottom horizontal line
        ax.plot([bracket_x - bracket_width/2, bracket_x + bracket_width/2], [min_height, min_height], 
                'k-', linewidth=1.5)
        
        # Position error text beside the bracket
        mid_height = (min_height + max_height) / 2
        ax.text(bracket_x, max_height, f'err: {overall_error:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the subplot
    # ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis='y', alpha=0.3)

if __name__ == "__main__":
    hw_config, model_config = configs()
    gemm_calc = _create_gemm_calc((hw_config, model_config))
    csv_data = csv_validation_data()
    shapes_file = Path("validation_scripts") / "llm_shapes.txt"
    
    results = compare(gemm_calc, csv_data, shapes_file)
    
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(results)
    
    # Create visualization
    plot_stacked_bar_chart(results)