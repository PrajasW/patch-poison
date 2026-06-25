#!/usr/bin/env python3
"""
Script to generate a results table from geometric attack experiments.
Extracts SSIM and PSNR metrics from:
- benchmark_recon_quality.log (GT vs Render)
- benchmark_dataset_comparison.log (Poisoned vs Actual)

Outputs a printed table and LaTeX code.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def parse_log_file(log_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse a log file and extract the last mean_ssim, mean_psnr, and mean_lpips values.
    
    Args:
        log_path: Path to the log file
        
    Returns:
        Tuple of (ssim, psnr, lpips) or (None, None, None) if file doesn't exist
    """
    if not os.path.exists(log_path):
        return None, None, None
    
    ssim = None
    psnr = None
    lpips = None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all matches and take the last one
    ssim_matches = re.findall(r'mean_ssim=([0-9.]+|inf|-inf)', content, re.IGNORECASE)
    psnr_matches = re.findall(r'mean_psnr=([0-9.]+|inf|-inf)', content, re.IGNORECASE)
    lpips_matches = re.findall(r'mean_lpips=([0-9.]+|inf|-inf)', content, re.IGNORECASE)
    
    if ssim_matches:
        val = ssim_matches[-1].lower()
        ssim = float('inf') if val == 'inf' else float('-inf') if val == '-inf' else float(val)
    if psnr_matches:
        val = psnr_matches[-1].lower()
        psnr = float('inf') if val == 'inf' else float('-inf') if val == '-inf' else float(val)
    if lpips_matches:
        val = lpips_matches[-1].lower()
        lpips = float('inf') if val == 'inf' else float('-inf') if val == '-inf' else float(val)
    
    return ssim, psnr, lpips


def collect_results_from_dir(base_dir: str, exp_type: str = "", prefix: str = "") -> List[Dict]:
    """
    Collect results from all dataset directories in a single base directory.
    
    Args:
        base_dir: Base directory containing dataset folders
        exp_type: Experiment type label (e.g., 'geometric', 'da3')
        prefix: Optional prefix to add to dataset names (e.g., 'da3' -> 'da3_chair')
        
    Returns:
        List of dictionaries containing results for each dataset
    """
    results = []
    
    # Get all dataset directories
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return results
    
    datasets = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))])
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        exp_path = os.path.join(dataset_path, 'exp_run_1')
        
        if not os.path.exists(exp_path):
            print(f"Warning: {exp_path} does not exist, skipping...")
            continue
        
        # Parse reconstruction quality (GT vs Render)
        recon_log = os.path.join(exp_path, 'benchmark_recon_quality.log')
        recon_ssim, recon_psnr, recon_lpips = parse_log_file(recon_log)
        
        # Parse dataset comparison (Poisoned vs Actual)
        dataset_log = os.path.join(exp_path, 'benchmark_dataset_comparison.log')
        dataset_ssim, dataset_psnr, dataset_lpips = parse_log_file(dataset_log)
        
        # Apply prefix to dataset name if specified
        display_name = f"{prefix}_{dataset}" if prefix else dataset
        
        results.append({
            'exp_type': exp_type,
            'dataset': display_name,
            'recon_ssim': recon_ssim,
            'recon_psnr': recon_psnr,
            'recon_lpips': recon_lpips,
            'dataset_ssim': dataset_ssim,
            'dataset_psnr': dataset_psnr,
            'dataset_lpips': dataset_lpips
        })
    
    return results


def collect_results(base_dirs: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    Collect results from multiple base directories.
    
    Args:
        base_dirs: List of tuples (base_dir_path, exp_type_label, prefix)
        
    Returns:
        List of dictionaries containing results for each dataset
    """
    all_results = []
    
    for base_dir, exp_type, prefix in base_dirs:
        print(f"Scanning directory: {base_dir} (type: {exp_type}, prefix: {prefix or 'none'})")
        results = collect_results_from_dir(base_dir, exp_type, prefix)
        all_results.extend(results)
    
    return all_results


def plot_scatter(results: List[Dict], out_dir: str) -> None:
    """Create scatter plots for SSIM and PSNR (GT vs Render on x, Poisoned vs Original on y)."""
    os.makedirs(out_dir, exist_ok=True)

    def _plot(metric_key_x: str, metric_key_y: str, title: str, fname: str, xlabel: str, ylabel: str):
        xs = []
        ys = []
        labels = []
        # Track infinite values separately
        inf_xs = []
        inf_ys = []
        inf_labels = []
        inf_x_is_inf = []  # Track which axis is infinite
        inf_y_is_inf = []
        
        for r in results:
            x = r.get(metric_key_x)
            y = r.get(metric_key_y)
            if x is None or y is None:
                continue
            
            x_is_inf = (x == float('inf') or x == float('-inf'))
            y_is_inf = (y == float('inf') or y == float('-inf'))
            
            if x_is_inf or y_is_inf:
                inf_xs.append(x if not x_is_inf else 0)  # Placeholder, will be replaced
                inf_ys.append(y if not y_is_inf else 0)
                inf_labels.append(r['dataset'])
                inf_x_is_inf.append(x_is_inf)
                inf_y_is_inf.append(y_is_inf)
            else:
                xs.append(x)
                ys.append(y)
                labels.append(r['dataset'])

        if not xs and not inf_xs:
            print(f"No data available for plot {fname}, skipping.")
            return

        plt.figure(figsize=(7, 7))
        
        # Determine axis limits from finite values - use same scale for both axes
        if xs:
            all_vals = xs + ys
            data_min, data_max = min(all_vals), max(all_vals)
        else:
            data_min, data_max = 0, 1
        
        # Use same min/max for both axes
        x_min, x_max = data_min, data_max
        y_min, y_max = data_min, data_max
        
        # Add padding for infinite value markers
        data_range = data_max - data_min if data_max > data_min else 1
        x_range = data_range
        y_range = data_range
        x_inf_pos = data_max + data_range * 0.15  # Position for infinite x values
        y_inf_pos = data_max + data_range * 0.15  # Position for infinite y values
        
        # Plot finite values
        if xs:
            plt.scatter(xs, ys, c='tab:blue', alpha=0.8, edgecolors='k')
            for x, y, label in zip(xs, ys, labels):
                plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
        
        # Plot infinite values with special marker
        if inf_xs:
            plot_inf_xs = []
            plot_inf_ys = []
            for i, (x, y, x_inf, y_inf) in enumerate(zip(inf_xs, inf_ys, inf_x_is_inf, inf_y_is_inf)):
                plot_x = x_inf_pos if x_inf else x
                plot_y = y_inf_pos if y_inf else y
                plot_inf_xs.append(plot_x)
                plot_inf_ys.append(plot_y)
            
            plt.scatter(plot_inf_xs, plot_inf_ys, c='tab:red', alpha=0.8, edgecolors='k', 
                       marker='^', s=100)
            for px, py, label, x_inf, y_inf in zip(plot_inf_xs, plot_inf_ys, inf_labels, inf_x_is_inf, inf_y_is_inf):
                suffix = ""
                if x_inf and y_inf:
                    suffix = " (x=∞, y=∞)"
                elif x_inf:
                    suffix = " (x=∞)"
                elif y_inf:
                    suffix = " (y=∞)"
                plt.annotate(f"{label}{suffix}", (px, py), textcoords="offset points", 
                           xytext=(5, 5), ha='left', fontsize=8, color='tab:red')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Set equal axis limits for both x and y
        padding = data_range * 0.1
        
        # Extend axis limits to show infinite markers
        if inf_xs:
            plt.xlim(data_min - padding, x_inf_pos + padding)
            plt.ylim(data_min - padding, y_inf_pos + padding)
            # Add break indicator lines
            ax = plt.gca()
            if any(inf_x_is_inf):
                ax.axvline(x=data_max + data_range * 0.08, color='gray', linestyle=':', alpha=0.5)
            if any(inf_y_is_inf):
                ax.axhline(y=data_max + data_range * 0.08, color='gray', linestyle=':', alpha=0.5)
        else:
            # No infinite values - just set equal limits with padding
            plt.xlim(data_min - padding, data_max + padding)
            plt.ylim(data_min - padding, data_max + padding)
        
        # Ensure aspect ratio is equal
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    _plot(
        metric_key_x='recon_ssim',
        metric_key_y='dataset_ssim',
        title='SSIM: GT vs Render (x) vs Poisoned vs Original (y)',
        fname='scatter_ssim.png',
        xlabel='GT vs Render SSIM',
        ylabel='Poisoned vs Original SSIM'
    )

    _plot(
        metric_key_x='recon_psnr',
        metric_key_y='dataset_psnr',
        title='PSNR: GT vs Render (x) vs Poisoned vs Original (y)',
        fname='scatter_psnr.png',
        xlabel='GT vs Render PSNR',
        ylabel='Poisoned vs Original PSNR'
    )

    _plot(
        metric_key_x='recon_lpips',
        metric_key_y='dataset_lpips',
        title='LPIPS: GT vs Render (x) vs Poisoned vs Original (y)',
        fname='scatter_lpips.png',
        xlabel='GT vs Render LPIPS',
        ylabel='Poisoned vs Original LPIPS'
    )


def format_value(value: Optional[float], decimals: int = 4) -> str:
    """Format a value for display, handling None values and infinity."""
    if value is None:
        return "N/A"
    if value == float('inf'):
        return "∞"
    if value == float('-inf'):
        return "-∞"
    return f"{value:.{decimals}f}"


def print_table(results: List[Dict]) -> None:
    """
    Print a formatted table of results.
    
    Args:
        results: List of result dictionaries
    """
    # Table header
    header = ["Exp Type", "Dataset", "Recon SSIM ↓", "Recon PSNR ↓", "Recon LPIPS ↑", "Dataset SSIM ↑", "Dataset PSNR ↑", "Dataset LPIPS ↓"]
    
    # Calculate column widths
    col_widths = [len(h) for h in header]
    
    # Update widths based on data
    for r in results:
        col_widths[0] = max(col_widths[0], len(r.get('exp_type', '')))
        col_widths[1] = max(col_widths[1], len(r['dataset']))
        col_widths[2] = max(col_widths[2], len(format_value(r['recon_ssim'])))
        col_widths[3] = max(col_widths[3], len(format_value(r['recon_psnr'])))
        col_widths[4] = max(col_widths[4], len(format_value(r.get('recon_lpips'))))
        col_widths[5] = max(col_widths[5], len(format_value(r['dataset_ssim'])))
        col_widths[6] = max(col_widths[6], len(format_value(r['dataset_psnr'])))
        col_widths[7] = max(col_widths[7], len(format_value(r.get('dataset_lpips'))))
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    
    # Print separator
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"
    
    # Print header
    print("\n" + "=" * 100)
    print("ATTACK RESULTS TABLE")
    print("=" * 100)
    print()
    print(separator)
    header_row = "|" + "|".join([h.center(col_widths[i]) for i, h in enumerate(header)]) + "|"
    print(header_row)
    print(separator)
    
    # Print data rows
    for r in results:
        row = [
            r.get('exp_type', '').center(col_widths[0]),
            r['dataset'].center(col_widths[1]),
            format_value(r['recon_ssim']).center(col_widths[2]),
            format_value(r['recon_psnr']).center(col_widths[3]),
            format_value(r.get('recon_lpips')).center(col_widths[4]),
            format_value(r['dataset_ssim']).center(col_widths[5]),
            format_value(r['dataset_psnr']).center(col_widths[6]),
            format_value(r.get('dataset_lpips')).center(col_widths[7])
        ]
        print("|" + "|".join(row) + "|")
    
    print(separator)
    print()


def generate_latex(results: List[Dict]) -> str:
    """
    Generate LaTeX code for the results table.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        LaTeX code as a string
    """
    latex_lines = [
        "% =============================================================================",
        "% LATEX TABLE - Copy and paste this into your Overleaf document",
        "% =============================================================================",
        "",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Attack Results: Reconstruction Quality and Dataset Perturbation Metrics}",
        "    \\label{tab:attack_results}",
        "    \\begin{tabular}{llcccccc}",
        "        \\toprule",
        "        \\textbf{Exp Type} & \\textbf{Dataset} & \\multicolumn{3}{c}{\\textbf{GT vs Render}} & \\multicolumn{3}{c}{\\textbf{Poisoned vs Original}} \\\\",
        "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "        & & SSIM $\\downarrow$ & PSNR $\\downarrow$ & LPIPS $\\uparrow$ & SSIM $\\uparrow$ & PSNR $\\uparrow$ & LPIPS $\\downarrow$ \\\\",
        "        \\midrule",
    ]

    latex = "\n".join(latex_lines) + "\n"
    
    for r in results:
        # Clean up names for LaTeX (replace underscores)
        exp_type = r.get('exp_type', '').replace('_', r'\_')
        dataset_name = r['dataset'].replace('_', r'\_')
        
        recon_ssim = format_value(r['recon_ssim'], 4) if r['recon_ssim'] is not None else "N/A"
        recon_psnr = format_value(r['recon_psnr'], 2) if r['recon_psnr'] is not None else "N/A"
        recon_lpips = format_value(r.get('recon_lpips'), 4) if r.get('recon_lpips') is not None else "N/A"
        dataset_ssim = format_value(r['dataset_ssim'], 4) if r['dataset_ssim'] is not None else "N/A"
        dataset_psnr = format_value(r['dataset_psnr'], 2) if r['dataset_psnr'] is not None else "N/A"
        dataset_lpips = format_value(r.get('dataset_lpips'), 4) if r.get('dataset_lpips') is not None else "N/A"
        
        latex += f"        {exp_type} & {dataset_name} & {recon_ssim} & {recon_psnr} & {recon_lpips} & {dataset_ssim} & {dataset_psnr} & {dataset_lpips} \\\\\n"
    
    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "    \\begin{tablenotes}\n"
    latex += "        \\small\n"
    latex += "        \\item \\textbf{GT vs Render}: Ground truth images compared to rendered images from the trained model. Higher values indicate better reconstruction quality.\n"
    latex += "        \\item \\textbf{Poisoned vs Original}: Poisoned dataset images compared to original clean images. Lower values indicate more effective perturbation/attack.\n"
    latex += "    \\end{tablenotes}\n"
    latex += "\\end{table}\n\n"
    latex += "% Don't forget to include the booktabs package in your preamble:\n"
    latex += "% \\usepackage{booktabs}\n"
    latex += "% \\usepackage{threeparttable}  % Optional: for table notes\n"
    
    return latex


def main():
    # Default directories for experiments (path, experiment_type_label, prefix)
    # prefix is added to dataset names (e.g., 'da3' -> 'da3_chair')
    default_dirs = [
        # ("/home2/prajas.wadekar/patch-poison/log/geometric/Nerf_Synthetic", "geometric", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/da3/Nerf_Synthetic", "da3", "da3"),
        # ("/home2/prajas.wadekar/patch-poison/log/patched", "patched", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_variants", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_checkerboard_sizes", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_size", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_partial", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_colour", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_colour", "", ""),
        # ("/home2/prajas.wadekar/patch-poison/log/PatchPoison_checkerboard_alpha_useless", "", ""),

    ]
    
    plot_dir = None
    
    # Allow override from command line
    # Usage: python generate_results_table.py [dir:type:prefix] [dir:type] ... [--plot-dir DIR]
    # Examples:
    #   dir:type:prefix  - full specification with prefix
    #   dir:type         - no prefix
    #   dir              - uses parent dir name as type, no prefix
    import sys
    
    base_dirs = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--plot-dir' and i + 1 < len(sys.argv):
            plot_dir = sys.argv[i + 1]
            i += 2
            continue
        
        # Parse dir:type:prefix format
        parts = arg.split(':')
        if len(parts) >= 3:
            path, exp_type, prefix = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            path, exp_type = parts
            prefix = ""
        else:
            path = parts[0]
            # Use parent directory name as experiment type
            exp_type = os.path.basename(os.path.dirname(path))
            prefix = ""
        base_dirs.append((path, exp_type, prefix))
        i += 1
    
    # Use defaults if no directories specified
    if not base_dirs:
        base_dirs = default_dirs
    
    # Collect results
    results = collect_results(base_dirs)
    
    if not results:
        print("No results found!")
        return
    
    # Print table
    print_table(results)
    
    # Generate and print LaTeX
    latex_code = generate_latex(results)
    print("\n" + "=" * 100)
    print("LATEX CODE (Copy and paste into Overleaf)")
    print("=" * 100)
    print(latex_code)

    # Plots
    if plot_dir is None:
        plot_dir = "/home2/prajas.wadekar/patch-poison/plots/combined"
    plot_scatter(results, plot_dir)


if __name__ == "__main__":
    main()
