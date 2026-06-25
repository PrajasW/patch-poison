#!/usr/bin/env python3
"""
Generate a results table for the full_dataset_clean folder structure:
log/full_dataset_clean/<dataset>/exp_run_{1..3}

Extracts SSIM/PSNR/LPIPS metrics from:
- benchmark_recon_quality.log (GT vs Render)
- benchmark_dataset_comparison.log (Poisoned vs Actual - though for clean data this should be ~1.0)

Outputs a printed table and LaTeX code with mean ± std across exp runs.
"""

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_log_file(log_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not os.path.exists(log_path):
        return None, None, None

    ssim = None
    psnr = None
    lpips = None

    with open(log_path, "r") as f:
        content = f.read()

    ssim_matches = re.findall(r"mean_ssim=([0-9.]+|inf|-inf)", content, re.IGNORECASE)
    psnr_matches = re.findall(r"mean_psnr=([0-9.]+|inf|-inf)", content, re.IGNORECASE)
    lpips_matches = re.findall(r"mean_lpips=([0-9.]+|inf|-inf)", content, re.IGNORECASE)

    if ssim_matches:
        val = ssim_matches[-1].lower()
        ssim = float("inf") if val == "inf" else float("-inf") if val == "-inf" else float(val)
    if psnr_matches:
        val = psnr_matches[-1].lower()
        psnr = float("inf") if val == "inf" else float("-inf") if val == "-inf" else float(val)
    if lpips_matches:
        val = lpips_matches[-1].lower()
        lpips = float("inf") if val == "inf" else float("-inf") if val == "-inf" else float(val)

    return ssim, psnr, lpips


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if any(v in (float("inf"), float("-inf")) for v in values):
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def collect_results_clean_dataset(base_dir: str) -> List[Dict]:
    results: List[Dict] = []

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return results

    datasets = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        
        recon_ssims: List[float] = []
        recon_psnrs: List[float] = []
        recon_lpips_vals: List[float] = []
        dataset_ssims: List[float] = []
        dataset_psnrs: List[float] = []
        dataset_lpips_vals: List[float] = []

        for run_idx in (1, 2, 3):
            exp_path = os.path.join(dataset_path, f"exp_run_{run_idx}")
            if not os.path.exists(exp_path):
                print(f"Warning: {exp_path} does not exist, skipping...")
                continue

            recon_log = os.path.join(exp_path, "benchmark_recon_quality.log")
            recon_ssim, recon_psnr, recon_lpips = parse_log_file(recon_log)

            dataset_log = os.path.join(exp_path, "benchmark_dataset_comparison.log")
            dataset_ssim, dataset_psnr, dataset_lpips = parse_log_file(dataset_log)

            if recon_ssim is not None:
                recon_ssims.append(recon_ssim)
            if recon_psnr is not None:
                recon_psnrs.append(recon_psnr)
            if recon_lpips is not None:
                recon_lpips_vals.append(recon_lpips)

            if dataset_ssim is not None:
                dataset_ssims.append(dataset_ssim)
            if dataset_psnr is not None:
                dataset_psnrs.append(dataset_psnr)
            if dataset_lpips is not None:
                dataset_lpips_vals.append(dataset_lpips)

        recon_ssim_mean, recon_ssim_std = _mean_std(recon_ssims)
        recon_psnr_mean, recon_psnr_std = _mean_std(recon_psnrs)
        recon_lpips_mean, recon_lpips_std = _mean_std(recon_lpips_vals)
        dataset_ssim_mean, dataset_ssim_std = _mean_std(dataset_ssims)
        dataset_psnr_mean, dataset_psnr_std = _mean_std(dataset_psnrs)
        dataset_lpips_mean, dataset_lpips_std = _mean_std(dataset_lpips_vals)

        results.append(
            {
                "dataset": dataset,
                "recon_ssim_mean": recon_ssim_mean,
                "recon_ssim_std": recon_ssim_std,
                "recon_psnr_mean": recon_psnr_mean,
                "recon_psnr_std": recon_psnr_std,
                "recon_lpips_mean": recon_lpips_mean,
                "recon_lpips_std": recon_lpips_std,
                "dataset_ssim_mean": dataset_ssim_mean,
                "dataset_ssim_std": dataset_ssim_std,
                "dataset_psnr_mean": dataset_psnr_mean,
                "dataset_psnr_std": dataset_psnr_std,
                "dataset_lpips_mean": dataset_lpips_mean,
                "dataset_lpips_std": dataset_lpips_std,
            }
        )

    return results


def format_mean_std(mean: Optional[float], std: Optional[float], decimals: int = 4) -> str:
    if mean is None:
        return "N/A"
    if std is None:
        return format_value(mean, decimals)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_value(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:.{decimals}f}"


def print_table(results: List[Dict]) -> None:
    header = [
        "Dataset",
        "SSIM",
        "PSNR",
        "LPIPS",
        "Dataset SSIM",
        "Dataset PSNR",
        "Dataset LPIPS",
    ]

    col_widths = [len(h) for h in header]
    for r in results:
        col_widths[0] = max(col_widths[0], len(r["dataset"]))
        col_widths[1] = max(col_widths[1], len(format_mean_std(r["recon_ssim_mean"], r["recon_ssim_std"])))
        col_widths[2] = max(col_widths[2], len(format_mean_std(r["recon_psnr_mean"], r["recon_psnr_std"], 2)))
        col_widths[3] = max(col_widths[3], len(format_mean_std(r["recon_lpips_mean"], r["recon_lpips_std"])))
        col_widths[4] = max(col_widths[4], len(format_mean_std(r["dataset_ssim_mean"], r["dataset_ssim_std"])))
        col_widths[5] = max(col_widths[5], len(format_mean_std(r["dataset_psnr_mean"], r["dataset_psnr_std"], 2)))
        col_widths[6] = max(col_widths[6], len(format_mean_std(r["dataset_lpips_mean"], r["dataset_lpips_std"])))

    col_widths = [w + 2 for w in col_widths]
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"

    print("\n" + "=" * 100)
    print("CLEAN DATASET RESULTS TABLE (mean ± std across exp runs)")
    print("=" * 100)
    print()
    print(separator)
    header_row = "|" + "|".join([h.center(col_widths[i]) for i, h in enumerate(header)]) + "|"
    print(header_row)
    print(separator)

    for r in results:
        row = [
            r["dataset"].center(col_widths[0]),
            format_mean_std(r["recon_ssim_mean"], r["recon_ssim_std"]).center(col_widths[1]),
            format_mean_std(r["recon_psnr_mean"], r["recon_psnr_std"], 2).center(col_widths[2]),
            format_mean_std(r["recon_lpips_mean"], r["recon_lpips_std"]).center(col_widths[3]),
            format_mean_std(r["dataset_ssim_mean"], r["dataset_ssim_std"]).center(col_widths[4]),
            format_mean_std(r["dataset_psnr_mean"], r["dataset_psnr_std"], 2).center(col_widths[5]),
            format_mean_std(r["dataset_lpips_mean"], r["dataset_lpips_std"]).center(col_widths[6]),
        ]
        print("|" + "|".join(row) + "|")

    print(separator)
    print()


def generate_latex(results: List[Dict]) -> str:
    latex_lines = [
        "% =============================================================================",
        "% LATEX TABLE - Copy and paste this into your Overleaf document",
        "% =============================================================================",
        "",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Baseline Results on Clean Dataset: Reconstruction Quality}",
        "    \\label{tab:clean_dataset_results}",
        "    \\begin{threeparttable}",
        "    \\begin{tabular}{lccc|ccc}",
        "        \\toprule",
        "        \\textbf{Dataset} & \\multicolumn{3}{c}{\\textbf{GT vs Render}} & \\multicolumn{3}{c}{\\textbf{Dataset Comparison}} \\\\",
        "        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "        & SSIM $\\uparrow$ & PSNR $\\uparrow$ & LPIPS $\\downarrow$ & SSIM & PSNR & LPIPS \\\\",
        "        \\midrule",
    ]

    latex = "\n".join(latex_lines) + "\n"

    def _format_latex_mean_pm(mean: Optional[float], std: Optional[float], decimals: int) -> str:
        if mean is None:
            return "N/A"
        if std is None:
            if mean == float("inf"):
                return "$\\infty$"
            if mean == float("-inf"):
                return "$-\\infty$"
            return f"${mean:.{decimals}f}$"
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    for r in results:
        dataset_name = r["dataset"].replace("_", r"\_")

        recon_ssim = _format_latex_mean_pm(r["recon_ssim_mean"], r["recon_ssim_std"], 4)
        recon_psnr = _format_latex_mean_pm(r["recon_psnr_mean"], r["recon_psnr_std"], 2)
        recon_lpips = _format_latex_mean_pm(r["recon_lpips_mean"], r["recon_lpips_std"], 4)
        dataset_ssim = _format_latex_mean_pm(r["dataset_ssim_mean"], r["dataset_ssim_std"], 4)
        dataset_psnr = _format_latex_mean_pm(r["dataset_psnr_mean"], r["dataset_psnr_std"], 2)
        dataset_lpips = _format_latex_mean_pm(r["dataset_lpips_mean"], r["dataset_lpips_std"], 4)

        latex += (
            f"        {dataset_name} & {recon_ssim} & {recon_psnr} & {recon_lpips} "
            f"& {dataset_ssim} & {dataset_psnr} & {dataset_lpips} \\\\\n"
        )

    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "    \\begin{tablenotes}\n"
    latex += "        \\small\n"
    latex += "        \\item \\textbf{GT vs Render}: Ground truth images compared to rendered images from the trained model.\n"
    latex += "        \\item \\textbf{Dataset Comparison}: Comparison between dataset images (should be 1.0/perfect for clean data).\n"
    latex += "        \\item Values are reported as mean $\\pm$ std across exp\\_run\\_1, exp\\_run\\_2, and exp\\_run\\_3.\n"
    latex += "    \\end{tablenotes}\n"
    latex += "    \\end{threeparttable}\n"
    latex += "\\end{table}\n\n"
    latex += "% Don't forget to include the booktabs package in your preamble:\n"
    latex += "% \\usepackage{booktabs}\n"
    latex += "% \\usepackage{threeparttable}  % Optional: for table notes\n"

    return latex


def plot_results(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Bar chart for SSIM across datasets
    datasets = [r["dataset"] for r in results]
    ssim_means = [r["recon_ssim_mean"] if r["recon_ssim_mean"] is not None else 0 for r in results]
    ssim_stds = [r["recon_ssim_std"] if r["recon_ssim_std"] is not None else 0 for r in results]
    psnr_means = [r["recon_psnr_mean"] if r["recon_psnr_mean"] is not None else 0 for r in results]
    psnr_stds = [r["recon_psnr_std"] if r["recon_psnr_std"] is not None else 0 for r in results]
    lpips_means = [r["recon_lpips_mean"] if r["recon_lpips_mean"] is not None else 0 for r in results]
    lpips_stds = [r["recon_lpips_std"] if r["recon_lpips_std"] is not None else 0 for r in results]

    # SSIM bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(datasets)), ssim_means, yerr=ssim_stds, capsize=5, alpha=0.7, edgecolor="k")
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    plt.ylabel("SSIM")
    plt.title("Reconstruction Quality: SSIM across Datasets")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ssim_comparison.png"), dpi=200)
    plt.close()
    print(f"Saved plot: {os.path.join(out_dir, 'ssim_comparison.png')}")

    # PSNR bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(datasets)), psnr_means, yerr=psnr_stds, capsize=5, alpha=0.7, edgecolor="k", color="tab:orange")
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    plt.ylabel("PSNR (dB)")
    plt.title("Reconstruction Quality: PSNR across Datasets")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_comparison.png"), dpi=200)
    plt.close()
    print(f"Saved plot: {os.path.join(out_dir, 'psnr_comparison.png')}")

    # LPIPS bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(datasets)), lpips_means, yerr=lpips_stds, capsize=5, alpha=0.7, edgecolor="k", color="tab:green")
    plt.xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    plt.ylabel("LPIPS")
    plt.title("Reconstruction Quality: LPIPS across Datasets")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lpips_comparison.png"), dpi=200)
    plt.close()
    print(f"Saved plot: {os.path.join(out_dir, 'lpips_comparison.png')}")


def main() -> None:
    base_dir = "/home2/prajas.wadekar/patch-poison/log/full_dataset_clean"
    results = collect_results_clean_dataset(base_dir)

    if not results:
        print("No results found!")
        return

    print_table(results)

    latex_code = generate_latex(results)
    print("\n" + "=" * 100)
    print("LATEX CODE (Copy and paste into Overleaf)")
    print("=" * 100)
    print(latex_code)

    plot_dir = "/home2/prajas.wadekar/patch-poison/plots/full_dataset_clean"
    plot_results(results, plot_dir)


if __name__ == "__main__":
    main()
