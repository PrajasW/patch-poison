#!/usr/bin/env python3
"""
Generate a results table for the PatchPoison_bottom_left folder structure:
log/PatchPoison_bottom_left/<dataset>/patch_size_<N>/exp_run_{1..3}

Extracts SSIM/PSNR/LPIPS metrics from:
- benchmark_recon_quality.log (GT vs Render)
- benchmark_dataset_comparison.log (Poisoned vs Actual)

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


def collect_results_PatchPoison_bottom_left(base_dir: str) -> List[Dict]:
    results: List[Dict] = []

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return results

    datasets = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        patch_sizes = sorted(
            [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]
        )

        for patch in patch_sizes:
            recon_ssims: List[float] = []
            recon_psnrs: List[float] = []
            recon_lpips_vals: List[float] = []
            dataset_ssims: List[float] = []
            dataset_psnrs: List[float] = []
            dataset_lpips_vals: List[float] = []

            for run_idx in (1, 2, 3):
                exp_path = os.path.join(dataset_path, patch, f"exp_run_{run_idx}")
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
                    "exp_type": patch,
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
        "Patch Size",
        "Dataset",
        "Recon SSIM ↓",
        "Recon PSNR ↓",
        "Recon LPIPS ↑",
        "Dataset SSIM ↑",
        "Dataset PSNR ↑",
        "Dataset LPIPS ↓",
    ]

    col_widths = [len(h) for h in header]
    for r in results:
        col_widths[0] = max(col_widths[0], len(r.get("exp_type", "")))
        col_widths[1] = max(col_widths[1], len(r["dataset"]))
        col_widths[2] = max(col_widths[2], len(format_mean_std(r["recon_ssim_mean"], r["recon_ssim_std"])) )
        col_widths[3] = max(col_widths[3], len(format_mean_std(r["recon_psnr_mean"], r["recon_psnr_std"], 2)))
        col_widths[4] = max(col_widths[4], len(format_mean_std(r["recon_lpips_mean"], r["recon_lpips_std"])) )
        col_widths[5] = max(col_widths[5], len(format_value(r["dataset_ssim_mean"])))
        col_widths[6] = max(col_widths[6], len(format_value(r["dataset_psnr_mean"], 2)))
        col_widths[7] = max(col_widths[7], len(format_value(r["dataset_lpips_mean"])))

    col_widths = [w + 2 for w in col_widths]
    separator = "+" + "+".join(["-" * w for w in col_widths]) + "+"

    print("\n" + "=" * 100)
    print("ATTACK RESULTS TABLE (mean ± std across exp runs)")
    print("=" * 100)
    print()
    print(separator)
    header_row = "|" + "|".join([h.center(col_widths[i]) for i, h in enumerate(header)]) + "|"
    print(header_row)
    print(separator)

    for r in results:
        row = [
            r.get("exp_type", "").center(col_widths[0]),
            r["dataset"].center(col_widths[1]),
            format_mean_std(r["recon_ssim_mean"], r["recon_ssim_std"]).center(col_widths[2]),
            format_mean_std(r["recon_psnr_mean"], r["recon_psnr_std"], 2).center(col_widths[3]),
            format_mean_std(r["recon_lpips_mean"], r["recon_lpips_std"]).center(col_widths[4]),
            format_value(r["dataset_ssim_mean"]).center(col_widths[5]),
            format_value(r["dataset_psnr_mean"], 2).center(col_widths[6]),
            format_value(r["dataset_lpips_mean"]).center(col_widths[7]),
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
        "    \\caption{Attack Results: Reconstruction Quality and Dataset Perturbation Metrics}",
        "    \\label{tab:attack_results}",
        "    \\begin{threeparttable}",
        "    \\begin{tabular}{llcccccc}",
        "        \\toprule",
        "        \\textbf{Exp Type} & \\textbf{Dataset} & \\multicolumn{3}{c}{\\textbf{GT vs Render}} & \\multicolumn{3}{c}{\\textbf{Poisoned vs Original}} \\\\",
        "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "        & & SSIM $\\downarrow$ & PSNR $\\downarrow$ & LPIPS $\\uparrow$ & SSIM $\\uparrow$ & PSNR $\\uparrow$ & LPIPS $\\downarrow$ \\\\",
        "        \\midrule",
    ]

    latex = "\n".join(latex_lines) + "\n"

    def _format_latex_value(value: Optional[float], decimals: int) -> str:
        if value is None:
            return "N/A"
        if value == float("inf"):
            return "$\\infty$"
        if value == float("-inf"):
            return "$-\\infty$"
        return f"${value:.{decimals}f}$"

    def _format_latex_mean_pm(mean: Optional[float], std: Optional[float], decimals: int) -> str:
        if mean is None:
            return "N/A"
        if std is None:
            return _format_latex_value(mean, decimals)
        return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"

    for r in results:
        exp_type = r.get("exp_type", "").replace("_", r"\_")
        dataset_name = r["dataset"].replace("_", r"\_")

        recon_ssim = _format_latex_mean_pm(r["recon_ssim_mean"], r["recon_ssim_std"], 4)
        recon_psnr = _format_latex_mean_pm(r["recon_psnr_mean"], r["recon_psnr_std"], 2)
        recon_lpips = _format_latex_mean_pm(r["recon_lpips_mean"], r["recon_lpips_std"], 4)
        dataset_ssim = _format_latex_value(r["dataset_ssim_mean"], 4)
        dataset_psnr = _format_latex_value(r["dataset_psnr_mean"], 2)
        dataset_lpips = _format_latex_value(r["dataset_lpips_mean"], 4)

        latex += (
            f"        {exp_type} & {dataset_name} & {recon_ssim} & {recon_psnr} & {recon_lpips} "
            f"& {dataset_ssim} & {dataset_psnr} & {dataset_lpips} \\\\\n"
        )

    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    latex += "    \\begin{tablenotes}\n"
    latex += "        \\small\n"
    latex += "        \\item \\textbf{GT vs Render}: Ground truth images compared to rendered images from the trained model. Higher values indicate better reconstruction quality.\n"
    latex += "        \\item \\textbf{Poisoned vs Original}: Poisoned dataset images compared to original clean images. Lower values indicate more effective perturbation/attack.\n"
    latex += "        \\item Values are reported as mean $\\pm$ std for GT vs Render over exp\_run\_1..3.\n"
    latex += "    \\end{tablenotes}\n"
    latex += "    \\end{threeparttable}\n"
    latex += "\\end{table}\n\n"
    latex += "% Don't forget to include the booktabs package in your preamble:\n"
    latex += "% \\usepackage{booktabs}\n"
    latex += "% \\usepackage{threeparttable}  % Optional: for table notes\n"

    return latex


def plot_scatter(results: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _plot(metric_key_x: str, metric_key_y: str, title: str, fname: str, xlabel: str, ylabel: str):
        xs = []
        ys = []
        labels = []
        for r in results:
            x = r.get(metric_key_x)
            y = r.get(metric_key_y)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
            labels.append(r["dataset"])

        if not xs:
            print(f"No data available for plot {fname}, skipping.")
            return

        plt.figure(figsize=(7, 7))
        plt.scatter(xs, ys, c="tab:blue", alpha=0.8, edgecolors="k")
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    _plot(
        metric_key_x="recon_ssim_mean",
        metric_key_y="dataset_ssim_mean",
        title="SSIM: GT vs Render (x) vs Poisoned vs Original (y)",
        fname="scatter_ssim.png",
        xlabel="GT vs Render SSIM",
        ylabel="Poisoned vs Original SSIM",
    )

    _plot(
        metric_key_x="recon_psnr_mean",
        metric_key_y="dataset_psnr_mean",
        title="PSNR: GT vs Render (x) vs Poisoned vs Original (y)",
        fname="scatter_psnr.png",
        xlabel="GT vs Render PSNR",
        ylabel="Poisoned vs Original PSNR",
    )

    _plot(
        metric_key_x="recon_lpips_mean",
        metric_key_y="dataset_lpips_mean",
        title="LPIPS: GT vs Render (x) vs Poisoned vs Original (y)",
        fname="scatter_lpips.png",
        xlabel="GT vs Render LPIPS",
        ylabel="Poisoned vs Original LPIPS",
    )


def main() -> None:
    base_dir = "/home2/prajas.wadekar/patch-poison/log/PatchPoison_bottom_right"
    results = collect_results_PatchPoison_bottom_left(base_dir)

    if not results:
        print("No results found!")
        return

    print_table(results)

    latex_code = generate_latex(results)
    print("\n" + "=" * 100)
    print("LATEX CODE (Copy and paste into Overleaf)")
    print("=" * 100)
    print(latex_code)

    plot_dir = "/home2/prajas.wadekar/patch-poison/plots/PatchPoison_bottom_right"
    plot_scatter(results, plot_dir)


if __name__ == "__main__":
    main()
