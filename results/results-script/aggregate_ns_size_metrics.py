#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


METRIC_KEYS = ("mean_psnr", "mean_ssim", "mean_lpips")


def parse_metric_log(log_path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k in METRIC_KEYS:
                values[k] = float(v)

    missing = [k for k in METRIC_KEYS if k not in values]
    if missing:
        raise ValueError(f"Missing keys {missing} in {log_path}")
    return values


def mean_std_str(xs: List[float]) -> str:
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.4f} +- {s:.4f}"


def build_table(root: Path) -> List[Dict[str, str]]:
    # key: (scene, patch_size) -> lists of run metrics
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    pattern = re.compile(r"patch_size_(\d+)")
    for rq in root.glob("*/patch_size_*/exp_run_*/benchmark_recon_quality.log"):
        scene = rq.parts[-4]
        patch_dir = rq.parts[-3]
        m = pattern.fullmatch(patch_dir)
        if not m:
            continue
        patch_size = int(m.group(1))

        dc = rq.parent / "benchmark_dataset_comparison.log"
        if not dc.exists():
            continue

        pr = parse_metric_log(rq)  # Poisoned vs Render
        gp = parse_metric_log(dc)  # GT vs Poisoned

        key = (scene, patch_size)
        if key not in grouped:
            grouped[key] = {
                "pr_psnr": [],
                "pr_ssim": [],
                "pr_lpips": [],
                "gp_psnr": [],
                "gp_ssim": [],
                "gp_lpips": [],
            }

        grouped[key]["pr_psnr"].append(pr["mean_psnr"])
        grouped[key]["pr_ssim"].append(pr["mean_ssim"])
        grouped[key]["pr_lpips"].append(pr["mean_lpips"])
        grouped[key]["gp_psnr"].append(gp["mean_psnr"])
        grouped[key]["gp_ssim"].append(gp["mean_ssim"])
        grouped[key]["gp_lpips"].append(gp["mean_lpips"])

    rows: List[Dict[str, str]] = []
    for (scene, patch_size) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        g = grouped[(scene, patch_size)]
        n_runs = len(g["pr_psnr"])

        rows.append(
            {
                "scene": scene,
                "patch_size": str(patch_size),
                "n_runs": str(n_runs),
                "pr_psnr": mean_std_str(g["pr_psnr"]),
                "pr_ssim": mean_std_str(g["pr_ssim"]),
                "pr_lpips": mean_std_str(g["pr_lpips"]),
                "gp_psnr": mean_std_str(g["gp_psnr"]),
                "gp_ssim": mean_std_str(g["gp_ssim"]),
                "gp_lpips": mean_std_str(g["gp_lpips"]),
            }
        )

    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    fieldnames = [
        "scene",
        "patch_size",
        "n_runs",
        "pr_psnr",
        "pr_ssim",
        "pr_lpips",
        "gp_psnr",
        "gp_ssim",
        "gp_lpips",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], out_md: Path) -> None:
    headers = [
        "scene",
        "patch_size",
        "n_runs",
        "PR PSNR",
        "PR SSIM",
        "PR LPIPS",
        "GP PSNR",
        "GP SSIM",
        "GP LPIPS",
    ]

    with out_md.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {scene} | {patch_size} | {n_runs} | {pr_psnr} | {pr_ssim} | {pr_lpips} | {gp_psnr} | {gp_ssim} | {gp_lpips} |\n".format(
                    **r
                )
            )


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


_PM_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))\s*\+\-\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))\s*$")


def metric_to_latex_pm(s: str) -> str:
    m = _PM_RE.match(s)
    if not m:
        return latex_escape(s)
    return f"${m.group(1)}\\pm{m.group(2)}$"


def metric_to_latex_mean_only(s: str) -> str:
    """Convert a 'mean +- std' string to LaTeX mean-only '$mean$'."""
    m = _PM_RE.match(s)
    if not m:
        return latex_escape(s)
    return f"${m.group(1)}$"


def write_latex(rows: List[Dict[str, str]], out_tex: Path) -> None:
    # Paper-ready LaTeX table using booktabs + threeparttable.
    # Requires the LaTeX preamble to include:
    #   \usepackage{booktabs}
    #   \usepackage{threeparttable}
    #
    # Column semantics:
    #   PR = Poisoned vs Render (reconstruction quality; higher SSIM/PSNR and lower LPIPS = better)
    #   GP = GT vs Poisoned (attack similarity; lower SSIM/PSNR and higher LPIPS = stronger attack)
    colspec = "llcccccc"  # Patch, Scene, then 6 metrics

    caption = "ns\_size results across patch sizes (mean $\\pm$ std over runs)."
    label = "tab:ns_size_patch_metrics"

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by aggregate_ns_size_metrics.py\n")
        f.write("% Requires: \\usepackage{booktabs}, \\usepackage{threeparttable}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write(f"    \\caption{{{caption}}}\n")
        f.write(f"    \\label{{{label}}}\n")
        f.write("    \\begin{threeparttable}\n")
        f.write(f"    \\begin{{tabular}}{{{colspec}}}\n")
        f.write("        \\toprule\n")
        f.write(
            "        \\textbf{Patch} & \\textbf{Scene} "
            "& \\multicolumn{3}{c}{\\textbf{Poisoned vs Render}} "
            "& \\multicolumn{3}{c}{\\textbf{GT vs Poisoned}} \\\\" 
        )
        f.write("\n")
        f.write("        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
        f.write(
            "        & "
            "& SSIM $\\uparrow$ & PSNR $\\uparrow$ & LPIPS $\\downarrow$ "
            "& SSIM $\\downarrow$ & PSNR $\\downarrow$ & LPIPS $\\uparrow$ \\\\" 
        )
        f.write("\n")
        f.write("        \\midrule\n")

        for r in rows:
            patch = latex_escape(r["patch_size"])
            scene = latex_escape(r["scene"])
            pr_ssim = metric_to_latex_pm(r["pr_ssim"])
            pr_psnr = metric_to_latex_pm(r["pr_psnr"])
            pr_lpips = metric_to_latex_pm(r["pr_lpips"])
            gp_ssim = metric_to_latex_mean_only(r["gp_ssim"])
            gp_psnr = metric_to_latex_mean_only(r["gp_psnr"])
            gp_lpips = metric_to_latex_mean_only(r["gp_lpips"])

            f.write(
                "        "
                + " & ".join(
                    [
                        patch,
                        scene,
                        pr_ssim,
                        pr_psnr,
                        pr_lpips,
                        gp_ssim,
                        gp_psnr,
                        gp_lpips,
                    ]
                )
                + r" \\"
                + "\n"
            )

        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("    \\begin{tablenotes}\n")
        f.write("        \\small\n")
        f.write(
            "        \\item \\textbf{Poisoned vs Render}: poisoned images compared to renders from the trained model (reconstruction quality).\n"
        )
        f.write(
            "        \\item \\textbf{GT vs Poisoned}: GT images compared to poisoned images (attack similarity).\n"
        )
        f.write(
            "        \\item GP columns report mean only; PR columns report mean $\\pm$ std over exp\\_run\\_1..$n$ (see column $n$ in the CSV/MD outputs).\n"
        )
        f.write("    \\end{tablenotes}\n")
        f.write("    \\end{threeparttable}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ns_size metrics from benchmark_recon_quality.log and "
            "benchmark_dataset_comparison.log into mean+-std tables."
        )
    )
    parser.add_argument(
        "--ns-size-dir",
        default="ns_size",
        help="Path to ns_size directory (default: ns_size)",
    )
    parser.add_argument(
        "--out-csv",
        default="results_size/ns_size_patch_metrics_mean_std.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-md",
        default="results_size/ns_size_patch_metrics_mean_std.md",
        help="Output Markdown path",
    )
    parser.add_argument(
        "--out-tex",
        default="results_size/ns_size_patch_metrics_mean_std.tex",
        help="Output LaTeX tabular path",
    )
    args = parser.parse_args()

    root = Path(args.ns_size_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    rows = build_table(root)
    if not rows:
        raise RuntimeError("No metrics found. Check directory structure and log files.")

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)
    for p in (out_csv, out_md, out_tex):
        p.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    write_latex(rows, out_tex)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote markdown table to {out_md}")
    print(f"Wrote LaTeX tabular to {out_tex}")


if __name__ == "__main__":
    main()
