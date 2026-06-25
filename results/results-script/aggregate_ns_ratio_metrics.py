#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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
                try:
                    values[k] = float(v)
                except ValueError:
                    values[k] = float("inf") if "inf" in v.lower() else float("nan")

    missing = [k for k in METRIC_KEYS if k not in values]
    if missing:
        raise ValueError(f"Missing keys {missing} in {log_path}")
    return values


def mean_std_str(xs: List[float]) -> str:
    if any(math.isinf(x) for x in xs):
        return "inf +- 0.0000"
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.4f} +- {s:.4f}"


def _parse_mean(s: str) -> float:
    """Extract the mean value from a 'mean +- std' string."""
    if s.startswith("inf"):
        return float("inf")
    return float(s.split("+-")[0].strip())


def build_table(root: Path) -> List[Dict[str, str]]:
    # key: (scene, poison_pct) -> lists of run metrics
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    pattern = re.compile(r"poison_(\d+)pct")
    for rq in root.glob("*/poison_*pct/exp_run_*/benchmark_recon_quality.log"):
        scene = rq.parts[-4]
        poison_dir = rq.parts[-3]
        m = pattern.fullmatch(poison_dir)
        if not m:
            continue
        poison_pct = int(m.group(1))

        dc = rq.parent / "benchmark_dataset_comparison.log"
        if not dc.exists():
            continue

        pr = parse_metric_log(rq)  # Poisoned vs Render
        gp = parse_metric_log(dc)  # GT vs Poisoned

        key = (scene, poison_pct)
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
    for (scene, poison_pct) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        g = grouped[(scene, poison_pct)]
        n_runs = len(g["pr_psnr"])

        rows.append(
            {
                "scene": scene,
                "poison_pct": str(poison_pct),
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


def build_across_scene_table(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Aggregate per-scene rows into across-scene mean+-std rows, one per poison_pct.

    For each poison_pct, we take the *mean* value from each scene's
    'mean +- std' string and compute a new mean +- std across scenes.
    """
    # poison_pct -> metric_name -> list of per-scene means
    by_pct: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        pct = r["poison_pct"]
        if pct not in by_pct:
            by_pct[pct] = {
                "pr_psnr": [],
                "pr_ssim": [],
                "pr_lpips": [],
                "gp_psnr": [],
                "gp_ssim": [],
                "gp_lpips": [],
            }
        for mk in ("pr_psnr", "pr_ssim", "pr_lpips", "gp_psnr", "gp_ssim", "gp_lpips"):
            by_pct[pct][mk].append(_parse_mean(r[mk]))

    across_rows: List[Dict[str, str]] = []
    for pct in sorted(by_pct.keys(), key=int):
        g = by_pct[pct]
        n_scenes = len(g["pr_psnr"])
        across_rows.append(
            {
                "scene": "ALL",
                "poison_pct": pct,
                "n_scenes": str(n_scenes),
                "pr_psnr": mean_std_str(g["pr_psnr"]),
                "pr_ssim": mean_std_str(g["pr_ssim"]),
                "pr_lpips": mean_std_str(g["pr_lpips"]),
                "gp_psnr": mean_std_str(g["gp_psnr"]),
                "gp_ssim": mean_std_str(g["gp_ssim"]),
                "gp_lpips": mean_std_str(g["gp_lpips"]),
            }
        )
    return across_rows


# ── Per-scene writers ─────────────────────────────────────────────────

def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    fieldnames = [
        "scene",
        "poison_pct",
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
        "poison_pct",
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
                "| {scene} | {poison_pct} | {n_runs} | {pr_psnr} | {pr_ssim} | {pr_lpips} | {gp_psnr} | {gp_ssim} | {gp_lpips} |\n".format(
                    **r
                )
            )


# ── Across-scene writers ─────────────────────────────────────────────

def write_across_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    fieldnames = [
        "scene",
        "poison_pct",
        "n_scenes",
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


def write_across_markdown(rows: List[Dict[str, str]], out_md: Path) -> None:
    headers = [
        "scene",
        "poison_pct",
        "n_scenes",
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
                "| {scene} | {poison_pct} | {n_scenes} | {pr_psnr} | {pr_ssim} | {pr_lpips} | {gp_psnr} | {gp_ssim} | {gp_lpips} |\n".format(
                    **r
                )
            )


# ── LaTeX helpers ─────────────────────────────────────────────────────

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
    colspec = "llcccccc"  # Poison%, Scene, then 6 metrics

    caption = "ns\\_ratio results across poison ratios (mean $\\pm$ std over runs)."
    label = "tab:ns_ratio_poison_metrics"

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by aggregate_ns_ratio_metrics.py\n")
        f.write("% Requires: \\usepackage{booktabs}, \\usepackage{threeparttable}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write(f"    \\caption{{{caption}}}\n")
        f.write(f"    \\label{{{label}}}\n")
        f.write("    \\begin{threeparttable}\n")
        f.write(f"    \\begin{{tabular}}{{{colspec}}}\n")
        f.write("        \\toprule\n")
        f.write(
            "        \\textbf{Poison\\%} & \\textbf{Scene} "
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
            poison = latex_escape(r["poison_pct"])
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
                        poison,
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


def write_across_latex(rows: List[Dict[str, str]], out_tex: Path) -> None:
    colspec = "lcccccc"  # Poison%, then 6 metrics

    caption = "ns\\_ratio across-scene results (mean $\\pm$ std over scenes)."
    label = "tab:ns_ratio_across_scene"

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by aggregate_ns_ratio_metrics.py\n")
        f.write("% Requires: \\usepackage{booktabs}, \\usepackage{threeparttable}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write(f"    \\caption{{{caption}}}\n")
        f.write(f"    \\label{{{label}}}\n")
        f.write("    \\begin{threeparttable}\n")
        f.write(f"    \\begin{{tabular}}{{{colspec}}}\n")
        f.write("        \\toprule\n")
        f.write(
            "        \\textbf{Poison\\%} "
            "& \\multicolumn{3}{c}{\\textbf{Poisoned vs Render}} "
            "& \\multicolumn{3}{c}{\\textbf{GT vs Poisoned}} \\\\" 
        )
        f.write("\n")
        f.write("        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(
            "        "
            "& SSIM $\\uparrow$ & PSNR $\\uparrow$ & LPIPS $\\downarrow$ "
            "& SSIM $\\downarrow$ & PSNR $\\downarrow$ & LPIPS $\\uparrow$ \\\\" 
        )
        f.write("\n")
        f.write("        \\midrule\n")

        for r in rows:
            poison = latex_escape(r["poison_pct"])
            pr_ssim = metric_to_latex_pm(r["pr_ssim"])
            pr_psnr = metric_to_latex_pm(r["pr_psnr"])
            pr_lpips = metric_to_latex_pm(r["pr_lpips"])
            gp_ssim = metric_to_latex_pm(r["gp_ssim"])
            gp_psnr = metric_to_latex_pm(r["gp_psnr"])
            gp_lpips = metric_to_latex_pm(r["gp_lpips"])

            f.write(
                "        "
                + " & ".join(
                    [
                        poison,
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
            "        \\item Across-scene aggregation: for each poison ratio, the per-scene mean is computed first, then mean $\\pm$ std is taken over all scenes.\n"
        )
        f.write("    \\end{tablenotes}\n")
        f.write("    \\end{threeparttable}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ns_ratio metrics from benchmark_recon_quality.log and "
            "benchmark_dataset_comparison.log into mean+-std tables."
        )
    )
    parser.add_argument(
        "--ns-ratio-dir",
        default="ns_ratio",
        help="Path to ns_ratio directory (default: ns_ratio)",
    )
    parser.add_argument(
        "--out-csv",
        default="results_ratio/ns_ratio_metrics_mean_std.csv",
        help="Output CSV path (per-scene)",
    )
    parser.add_argument(
        "--out-md",
        default="results_ratio/ns_ratio_metrics_mean_std.md",
        help="Output Markdown path (per-scene)",
    )
    parser.add_argument(
        "--out-tex",
        default="results_ratio/ns_ratio_metrics_mean_std.tex",
        help="Output LaTeX tabular path (per-scene)",
    )
    parser.add_argument(
        "--out-across-csv",
        default="results_ratio/ns_ratio_across_scene_mean_std.csv",
        help="Output CSV path (across-scene)",
    )
    parser.add_argument(
        "--out-across-md",
        default="results_ratio/ns_ratio_across_scene_mean_std.md",
        help="Output Markdown path (across-scene)",
    )
    parser.add_argument(
        "--out-across-tex",
        default="results_ratio/ns_ratio_across_scene_mean_std.tex",
        help="Output LaTeX tabular path (across-scene)",
    )
    args = parser.parse_args()

    root = Path(args.ns_ratio_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    # ── Per-scene tables ──────────────────────────────────────────────
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

    print(f"[per-scene] Wrote {len(rows)} rows to {out_csv}")
    print(f"[per-scene] Wrote markdown table to {out_md}")
    print(f"[per-scene] Wrote LaTeX tabular to {out_tex}")

    # ── Across-scene tables ───────────────────────────────────────────
    across_rows = build_across_scene_table(rows)

    out_across_csv = Path(args.out_across_csv)
    out_across_md = Path(args.out_across_md)
    out_across_tex = Path(args.out_across_tex)
    for p in (out_across_csv, out_across_md, out_across_tex):
        p.parent.mkdir(parents=True, exist_ok=True)
    write_across_csv(across_rows, out_across_csv)
    write_across_markdown(across_rows, out_across_md)
    write_across_latex(across_rows, out_across_tex)

    print(f"[across-scene] Wrote {len(across_rows)} rows to {out_across_csv}")
    print(f"[across-scene] Wrote markdown table to {out_across_md}")
    print(f"[across-scene] Wrote LaTeX tabular to {out_across_tex}")


if __name__ == "__main__":
    main()
