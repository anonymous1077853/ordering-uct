#!/usr/bin/env python3
"""Create a consistent Sachs comparison figure and terminal summaries."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ALGO_COMPARISONS = {
    "ordering_vs_random": ("ordering_uct", "random_ordering"),
    "ordering_vs_uctfull": ("ordering_uct", "uctfull"),
}

DEFAULT_EXPERIMENT_DIRS = {
    "ordering_vs_random": "sachs_budget_modifier_50_ordering_vs_random",
    "ordering_vs_uctfull": "sachs_time_budget_900s",
}

RESULT_TYPES = ("construct", "prune_cam")

ALGO_LABELS = {
    "ordering_uct": "Ordering-UCT",
    "uctfull": "CD-UCT",
    "random_ordering": "Random Ordering",
}

METRICS = {
    "shd": "SHD (↓)",
    "tpr": "TPR (↑)",
    "fdr": "FDR (↓)",
    "reward": "Reward (↑)",
    "pred_density": "Pred Density (↓)",
}

COLORS = {
    "ordering_uct": "#ff7f0e",
    "uctfull": "#1f77b4",
    "random_ordering": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Sachs algorithm comparison metrics.")
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Experiment directory under root (default depends on --algorithm-comparison)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root path that contains the experiment directory (default: current dir)",
    )
    parser.add_argument(
        "--result-type",
        choices=list(RESULT_TYPES),
        default="construct",
        help="Which result type to use for the saved figure (default: construct)",
    )
    parser.add_argument(
        "--metrics-subdir",
        choices=["eval_results", "hyperopt_results"],
        default="eval_results",
        help="Which metrics subdirectory under models/ to read (default: eval_results)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Directory for output plots (default: <experiment-dir>/figures)",
    )
    parser.add_argument(
        "--algorithm-comparison",
        choices=sorted(ALGO_COMPARISONS),
        default="ordering_vs_random",
        help="Algorithm pair to include in plots and summaries (default: ordering_vs_random)",
    )
    return parser.parse_args()


def get_edge_count(edges: Any) -> int | None:
    if isinstance(edges, list):
        return len(edges)
    return None


def infer_num_nodes(record: dict[str, Any]) -> int:
    max_node = -1
    results = record.get("results", {})
    if not isinstance(results, dict):
        return 0

    for section in ("construct", "prune_cam", "ground_truth"):
        section_data = results.get(section, {})
        if not isinstance(section_data, dict):
            continue
        edges = section_data.get("edges")
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if (
                isinstance(edge, list)
                and len(edge) == 2
                and isinstance(edge[0], int)
                and isinstance(edge[1], int)
            ):
                max_node = max(max_node, edge[0], edge[1])
    return max_node + 1 if max_node >= 0 else 0


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def ci95_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    stderr = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(1.96 * stderr)


def format_mean_ci95(values: list[float], precision: int = 4) -> str:
    if not values:
        return "N/A"
    mean, _ = mean_sd(values)
    ci95 = ci95_half_width(values)
    return f"{mean:.{precision}f} +/- {ci95:.{precision}f}"


def load_metrics(
    metrics_dir: str,
    result_type: str,
    algo_order: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    duration_construct: dict[str, list[float]] = defaultdict(list)
    ordering_evals: dict[str, list[float]] = defaultdict(list)
    pred_densities: dict[str, list[float]] = defaultdict(list)
    gt_densities: dict[str, list[float]] = defaultdict(list)

    metric_files = sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.json")))
    if not metric_files:
        raise FileNotFoundError(f"No metric files found in: {metrics_dir}")

    for path in metric_files:
        algo = os.path.basename(path).split("-bic-")[0]
        if algo not in algo_order:
            continue

        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)

        results = record.get("results", {}).get(result_type, {})
        if not isinstance(results, dict):
            continue

        for metric in ("shd", "tpr", "fdr", "reward"):
            value = results.get(metric)
            if isinstance(value, (int, float)):
                metric_values[algo][metric].append(float(value))

        n_nodes = infer_num_nodes(record)
        max_dag_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes >= 2 else 0
        if max_dag_edges > 0:
            pred_edges = get_edge_count(results.get("edges"))
            if pred_edges is not None:
                pred_densities[algo].append(pred_edges / max_dag_edges)
                metric_values[algo]["pred_density"].append(pred_edges / max_dag_edges)

            gt_edges = get_edge_count(record.get("results", {}).get("ground_truth", {}).get("edges"))
            if gt_edges is not None:
                gt_densities[algo].append(gt_edges / max_dag_edges)

        duration_value = record.get("duration_construct_s")
        if isinstance(duration_value, (int, float)):
            duration_construct[algo].append(float(duration_value))

        ordering_value = record.get("num_ordering_evals")
        if isinstance(ordering_value, (int, float)):
            ordering_evals[algo].append(float(ordering_value))

    return metric_values, duration_construct, ordering_evals, pred_densities, gt_densities


def print_metric_summary(metric_values: dict[str, dict[str, list[float]]], result_type: str, algo_order: tuple[str, ...]) -> None:
    display_result_type = "Prune" if result_type == "prune_cam" else "Construct"
    print(f"\n{display_result_type} metrics by algorithm (mean +/- 95% CI):")
    header = f"{'Algorithm':<18} {'N':>5} " + " ".join(f"{METRICS[metric]:>18}" for metric in METRICS)
    print(header)
    print("-" * len(header))

    for algo in algo_order:
        lengths = [len(metric_values.get(algo, {}).get(metric, [])) for metric in METRICS]
        n_values = max(lengths, default=0)
        row = [f"{ALGO_LABELS[algo]:<18} {n_values:>5}"]
        for metric in METRICS:
            row.append(format_mean_ci95(metric_values.get(algo, {}).get(metric, [])).rjust(18))
        print(" ".join(row))


def print_duration_summary(duration_construct: dict[str, list[float]], algo_order: tuple[str, ...]) -> None:
    print("\nAverage duration_construct_s by algorithm (mean +/- 95% CI):")
    print(f"{'Algorithm':<18} {'N':>5} {'Mean(s)':>12} {'CI95(s)':>12}")
    print("-" * 50)
    for algo in algo_order:
        values = duration_construct.get(algo, [])
        if values:
            mean, _ = mean_sd(values)
            ci95 = ci95_half_width(values)
            print(f"{ALGO_LABELS[algo]:<18} {len(values):>5} {mean:>12.4f} {ci95:>12.4f}")
        else:
            print(f"{ALGO_LABELS[algo]:<18} {0:>5} {'N/A':>12} {'N/A':>12}")


def print_additional_summary(
    ordering_evals: dict[str, list[float]],
    pred_densities: dict[str, list[float]],
    gt_densities: dict[str, list[float]],
    algo_order: tuple[str, ...],
) -> None:
    print("\nAdditional metrics by algorithm (mean +/- 95% CI):")
    print(
        f"{'Algorithm':<18} {'N':>5} {'OrderingEvals':>18} "
        f"{'PredDensity':>18} {'GTDensity':>18}"
    )
    print("-" * 84)

    for algo in algo_order:
        n_values = max(
            len(ordering_evals.get(algo, [])),
            len(pred_densities.get(algo, [])),
            len(gt_densities.get(algo, [])),
        )
        print(
            f"{ALGO_LABELS[algo]:<18} {n_values:>5} "
            f"{format_mean_ci95(ordering_evals.get(algo, []), precision=1).rjust(18)} "
            f"{format_mean_ci95(pred_densities.get(algo, [])).rjust(18)} "
            f"{format_mean_ci95(gt_densities.get(algo, [])).rjust(18)}"
        )


def plot_metrics(
    metric_values: dict[str, dict[str, list[float]]],
    result_type: str,
    out_path: str,
    algo_order: tuple[str, ...],
) -> bool:
    plot_metrics_items = {k: v for k, v in METRICS.items() if k != "pred_density"}
    n_metrics = len(plot_metrics_items)
    n_cols = 2
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    display_result_type = "Prune" if result_type == "prune_cam" else "Construct"

    fig.suptitle(f"Sachs comparison: {display_result_type}", fontsize=14, fontweight="bold")
    any_plotted = False

    for ax, (metric, label) in zip(axes, plot_metrics_items.items()):
        labels: list[str] = []
        means: list[float] = []
        ci95_half_widths: list[float] = []
        colors: list[str] = []

        for algo in algo_order:
            values = metric_values.get(algo, {}).get(metric, [])
            if not values:
                continue
            mean, _ = mean_sd(values)
            labels.append(ALGO_LABELS[algo])
            means.append(mean)
            ci95_half_widths.append(ci95_half_width(values))
            colors.append(COLORS[algo])

        if not labels:
            ax.set_title(f"{label} (no data)")
            ax.axis("off")
            continue

        any_plotted = True
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=ci95_half_widths, capsize=6, color=colors, zorder=3)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Algorithm", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_box_aspect(1)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.5)

        for index, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{means[index]:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for ax in axes[n_metrics:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if any_plotted:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return any_plotted


def main() -> int:
    args = parse_args()
    algo_order = ALGO_COMPARISONS[args.algorithm_comparison]
    experiment_dir = args.experiment_dir or DEFAULT_EXPERIMENT_DIRS[args.algorithm_comparison]
    exp_dir = os.path.join(args.root, experiment_dir)
    metrics_dir = os.path.join(exp_dir, "models", args.metrics_subdir)
    outdir = args.outdir or os.path.join(exp_dir, "figures")

    print(f"Loaded metrics from: {metrics_dir}")
    print(f"Algorithm comparison: {args.algorithm_comparison} ({', '.join(algo_order)})")

    terminal_metric_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    cached_duration_construct: dict[str, list[float]] | None = None
    cached_ordering_evals: dict[str, list[float]] | None = None
    cached_pred_densities: dict[str, list[float]] | None = None
    cached_gt_densities: dict[str, list[float]] | None = None

    for result_type in RESULT_TYPES:
        (
            metric_values,
            duration_construct,
            ordering_evals,
            pred_densities,
            gt_densities,
        ) = load_metrics(metrics_dir, result_type, algo_order)
        terminal_metric_values[result_type] = metric_values

        if cached_duration_construct is None:
            cached_duration_construct = duration_construct
            cached_ordering_evals = ordering_evals
            cached_pred_densities = pred_densities
            cached_gt_densities = gt_densities

    out_path = os.path.join(
        outdir,
        f"{experiment_dir}_{args.metrics_subdir}_{args.result_type}_{args.algorithm_comparison}_combined.png",
    )
    ok = plot_metrics(terminal_metric_values[args.result_type], args.result_type, out_path, algo_order)

    print("\nCreated figure:")
    if ok:
        print(f"  - {out_path}")
    else:
        print("  - No data available to plot.")

    for result_type in RESULT_TYPES:
        print_metric_summary(terminal_metric_values[result_type], result_type, algo_order)
    print_duration_summary(cached_duration_construct or {}, algo_order)
    print_additional_summary(
        cached_ordering_evals or {},
        cached_pred_densities or {},
        cached_gt_densities or {},
        algo_order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
