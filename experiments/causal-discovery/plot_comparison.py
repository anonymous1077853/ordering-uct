"""Plot synthetic experiment comparison metrics across node counts.

Usage:
    python plot_comparison.py [--result-type {construct,prune_cam}]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import get_backend

ALGORITHMS = ("ordering_uct", "uctfull")
RESULT_TYPES = ("construct", "prune_cam")

ALGO_LABELS = {
    "ordering_uct": "Ordering-UCT",
    "uctfull": "CD-UCT",
}

METRICS = {
    "shd": "SHD (↓)",
    "tpr": "TPR (↑)",
    "fdr": "FDR (↓)",
    "reward": "Reward (↑)",
    "pred_avg_degree": "Pred Avg Degree",
}

COLORS = {
    "ordering_uct": "#ff7f0e",
    "uctfull": "#1f77b4",
}

MARKERS = {
    "ordering_uct": "s",
    "uctfull": "o",
}


def ci95_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    stderr = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(1.96 * stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic experiment comparison metrics.")
    parser.add_argument(
        "--result-type",
        choices=list(RESULT_TYPES),
        default="prune_cam",
        help="Which result type to use for the saved figure (default: prune_cam)",
    )
    return parser.parse_args()


def discover_experiments(base_dir: str) -> list[tuple[int, str]]:
    pattern = os.path.join(base_dir, "synth*lr_time_budget_scaling")
    experiments: list[tuple[int, str]] = []
    for exp_dir in glob.glob(pattern):
        match = re.search(r"synth(\d+)lr_time_budget_scaling", os.path.basename(exp_dir))
        if match:
            experiments.append((int(match.group(1)), exp_dir))
    return sorted(experiments)


def load_metric_record(path: str, result_type: str, n_nodes: int) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    results = record.get("results", {}).get(result_type, {})
    if not isinstance(results, dict):
        return {}

    payload: dict[str, float] = {}
    for metric in METRICS:
        if metric == "pred_avg_degree":
            continue
        value = results.get(metric)
        if isinstance(value, (int, float)):
            payload[metric] = float(value)

    pred_avg_degree = infer_pred_avg_degree(results, n_nodes)
    if pred_avg_degree is not None:
        payload["pred_avg_degree"] = pred_avg_degree
    return payload


def infer_pred_avg_degree(results: dict[str, Any], n_nodes: int) -> float | None:
    edges = results.get("edges")
    if not isinstance(edges, list) or n_nodes < 1:
        return None
    return len(edges) / n_nodes


def load_result_data(experiments: list[tuple[int, str]], result_type: str) -> dict[str, dict[str, list[tuple[int, float]]]]:
    data: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for n_nodes, exp_dir in experiments:
        results_dir = os.path.join(exp_dir, "models", "eval_results")
        for algo in ALGORITHMS:
            paths = sorted(glob.glob(os.path.join(results_dir, f"{algo}-bic-hardcoded-*_metrics.json")))
            if not paths:
                print(f"  Warning: no {algo} metrics files found in {results_dir}")
                continue

            for path in paths:
                metric_record = load_metric_record(path, result_type, n_nodes)
                for metric, value in metric_record.items():
                    data[algo][metric].append((n_nodes, value))

    return data


def aggregate_by_node(
    data: dict[str, dict[str, list[tuple[int, float]]]],
    experiments: list[tuple[int, str]],
) -> dict[str, dict[str, tuple[list[int], list[float], list[float]]]]:
    node_counts_all = sorted({n_nodes for n_nodes, _ in experiments})
    aggregated: dict[str, dict[str, tuple[list[int], list[float], list[float]]]] = {}

    for algo, metric_data in data.items():
        aggregated[algo] = {}
        for metric, pairs in metric_data.items():
            by_node: dict[int, list[float]] = defaultdict(list)
            for n_nodes, value in pairs:
                by_node[n_nodes].append(value)

            node_counts: list[int] = []
            means: list[float] = []
            ci95_half_widths: list[float] = []
            for n_nodes in node_counts_all:
                values = by_node.get(n_nodes, [])
                if not values:
                    continue
                arr = np.array(values, dtype=float)
                node_counts.append(n_nodes)
                means.append(float(arr.mean()))
                ci95_half_widths.append(ci95_half_width(values))

            aggregated[algo][metric] = (node_counts, means, ci95_half_widths)

    return aggregated


def format_mean_ci95(mean: float, ci95: float) -> str:
    return f"{mean:.4f} +/- {ci95:.4f}"


def print_metric_summary(
    aggregated: dict[str, dict[str, tuple[list[int], list[float], list[float]]]],
    result_type: str,
) -> None:
    display_result_type = "Prune" if result_type == "prune_cam" else "Construct"
    print(f"\n{display_result_type} metric summary (mean +/- 95% CI):")
    col_width = 21
    header = f"\n{'Algorithm':<16} {'Nodes':>6}  " + "  ".join(f"{metric:>{col_width}}" for metric in METRICS)
    print(header)
    print("-" * len(header))

    for algo in ALGORITHMS:
        algo_metrics = aggregated.get(algo, {})
        available_metrics = [metric for metric in METRICS if metric in algo_metrics]
        if not available_metrics:
            continue

        node_counts = algo_metrics[available_metrics[0]][0]
        for index, n_nodes in enumerate(node_counts):
            values: list[str] = []
            for metric in METRICS:
                payload = algo_metrics.get(metric)
                if payload is None:
                    values.append(f"{'N/A':>{col_width}}")
                    continue
                metric_nodes, means, ci95_half_widths = payload
                metric_index = metric_nodes.index(n_nodes) if n_nodes in metric_nodes else None
                if metric_index is None:
                    values.append(f"{'N/A':>{col_width}}")
                else:
                    values.append(
                        f"{format_mean_ci95(means[metric_index], ci95_half_widths[metric_index]):>{col_width}}"
                    )
            algo_label = ALGO_LABELS[algo] if index == 0 else ""
            print(f"{algo_label:<16} {n_nodes:>6}  " + "  ".join(values))
        print()


def plot_metrics(
    aggregated: dict[str, dict[str, tuple[list[int], list[float], list[float]]]],
    result_type: str,
    out_path: str,
) -> None:
    plot_metrics_items = {k: v for k, v in METRICS.items() if k != "pred_avg_degree"}
    n_metrics = len(plot_metrics_items)
    n_cols = 2
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    display_result_type = "Prune" if result_type == "prune_cam" else "Construct"

    fig.suptitle(f"Algorithm Comparison: {display_result_type}", fontsize=14, fontweight="bold")

    for ax, (metric, label) in zip(axes, plot_metrics_items.items()):
        for algo in ALGORITHMS:
            payload = aggregated.get(algo, {}).get(metric)
            if payload is None:
                continue

            node_counts, means, ci95_half_widths = payload
            means_arr = np.array(means, dtype=float)
            ci95_arr = np.array(ci95_half_widths, dtype=float)

            ax.plot(
                node_counts,
                means_arr,
                marker=MARKERS[algo],
                color=COLORS[algo],
                label=ALGO_LABELS[algo],
                linewidth=2,
            )
            ax.fill_between(
                node_counts,
                means_arr - ci95_arr,
                means_arr + ci95_arr,
                alpha=0.2,
                color=COLORS[algo],
            )

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Number of nodes", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_box_aspect(1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(sorted({n for algo in ALGORITHMS for n in aggregated.get(algo, {}).get(metric, ([], [], []))[0]}))
        ax.legend(fontsize=10)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    if "agg" not in get_backend().lower():
        plt.show()


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Scanning for experiments in: {base_dir}")
    experiments = discover_experiments(base_dir)
    if not experiments:
        print("No experiments found matching synth*lr_time_budget_scaling/")
        return

    print(f"Found {len(experiments)} experiment(s): " + ", ".join(f"{n_nodes} nodes" for n_nodes, _ in experiments))
    print(f"Result type: {args.result_type}")

    aggregated_by_result: dict[str, dict[str, tuple[list[int], list[float], list[float]]]] = {}
    for result_type in RESULT_TYPES:
        data = load_result_data(experiments, result_type)
        aggregated = aggregate_by_node(data, experiments)
        aggregated_by_result[result_type] = aggregated
        print_metric_summary(aggregated, result_type)

    out_path = os.path.join(base_dir, "figures", f"comparison_{args.result_type}.png")
    plot_metrics(aggregated_by_result[args.result_type], args.result_type, out_path)


if __name__ == "__main__":
    main()
