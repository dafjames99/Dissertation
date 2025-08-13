"""
Minimal utilities to inspect and visualize evaluation CSVs.

Usage:
  python -m src.models.weak.inspect_evaluations [--run-id RUN_ID]

This will:
- Load compiled CSVs written by evaluate.py
- Show a bar chart of aggregate_ap_weighted per configuration (run_id)
- Optionally show per-category AP_mean for a selected run (or latest)
"""

from __future__ import annotations

CAT_METRICS = ["AP_mean", "NDCG_mean", "jobs_evaluated", "P_at_1", "P_at_5", "P_at_10"]
AGG_METRICS = [
    "aggregate_ap_weighted",
    "macro_ap",
    "mse_weighted_vs_similarity",
    "micro_jobs_evaluated",
    "micro_ap_mean",
    "micro_ndcg10_mean",
    "micro_p_at_1",
    "micro_p_at_5",
    "micro_p_at_10",
]
import numpy as np
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

src_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(src_path))

from utils.paths import EVAL_DIR  # noqa: E402


def load_data(eval_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    runs_path = eval_dir / "compiled_eval_runs.csv"
    per_cat_path = eval_dir / "compiled_eval_per_category.csv"

    runs_df = pd.read_csv(runs_path) if runs_path.exists() else None
    per_cat_df = pd.read_csv(per_cat_path) if per_cat_path.exists() else None
    return runs_df, per_cat_df


def plot_metric_bars(ax, runs_df: pd.DataFrame, metric: str, title: str | None = None):
    if runs_df is None or runs_df.empty:
        ax.set_visible(False)
        return
    df = runs_df.copy()
    if metric not in df.columns:
        ax.set_title(f"Missing metric: {metric}")
        ax.set_visible(False)
        return
    # Sort descending for most metrics; for MSE-like metrics, ascending is more intuitive
    ascending = metric.lower().startswith("mse")
    df = df.sort_values(metric, ascending=ascending)
    ax.bar(df["run_id"], df[metric])
    ax.set_title(title or metric)
    ax.set_xlabel("run_id")
    ax.set_ylabel(metric)
    ax.set_xticklabels(df["run_id"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_per_category(
    ax, per_cat_df: pd.DataFrame | None, run_id: str | None, top_n: int = 25
):
    if per_cat_df is None or per_cat_df.empty:
        ax.set_visible(False)
        return

    df = per_cat_df.copy()
    chosen_run = run_id
    if chosen_run is None:
        if "timestamp" in df.columns:
            latest_ts = df["timestamp"].max()
            chosen_run = df.loc[df["timestamp"] == latest_ts, "run_id"].iloc[0]
        else:
            chosen_run = df["run_id"].iloc[-1]

    dfr = df[df["run_id"] == chosen_run]
    if dfr.empty:
        ax.set_title(f"No per-category rows for run_id='{chosen_run}'.")
        ax.set_visible(False)
        return

    if "AP_mean" in dfr.columns:
        dfr = dfr.sort_values("AP_mean", ascending=False)
    dfr_top = dfr.head(min(top_n, len(dfr)))

    ax.barh(dfr_top["category"], dfr_top.get("AP_mean", pd.Series([0] * len(dfr_top))))
    ax.invert_yaxis()
    ax.set_title(f"Per-category AP_mean (Top {len(dfr_top)}) — {chosen_run}")
    ax.set_xlabel("AP_mean")
    ax.set_ylabel("category")
    ax.grid(axis="x", linestyle="--", alpha=0.4)


def plot_category_heatmap(
    ax,
    per_cat_df: pd.DataFrame | None,
    metric: str = "AP_mean",
    top_categories: int = 30,
):
    if per_cat_df is None or per_cat_df.empty:
        ax.set_visible(False)
        return
    if metric not in per_cat_df.columns:
        ax.set_title(f"Missing metric in per-category CSV: {metric}")
        ax.set_visible(False)
        return

    df = per_cat_df.copy()
    # Pivot to run_id (rows) x category (cols)
    H = df.pivot(index="run_id", columns="category", values=metric)
    # Select top-N categories by overall mean (descending)
    means = H.mean(axis=0, skipna=True).sort_values(ascending=False)
    keep_cols = means.head(min(top_categories, len(means))).index
    H = H.loc[:, keep_cols]

    # Coerce to numeric for plotting safety
    H = H.apply(pd.to_numeric, errors="coerce")
    sns.heatmap(H, cmap="viridis", annot=False, cbar=True, ax=ax)
    ax.set_title(f"{metric} heatmap (run_id x category)")
    ax.set_xlabel("category")
    ax.set_ylabel("run_id")
    ax.tick_params(axis="x", labelrotation=90)


def _parse_list_arg(csv: str) -> list[str]:
    if csv is None:
        return None
    return [s.strip() for s in csv.split(",") if s.strip()]


def plot_version_comparison_dashboard(
    runs_df: pd.DataFrame,
    versions: tuple[str, str] = ("v1", "v2"),
    variants: list[str] | None = None,
    pca_dims: list[str] | None = None,
    metrics: list[str] | None = None,
):
    """Side-by-side compare two versions across variants and PCA settings.

    Each subplot is one (variant, pca) configuration; x-axis lists metrics, with two
    bars per metric (one per version).
    """
    if runs_df is None or runs_df.empty:
        return None

    ver1, ver2 = versions
    df = runs_df.copy()

    if variants is None:
        variants = (
            df["variant"].dropna().astype(str).sort_values().unique().tolist()
            if "variant" in df.columns
            else []
        )
    if pca_dims is None:
        dims = []
        if "pca_dim" in df.columns:
            dims = df["pca_dim"].dropna().astype(int).sort_values().unique().tolist()
        pca_dims = ["none"] + [str(d) for d in dims]
    if metrics is None:
        metrics = AGG_METRICS
    try:
        metrics.remove("micro_jobs_evaluated")
    except:
        pass

    comparisons: list[tuple[str, str, pd.Series, pd.Series]] = []
    for var in variants:
        for p in pca_dims:
            if p == "none":
                mask1 = (
                    (df.get("version") == ver1)
                    & (df.get("variant") == var)
                    & (df.get("pca_dim").isna())
                )
                mask2 = (
                    (df.get("version") == ver2)
                    & (df.get("variant") == var)
                    & (df.get("pca_dim").isna())
                )
            else:
                try:
                    p_int = int(p)
                except Exception:
                    continue
                mask1 = (
                    (df.get("version") == ver1)
                    & (df.get("variant") == var)
                    & (df.get("pca_dim") == p_int)
                )
                mask2 = (
                    (df.get("version") == ver2)
                    & (df.get("variant") == var)
                    & (df.get("pca_dim") == p_int)
                )

            rows1 = df[mask1][metrics]
            rows2 = df[mask2][metrics]
            if rows1.empty or rows2.empty:
                continue
            comparisons.append((var, p, rows1.iloc[0], rows2.iloc[0]))
    if not comparisons:
        return None

    num = len(comparisons)
    rows = num
    fig, axes = plt.subplots(rows, 1, figsize=(14, max(3.5 * rows, 4.5)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Precompute common x positions and labels
    x = np.arange(len(metrics))
    width = 0.4
    handles_ref = None
    for idx, (var, p, row1, row2) in enumerate(comparisons):
        ax = axes[idx]
        vals1 = [float(row1.get(m, np.nan)) for m in metrics]
        vals2 = [float(row2.get(m, np.nan)) for m in metrics]

        # Determine which version is better for each metric
        # For most metrics, higher is better; for MSE-like, lower is better
        better_v1 = []
        for i, (v1, v2) in enumerate(zip(vals1, vals2)):
            if pd.isna(v1) or pd.isna(v2):
                better_v1.append(None)
            elif metrics[i].lower().startswith("mse"):
                better_v1.append(v1 < v2)  # Lower MSE is better
            else:
                better_v1.append(v1 > v2)  # Higher is better for most metrics

        # Create bars with color coding: green for better, red for worse
        colors1 = [
            "green" if b else "red" if b is not None else "gray" for b in better_v1
        ]
        colors2 = [
            "red" if b else "green" if b is not None else "gray" for b in better_v1
        ]

        b1 = ax.bar(
            x - width / 2,
            vals1,
            width,
            label=ver1,
            color=colors1,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )
        b2 = ax.bar(
            x + width / 2,
            vals2,
            width,
            label=ver2,
            color=colors2,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add version labels above the bars to show left=ver1, right=ver2
        ax.text(
            -0.5,
            ax.get_ylim()[1] * 0.95,
            f"← {ver1}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.5,
            ax.get_ylim()[1] * 0.95,
            f"{ver2} →",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        # Keep legend handles from the first subplot only
        if handles_ref is None:
            # Create custom legend handles with the color scheme
            from matplotlib.patches import Rectangle

            green_patch = Rectangle(
                (0, 0),
                1,
                1,
                facecolor="green",
                alpha=0.7,
                edgecolor="black",
                linewidth=1,
            )
            red_patch = Rectangle(
                (0, 0), 1, 1, facecolor="red", alpha=0.7, edgecolor="black", linewidth=1
            )
            handles_ref = (green_patch, red_patch)

        p_label = "no PCA" if p == "none" else f"PCA {p}"
        # Y-label should summarize the contents per request
        ax.set_ylabel(f"variant {var}, {p_label}")
        ax.set_xticks(x)
        # Only show tick labels on the bottom axis
        if idx < num - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(
                [m.replace("_", " ") for m in metrics], rotation=45, ha="right"
            )
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Single global legend (top-right)
    if handles_ref is not None:
        fig.legend(
            handles_ref, ["Better Performance", "Worse Performance"], loc="upper right"
        )

    fig.suptitle("Version Comparison Dashboard", fontsize=14)
    fig.supxlabel("Metric")
    fig.tight_layout(pad=1)
    return fig


def plot_p_at_k(
    ax,
    run_df: pd.DataFrame,
    top_n_runs: int = 5,
    sort_by: str = "micro_ap_mean",
):
    # Guard: required columns present
    required_cols = {"run_id", "micro_p_at_1", "micro_p_at_5", "micro_p_at_10"}
    if run_df is None or run_df.empty or not required_cols.issubset(run_df.columns):
        ax.set_visible(False)
        return

    # Fallback for sort metric if missing
    sort_metric = (
        sort_by
        if sort_by in run_df.columns
        else (
            "aggregate_ap_weighted"
            if "aggregate_ap_weighted" in run_df.columns
            else "micro_ap_mean"
        )
    )

    # Select top-N runs by sort metric (descending)
    df = run_df.copy()
    df = df.sort_values(sort_metric, ascending=False, na_position="last")
    df = df.head(min(top_n_runs, len(df)))

    run_ids = df["run_id"].tolist()
    p1 = df["micro_p_at_1"].astype(float).to_numpy()
    p5 = df["micro_p_at_5"].astype(float).to_numpy()
    p10 = df["micro_p_at_10"].astype(float).to_numpy()

    # Grouped bars: per run_id group, bars for K in {1,5,10}
    x = np.arange(len(run_ids))
    width = 0.25
    offsets = (-width, 0.0, width)

    ax.bar(x + offsets[0], p1, width, label="P@1")
    ax.bar(x + offsets[1], p5, width, label="P@5")
    ax.bar(x + offsets[2], p10, width, label="P@10")

    ax.set_ylabel("Score")
    ax.set_title(f"Top {len(run_ids)} runs by {sort_metric} — P@K")
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend(title="K", loc="upper left", ncols=3)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def main():
    parser = argparse.ArgumentParser(description="Inspect evaluation CSVs")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to visualize per-category results for",
    )
    parser.add_argument(
        "--top-n-categories",
        type=int,
        default=25,
        help="Top-N categories to show in the per-category bar chart",
    )
    parser.add_argument(
        "--heatmap-metric",
        type=str,
        default="AP_mean",
        help=f"Per-category metric to visualize in the heatmap: {CAT_METRICS}",
    )
    parser.add_argument(
        "--heatmap-top-categories",
        type=int,
        default=30,
        help="Top-N categories to include as columns in the heatmap (by mean value)",
    )
    parser.add_argument(
        "--compare-versions",
        nargs=2,
        metavar=("VER1", "VER2"),
        default=None,
        help="If provided, show a separate comparison dashboard for these versions (e.g., v1 v2)",
    )
    parser.add_argument(
        "--compare-variants",
        type=str,
        default="a,b,c",
        help="Comma-separated variant letters to include for version comparison (default: a,b,c)",
    )
    parser.add_argument(
        "--compare-pca",
        type=str,
        default="none,64,128,256",
        help="Comma-separated PCA dims to include (use 'none' for no PCA). Default: none,64,128,256",
    )
    parser.add_argument(
        "--compare-metrics",
        type=str,
        default=None,
        help=f"Comma-separated metrics to compare (default: {AGG_METRICS})",
    )
    parser.add_argument(
        "--top-run-count",
        type=int,
        default=5,
        help="Top-N runs to show in the P@K grouped bar chart (by --sort-by)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="micro_ap_mean",
        help=f"Metric column to rank runs by for P@K (default: micro_ap_mean). Options include: {AGG_METRICS}",
    )
    args = parser.parse_args()

    print(f"Evaluation directory: {EVAL_DIR}")
    runs_df, per_cat_df = load_data(EVAL_DIR)

    if runs_df is not None:
        print(f"Loaded runs: {len(runs_df)}")
        print(runs_df.head(3))
    if per_cat_df is not None:
        print(f"Loaded per-category rows: {len(per_cat_df)}")
        print(per_cat_df.head(3))
    # Metrics to visualize
    bar_metrics = [
        "aggregate_ap_weighted",
        "macro_ap",
        "mse_weighted_vs_similarity",
        "micro_ap_mean",
    ]

    # Build a single figure with multiple subplots
    fig = plt.figure(figsize=(30, 20))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    # Top row: up to 4 bar charts (truncate if >4)
    for i, metric in enumerate(bar_metrics[:4]):
        ax = fig.add_subplot(gs[i, 0])
        pretty = metric.replace("_", " ")
        plot_metric_bars(ax, runs_df, metric, title=pretty.title())

    # Bottom row: run_id x category heatmap (span cols 0-2) + per-category (col 3)
    ax_heat = fig.add_subplot(gs[:2, 1:])
    plot_category_heatmap(
        ax_heat,
        per_cat_df,
        metric=args.heatmap_metric,
        top_categories=args.heatmap_top_categories,
    )

    # ax_cat = fig.add_subplot(gs[1, 1:])
    # plot_per_category(ax_cat, per_cat_df, args.run_id, top_n=args.top_n_categories)

    ax_patk = fig.add_subplot(gs[2:, 1:])
    plot_p_at_k(
        ax_patk,
        runs_df,
        top_n_runs=int(args.top_run_count),
        sort_by=str(args.sort_by),
    )
    fig.suptitle("Evaluation Dashboard", fontsize=14)
    plt.tight_layout(pad=10)
    plt.show()

    # Optional version comparison dashboard
    if args.compare_versions and runs_df is not None:
        variants = _parse_list_arg(args.compare_variants)
        pca_dims = _parse_list_arg(args.compare_pca)
        metrics = _parse_list_arg(args.compare_metrics)
        plot_version_comparison_dashboard(
            runs_df,
            versions=(args.compare_versions[0], args.compare_versions[1]),
            variants=variants,
            pca_dims=pca_dims,
            metrics=metrics,
        )
        plt.show()


if __name__ == "__main__":
    main()
