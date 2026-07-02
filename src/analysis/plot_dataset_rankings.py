from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PIVOT_CSV = PROJECT_ROOT / "results" / "analysis" / "round_metrics_pivot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "rankings"

MODEL_COLORS = {
    "qwen3_5_9b": "#1f77b4",
    "gpt_oss_20b": "#ff7f0e",
    "deepseek_v4_flash": "#2ca02c",
    "glm_5_1": "#d62728",
    "owl_alpha": "#9467bd",
    "unknown_model": "#7f7f7f",
}
REASONING_HATCHES = {
    "high": "///",
    "none": "",
    "medium": "\\\\\\",
    "low": "..",
    "unknown": "xx",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ranking plots by dataset from round_metrics_pivot.csv."
    )
    parser.add_argument(
        "--pivot-csv",
        default=str(DEFAULT_PIVOT_CSV),
        help="Path to round_metrics_pivot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PDF ranking plots will be written.",
    )
    parser.add_argument(
        "--metric-column",
        default="selected_metric",
        help="Column from the pivot CSV to rank by. Default: selected_metric.",
    )
    parser.add_argument(
        "--include-mismatched-datasets",
        action="store_true",
        help="Include runs where the actual dataset did not match the expected dataset.",
    )
    return parser.parse_args()


def _load_frame(csv_path: Path, metric_column: str, include_mismatched: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        raise ValueError(f"Column `{metric_column}` was not found in {csv_path}.")

    if "all_rounds_match_expected" in df.columns and not include_mismatched:
        mask = df["all_rounds_match_expected"].astype(str).str.lower() == "true"
        df = df.loc[mask].copy()

    df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
    df = df.dropna(subset=[metric_column]).copy()
    model_col = "model_display" if "model_display" in df.columns else "model_slug"
    dataset_col = "dataset_display" if "dataset_display" in df.columns else "dataset_slug"
    df["label"] = df[model_col].astype(str) + " (" + df["reasoning"].astype(str) + ")"
    df["dataset_label"] = df[dataset_col].astype(str)
    return df


def _plot_single_dataset(
    dataset_df: pd.DataFrame,
    dataset_slug: str,
    metric_column: str,
    output_dir: Path,
) -> Path:
    ordered = dataset_df.sort_values(metric_column, ascending=True).copy()

    height = max(3.0, 0.55 * len(ordered) + 1.3)
    fig, ax = plt.subplots(figsize=(9, height))

    y_positions = list(range(len(ordered)))
    bars = ax.barh(
        y_positions,
        ordered[metric_column].tolist(),
        color=[MODEL_COLORS.get(model, "#7f7f7f") for model in ordered["model_slug"]],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, reasoning in zip(bars, ordered["reasoning"], strict=False):
        bar.set_hatch(REASONING_HATCHES.get(str(reasoning), "xx"))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["label"].tolist())
    ax.set_xlabel(metric_column)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    min_value = float(ordered[metric_column].min())
    max_value = float(ordered[metric_column].max())
    span = max(max_value - min_value, 0.02)
    ax.set_xlim(max(0.0, min_value - 0.1 * span), min(1.0, max_value + 0.15 * span))

    for y_pos, value in zip(y_positions, ordered[metric_column].tolist(), strict=False):
        ax.text(value + 0.003, y_pos, f"{value:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    output_path = output_dir / f"ranking_{dataset_slug}.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_combined_grid(
    df: pd.DataFrame,
    metric_column: str,
    output_dir: Path,
) -> Path:
    datasets = sorted(df["dataset_slug"].dropna().unique().tolist())
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, max(4 * n_rows, 4)),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, dataset_slug in zip(axes.flat, datasets, strict=False):
        ax.set_visible(True)
        dataset_df = df.loc[df["dataset_slug"] == dataset_slug].sort_values(
            metric_column, ascending=True
        )
        y_positions = list(range(len(dataset_df)))
        bars = ax.barh(
            y_positions,
            dataset_df[metric_column].tolist(),
            color=[
                MODEL_COLORS.get(model, "#7f7f7f")
                for model in dataset_df["model_slug"].tolist()
            ],
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, reasoning in zip(bars, dataset_df["reasoning"], strict=False):
            bar.set_hatch(REASONING_HATCHES.get(str(reasoning), "xx"))

        ax.set_yticks(y_positions)
        ax.set_yticklabels(dataset_df["label"].tolist(), fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = output_dir / "ranking_all_datasets.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    pivot_csv = Path(args.pivot_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_frame(
        pivot_csv,
        metric_column=args.metric_column,
        include_mismatched=args.include_mismatched_datasets,
    )
    if df.empty:
        print("No rows available for plotting after filtering.")
        return 1

    written_paths: list[Path] = []
    for dataset_slug, dataset_df in df.groupby("dataset_slug", sort=True):
        written_paths.append(
            _plot_single_dataset(
                dataset_df=dataset_df,
                dataset_slug=str(dataset_slug),
                metric_column=args.metric_column,
                output_dir=output_dir,
            )
        )

    combined_path = _plot_combined_grid(
        df=df,
        metric_column=args.metric_column,
        output_dir=output_dir,
    )

    print(f"Wrote {len(written_paths)} per-dataset ranking plots to {output_dir}")
    print(f"Wrote combined plot to {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
