from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIVOT_CSV = PROJECT_ROOT / "results" / "analysis" / "round_metrics_pivot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

METRIC_COLUMN = "selected_metric"

MODEL_COLORS = {
    "qwen3_5_9b": "#1f77b4",
    "gpt_oss_20b": "#ff7f0e",
    "deepseek_v4_flash": "#2ca02c",
    "glm_5_1": "#d62728",
    "owl_alpha": "#9467bd",
    "unknown_model": "#7f7f7f",
}
REASONING_DISPLAY = {
    "none": "No Thinking",
    "high": "High Thinking",
    "medium": "Medium Thinking",
    "low": "Low Thinking",
    "unknown": "Unknown",
}
REASONING_OFFSETS = {
    "none": -0.18,
    "high": 0.18,
    "low": -0.06,
    "medium": 0.06,
    "unknown": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build summary tables and plots for dataset selection accuracy and final performance."
    )
    parser.add_argument(
        "--pivot-csv",
        default=str(DEFAULT_PIVOT_CSV),
        help="Path to round_metrics_pivot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where analysis tables and plots will be written.",
    )
    return parser.parse_args()


def _load_runs(pivot_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pivot_csv)
    df[METRIC_COLUMN] = pd.to_numeric(df[METRIC_COLUMN], errors="coerce")
    df["dataset_correct"] = (
        df["all_rounds_match_expected"].astype(str).str.lower() == "true"
    )
    df["effective_score"] = df[METRIC_COLUMN].where(df["dataset_correct"], 0.0)
    df["reasoning_display"] = df["reasoning"].astype(str).map(
        lambda value: REASONING_DISPLAY.get(value, value)
    )
    df["model_reasoning"] = (
        (
            df["model_display"].astype(str)
            if "model_display" in df.columns
            else df["model_slug"].astype(str)
        )
        + " ("
        + df["reasoning_display"].astype(str)
        + ")"
    )
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _dataset_selection_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby(["model_slug", "reasoning"], dropna=False)
        .agg(
            runs_total=("run_dir_name", "count"),
            dataset_match_count=("dataset_correct", "sum"),
            dataset_match_rate=("dataset_correct", "mean"),
        )
        .reset_index()
        .sort_values(["dataset_match_rate", "dataset_match_count"], ascending=[False, False])
    )
    if "model_display" in df.columns:
        model_display_map = (
            df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        overall.insert(1, "model_display", overall["model_slug"].map(model_display_map))

    per_dataset = (
        df.groupby(["expected_dataset_slug", "model_slug", "reasoning"], dropna=False)
        .agg(
            runs_total=("run_dir_name", "count"),
            dataset_match_count=("dataset_correct", "sum"),
            dataset_match_rate=("dataset_correct", "mean"),
        )
        .reset_index()
        .sort_values(
            ["expected_dataset_slug", "dataset_match_rate", "dataset_match_count"],
            ascending=[True, False, False],
        )
    )
    if "expected_dataset_display" in df.columns:
        dataset_display_map = (
            df.drop_duplicates("expected_dataset_slug")
            .set_index("expected_dataset_slug")["expected_dataset_display"]
            .to_dict()
        )
        per_dataset.insert(
            1,
            "expected_dataset_display",
            per_dataset["expected_dataset_slug"].map(dataset_display_map),
        )
    if "model_display" in df.columns:
        model_display_map = (
            df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        per_dataset.insert(3, "model_display", per_dataset["model_slug"].map(model_display_map))
    return overall, per_dataset


def _valid_performance_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = df.loc[df["dataset_correct"]].copy()

    overall = (
        valid_df.groupby(["model_slug", "reasoning"], dropna=False)
        .agg(
            valid_runs=("run_dir_name", "count"),
            mean_f1_macro=(METRIC_COLUMN, "mean"),
            median_f1_macro=(METRIC_COLUMN, "median"),
            min_f1_macro=(METRIC_COLUMN, "min"),
            max_f1_macro=(METRIC_COLUMN, "max"),
        )
        .reset_index()
        .sort_values(["mean_f1_macro", "valid_runs"], ascending=[False, False])
    )
    if "model_display" in valid_df.columns:
        model_display_map = (
            valid_df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        overall.insert(1, "model_display", overall["model_slug"].map(model_display_map))

    per_dataset = (
        valid_df.groupby(["expected_dataset_slug", "model_slug", "reasoning"], dropna=False)
        .agg(
            valid_runs=("run_dir_name", "count"),
            mean_f1_macro=(METRIC_COLUMN, "mean"),
            median_f1_macro=(METRIC_COLUMN, "median"),
            min_f1_macro=(METRIC_COLUMN, "min"),
            max_f1_macro=(METRIC_COLUMN, "max"),
        )
        .reset_index()
        .sort_values(
            ["expected_dataset_slug", "mean_f1_macro"],
            ascending=[True, False],
        )
    )
    if "expected_dataset_display" in valid_df.columns:
        dataset_display_map = (
            valid_df.drop_duplicates("expected_dataset_slug")
            .set_index("expected_dataset_slug")["expected_dataset_display"]
            .to_dict()
        )
        per_dataset.insert(
            1,
            "expected_dataset_display",
            per_dataset["expected_dataset_slug"].map(dataset_display_map),
        )
    if "model_display" in valid_df.columns:
        model_display_map = (
            valid_df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        per_dataset.insert(3, "model_display", per_dataset["model_slug"].map(model_display_map))
    return overall, per_dataset


def _effective_score_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby(["model_slug", "reasoning"], dropna=False)
        .agg(
            runs_total=("run_dir_name", "count"),
            dataset_match_rate=("dataset_correct", "mean"),
            mean_f1_macro_valid=(METRIC_COLUMN, lambda s: s[df.loc[s.index, "dataset_correct"]].mean()),
            mean_effective_f1_macro=("effective_score", "mean"),
        )
        .reset_index()
        .sort_values(["mean_effective_f1_macro", "dataset_match_rate"], ascending=[False, False])
    )
    if "model_display" in df.columns:
        model_display_map = (
            df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        overall.insert(1, "model_display", overall["model_slug"].map(model_display_map))

    per_dataset = (
        df.groupby(["expected_dataset_slug", "model_slug", "reasoning"], dropna=False)
        .agg(
            runs_total=("run_dir_name", "count"),
            dataset_match_rate=("dataset_correct", "mean"),
            mean_f1_macro_valid=(METRIC_COLUMN, lambda s: s[df.loc[s.index, "dataset_correct"]].mean()),
            mean_effective_f1_macro=("effective_score", "mean"),
        )
        .reset_index()
        .sort_values(
            ["expected_dataset_slug", "mean_effective_f1_macro"],
            ascending=[True, False],
        )
    )
    if "expected_dataset_display" in df.columns:
        dataset_display_map = (
            df.drop_duplicates("expected_dataset_slug")
            .set_index("expected_dataset_slug")["expected_dataset_display"]
            .to_dict()
        )
        per_dataset.insert(
            1,
            "expected_dataset_display",
            per_dataset["expected_dataset_slug"].map(dataset_display_map),
        )
    if "model_display" in df.columns:
        model_display_map = (
            df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )
        per_dataset.insert(3, "model_display", per_dataset["model_slug"].map(model_display_map))
    return overall, per_dataset


def _plot_dataset_match_rate(overall_df: pd.DataFrame, output_dir: Path) -> Path:
    ordered = overall_df.sort_values("dataset_match_rate", ascending=True).copy()
    if "model_display" in ordered.columns:
        ordered["plot_label"] = (
            ordered["model_display"].astype(str)
            + " ("
            + ordered["reasoning"].astype(str).map(
                lambda value: REASONING_DISPLAY.get(value, value)
            )
            + ")"
        )
    else:
        ordered["plot_label"] = (
            ordered["model_slug"].astype(str)
            + " ("
            + ordered["reasoning"].astype(str).map(
                lambda value: REASONING_DISPLAY.get(value, value)
            )
            + ")"
        )
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.55 * len(ordered) + 1.2)))

    bars = ax.barh(
        ordered["plot_label"],
        ordered["dataset_match_rate"],
        color=[MODEL_COLORS.get(model, "#7f7f7f") for model in ordered["model_slug"]],
        edgecolor="black",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Dataset match rate")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ordered["dataset_match_rate"], strict=False):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )
    fig.tight_layout()
    output_path = output_dir / "dataset_match_rate_by_model_reasoning.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_dataset_metric(
    df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    legend_below: bool = False,
) -> Path:
    datasets = sorted(df["expected_dataset_slug"].dropna().unique().tolist())
    models = sorted(df["model_slug"].dropna().unique().tolist())
    dataset_labels = {}
    if "expected_dataset_display" in df.columns:
        dataset_labels = (
            df.drop_duplicates("expected_dataset_slug")
            .set_index("expected_dataset_slug")["expected_dataset_display"]
            .to_dict()
        )
    model_labels = {}
    if "model_display" in df.columns:
        model_labels = (
            df.drop_duplicates("model_slug")
            .set_index("model_slug")["model_display"]
            .to_dict()
        )

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(datasets)), 6))
    base_positions = list(range(len(datasets)))
    bar_width = 0.16
    combo_order = [(model, reasoning) for model in models for reasoning in ("none", "high")]
    plotted_values: list[float] = []
    pending_labels: list[tuple[plt.Rectangle, float]] = []

    for combo_index, (model, reasoning) in enumerate(combo_order):
        subset = df.loc[
            (df["model_slug"] == model) & (df["reasoning"] == reasoning)
        ].copy()
        values = []
        for dataset in datasets:
            match = subset.loc[subset["expected_dataset_slug"] == dataset, value_column]
            values.append(float(match.iloc[0]) if not match.empty and pd.notna(match.iloc[0]) else 0.0)
        plotted_values.extend(value for value in values if value > 0)
        offset = (combo_index - (len(combo_order) - 1) / 2) * bar_width
        positions = [base + offset for base in base_positions]
        label = f"{model_labels.get(model, model)} ({REASONING_DISPLAY.get(reasoning, reasoning)})"
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=label,
            color=MODEL_COLORS.get(model, "#7f7f7f"),
            alpha=0.45 if reasoning == "none" else 0.9,
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values, strict=False):
            if value <= 0:
                continue
            pending_labels.append((bar, value))

    if plotted_values:
        max_value = max(plotted_values)
        upper_margin = max(0.06, max_value * 0.12)
        ax.set_ylim(0, max_value + upper_margin)
    else:
        ax.set_ylim(0, 1.0)

    ylim_top = ax.get_ylim()[1]
    label_offset = max(0.008, ylim_top * 0.012)
    label_ceiling = ylim_top - max(0.01, ylim_top * 0.03)
    for bar, value in pending_labels:
        label_y = min(value + label_offset, label_ceiling)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(
        [dataset_labels.get(dataset, dataset) for dataset in datasets],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if legend_below:
        legend_columns = max(1, len(combo_order))
        ax.legend(
            fontsize=8,
            ncol=legend_columns,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.11, 1, 1))
    else:
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    pivot_csv = Path(args.pivot_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = _load_runs(pivot_csv)
    runs_df["model_reasoning"] = (
        runs_df["model_slug"].astype(str) + " (" + runs_df["reasoning"].astype(str) + ")"
    )

    dataset_overall, dataset_per_dataset = _dataset_selection_summary(runs_df)
    valid_overall, valid_per_dataset = _valid_performance_summary(runs_df)
    effective_overall, effective_per_dataset = _effective_score_summary(runs_df)

    _write_csv(dataset_overall, output_dir / "dataset_selection_overall.csv")
    _write_csv(dataset_per_dataset, output_dir / "dataset_selection_by_dataset.csv")
    _write_csv(valid_overall, output_dir / "valid_performance_overall.csv")
    _write_csv(valid_per_dataset, output_dir / "valid_performance_by_dataset.csv")
    _write_csv(effective_overall, output_dir / "effective_score_overall.csv")
    _write_csv(effective_per_dataset, output_dir / "effective_score_by_dataset.csv")

    dataset_match_plot = _plot_dataset_match_rate(dataset_overall, output_dir)
    valid_plot = _plot_dataset_metric(
        valid_per_dataset,
        value_column="mean_f1_macro",
        ylabel="F1-Macro",
        filename="valid_performance_by_dataset.pdf",
        output_dir=output_dir,
        legend_below=True,
    )
    effective_plot = _plot_dataset_metric(
        effective_per_dataset,
        value_column="mean_effective_f1_macro",
        ylabel="Effective F1-Macro",
        filename="effective_score_by_dataset.pdf",
        output_dir=output_dir,
        legend_below=True,
    )

    print(f"Saved dataset selection tables to {output_dir}")
    print(f"Saved valid-run performance tables to {output_dir}")
    print(f"Saved effective-score tables to {output_dir}")
    print(f"Saved plot: {dataset_match_plot}")
    print(f"Saved plot: {valid_plot}")
    print(f"Saved plot: {effective_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
