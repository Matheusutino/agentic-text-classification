from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONG_CSV = PROJECT_ROOT / "results" / "analysis" / "round_metrics_long.csv"
DEFAULT_PIVOT_CSV = PROJECT_ROOT / "results" / "analysis" / "round_metrics_pivot.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export runs/rounds where the agent used a dataset different from the expected dataset."
    )
    parser.add_argument(
        "--long-csv",
        default=str(DEFAULT_LONG_CSV),
        help="Path to round_metrics_long.csv.",
    )
    parser.add_argument(
        "--pivot-csv",
        default=str(DEFAULT_PIVOT_CSV),
        help="Path to round_metrics_pivot.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where mismatch CSVs will be written.",
    )
    return parser.parse_args()


def _write_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    args = parse_args()
    long_csv = Path(args.long_csv).expanduser().resolve()
    pivot_csv = Path(args.pivot_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    long_df = pd.read_csv(long_csv)
    pivot_df = pd.read_csv(pivot_csv)

    wrong_rounds = long_df.loc[
        long_df["dataset_matches_expected"].astype(str).str.lower() == "false"
    ].copy()
    wrong_runs = pivot_df.loc[
        pivot_df["all_rounds_match_expected"].astype(str).str.lower() == "false"
    ].copy()

    wrong_rounds_path = output_dir / "dataset_mismatches_rounds.csv"
    wrong_runs_path = output_dir / "dataset_mismatches_runs.csv"

    _write_dataframe_csv(wrong_rounds, wrong_rounds_path)
    _write_dataframe_csv(wrong_runs, wrong_runs_path)

    print(f"Mismatched runs: {len(wrong_runs)}")
    print(f"Mismatched rounds: {len(wrong_rounds)}")
    print(f"Saved run-level mismatches to: {wrong_runs_path}")
    print(f"Saved round-level mismatches to: {wrong_rounds_path}")

    if not wrong_runs.empty:
        print()
        print("Run-level mismatches:")
        for _, row in wrong_runs.iterrows():
            print(
                f"- {row['run_dir_name']}: expected={row['expected_dataset_slug']} "
                f"actual={row['actual_dataset_slugs']} model={row['model_slug']} reasoning={row['reasoning']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
