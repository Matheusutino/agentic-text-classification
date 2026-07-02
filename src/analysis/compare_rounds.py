from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REASONING_LEVELS = ("none", "low", "medium", "high")
KNOWN_MODEL_SLUGS = (
    "qwen3_5_9b",
    "gpt_oss_20b",
    "deepseek_v4_flash",
    "glm_5_1",
    "owl_alpha",
)
TIMESTAMP_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<reasoning>none|low|medium|high)_(?P<date>\d{8})_(?P<time>\d{6})$"
)
METRIC_FIELDS = ("accuracy", "f1_macro", "precision_macro", "recall_macro")
MODEL_DISPLAY_NAMES = {
    "qwen3_5_9b": "Qwen3.5-9B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "glm_5_1": "GLM-5.1",
    "owl_alpha": "Owl Alpha",
    "unknown_model": "Unknown Model",
}
DATASET_DISPLAY_NAMES = {
    "cstr": "CSTR",
    "syskillwebert": "SyskillWebert",
    "review_polarity": "Review Polarity",
    "sms_spam": "SMS Spam",
    "dmoz_science": "DMOZ Science",
    "dmoz_health": "DMOZ Health",
    "classic4": "Classic4",
    "re8": "Re8",
    "dmoz_computers": "DMOZ Computers",
    "nsf": "NSF",
    "dmoz_sports": "DMOZ Sports",
    "unknown_dataset": "Unknown Dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-round metrics from experiment results."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory containing experiment result folders.",
    )
    parser.add_argument(
        "--metric",
        default="f1_macro",
        choices=METRIC_FIELDS,
        help="Metric used in the pivot comparison output.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset slug filter, e.g. dmoz_computers or review_polarity.",
    )
    parser.add_argument(
        "--long-output",
        default=None,
        help="Optional CSV path for the long per-round table.",
    )
    parser.add_argument(
        "--pivot-output",
        default=None,
        help="Optional CSV path for the pivot comparison table.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _dataset_slug_from_history(payload: dict[str, Any]) -> str:
    rounds = payload.get("rounds", [])
    for round_payload in rounds:
        result_payload = round_payload.get("result")
        if not result_payload:
            continue
        dataset_path = result_payload.get("dataset_path")
        if dataset_path:
            return _slugify(Path(dataset_path).stem)
    return "unknown_dataset"


def _parse_run_name(run_dir_name: str) -> dict[str, str]:
    match = TIMESTAMP_RE.match(run_dir_name)
    if match:
        base_name = match.group("prefix")
        timestamp = f"{match.group('date')}_{match.group('time')}"
    else:
        base_name = run_dir_name
        timestamp = ""

    for reasoning in REASONING_LEVELS:
        for model_slug in KNOWN_MODEL_SLUGS:
            suffix = f"_{model_slug}_{reasoning}"
            if base_name.endswith(suffix):
                expected_dataset_slug = base_name[: -len(suffix)]
                return {
                    "expected_dataset_slug": expected_dataset_slug or "unknown_dataset",
                    "model_slug": model_slug,
                    "reasoning": reasoning,
                    "timestamp": timestamp,
                }

    return {
        "expected_dataset_slug": "unknown_dataset",
        "model_slug": "unknown_model",
        "reasoning": "unknown",
        "timestamp": timestamp,
    }


def _iter_history_files(results_root: Path) -> Iterable[Path]:
    return sorted(results_root.glob("*/optimization_history.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_error(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return " | ".join(part.strip() for part in text.splitlines() if part.strip())


def _model_display_name(model_slug: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_slug, model_slug)


def _dataset_display_name(dataset_slug: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset_slug, dataset_slug)


def _round_rows(history_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(history_path)
    actual_dataset_slug = _dataset_slug_from_history(payload)
    run_info = _parse_run_name(history_path.parent.name)
    expected_dataset_slug = run_info["expected_dataset_slug"]
    dataset_matches_expected = (
        expected_dataset_slug == "unknown_dataset"
        or expected_dataset_slug == actual_dataset_slug
    )
    selected_round_index = payload.get("selected_round_index")

    rows: list[dict[str, Any]] = []
    for round_payload in payload.get("rounds", []):
        metrics = round_payload.get("metrics") or {}
        result_payload = round_payload.get("result") or {}
        row = {
            "run_dir_name": history_path.parent.name,
            "dataset_slug": (
                expected_dataset_slug
                if expected_dataset_slug != "unknown_dataset"
                else actual_dataset_slug
            ),
            "expected_dataset_slug": expected_dataset_slug,
            "actual_dataset_slug": actual_dataset_slug,
            "dataset_matches_expected": dataset_matches_expected,
            "model_slug": run_info["model_slug"],
            "dataset_display": _dataset_display_name(
                expected_dataset_slug
                if expected_dataset_slug != "unknown_dataset"
                else actual_dataset_slug
            ),
            "expected_dataset_display": _dataset_display_name(expected_dataset_slug),
            "actual_dataset_display": _dataset_display_name(actual_dataset_slug),
            "model_display": _model_display_name(run_info["model_slug"]),
            "reasoning": run_info["reasoning"],
            "timestamp": run_info["timestamp"],
            "task": payload.get("task"),
            "finished_reason": payload.get("finished_reason"),
            "max_rounds": payload.get("max_rounds"),
            "max_minutes": payload.get("max_minutes"),
            "round_index": round_payload.get("round_index"),
            "status": round_payload.get("status"),
            "selected_round_index": selected_round_index,
            "is_selected_round": round_payload.get("round_index") == selected_round_index,
            "representation": result_payload.get("representation"),
            "classifier": result_payload.get("model"),
            "round_dir": round_payload.get("round_dir"),
            "error": _normalize_error(round_payload.get("error")),
        }
        for metric_name in METRIC_FIELDS:
            row[metric_name] = metrics.get(metric_name)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def _build_pivot(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    max_round = max((int(row["round_index"]) for row in rows), default=0)
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for row in rows:
        key = (
            str(row["dataset_slug"]),
            str(row["model_slug"]),
            str(row["reasoning"]),
            str(row["run_dir_name"]),
        )
        group = grouped.setdefault(
            key,
            {
                "dataset_slug": row["dataset_slug"],
                "expected_dataset_slug": row["expected_dataset_slug"],
                "dataset_display": row["dataset_display"],
                "expected_dataset_display": row["expected_dataset_display"],
                "model_slug": row["model_slug"],
                "model_display": row["model_display"],
                "reasoning": row["reasoning"],
                "run_dir_name": row["run_dir_name"],
                "finished_reason": row["finished_reason"],
                "selected_round_index": row["selected_round_index"],
                "selected_metric": "",
                "actual_dataset_slug_set": set(),
                "actual_dataset_display_set": set(),
                "all_rounds_match_expected": True,
            },
        )
        round_column = f"round_{int(row['round_index']):02d}"
        group[round_column] = row.get(metric_name)
        if row.get("is_selected_round"):
            group["selected_metric"] = row.get(metric_name)
        group["actual_dataset_slug_set"].add(row["actual_dataset_slug"])
        group["actual_dataset_display_set"].add(row["actual_dataset_display"])
        group["all_rounds_match_expected"] = (
            group["all_rounds_match_expected"] and bool(row["dataset_matches_expected"])
        )

    pivot_rows = sorted(
        grouped.values(),
        key=lambda row: (
            str(row["dataset_slug"]),
            str(row["model_slug"]),
            str(row["reasoning"]),
            str(row["run_dir_name"]),
        ),
    )

    for row in pivot_rows:
        for round_index in range(1, max_round + 1):
            column = f"round_{round_index:02d}"
            row.setdefault(column, "")
        row["actual_dataset_slugs"] = "|".join(sorted(row.pop("actual_dataset_slug_set")))
        row["actual_dataset_displays"] = "|".join(
            sorted(row.pop("actual_dataset_display_set"))
        )
    return pivot_rows


def _print_preview(rows: list[dict[str, Any]], pivot_rows: list[dict[str, Any]], metric_name: str) -> None:
    print("Long rows:", len(rows))
    print("Pivot rows:", len(pivot_rows))
    print()
    print(f"Per-run {metric_name} by round:")
    for row in pivot_rows:
        round_values = []
        round_columns = sorted(
            column for column in row.keys() if column.startswith("round_")
        )
        for column in round_columns:
            value = _format_metric(row[column])
            if value:
                round_values.append(f"{column}={value}")
        selected = _format_metric(row.get("selected_metric"))
        selected_suffix = f" selected={selected}" if selected else ""
        print(
            f"- {row['dataset_slug']} | {row['model_slug']} | {row['reasoning']} | "
            + " ".join(round_values)
            + selected_suffix
        )


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()

    all_rows: list[dict[str, Any]] = []
    for history_path in _iter_history_files(results_root):
        all_rows.extend(_round_rows(history_path))

    if args.dataset is not None:
        dataset_filter = _slugify(args.dataset)
        all_rows = [
            row for row in all_rows if row.get("dataset_slug") == dataset_filter
        ]

    if not all_rows:
        print("No matching optimization_history.json files were found.")
        return 1

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            str(row["dataset_slug"]),
            str(row["model_slug"]),
            str(row["reasoning"]),
            str(row["timestamp"]),
            int(row["round_index"]),
        ),
    )

    long_fieldnames = [
        "run_dir_name",
        "dataset_slug",
        "dataset_display",
        "expected_dataset_slug",
        "expected_dataset_display",
        "actual_dataset_slug",
        "actual_dataset_display",
        "dataset_matches_expected",
        "model_slug",
        "model_display",
        "reasoning",
        "timestamp",
        "task",
        "finished_reason",
        "max_rounds",
        "max_minutes",
        "round_index",
        "status",
        "selected_round_index",
        "is_selected_round",
        "representation",
        "classifier",
        *METRIC_FIELDS,
        "error",
        "round_dir",
    ]

    pivot_rows = _build_pivot(all_rows, args.metric)
    max_round = max((int(row["round_index"]) for row in all_rows), default=0)
    pivot_fieldnames = [
        "dataset_slug",
        "dataset_display",
        "expected_dataset_slug",
        "expected_dataset_display",
        "actual_dataset_slugs",
        "actual_dataset_displays",
        "all_rounds_match_expected",
        "model_slug",
        "model_display",
        "reasoning",
        "run_dir_name",
        "finished_reason",
        "selected_round_index",
        *[f"round_{round_index:02d}" for round_index in range(1, max_round + 1)],
        "selected_metric",
    ]

    long_output = (
        Path(args.long_output)
        if args.long_output is not None
        else PROJECT_ROOT / "results" / "analysis" / "round_metrics_long.csv"
    )
    pivot_output = (
        Path(args.pivot_output)
        if args.pivot_output is not None
        else PROJECT_ROOT / "results" / "analysis" / "round_metrics_pivot.csv"
    )

    _write_csv(long_output, all_rows, long_fieldnames)
    _write_csv(pivot_output, pivot_rows, pivot_fieldnames)
    _print_preview(all_rows, pivot_rows, args.metric)
    print()
    print(f"Saved long CSV to: {long_output}")
    print(f"Saved pivot CSV to: {pivot_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
