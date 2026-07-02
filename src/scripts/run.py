from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.langgraph_runner import optimize_pipeline_langgraph
from src.types import OptimizationHistory, RoundSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fully agentic text classification pipeline."
    )
    parser.add_argument("--task", required=True, help="Task description for the agent.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Base directory where the run folder will be created.",
    )
    parser.add_argument(
        "--llm-provider",
        default="openrouter",
        choices=["openrouter"],
        help="LLM provider backend.",
    )
    parser.add_argument("--llm-model", required=True, help="Model name.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name used only for output directory naming.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Maximum number of optimization rounds.",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=10,
        help="Wall-clock budget for all rounds.",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="Reasoning level for the agent (none, low, medium, high). Omit to leave it unset.",
    )
    parser.add_argument(
        "--max-tool-errors-per-round",
        type=int,
        default=3,
        help="Maximum number of tool-related errors allowed within a single round.",
    )
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _slugify_model_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _model_suffix(value: str) -> str:
    return value.split("/", 1)[-1]


def _dataset_slug(dataset_path: str | None) -> str:
    if not dataset_path:
        return "unknown_dataset"
    return _slugify_model_name(Path(dataset_path).stem)


def _reasoning_slug(thinking_effort: str | None) -> str:
    return thinking_effort if thinking_effort is not None else "no_reasoning"


def _build_output_dir(base_root: Path, llm_model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pending_{_slugify_model_name(_model_suffix(llm_model))}_{timestamp}"
    return base_root / run_name


def _build_final_output_dir(
    base_root: Path,
    dataset_path: str | None,
    llm_model: str,
    thinking_effort: str | None,
) -> Path:
    run_name = "_".join(
        [
            _dataset_slug(dataset_path),
            _slugify_model_name(_model_suffix(llm_model)),
            _reasoning_slug(thinking_effort),
        ]
    )
    return base_root / run_name


def _selected_round(history: OptimizationHistory) -> RoundSummary | None:
    if history.selected_round_index is None:
        return None
    for round_summary in history.rounds:
        if round_summary.round_index == history.selected_round_index:
            return round_summary
    return None


def _build_final_report(history: OptimizationHistory) -> str:
    lines = [
        "# Optimization Summary",
        "",
        f"- Task: `{history.task}`",
        f"- Finished reason: `{history.finished_reason}`",
        f"- Max rounds: `{history.max_rounds}`",
        f"- Max minutes: `{history.max_minutes}`",
        "",
        "## Rounds",
    ]
    for round_summary in history.rounds:
        lines.append(
            f"- Round {round_summary.round_index:02d}: status=`{round_summary.status}` dir=`{round_summary.round_dir}`"
        )
        if round_summary.result is not None:
            lines.append(
                f"  choices: dataset=`{round_summary.result.dataset_path}` text=`{round_summary.result.text_column}` label=`{round_summary.result.label_column}` representation=`{round_summary.result.representation}` model=`{round_summary.result.model}`"
            )
        if round_summary.metrics is not None:
            lines.append(
                "  metrics: "
                f"accuracy={round_summary.metrics.accuracy} "
                f"f1_macro={round_summary.metrics.f1_macro} "
                f"precision_macro={round_summary.metrics.precision_macro} "
                f"recall_macro={round_summary.metrics.recall_macro}"
            )
        if round_summary.error:
            lines.append(f"  error: {round_summary.error}")

    selected = _selected_round(history)
    lines.extend(["", "## Final Selection"])
    if selected is None or selected.result is None:
        lines.append("- No successful round was selected.")
    else:
        lines.append(f"- Selected round: `{selected.round_index:02d}`")
        lines.append(
            f"- Dataset: `{selected.result.dataset_path}` | Text: `{selected.result.text_column}` | Label: `{selected.result.label_column}`"
        )
        lines.append(
            f"- Representation: `{selected.result.representation}` | Model: `{selected.result.model}`"
        )
        if selected.metrics is not None:
            lines.append(
                f"- Best metrics: accuracy={selected.metrics.accuracy} f1_macro={selected.metrics.f1_macro}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_dotenv()
    args = parse_args()
    base_output_root = Path(args.output_root).expanduser().resolve()
    base_output_root.mkdir(parents=True, exist_ok=True)
    if args.dataset_name is not None:
        requested_output_root = _build_final_output_dir(
            base_root=base_output_root,
            dataset_path=args.dataset_name,
            llm_model=args.llm_model,
            thinking_effort=args.thinking_effort,
        )
        if requested_output_root.exists():
            print(
                f"Skipping existing result directory: {requested_output_root}",
                file=sys.stderr,
            )
            return 0
    output_root = _build_output_dir(base_root=base_output_root, llm_model=args.llm_model)
    output_root.mkdir(parents=True, exist_ok=False)

    history = optimize_pipeline_langgraph(
        task=args.task,
        model_name=args.llm_model,
        output_dir=output_root,
        llm_provider=args.llm_provider,
        max_rounds=args.max_rounds,
        max_minutes=args.max_minutes,
        verbose=True,
        thinking_effort=args.thinking_effort,
        max_tool_errors_per_round=args.max_tool_errors_per_round,
    )

    if history is None:
        print("Pipeline failed — no result produced.", file=sys.stderr)
        return 1

    selected = _selected_round(history)
    final_output_root = _build_final_output_dir(
        base_root=base_output_root,
        dataset_path=(
            args.dataset_name
            if args.dataset_name is not None
            else None if selected is None or selected.result is None else selected.result.dataset_path
        ),
        llm_model=args.llm_model,
        thinking_effort=args.thinking_effort,
    )
    if final_output_root.exists():
        print(
            f"Refusing to overwrite existing result directory: {final_output_root}",
            file=sys.stderr,
        )
        return 1
    output_root.rename(final_output_root)
    output_root = final_output_root
    if selected is not None:
        save_json(output_root / "result.json", selected.model_dump(mode="json"))
    final_report = _build_final_report(history)
    (output_root / "final_report.md").write_text(final_report, encoding="utf-8")

    if selected is None:
        print(
            f"Run finished without a successful round. See {output_root / 'optimization_history.json'}",
            file=sys.stderr,
        )
        return 1

    print(f"Done. Results saved to {output_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
