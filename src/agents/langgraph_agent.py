from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv

from src.pipeline.data import dataset_profile, discover_datasets
from src.pipeline.evaluation import evaluate_classifier
from src.pipeline.modeling import train_classifier
from src.pipeline.preprocessing import preprocess_dataset
from src.pipeline.report import generate_report
from src.pipeline.representation import build_representation
from src.pipeline.web import fetch_url_content, search_arxiv, search_ddg
from src.types import PipelineResult

LANGGRAPH_SYSTEM_PROMPT = """You are an autonomous text classification pipeline agent.

Your objective is to solve the requested text classification task end-to-end by selecting the most appropriate available dataset, identifying the text and label columns, choosing a suitable feature representation, selecting a classifier, running the pipeline, and then calling the `final_result` tool with a valid PipelineResult payload.

Use the available tools to inspect datasets, execute the pipeline, and revise your next action based on tool outputs. You must use exactly one tool call at a time. Do not emit XML-like tool tags, fake tool calls, plain JSON, or prose instead of a real tool call. Use DuckDuckGo search only when general web search is genuinely useful for selecting or justifying a method.

Before calling `final_result`, make sure preprocessing, representation building, training, evaluation, and report generation have already succeeded in the current run directory.

Each round corresponds to exactly one pipeline configuration/hypothesis. Inside a single round, do not restart from preprocessing with a new configuration after you have already begun executing a pipeline. If you have already produced metrics for the current round, finish that round by generating the report and calling `final_result`. Strategy changes belong to the next outer round, not to the current one.

More concretely: once you choose the dataset/text column/label column for a round, keep them fixed for the rest of that round. Once you build a representation in a round, keep that representation configuration fixed. Once you train a model in a round, keep that model configuration fixed. If you want to try a different dataset, columns, representation, or model, that must happen in the next outer round, not inside the current one.

If a tool returns an error:
- inspect the exact error text carefully;
- keep the same overall round strategy unless the error proves it is impossible;
- fix the next tool call based on the observed failure;
- continue within the same round until the pipeline succeeds or the round is exhausted.

When revision context is provided, use it to improve the strategy over the previous round. Inside a single round, focus on recovering from operational errors rather than inventing a completely new plan after every failure.
"""

REQUIRED_PIPELINE_ARTIFACTS = (
    "dataset_info.json",
    "representation_metadata.json",
    "model.joblib",
    "metrics.json",
    "report.md",
)


class LangGraphAgentState(TypedDict):
    task: str
    run_dir: str
    messages: list[Any]
    selected_plan: dict[str, Any] | None
    pending_tool_call: dict[str, Any] | None
    last_tool_result: dict[str, Any] | None
    last_tool_error: str | None
    tool_error_count: int
    completed_steps: list[str]
    final_result: dict[str, Any] | None
    status: Literal["running", "success", "failed"]
    failure_reason: str | None


@dataclasses.dataclass
class LangGraphExecutionRecord:
    result: PipelineResult
    events: list[dict[str, Any]]


def _print_event(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _extract_reasoning_text(ai_message: Any) -> str:
    content_blocks = getattr(ai_message, "content_blocks", None) or []
    parts: list[str] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "reasoning":
            continue
        reasoning = block.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append(reasoning.strip())
    return "\n".join(parts)


def _serialize_tool_output(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), ensure_ascii=False)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pipeline_tool_mapping() -> dict[str, Any]:
    return {
        "discover_datasets": discover_datasets,
        "dataset_profile": dataset_profile,
        "search_ddg": search_ddg,
        "fetch_url_content": fetch_url_content,
        "search_arxiv": search_arxiv,
        "preprocess_dataset": preprocess_dataset,
        "build_representation": build_representation,
        "train_classifier": train_classifier,
        "evaluate_classifier": evaluate_classifier,
        "generate_report": generate_report,
    }


def _normalize_dataset_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _extract_round_plan_fragment(
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "preprocess_dataset":
        return {
            "dataset_path": _normalize_dataset_path(tool_args["dataset_path"]),
            "text_column": tool_args["text_column"],
            "label_column": tool_args["label_column"],
        }
    if tool_name == "build_representation":
        return {"representation_config": tool_args["config"]}
    if tool_name == "train_classifier":
        return {"model_config": tool_args["config"]}
    return {}


def _round_plan_conflict(
    selected_plan: dict[str, Any] | None,
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    if selected_plan is None:
        return None

    incoming = _extract_round_plan_fragment(tool_name, tool_args)
    if not incoming:
        return None

    for key, value in incoming.items():
        if key not in selected_plan:
            continue
        if _canonical_json(selected_plan[key]) != _canonical_json(value):
            return (
                "This round is locked to a single pipeline configuration. "
                f"You tried to change `{key}` inside the same round. "
                "Keep the same dataset/columns/representation/model for this round "
                "and finish it, or let the outer loop start the next round."
            )
    return None


def _merge_round_plan(
    selected_plan: dict[str, Any] | None,
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any] | None:
    fragment = _extract_round_plan_fragment(tool_name, tool_args)
    if not fragment:
        return selected_plan
    merged = {} if selected_plan is None else dict(selected_plan)
    merged.update(fragment)
    return merged


def _normalize_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
    run_dir: str,
) -> dict[str, Any]:
    normalized = dict(tool_args)
    tools_with_fixed_run_dir = {
        "preprocess_dataset",
        "build_representation",
        "train_classifier",
        "evaluate_classifier",
        "generate_report",
    }
    if tool_name in tools_with_fixed_run_dir:
        normalized["run_dir"] = run_dir
    return normalized


def _missing_required_artifacts(run_dir: str) -> list[str]:
    run_path = Path(run_dir).expanduser().resolve()
    return [name for name in REQUIRED_PIPELINE_ARTIFACTS if not (run_path / name).exists()]


def _build_chat_model(model_name: str, thinking_effort: str | None = None):
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file or environment."
        )

    try:
        from langchain_openrouter import ChatOpenRouter
    except Exception as exc:
        raise ImportError(
            "LangGraph OpenRouter support requires `langgraph` and `langchain-openrouter`."
        ) from exc

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0,
        "max_retries": 2,
    }
    if thinking_effort is not None:
        kwargs["reasoning"] = {"effort": thinking_effort, "summary": "auto"}
    return ChatOpenRouter(**kwargs)


def _build_tool_schemas():
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:
        raise ImportError(
            "LangGraph OpenRouter support requires `langgraph` and `langchain-openrouter`."
        ) from exc

    def final_result(
        dataset_path: str,
        text_column: str,
        label_column: str,
        representation: Literal["tfidf", "bow", "sentence_transformer"],
        model: Literal[
            "logistic_regression",
            "linear_svm",
            "multinomial_nb",
            "decision_tree",
            "random_forest",
            "knn",
        ],
        metrics_requested: list[
            Literal["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        ],
        assumptions: list[str],
        justification: str,
    ) -> str:
        """Call this only after the pipeline has completed and all required artifacts exist."""
        return "final_result"

    tools = [
        StructuredTool.from_function(discover_datasets),
        StructuredTool.from_function(dataset_profile),
        StructuredTool.from_function(search_ddg),
        StructuredTool.from_function(fetch_url_content),
        StructuredTool.from_function(search_arxiv),
        StructuredTool.from_function(preprocess_dataset),
        StructuredTool.from_function(build_representation),
        StructuredTool.from_function(train_classifier),
        StructuredTool.from_function(evaluate_classifier),
        StructuredTool.from_function(generate_report),
        StructuredTool.from_function(final_result, name="final_result"),
    ]
    return tools


def run_langgraph_round(
    task: str,
    prompt: str,
    round_dir: str,
    model_name: str,
    verbose: bool = False,
    thinking_effort: str | None = None,
    max_tool_errors: int = 3,
) -> LangGraphExecutionRecord:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import RetryPolicy
    except Exception as exc:
        raise ImportError(
            "LangGraph runtime requires `langgraph` and `langchain-openrouter`."
        ) from exc

    model = _build_chat_model(model_name=model_name, thinking_effort=thinking_effort)
    tool_schemas = _build_tool_schemas()
    tool_mapping = _pipeline_tool_mapping()
    model_with_tools = model.bind_tools(tool_schemas)
    event_log: list[dict[str, Any]] = []

    def log_event(event_type: str, **payload: Any) -> None:
        event_log.append({"type": event_type, **payload})

    def agent_node(state: LangGraphAgentState) -> LangGraphAgentState:
        messages = [SystemMessage(content=LANGGRAPH_SYSTEM_PROMPT), *state["messages"]]
        ai_message = model_with_tools.invoke(messages)
        updated_messages = [*state["messages"], ai_message]
        reasoning_text = _extract_reasoning_text(ai_message)
        if reasoning_text:
            _print_event("[langgraph/thinking]", verbose=verbose)
            _print_event(reasoning_text, verbose=verbose)
            log_event("thinking", content=reasoning_text)
        if ai_message.content:
            _print_event(f"[langgraph/agent] {ai_message.content}", verbose=verbose)
            log_event("agent", content=ai_message.content)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 1:
            error_count = state["tool_error_count"] + 1
            correction = HumanMessage(
                content=(
                    "You called multiple tools at once. Call exactly one tool at a time "
                    "and continue the same round."
                )
            )
            return {
                **state,
                "messages": [*updated_messages, correction],
                "last_tool_error": "Multiple tool calls emitted in a single agent turn.",
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
                "pending_tool_call": None,
            }

        if not tool_calls:
            error_count = state["tool_error_count"] + 1
            correction = HumanMessage(
                content=(
                    "Your previous response did not include a valid tool call. "
                    "Call one real tool next, or call `final_result` only if the "
                    "pipeline has fully completed."
                )
            )
            return {
                **state,
                "messages": [*updated_messages, correction],
                "last_tool_error": "Model response did not include a valid tool call.",
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
                "pending_tool_call": None,
            }

        tool_call = tool_calls[0]
        normalized_preview_args = _normalize_tool_args(
            tool_name=tool_call["name"],
            tool_args=tool_call.get("args", {}),
            run_dir=state["run_dir"],
        )
        _print_event(
            f"[langgraph/tool-call] {tool_call['name']}({json.dumps(normalized_preview_args, ensure_ascii=False)})",
            verbose=verbose,
        )
        log_event(
            "tool_call",
            tool_name=tool_call["name"],
            args=normalized_preview_args,
        )
        return {
            **state,
            "messages": updated_messages,
            "pending_tool_call": tool_call,
            "last_tool_error": None,
        }

    def execute_tool_node(state: LangGraphAgentState) -> LangGraphAgentState:
        pending = state["pending_tool_call"]
        if pending is None:
            return state

        tool_name = pending["name"]
        tool_args = _normalize_tool_args(
            tool_name=tool_name,
            tool_args=pending.get("args", {}),
            run_dir=state["run_dir"],
        )
        plan_error = _round_plan_conflict(
            selected_plan=state["selected_plan"],
            tool_name=tool_name,
            tool_args=tool_args,
        )
        if plan_error is not None:
            error_count = state["tool_error_count"] + 1
            tool_message = ToolMessage(
                content=f"Tool error: {plan_error}",
                tool_call_id=pending["id"],
            )
            _print_event(f"[langgraph/tool-error] {plan_error}", verbose=verbose)
            log_event("tool_error", tool_name=tool_name, error=plan_error)
            return {
                **state,
                "messages": [*state["messages"], tool_message],
                "pending_tool_call": None,
                "last_tool_error": plan_error,
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
            }
        if tool_name == "final_result":
            return {
                **state,
                "pending_tool_call": None,
                "final_result": tool_args,
                "selected_plan": tool_args,
            }

        tool_func = tool_mapping.get(tool_name)
        if tool_func is None:
            error_count = state["tool_error_count"] + 1
            tool_message = ToolMessage(
                content=f"Tool error: Unknown tool `{tool_name}`.",
                tool_call_id=pending["id"],
            )
            return {
                **state,
                "messages": [*state["messages"], tool_message],
                "pending_tool_call": None,
                "last_tool_error": f"Unknown tool `{tool_name}`.",
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
            }

        try:
            result = tool_func(**tool_args)
        except Exception as exc:
            error_count = state["tool_error_count"] + 1
            tool_message = ToolMessage(
                content=f"Tool error: {exc}",
                tool_call_id=pending["id"],
            )
            _print_event(f"[langgraph/tool-error] {exc}", verbose=verbose)
            log_event("tool_error", tool_name=tool_name, error=str(exc))
            return {
                **state,
                "messages": [*state["messages"], tool_message],
                "pending_tool_call": None,
                "last_tool_error": str(exc),
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
            }

        result_text = _serialize_tool_output(result)
        _print_event(
            f"[langgraph/tool-result] {result_text[:120].replace(chr(10), ' ')}",
            verbose=verbose,
        )
        log_event("tool_result", tool_name=tool_name, content=result_text)
        tool_message = ToolMessage(content=result_text, tool_call_id=pending["id"])
        return {
            **state,
            "messages": [*state["messages"], tool_message],
            "pending_tool_call": None,
            "last_tool_result": {"tool_name": tool_name, "content": result_text},
            "completed_steps": [*state["completed_steps"], tool_name],
            "selected_plan": _merge_round_plan(
                selected_plan=state["selected_plan"],
                tool_name=tool_name,
                tool_args=tool_args,
            ),
        }

    def finalize_node(state: LangGraphAgentState) -> LangGraphAgentState:
        if state["final_result"] is None:
            return {
                **state,
                "status": "failed",
                "failure_reason": "final_result was requested without a final result payload.",
            }

        try:
            final_result = PipelineResult.model_validate(state["final_result"])
        except Exception as exc:
            error_count = state["tool_error_count"] + 1
            correction = HumanMessage(
                content=(
                    "Your previous `final_result` payload was invalid. Correct the fields "
                    "and call `final_result` again only after the pipeline is complete. "
                    f"Validation error: {exc}"
                )
            )
            return {
                **state,
                "messages": [*state["messages"], correction],
                "final_result": None,
                "last_tool_error": str(exc),
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
            }

        missing = _missing_required_artifacts(state["run_dir"])
        if missing:
            error_count = state["tool_error_count"] + 1
            correction = HumanMessage(
                content=(
                    "Do not call `final_result` yet. The pipeline is incomplete. "
                    f"Missing required artifacts in `{state['run_dir']}`: {', '.join(missing)}. "
                    "Continue by calling the remaining tools until all required artifacts exist."
                )
            )
            return {
                **state,
                "messages": [*state["messages"], correction],
                "final_result": None,
                "last_tool_error": f"Missing required artifacts: {', '.join(missing)}",
                "tool_error_count": error_count,
                "status": "failed" if error_count >= max_tool_errors else "running",
                "failure_reason": (
                    "Exceeded tool error budget."
                    if error_count >= max_tool_errors
                    else None
                ),
            }

        return {
            **state,
            "final_result": final_result.model_dump(mode="json"),
            "status": "success",
            "failure_reason": None,
        }

    def route_after_agent(state: LangGraphAgentState) -> str:
        if state["status"] == "failed":
            return "end"
        if state["final_result"] is not None:
            return "finalize"
        if state["pending_tool_call"] is not None:
            return "execute_tool"
        return "agent"

    def route_after_execute_tool(state: LangGraphAgentState) -> str:
        if state["status"] == "failed":
            return "end"
        if state["final_result"] is not None:
            return "finalize"
        return "agent"

    def route_after_finalize(state: LangGraphAgentState) -> str:
        if state["status"] == "success":
            return "end"
        if state["status"] == "failed":
            return "end"
        return "agent"

    builder = StateGraph(LangGraphAgentState)
    builder.add_node("agent", agent_node, retry_policy=RetryPolicy(max_attempts=3))
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"agent": "agent", "execute_tool": "execute_tool", "finalize": "finalize", "end": END},
    )
    builder.add_conditional_edges(
        "execute_tool",
        route_after_execute_tool,
        {"agent": "agent", "finalize": "finalize", "end": END},
    )
    builder.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {"agent": "agent", "end": END},
    )

    graph = builder.compile()
    initial_state: LangGraphAgentState = {
        "task": task,
        "run_dir": round_dir,
        "messages": [HumanMessage(content=prompt)],
        "selected_plan": None,
        "pending_tool_call": None,
        "last_tool_result": None,
        "last_tool_error": None,
        "tool_error_count": 0,
        "completed_steps": [],
        "final_result": None,
        "status": "running",
        "failure_reason": None,
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

    if final_state["status"] != "success" or final_state["final_result"] is None:
        raise RuntimeError(final_state.get("failure_reason") or final_state.get("last_tool_error") or "LangGraph round failed.")

    return LangGraphExecutionRecord(
        result=PipelineResult.model_validate(final_state["final_result"]),
        events=event_log,
    )
