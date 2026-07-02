#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_ROOT="${PROJECT_ROOT}/results"
LLM_PROVIDER="openrouter"
MAX_ROUNDS=5
MAX_MINUTES=30
MAX_TOOL_ERRORS_PER_ROUND=3

MODELS=(
  #"deepseek/deepseek-v4-flash"
  "openrouter/owl-alpha"
)

DATASETS=(
  "Dmoz-Computers.csv|Classify web pages in the computers domain into subcategories."
  "Dmoz-Health.csv|Classify web pages in the health domain into subcategories."
  "Dmoz-Science.csv|Classify web pages in the science domain into subcategories."
  "Dmoz-Sports.csv|Classify web pages in the sports domain into subcategories."
  "NSF.csv|Classify NSF project descriptions by research area."
  "SyskillWebert.csv|Classify web pages by a user's preference rating."
  "classic4.csv|Classify documents into one of the four Classic benchmark collections: CACM, CISI, CRAN, or MED."
  "re8.csv|Classify Reuters news articles into one of the available categories."
  "review_polarity.csv|Classify review texts by sentiment polarity."
  "sms_spam.csv|Classify SMS messages as spam or ham."
)

run_experiment() {
  local dataset_name="$1"
  local task="$2"
  local model="$3"
  local reasoning="$4"
  local dataset_slug model_slug target_dir

  dataset_slug="$(printf '%s' "${dataset_name%.csv}" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]/_/g; s/__/_/g; s/^_//; s/_$//')"
  model_slug="$(printf '%s' "${model#*/}" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]/_/g; s/__/_/g; s/^_//; s/_$//')"
  target_dir="${RESULTS_ROOT}/${dataset_slug}_${model_slug}_${reasoning}"

  if [[ -d "$target_dir" ]]; then
    echo
    echo "============================================================"
    echo "Skipping existing result:"
    echo "dataset:   $dataset_name"
    echo "model:     $model"
    echo "reasoning: $reasoning"
    echo "path:      $target_dir"
    echo "============================================================"
    echo
    return 0
  fi

  local -a cmd=(
    python
    "${PROJECT_ROOT}/src/scripts/run.py"
    --task "$task"
    --dataset-name "$dataset_name"
    --output-root "$RESULTS_ROOT"
    --llm-provider "$LLM_PROVIDER"
    --llm-model "$model"
    --thinking-effort "$reasoning"
    --max-rounds "$MAX_ROUNDS"
    --max-minutes "$MAX_MINUTES"
    --max-tool-errors-per-round "$MAX_TOOL_ERRORS_PER_ROUND"
  )

  echo
  echo "============================================================"
  echo "dataset:   $dataset_name"
  echo "task:      $task"
  echo "model:     $model"
  echo "reasoning: $reasoning"
  echo "============================================================"
  echo

  "${cmd[@]}"
}

main() {
  cd "$PROJECT_ROOT" || exit 1

  local dataset_name task model
  for entry in "${DATASETS[@]}"; do
    dataset_name="${entry%%|*}"
    task="${entry#*|}"

    for model in "${MODELS[@]}"; do
      run_experiment "$dataset_name" "$task" "$model" "none"
      run_experiment "$dataset_name" "$task" "$model" "high"
    done
  done
}

main "$@"
