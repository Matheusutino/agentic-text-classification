# Agentic Text Classification Pipeline

This project evaluates LLMs as end-to-end agents for text classification.

Instead of only selecting hyperparameters, the agent is responsible for the whole pipeline:

- choosing the most appropriate dataset for a natural-language task
- identifying text and label columns
- selecting a text representation
- selecting and configuring a classifier
- training and evaluating the model
- generating a report
- improving its strategy across multiple rounds

The current orchestration runtime is based on `LangGraph` and uses `OpenRouter` as the LLM backend.

## Overview

The agent receives a task such as:

`Classify review texts by sentiment polarity.`

It then interacts with local tools to inspect datasets, run preprocessing, build features, train a model, evaluate metrics, and write results to disk.

Each execution is organized into **rounds**:

- one round = one pipeline hypothesis/configuration
- the agent may recover from operational tool errors inside the same round
- strategy changes should happen in the next round

This structure makes it possible to study not only final performance, but also:

- whether the LLM chooses the correct dataset
- whether reasoning helps
- whether the LLM improves from one round to the next

## Repository Structure

```text
datasets/                Benchmark CSV datasets
results/                 Experiment outputs and analysis artifacts
src/agents/              LangGraph agent, prompts, history, and runner
src/pipeline/            Local pipeline tools used by the agent
src/types/               Pydantic types and configs
src/scripts/             CLI entrypoints and batch runner
src/analysis/            Result aggregation and plotting scripts
```

## Datasets

The repository includes the following datasets:

- `CSTR.csv`
- `Dmoz-Computers.csv`
- `Dmoz-Health.csv`
- `Dmoz-Science.csv`
- `Dmoz-Sports.csv`
- `NSF.csv`
- `SyskillWebert.csv`
- `classic4.csv`
- `re8.csv`
- `review_polarity.csv`
- `sms_spam.csv`

The batch runner associates each dataset with a task prompt in English.

## Agent Tools

The agent uses local Python tools defined under `src/pipeline/`.

### Dataset inspection

- `discover_datasets`
  - lists available CSV datasets in the project
- `dataset_profile`
  - inspects a dataset and returns columns, basic statistics, label distribution, and examples

### Pipeline execution

- `preprocess_dataset`
  - loads the selected dataset, chooses text and label columns, and creates train/test splits
- `build_representation`
  - builds text features
- `train_classifier`
  - trains a classifier from the persisted representation
- `evaluate_classifier`
  - computes evaluation metrics
- `generate_report`
  - writes a report for the round

### Web and literature search

- `search_arxiv`
  - queries arXiv for relevant academic papers
- `search_ddg`
  - performs a public web search through DuckDuckGo
- `fetch_url_content`
  - fetches the content of a web page and returns extracted markdown

These search tools are optional. The agent can use them to justify modeling decisions or inspect external references.

## Supported Representations

The current text representations are:

- `tfidf`
- `bow`
- `sentence_transformer`

## Supported Classifiers

The current classifiers are:

- `logistic_regression`
- `linear_svm`
- `multinomial_nb`
- `decision_tree`
- `random_forest`
- `knn`

The LLM chooses both the model family and the model configuration exposed by the typed schema.

## Runtime

The main runtime is implemented in:

- [src/agents/langgraph_agent.py](src/agents/langgraph_agent.py)
- [src/agents/langgraph_runner.py](src/agents/langgraph_runner.py)

Important characteristics:

- `LangGraph` orchestrates the round loop
- `OpenRouter` is the only supported LLM provider in the current CLI
- reasoning effort can be configured with `--thinking-effort`
- tool errors are bounded per round with `--max-tool-errors-per-round`
- the agent receives revision context from previous rounds

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:

```env
OPENROUTER_API_KEY=your_api_key_here
```

`run.py` loads environment variables via `python-dotenv`.

## Running a Single Experiment

Example with reasoning enabled:

```bash
python src/scripts/run.py \
  --task "Classify review texts by sentiment polarity." \
  --dataset-name "review_polarity.csv" \
  --output-root results \
  --llm-provider openrouter \
  --llm-model "deepseek/deepseek-v4-flash" \
  --thinking-effort high \
  --max-rounds 5 \
  --max-minutes 30 \
  --max-tool-errors-per-round 3
```

Example with reasoning disabled:

```bash
python src/scripts/run.py \
  --task "Classify SMS messages as spam or ham." \
  --dataset-name "sms_spam.csv" \
  --output-root results \
  --llm-provider openrouter \
  --llm-model "openrouter/owl-alpha" \
  --thinking-effort none \
  --max-rounds 5 \
  --max-minutes 30 \
  --max-tool-errors-per-round 3
```

## Batch Experiments

The batch script is:

- [src/scripts/run.sh](src/scripts/run.sh)

It iterates over:

- a list of datasets/tasks
- a list of models
- `none` and `high` reasoning settings

Run it with:

```bash
bash src/scripts/run.sh
```

The script skips experiments whose final result directory already exists.

## Output Structure

Each completed experiment is saved under `results/` with a name like:

```text
review_polarity_deepseek_v4_flash_high
sms_spam_owl_alpha_none
```

The directory name is based on:

- expected dataset from the experiment definition
- model name suffix
- reasoning level

Each experiment directory contains:

```text
optimization_history.json
final_report.md
result.json
round_01/
round_02/
...
```

Each round directory may contain:

```text
agent_trace.json
dataset_info.json
splits.npz
representation_metadata.json
model.joblib
model_config.json
metrics.json
classification_report.txt
report.md
result.json
```

## What Is Saved Per Round

### `agent_trace.json`

This is the chronological trace of the agent execution for that round, including:

- reasoning text when available
- tool calls
- tool results
- tool errors

### `metrics.json`

Contains the evaluation metrics for the round, typically including:

- `accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

### `optimization_history.json`

Contains the full run history:

- round summaries
- selected best round
- final result
- finished reason

## Analysis Scripts

### 1. Aggregate rounds

- [src/analysis/compare_rounds.py](src/analysis/compare_rounds.py)

Generates:

- `results/analysis/round_metrics_long.csv`
- `results/analysis/round_metrics_pivot.csv`

Usage:

```bash
python src/analysis/compare_rounds.py --results-root results
```

### 2. Find dataset mismatches

- [src/analysis/find_dataset_mismatches.py](src/analysis/find_dataset_mismatches.py)

Generates:

- `results/analysis/dataset_mismatches_runs.csv`
- `results/analysis/dataset_mismatches_rounds.csv`

This is useful for identifying cases where the LLM solved the wrong dataset for the requested task.

Usage:

```bash
python src/analysis/find_dataset_mismatches.py
```

### 3. Build summary analysis

- [src/analysis/analyze_experiments.py](src/analysis/analyze_experiments.py)

Generates summary CSVs and PDF plots for:

- dataset selection accuracy
- valid-only final performance
- effective score with dataset mismatches penalized to zero

Usage:

```bash
python src/analysis/analyze_experiments.py
```

Main outputs:

- `results/analysis/dataset_match_rate_by_model_reasoning.pdf`
- `results/analysis/valid_performance_by_dataset.pdf`
- `results/analysis/effective_score_by_dataset.pdf`

### 4. Ranking plots by dataset

- [src/analysis/plot_dataset_rankings.py](src/analysis/plot_dataset_rankings.py)

Generates per-dataset PDF ranking plots and a combined figure:

- `results/analysis/rankings/ranking_all_datasets.pdf`
- `results/analysis/rankings/ranking_<dataset>.pdf`

Usage:

```bash
python src/analysis/plot_dataset_rankings.py
```

## Interpreting the Main Analysis Outputs

### `valid_performance_by_dataset.pdf`

Shows the mean final `F1-Macro` for each `model + reasoning` combination, **only on runs where the agent selected the correct dataset**.

This answers:

- when the agent planned correctly, how well did it perform?

### `effective_score_by_dataset.pdf`

Shows the effective score where:

- correct dataset -> use final `F1-Macro`
- wrong dataset -> score becomes `0`

This answers:

- how good is the full end-to-end behavior when planning errors are penalized?

### `dataset_match_rate_by_model_reasoning.pdf`

Shows the dataset selection success rate by model and reasoning setting.

This answers:

- how often does the LLM choose the correct dataset for the requested task?

## Current Modeling Assumptions

The project is designed to test agentic behavior, not just pure classifier quality. In practice that means:

- the LLM selects the dataset
- the LLM selects the representation
- the LLM selects the classifier
- the LLM can revise the strategy over rounds

As a result, final performance depends on both:

- planning correctness
- modeling quality

That is why the analysis explicitly separates:

- valid-only performance
- dataset selection accuracy
- effective score

## Notes

- `OpenRouter` is currently the only supported provider in the CLI.
- Some models may support `--thinking-effort` better than others.
- `pending_*` directories are temporary execution directories created before the final result directory is renamed.
- The project uses PDF output for plots under `results/analysis`.

## Minimal Workflow

1. Install dependencies.
2. Add `OPENROUTER_API_KEY` to `.env`.
3. Run a single experiment with `src/scripts/run.py`.
4. Run the analysis scripts under `src/analysis/`.
5. Inspect results in `results/`.
