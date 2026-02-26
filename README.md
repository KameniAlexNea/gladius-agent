# gladius-agent

Fully autonomous multi-agent system for Kaggle-style ML competitions. The graph runs continuously without human intervention: it generates hypotheses, writes and reviews code, executes training runs, validates OOF arrays, decides whether to submit, tracks leaderboard scores, and synthesises findings back into strategy — all as a closed LangGraph loop. The human receives Telegram notifications and reads results.

---

## Architecture

18 specialised agents are wired as nodes in a LangGraph `StateGraph`. All routing goes through a single `router` node that reads `state["next_node"]` and `state["experiment_status"]`. LangGraph checkpointing (SQLite in production, MemorySaver in dev) lets the process resume from the last completed node after a crash.

```
router
  ├── strategy          → hypothesis → code_generator → code_reviewer
  │                                                           ├── versioning_agent → executor → watchdog
  │                                                           │                                    ├── validation_agent → submission_decider
  │                                                           │                                    │                           ├── submission_agent → lb_tracker → router
  │                                                           │                                    │                           └── router (held)
  │                                                           │                                    └── knowledge_extractor → router
  │                                                           ├── hypothesis (retry)
  │                                                           └── strategy (3rd failure)
  ├── ensemble_agent    → hypothesis
  ├── notifier          → router
  └── error_handler     → (original node)
```

### Agent Layers

| Layer | Agents |
|---|---|
| Orchestration | Conductor, Error Handler |
| Strategy | Strategy Agent, Hypothesis Generator, LB Tracker, Ensemble Agent, Knowledge Extractor |
| Code | Code Generator, Code Reviewer, Versioning Agent |
| Execution | Executor, Watchdog, Resource Manager |
| Validation & Submission | Validation Agent, Submission Decider, Submission Agent, Notifier |

---

## State

A single `GraphState` TypedDict is the source of truth for one competition run. Fields:

| Field | Type | Purpose |
|---|---|---|
| `competition` | dict | Config: metric, target, deadline, days remaining, sub limit |
| `current_experiment` | dict\|None | ExperimentSpec from Hypothesis Generator |
| `experiment_status` | str | `pending → queued → running → done/failed/killed → validated → submitted → complete` |
| `running_pid` | int\|None | PID of the training subprocess |
| `run_id` | str\|None | Version tag, e.g. `v42` |
| `oof_score` | float\|None | OOF metric for the current run |
| `lb_score` | float\|None | Public LB score after submission |
| `gap_history` | list | OOF−LB gaps for last 10 scored experiments |
| `submissions_today` | int | Incremented by Submission Agent |
| `last_submission_time` | float\|None | Unix timestamp, used for rate-limit buffer |
| `directive` | dict\|None | JSON from Strategy Agent |
| `exploration_flag` | bool | Forced exploitation when `days_remaining < 2` |
| `consecutive_same_directive` | int | Forces explore after 5 identical directives |
| `session_summary` | str\|None | Compressed strategy history |
| `generated_script_path` | str\|None | Path to script produced by Code Generator |
| `reviewer_feedback` | str\|None | Issues from Code Reviewer (`None` = pass) |
| `code_retry_count` | int | Resets on new directive; caps at 3 |
| `next_node` | str | Routing target for the router |
| `node_retry_counts` | dict | Per-node crash retry counter |
| `next_node_before_error` | str\|None | Where to resume after error handling |
| `error_message` | str\|None | Last error, cleared by Knowledge Extractor |

### Persistent files (survive restarts)

```
state/
  checkpoint.db           # LangGraph SqliteSaver (production)
  knowledge.json          # Append-only experiment findings
  leaderboard.json        # LB score history
  versions/               # Per-version metadata JSON
  scripts/                # Versioned training scripts
  oof/                    # <run_id>_oof.npy arrays
  predictions/            # <run_id>_submission.csv
  cache/                  # Model/feature caches (auto-evicted on disk pressure)
  experiments/            # Archived per-run records
logs/
  <run_id>.txt            # stdout/stderr from training subprocess
```

---

## Experiment Lifecycle

```
PENDING → QUEUED → RUNNING → DONE → VALIDATED → SUBMITTED → COMPLETE
                          ↘ FAILED → (knowledge_extractor)
                          ↘ KILLED → (knowledge_extractor)
                                              ↘ HELD → (router, no submission)
                                                            ↘ SCORE_TIMEOUT → notifier
```

---

## Agent Summaries

### Strategy Agent
LLM call with structured prompt. Outputs a `DirectiveJSON`:
```json
{
  "directive_type": "tune_existing | new_features | new_model_type | ensemble | seed_average",
  "target_model": "catboost | lgbm | xgboost | nn | blend",
  "rationale": "one sentence",
  "exploration_flag": true,
  "priority": 3
}
```
Exploration/exploitation enforced in code: forced exploit when `days_remaining < 2`, forced explore after 5 consecutive same directives.

### Hypothesis Generator
Takes the directive and outputs a concrete `ExperimentSpec`:
```json
{
  "parent_version": "v41",
  "changes": [
    {"type": "param_change", "param": "num_leaves", "old": "64", "new": "128"},
    {"type": "feature_add", "feature_name": "feat_x", "code_snippet": "..."},
    {"type": "feature_remove", "feature_name": "feat_y"}
  ],
  "estimated_runtime_multiplier": 1.2,
  "rationale": "..."
}
```

### Code Generator
Does **not** write from scratch. Applies the `changes` array to the parent script via regex (`param_change`), marker-based insertion (`feature_add`), and line filtering (`feature_remove`). Uses the LLM for complex non-trivial changes with ±20 line context.

### Code Reviewer
Static analysis pipeline: `py_compile` → `pylint --errors-only` → hardcoded-path scan. Returns `pass` (routes to Versioning Agent) or `fail` with specific issues (routes back to Hypothesis Generator, max 2 retries, then escalates to Strategy Agent).

### Versioning Agent
On reviewer pass: assigns `v{N}` tag, copies script to `state/scripts/`, writes metadata JSON, runs `git add + commit`.

### Executor
Launches the versioned script as a subprocess, redirects stdout/stderr to `logs/<run_id>.txt`, stores PID in state.

### Watchdog
Polls the log file every 30 s. Kills the process if: RAM > 90%, no log output for 30 min, or wall-clock exceeds `2× estimated_runtime_multiplier`. Routes to Validation Agent on clean exit, Knowledge Extractor on kill/failure.

### Resource Manager
Monitors disk usage. Evicts oldest cache files when usage > 90%, skipping files referenced by the current run or any running job.

### Validation Agent
Checks OOF array: non-empty, no NaN, values in `[0, 1]`.

### Submission Decider
Gates submission on: daily budget not exhausted AND OOF score available AND OOF-LB gap not widening for 3+ consecutive submissions.

### Submission Agent
Calls `kaggle competitions submit` with exponential backoff (max 3 retries: 30 s, 60 s, 120 s). Increments `submissions_today` on success.

### LB Tracker
Polls `kaggle competitions submissions` every 15 min for up to 3 hours. On score arrival: appends to `state/leaderboard.json`, updates `gap_history`. On timeout: routes to Notifier.

### Ensemble Agent
Scans `state/oof/*.npy`. Selects uncorrelated models (Pearson r < 0.97 on OOF). Requires ≥ 3 uncorrelated models before dispatching a blend directive to Hypothesis Generator.

### Knowledge Extractor
Runs after every terminal experiment event (failed, killed, validated, lb score received). Classifies the finding (`param_failure`, `overfitting_signal`, `model_failure`, `feature_failure`) and appends a structured JSON entry to `state/knowledge.json`. Hypothesis Generator reads this file to avoid previously failed parameter regions.

### Notifier
Sends Telegram messages for: LB score received, score timeout, run failures, agent degradation. Configured via `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` environment variables.

### Error Handler
Retries the failing node up to 3 times (per-node counter in `node_retry_counts`). After 3 failures, escalates to Strategy Agent and fires a Telegram alert.

---

## Setup

### Requirements

- Python 3.10+
- `kaggle` CLI configured (`~/.kaggle/kaggle.json`)
- OpenAI-compatible API key

### Install

```bash
git clone https://github.com/your-org/gladius-agent
cd gladius-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"             # default: gpt-4o
export TELEGRAM_BOT_TOKEN="..."       # optional
export TELEGRAM_CHAT_ID="..."         # optional
```

### Competition config

Create `competition.json` in the working directory:

```json
{
  "name": "my-kaggle-competition",
  "metric": "auc",
  "target": "target_column",
  "deadline": "2026-04-30",
  "days_remaining": 65,
  "submission_limit": 5
}
```

### Run

```bash
gladius competition.json
# or
python -m gladius.graph competition.json
```

The graph runs until interrupted. State is checkpointed after every node. On restart, it resumes from the last completed node.

---

## Development

```bash
# Format
tox -e format

# Tests
pip install -e ".[dev]"
pytest tests/ -v
```

### Project structure

```
gladius/
  graph.py                   # Graph wiring, entry point
  state.py                   # GraphState TypedDict, ExperimentStatus enum
  nodes/
    router.py                # Central routing logic
    error_handler.py         # Retry + escalation
    orchestration/
      conductor.py           # Top-level health monitor (stub)
    strategy/
      strategy_agent.py      # LLM directive generation
      hypothesis_generator.py # LLM experiment spec generation
      lb_tracker.py          # Kaggle LB polling
      ensemble_agent.py      # OOF correlation + blend dispatch
      knowledge_extractor.py # Structured findings → knowledge.json
    code/
      code_generator.py      # Script patching (param/feature changes)
      code_reviewer.py       # Static analysis gate
      versioning_agent.py    # git commit + metadata
    execution/
      executor.py            # Subprocess launch
      watchdog.py            # Log tail, kill on limits
      resource_manager.py    # Disk eviction
    validation/
      validation_agent.py    # OOF sanity checks
      submission_decider.py  # Submission gating logic
      submission_agent.py    # kaggle CLI submit
      notifier.py            # Telegram notifications
  utils/
    llm.py                   # OpenAI call wrapper (JSON mode)
    context_builder.py       # Context assembly for LLM nodes
    file_utils.py            # OOF file I/O with POSIX locking
```

---

## Known limitations / not yet implemented

- **Conductor** is a stub: health checks, Supervisor integration, and per-tick job monitoring are not implemented.
- **State Manager** (lock-gated read/write) is not implemented; agents write state directly.
- **Supervisor** (PID monitoring + agent restart on crash) is not implemented.
- **Production checkpointer**: graph uses `MemorySaver`; switch to `SqliteSaver` for persistence across restarts.
- **ContextBuilder** does not load the experiment archive (top-10 by OOF, last-5 chronological, model type distribution).
- **Code Generator** does not call the LLM for complex changes; only regex/marker substitution.
- **Code Reviewer** is missing the import-against-installed-packages check and the 1%-data sandbox run.
- **Submission Decider** is missing the `best_oof` comparison and the 2-hour rate-limit buffer.
- **Validation Agent** is missing the leakage check and the metric-vs-logged-score consistency check.
- **Ensemble Agent** does not optimise blend weights (Nelder-Mead / Optuna); it only selects base models.
- **Parallel execution** via LangGraph `Send` API is not implemented.
