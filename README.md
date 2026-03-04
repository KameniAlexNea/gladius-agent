# gladius-agent

Fully autonomous multi-agent system for ML competitions. Given a competition directory it runs a continuous loop without human intervention: plans experiments, writes and executes code, validates OOF results, decides whether to submit, and synthesises learnings into persistent memory — all driven by [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk).

---

## Architecture

Four specialised Claude agents run sequentially in a planning → implementing → validation loop. The orchestrator is plain Python `if/elif` — no LangGraph, no graph edges.

```
orchestrator.py
  │
  ├─ [iteration N] planning
  │    └─ planner agent           resumed session — accumulates competition understanding
  │         • reads CLAUDE.md + MEMORY.md
  │         • explores data directory and existing solutions
  │         • outputs: {approach_summary, plan[steps], expected_metric_delta}
  │
  ├─ [iteration N] implementing
  │    └─ implementer agent(s)    fresh session — focused on one plan
  │         • reads CLAUDE.md + plan
  │         • writes code, runs it, debugs, iterates until done
  │         • outputs: {status, oof_score, solution_files, submission_file, notes}
  │         └─ (--parallel N): up to N run concurrently via asyncio.gather
  │
  ├─ [iteration N] validation
  │    ├─ validation agent        stateless — compares OOF, checks submission format
  │    └─ summarizer agent        rewrites MEMORY.md with cumulative learnings
  │
  └─ [iteration N+1] planning  ← loop
```

### Agents

| Agent | Session | Tools | Max turns |
|---|---|---|---|
| Planner | **Resumed** — persistent memory across iterations | Read, Glob, Grep, Bash, WebSearch, Task | 40 |
| Implementer | **Fresh** each iteration | Read, Write, Edit, Bash, Glob, Grep | 80 |
| Validation | Stateless | Read, Bash | 10 |
| Summarizer | Stateless | Read, Write | 15 |

### State

`CompetitionState` is a Python dataclass persisted to a **normalised 7-table SQLite database** (`.gladius/state.db`). No JSON blobs except `current_plan` (a nested list with no clean column mapping). The agent resumes correctly after a crash.

| Table | Contents |
|---|---|
| `competition` | Static: competition_id, metric, direction, data_dir — written once |
| `current_state` | Mutable scalars: iteration, phase, best scores, session IDs |
| `experiments` | One row per completed experiment |
| `failed_runs` | One row per failed implementer run |
| `error_log` | One row per unhandled orchestrator error |
| `lb_scores` | Leaderboard scores after submission |
| `state_history` | Append-only audit log of every save |

### Claude Code native config (`.claude/`)

Each competition directory gets a bootstrapped `.claude/` layout on first run:

```
.claude/
  agents/
    planner.md               — agent definition (name, tools, model, maxTurns)
    implementer.md           — agent definition
  skills/
    ml-pipeline/SKILL.md     — CV pattern, baselines, submission format
    submit-check/SKILL.md    — submission validation checklist
    code-review/SKILL.md     — leakage, metric correctness, format checks (CRITICAL items)
    git-workflow/SKILL.md    — commit message format after each run
    uv-venv/SKILL.md         — how to install packages and run scripts
  agent-memory/
    planner/MEMORY.md        — persistent learnings (rewritten by summarizer each iteration)
  settings.json              — model, env vars, PostToolUse/PreToolUse hooks
scripts/
  after_edit.sh              — PostToolUse: py_compile immediately on Edit/Write
  validate_bash.sh           — PreToolUse: block rm -rf / and rm -rf ~
CLAUDE.md                    — live context refreshed every iteration by orchestrator
.mcp.json                    — MCP server registration (metric-tools)
```

**`CLAUDE.md`** is written by the orchestrator at the start of every iteration. It carries competition settings, best OOF, recent experiments, failed approaches, and a stagnation warning when the last 3 experiments moved the metric by < 0.001. Every agent reads it automatically (Claude Code loads it from the working directory at session start).

**`MEMORY.md`** is written by the summarizer after every validation. It accumulates: key data insights, what works ✅, what fails ❌, patterns and hypotheses, full score history, and suggested next directions. The planner reads it at the start of every session.

### Platform support

| Platform | Submit | `platform:` value |
|---|---|---|
| Kaggle | `kaggle competitions submit` CLI | `kaggle` |
| Zindi | `zindi` Python package | `zindi` |
| Fake (offline) | Local scoring vs `.answers.csv` | `fake` |

---

## Setup

### Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` set in environment
- `claude` CLI — bundled with `claude-agent-sdk` (no separate install)
- Platform credentials if not using `fake`: `~/.kaggle/kaggle.json` for Kaggle, `ZINDI_USERNAME` / `ZINDI_PASSWORD` for Zindi

### Install

```bash
git clone https://github.com/your-org/gladius-agent
cd gladius-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### First-time CLI setup

The bundled `claude` binary shows an interactive theme picker on first run. Complete it once before running headlessly:

```bash
echo "1" | .venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude \
    --output-format stream-json --setting-sources "" 2>/dev/null || true
```

### Environment

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Kaggle — or use ~/.kaggle/kaggle.json:
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
# Zindi:
export ZINDI_USERNAME="..."
export ZINDI_PASSWORD="..."
```

---

## Competition directory

Each competition lives in its own directory. The only required file is `README.md` with a YAML frontmatter block:

```yaml
---
competition_id: my-competition
platform: kaggle         # kaggle | zindi | fake
metric: auc_roc          # metric name (informational for agents)
direction: maximize      # maximize | minimize
data_dir: data           # relative path to data directory
---

# My Competition Title

...competition description and data documentation here...
```

`data_dir` must contain at minimum:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### Example: offline testing with fake competition

```bash
gladius --competition-dir examples/fake_competition
```

`examples/fake_competition` is a 1000-row binary classification dataset (Customer Churn Prediction). `platform: fake` — submissions scored locally against `data/.answers.csv` using AUC-ROC.

---

## CLI

```bash
gladius --competition-dir PATH [--iterations N] [--no-resume] [--no-submit] [--parallel N] [--mode MODE]
```

| Flag | Default | Description |
|---|---|---|
| `--competition-dir` | required | Competition directory (must contain README.md with frontmatter) |
| `--iterations` | 20 | Maximum planning+implementation iterations |
| `--no-resume` | false | Start fresh — ignore `.gladius/state.db` |
| `--no-submit` | false | Dry-run — skip platform submissions |
| `--parallel N` | 1 | Run N implementers concurrently with different approaches |
| `--mode` | experimental | `experimental` or `personal-production` (enables stricter defaults) |
| `--max-iteration-seconds` | unset | Optional runtime cap per iteration |
| `--max-agent-calls-per-iteration` | unset | Optional cap on total agent calls per iteration |
| `--max-failed-runs-total` | unset | Optional cap on cumulative failed runs before halt |

With `--parallel 2`, the planner generates 2 independent, structurally different approaches. Both implementers run concurrently. The best-OOF result advances to validation; all successful runs are recorded as experiments.

---

## Project structure

```
gladius/
  orchestrator.py          — Main loop + CLI (planning → implementing → validation)
  state.py                 — CompetitionState dataclass + StateStore (7-table SQLite)
  agents/
    _base.py               — run_agent(): retry, streaming console, bypassPermissions
    planner.py             — Explores data, produces ordered plan
    implementer.py         — Writes code, executes, reports OOF score
    validation.py          — Compares OOF, recommends submit/hold (stateless)
    summarizer.py          — Rewrites MEMORY.md with cumulative learnings
  tools/
    fake_platform_tools.py — Offline scoring MCP server (AUC-ROC vs answer key)
    kaggle_tools.py        — Kaggle API MCP server
    zindi_tools.py         — Zindi submission MCP server
  utils/
    competition_config.py  — Reads YAML frontmatter from competition README.md
    project_setup.py       — Bootstraps .claude/ layout, skills, hooks, CLAUDE.md
examples/
  fake_competition/        — Customer Churn Prediction (offline, AUC-ROC)
```

---

## Development

```bash
# Format
tox -e format

# Run example competition (1 iteration, fresh start)
gladius --competition-dir examples/fake_competition --iterations 1 --no-resume

# Parallel run (2 approaches per iteration)
gladius --competition-dir examples/fake_competition --parallel 2
```

---

## Design principles

1. **Agents are complete autonomous workers** — not thin LLM wrappers. The implementer reads, writes, runs, and debugs until the experiment completes. Tell it the goal; let it figure out the steps.

2. **Orchestrator owns all routing and state mutation** — agents output structured JSON; the orchestrator acts on it. Agents never write to `best_oof_score` directly. This was the fatal regression in the original LangGraph version (validation node was updating state it shouldn't own, permanently breaking the submission gate).

3. **One persistent agent, one fresh agent** — the planner resumes its session across iterations (accumulates deep understanding); the implementer starts fresh every time (clean context for one plan).

4. **Structured output via `output_format`** — every agent specifies a `json_schema`. The SDK guarantees the output conforms before returning. No `json.loads()` + exception handling in the orchestrator.

5. **Parallelism via `asyncio.gather()`** — not LangGraph `Send()` primitives. Bounded by `asyncio.Semaphore` when needed.

6. **CLAUDE.md as shared live context** — instead of injecting full state into every prompt, the orchestrator writes `CLAUDE.md` once per iteration. Every agent reads it at session start via Claude Code's native project-context loading.

7. **`bypassPermissions` for headless operation** — `acceptEdits` only covers file edits and still sends `can_use_tool` control requests for `Bash`/`WebSearch`/`Task`, which crash without a permission callback. `bypassPermissions` skips all permission checks.
