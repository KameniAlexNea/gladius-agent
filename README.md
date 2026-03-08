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
  │         • explores data directory and existing solutions (read-only)
  │         • outputs plan via ExitPlanMode
  │
  ├─ [iteration N] implementing
  │    └─ implementer agent(s)    fresh session — coordinator, NOT a code writer
  │         • reads CLAUDE.md + plan
  │         • routes via EXPERIMENT_STATE.json between phase-specialist subagents:
  │             ml-scaffolder  → bootstrap src/ project layout (once)
  │             ml-developer   → write-run-fix loop (handles execution errors)
  │             ml-evaluator   → extract OOF score from artifacts
  │             code-reviewer  → read-only logical review (leakage, metric bugs)
  │             ml-scientist   → fix logical ML bugs (spawned on CRITICAL review)
  │             submission-builder → generate test predictions + format CSV
  │         • outputs: {status, oof_score, solution_files, submission_file, notes}
  │         └─ (--parallel N): up to N run concurrently via asyncio.gather
  │
  ├─ [iteration N] validation
  │    ├─ validation agent        stateless — compares OOF, checks submission format
  │    │    • queries platform quota via MCP (Zindi/Kaggle)
  │    │    • orchestrator overrides with deterministic improvement check
  │    └─ summarizer agent        rewrites MEMORY.md with cumulative learnings
  │
  └─ [iteration N+1] planning  ← loop
```

### Agents

| Agent | Session | Tools | Max turns |
|---|---|---|---|
| Planner | **Resumed** — persistent across iterations | Read, Glob, Grep, WebSearch, TodoWrite | 40 |
| Implementer | **Fresh** each iteration — **coordinator** | Agent(), Read, Write, Glob, TodoWrite | 30 |
| Validation | Stateless | Read, Grep + platform MCP tool | 25 |
| Summarizer | Stateless | Read, Grep | 15 |

The **planner** runs in `permissionMode: plan` with a `can_use_tool` callback that blocks `Write`, `Edit`, `Bash`, and `Task` — it is strictly read-only and exits only via `ExitPlanMode`.

The **implementer** is a thin coordinator. It does not write code or run commands. Instead it spawns phase-specialist subagents via `Agent()` and reads `.claude/EXPERIMENT_STATE.json` after each one to decide routing. It runs with `bypassPermissions`.

### Implementer subagents

Subagents are `.claude/agents/*.md` files. The implementer can spawn only the six listed below — no unbounded delegation.

| Subagent | Model | Max turns | Preloaded skills | Purpose |
|---|---|---|---|---|
| `ml-scaffolder` | `GLADIUS_SMALL_MODEL` | 15 | ml-setup | Bootstrap `src/` package layout — runs once |
| `ml-developer` | inherit | 80 | ml-pipeline, feature-engineering, polars, hpo, ensembling | Write-run-fix loop; handles execution errors only |
| `ml-evaluator` | `GLADIUS_SMALL_MODEL` | 15 | ml-pipeline | Extract OOF score from `artifacts/oof_predictions.npy` |
| `code-reviewer` | inherit | 20 | code-review | Read-only logical review (leakage, metric, CV contamination) |
| `ml-scientist` | inherit | 40 | ml-pipeline, feature-engineering, code-review | Fix logical ML bugs — spawned only on CRITICAL review issues |
| `submission-builder` | inherit | 20 | submit-check | Generate test predictions, format + validate submission CSV |

`GLADIUS_SMALL_MODEL` defaults to `inherit` (same model as the coordinator) when unset. Set it to a faster/cheaper model (e.g. `claude-haiku-4-5`) in your `.env` to reduce cost on deterministic tasks.

**Routing** (directed graph with back-edges):
```
SCAFFOLD → DEVELOP → EVALUATE → REVIEW → SUBMIT
                ↑                   │
           execution error          │ CRITICAL logical bug
                                    ↓
                               ml-scientist → DEVELOP → EVALUATE → REVIEW
```

### Skills

Subagents have skills injected at spawn time via the `skills:` frontmatter field — no `Skill({...})` turn required.

| Skill | Used by |
|---|---|
| `ml-setup` | ml-scaffolder — canonical `src/` package layout |
| `ml-pipeline` | ml-developer, ml-evaluator, ml-scientist — CV patterns, baselines, metric formulas |
| `adversarial-validation` | ml-developer — detect train/test distribution shift |
| `feature-engineering` | ml-developer, ml-scientist — feature recipes, SHAP importance, pruning |
| `hpo` | ml-developer — Optuna Bayesian search |
| `ensembling` | ml-developer — OOF blending, hill-climbing model selection |
| `polars` | ml-developer — fast DataFrame ops (Arrow backend, lazy eval) |
| `code-review` | code-reviewer, ml-scientist — leakage, metric correctness, CV contamination |
| `submit-check` | submission-builder — validate submission CSV format before upload |
| `research` | (available on demand) WebSearch for SOTA techniques on ArXiv + Kaggle |
| `transformers` | (available on demand) HuggingFace Transformers for NLP/vision |
| `pytorch-lightning` | (available on demand) structured DL training loops |
| `timesfm` | (available on demand) Google TimesFM zero-shot time-series forecasting |
| `jupyter-mcp` | (available on demand) start Jupyter + MCP server for notebook work |
| `git-workflow` | commit after each working solution |
| `uv-venv` | create venv, install packages, run scripts |

### State

`CompetitionState` is a Python dataclass persisted to a **normalised 7-table SQLite database** (`.gladius/state.db`). The agent resumes correctly after a crash. On resume, if `best_oof_score` was never persisted (e.g. due to a validation crash), it is recalibrated from the recorded experiments.

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

Each competition directory gets a bootstrapped `.claude/` layout on first run (idempotent — safe to re-run on resume). All template content lives in `gladius/utils/templates/` and is copied at runtime.

```
.claude/
  agents/
    planner.md               — permissionMode: plan, tools: Read Glob Grep WebSearch TodoWrite
    implementer.md           — coordinator: permissionMode: bypassPermissions, tools: Agent() Read Write Glob TodoWrite
    ml-scaffolder.md         — subagent: bootstraps src/ layout, model: haiku
    ml-developer.md          — subagent: write-run-fix loop, model: inherit, maxTurns: 80
    ml-evaluator.md          — subagent: extract OOF score, model: haiku
    code-reviewer.md         — subagent: read-only review, permissionMode: plan
    ml-scientist.md          — subagent: fix logical ML bugs, model: inherit
    submission-builder.md    — subagent: generate + validate submission CSV
  EXPERIMENT_STATE.json      — artifact handshake: subagents write structured JSON; coordinator reads to route
  skills/
    ml-project-structure/    — canonical src/ package layout
    ml-pipeline/             — CV patterns, baselines, metric formulas
    adversarial-validation/  — train/test shift detection
    feature-engineering/     — feature recipes, SHAP importance
    hpo/                     — Optuna Bayesian search
    ensembling/              — OOF blending, hill-climbing
    research/                — ArXiv + Kaggle forum search
    polars/                  — fast DataFrame ops
    transformers/            — HuggingFace NLP/vision
    pytorch-lightning/       — structured DL training
    timesfm/                 — time-series forecasting
    code-review/             — leakage, metric correctness, format checks
    submit-check/            — submission validation checklist
    jupyter-mcp/             — Jupyter + MCP server
    git-workflow/            — commit format
    uv-venv/                 — venv creation, package management
  agent-memory/
    planner/MEMORY.md        — persistent learnings (rewritten by summarizer each iteration)
  settings.json              — env vars, PostToolUse/PreToolUse hooks
scripts/
  after_edit.sh              — PostToolUse: py_compile immediately on Edit/Write
  validate_bash.sh           — PreToolUse: block rm -rf / and rm -rf ~
CLAUDE.md                    — live context refreshed every iteration by orchestrator
.mcp.json                    — MCP server registration (Zindi/Kaggle/fake platform tools)
```

**`CLAUDE.md`** is written by the orchestrator at the start of every iteration. It carries competition settings, best OOF, recent experiments (top 5), failed approaches, a stagnation warning when the last 3 experiments moved the metric by < 0.001, and the available skills table. Every agent reads it automatically via Claude Code's project-context loading.

**`MEMORY.md`** is written by the summarizer after every validation. It accumulates: key data insights, what works ✅, what fails ❌, patterns and hypotheses, full score history, and suggested next directions. The planner reads it at the start of every session. Absolute path is injected into CLAUDE.md to prevent the model reading a global `~/.claude/agent-memory/` file by mistake.

### Platform support

| Platform | Submit | `platform:` value |
|---|---|---|
| Kaggle | `kaggle competitions submit` CLI | `kaggle` |
| Zindi | `zindi` Python package | `zindi` |
| Fake (offline) | Local scoring vs `.answers.csv` | `fake` |
| None | Record artifact only, no upload | `none` |

Platform MCP servers (Zindi, Kaggle) run as **subprocess stdio** servers — not in-process Python objects — so the Claude Code CLI subprocess can reach them over IPC.

---

## Setup

### Requirements

- Python 3.10+
- `uv` — fast Python package manager (`pip install uv` or `brew install uv`)
- Platform credentials if not using `fake`: `~/.kaggle/kaggle.json` for Kaggle, `ZINDI_USERNAME` / `ZINDI_PASSWORD` for Zindi

### Install

```bash
git clone https://github.com/your-org/gladius-agent
cd gladius-agent
uv sync
source .venv/bin/activate
```

### First-time CLI setup

The bundled `claude` binary shows an interactive theme picker on first run. Complete it once before running headlessly:

```bash
echo "1" | .venv/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude \
    --output-format stream-json --setting-sources "" 2>/dev/null || true
```

### Environment

Create a `.env` file in your competition directory (read automatically by `python-dotenv`):

```bash
# Required — set to your model:
GLADIUS_MODEL=claude-sonnet-4-5   # or an Ollama model: qwen3.5:35b

# Optional — cheaper model for deterministic subagents (ml-scaffolder, ml-evaluator):
# Defaults to 'inherit' (same model as the coordinator) when unset.
GLADIUS_SMALL_MODEL=claude-haiku-4-5

# Anthropic API (not needed for local Ollama models):
ANTHROPIC_API_KEY="sk-ant-..."

# Kaggle — or use ~/.kaggle/kaggle.json:
KAGGLE_USERNAME="..."
KAGGLE_KEY="..."

# Zindi:
ZINDI_USERNAME="..."
ZINDI_PASSWORD="..."

# Optional — 0-based index of the Zindi challenge to select:
ZINDI_CHALLENGE_INDEX=0
```

---

## Competition directory

Each competition lives in its own directory. The only required file is `README.md` with a YAML frontmatter block:

```yaml
---
competition_id: my-competition
platform: zindi          # kaggle | zindi | fake | none
metric: f1-score         # metric name (informational for agents)
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
cd examples/fake_competition
gladius --competition-dir ./ --iterations 5
```

`examples/fake_competition` is a 1000-row binary classification dataset. `platform: fake` — submissions scored locally against `data/.answers.csv` using AUC-ROC. No API key needed.

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
| `--mode` | experimental | `experimental` or `personal-production` (hard caps: 1800s/iter, 5 agent calls/iter) |
| `--max-iteration-seconds` | unset | Runtime cap per iteration |
| `--max-agent-calls-per-iteration` | unset | Cap on total agent calls per iteration |
| `--max-failed-runs-total` | unset | Cap on cumulative failed runs before halt |

With `--parallel 2`, the planner generates 2 independent approaches. Both implementers run concurrently via `asyncio.gather`. The best-OOF result advances to validation; all successful runs are recorded as experiments.

---

## Project structure

```
gladius/
  orchestrator.py          — Main loop + CLI (planning → implementing → validation)
  state.py                 — CompetitionState dataclass + StateStore (7-table SQLite)
  agents/
    _base.py               — run_agent() / run_planning_agent(): retry, streaming console
    planner.py             — Explores data, produces ordered plan via ExitPlanMode
    implementer.py         — Coordinator: spawns subagents, reports OOF score
    validation.py          — Compares OOF, queries platform quota, recommends submit/hold
    summarizer.py          — Rewrites MEMORY.md with cumulative learnings
    specs/
      implementer_spec.py  — Coordinator system prompt + output schema
  tools/
    fake_platform_tools.py — Offline scoring MCP server (AUC-ROC vs answer key)
    kaggle_tools.py        — Kaggle API MCP server (subprocess stdio)
    zindi_tools.py         — Zindi submission MCP server (subprocess stdio)
  utils/
    competition_config.py  — Reads YAML frontmatter from competition README.md
    project_setup.py       — Bootstraps .claude/ layout, copies skills + subagents, hooks, CLAUDE.md
    templates/
      agents/              — planner.md, implementer.md, + 6 subagent templates
      skills/              — one .md per skill (18 skills total)
      hooks/               — after_edit.sh, validate_bash.sh
      memory/              — MEMORY.md starter template
examples/
  fake_competition/        — Customer Churn Prediction (offline, AUC-ROC)
  data_dog/                — Zindi financial well-being competition (F1-score, multiclass)
```

---

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
python -m pytest

# Format
tox -e format

# Run example competition (1 iteration, fresh start)
gladius --competition-dir examples/fake_competition --iterations 1 --no-resume
```

---

## Design principles

1. **Agents are complete autonomous workers** — not thin LLM wrappers. The implementer's subagents (ml-developer, ml-scientist, etc.) each own a single phase and work to completion before returning control. Tell the coordinator the goal; let the subagents figure out the steps.

2. **Orchestrator owns all routing and state mutation** — agents output structured JSON; the orchestrator acts on it. Agents never mutate `best_oof_score` directly. Improvement is always re-verified deterministically in Python, regardless of what the validation agent says.

3. **One persistent agent, one fresh agent** — the planner resumes its session across iterations (accumulates deep understanding via MEMORY.md); the implementer starts fresh every time (clean context for one plan).

4. **Strict planning mode** — the planner runs in `permissionMode: plan` with a `can_use_tool` callback that actively denies `Bash`, `Write`, `Edit`, and `Task`. Planning is purely read-only; the only output channel is `ExitPlanMode`.

5. **Coordinator + subagents, not one monolith** — the implementer is a coordinator (30 turns, no Bash/Edit). It routes between six focused subagents via `Agent()`. Each subagent gets only the tools and skills it needs. Context contamination between phases is impossible.

6. **Artifact handshake, not free-text routing** — after every subagent completes, the coordinator reads `.claude/EXPERIMENT_STATE.json` (structured JSON written by the subagent) to decide the next phase. No parsing of conversation text.

7. **Skills preloaded in frontmatter** — domain knowledge is declared in the `skills:` field of each subagent's `.md` frontmatter. Claude Code injects the full skill content at spawn time. No `Skill({...})` turn required; no per-turn latency cost.

8. **Template-driven bootstrap** — all `.claude/` content is generated from `gladius/utils/templates/` at runtime. Subagent templates (`_write_subagents()`) are copied once and preserved — teams can customise them without their changes being overwritten.

9. **Resilient validation** — if the validation agent crashes (e.g. platform API down), a deterministic fallback fires and the summarizer still runs. `best_oof_score` is recalibrated from experiment history on resume if it was never persisted.

10. **Structured output via `output_format`** — every agent specifies a `json_schema`. The SDK guarantees the output conforms before returning. No `json.loads()` + exception handling in the orchestrator.

11. **CLAUDE.md as shared live context** — instead of injecting full state into every prompt, the orchestrator writes `CLAUDE.md` once per iteration. Every agent reads it at session start via Claude Code's native project-context loading. Memory path is absolute to prevent reading the wrong global file.
