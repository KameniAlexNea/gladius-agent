# Gladius Agent — Claude Agent SDK Framework

**Revision**: 2.0
**Status**: Implemented — reflects the working codebase as of 2025-02-27
**Previous revision**: 1.0 was a design/migration plan that was only partially realised. See diff below.

---

## 1. Why Replace LangGraph (retained from v1.0)

The LangGraph approach required:
- A hand-written Python function for every agent node
- A hand-written routing function for every conditional edge
- A TypedDict schema that had to be kept in sync with every node
- Custom JSON-schema validation inside `call_llm()` wrappers

Every refactor touched all four layers. Critical bugs were introduced in each session because the routing edges were disconnected from the nodes (see `code-review.md` revisions 1-3: `validation_agent→submission_decider` and `ensemble_agent→hypothesis` both used `add_edge` instead of `add_conditional_edges`). The `best_oof` gate was permanently broken, `_auc_roc` was O(n²), and the `resource_manager` node was never wired.

The core issue: a Kaggle competition pipeline is inherently complex, multi-step, and tool-heavy. Every LangGraph node was basically just wrapping an LLM call that should also do file I/O, run subprocess commands, and make decisions. These are exactly the things Claude Code is built to handle autonomously.

---

## 2. What Was Actually Built vs. the v1.0 Design

| v1.0 design | v2.0 reality |
|---|---|
| `strategy.py` — StrategyAgent, HypothesisGenerator, KnowledgeExtractor | **Not built.** Merged into `planner.py` |
| `code.py` — CodeAgent (generate + review + version) | **Not built.** Merged into `implementer.py` |
| `execution.py` — ExecutionAgent, resource manager | **Not built.** Implementer handles execution |
| `ensemble.py` — EnsembleAgent | **Not built** (too ambitious; dropped) |
| `validation.py` — ValidationAgent, SubmissionDecider | **Built.** Single stateless validation agent |
| `memory_tools.py` — ChromaDB similarity search | **Not built.** Replaced by MEMORY.md written by summarizer |
| `config.py` — AgentConfig dataclass | **Not built.** Config read from competition README.md frontmatter |
| 18 specialised LangGraph nodes | **4 Claude Code agents** |

### Lessons from Simplification

The most important lesson from v1.0 to v2.0: **agents are complete workers, not thin wrappers**. The implementer writes code, runs it, reads error output, edits, re-runs, and repeats — all inside one `run_agent()` call with `max_turns=80`. There is no separate CodeAgent, no separate ExecutionAgent, no separate ReviewAgent. The LLM handles the inner loop.

The second lesson: **orchestrator owns all state mutation**. Agents return structured JSON; the orchestrator decides what to update. This prevents the class of bug where a validation node writes to `best_oof_score` when it shouldn't.

---

## 3. Architecture Overview

```
orchestrator.py                 # plain Python while loop  (not a graph)
  │
  ├─ state.py                   # CompetitionState dataclass + StateStore (7-table SQLite)
  ├─ utils/competition_config.py # reads YAML frontmatter from competition README.md
  ├─ utils/project_setup.py     # bootstraps .claude/ layout on first run
  │
  ├─ agents/_base.py            # run_agent(): retry, streaming, bypassPermissions
  ├─ agents/planner.py          # resumed session — plans experiments
  ├─ agents/implementer.py      # fresh session — writes + runs code
  ├─ agents/validation.py       # stateless — compare OOF, recommend submit/hold
  └─ agents/summarizer.py       # stateless — rewrite MEMORY.md
```

The orchestrator loop is:
```
while iteration < max_iterations:
    plan   = await run_planner(state, ...)
    result = await run_implementer(state, plan, ...)   # or gather(N) for --parallel
    valid  = await run_validation(state, result, ...)
    await run_summarizer(state, result, valid, ...)
    update state + save + increment iteration
```

No LangGraph. No `StateGraph`. No conditional edges. Just Python.

---

## 4. State Management

`CompetitionState` is a Python `@dataclass` persisted to **7 normalised SQLite tables**. The v1.0 design used JSON blobs for everything; v2.0 only uses a JSON blob for `current_plan` (a list-of-dicts with no natural column mapping).

### Tables

| Table | Contents | Notes |
|---|---|---|
| `competition` | competition_id, metric, direction, data_dir | Written once on first run |
| `current_state` | iteration, phase, best_oof, best_lb, session IDs | Mutable scalars |
| `experiments` | id, iteration, approach, oof_score, solution_files, notes | One row per completed run |
| `failed_runs` | id, iteration, approach, error, traceback | One row per failed implementer |
| `error_log` | id, iteration, phase, error, traceback | Unhandled orchestrator errors |
| `lb_scores` | id, iteration, lb_score, submitted_at | Post-submission leaderboard |
| `state_history` | id, saved_at, iteration, phase, best_oof | Append-only audit log |

### Key design choices

- `StateStore.save()` does DELETE + INSERT for list tables (`experiments`, `failed_runs`, etc.). This is correct for < 50 iterations; at 100+ it becomes O(N) per save. An append-only write with upsert would scale better.
- `StateStore.load()` reconstructs all lists from their tables on startup. Resume after crash works correctly.
- `store.close()` is called after the main loop exits. **Known bug**: if the loop raises an unhandled exception, the connection leaks (not in a `finally` block).

---

## 5. Agent Definitions

All agents go through `run_agent()` in `agents/_base.py`.

```python
async def run_agent(
    prompt: str,
    options: ClaudeCodeOptions,
    *,
    timeout_seconds: float = 3600,
    max_errors: int = 3,
) -> str:
    """
    Streams a Claude Code agent, prints to console, retries up to max_errors on error.
    Returns the final text result.
    Raises RuntimeError if max_errors exceeded.
    """
```

`permission_mode` is always `"bypassPermissions"`. `"acceptEdits"` was the original choice but it still sends `can_use_tool` control requests for `Bash`/`WebSearch`/`Task` tools — these require a `can_use_tool` callback in the SDK, which we don't register. Without it a `CLIConnectionError` is raised. `bypassPermissions` skips all permission checks.

### Planner (`agents/planner.py`)

```python
async def run_planner(
    state: CompetitionState,
    data_dir: Path,
    project_dir: Path,
    platform: str,
    n_parallel: int = 1,
) -> dict:
```

- **Session**: resumed via `state.planner_session_id`. If `None`, starts a new session and stores the ID.
- **MCP servers**: none (`mcp_servers: dict = {}`). The original design used an SDK MCP server for `fake_server` but this triggered a race condition: `end_input()` closes stdin before `_handle_control_request` can write back the MCP response. See §9 for details.
- **Output schema**: `{approach_summary, plan[steps], expected_metric_delta}`. If `n_parallel > 1`, the planner is asked to produce `n_parallel` structurally different approaches.

### Implementer (`agents/implementer.py`)

```python
async def run_implementer(
    plan: dict,
    project_dir: Path,
    data_dir: Path,
    iteration: int,
) -> dict:
```

- **Session**: always fresh. Context from CLAUDE.md and the plan dict injected into prompt.
- **Tools**: Read, Write, Edit, Bash, Glob, Grep.
- **Output schema**: `{status, oof_score, solution_files, submission_file, notes}`.
- `max_turns=80` — implementer runs until it finishes or exhausts turns.

### Validation (`agents/validation.py`)

- **Stateless** — no session ID, no persistent state.
- **Output schema**: `{should_submit, reasoning, confidence}`.
- **Known gap**: `orchestrator.py` reads `validation.get("notes", "")` but the schema has `additionalProperties: False` with no `notes` field. Claude cannot add it. The notes the orchestrator passes to `run_summarizer` are always empty.

### Summarizer (`agents/summarizer.py`)

- **Stateless** — rewrites `MEMORY.md` completely each iteration.
- Called after `state.iteration += 1` has already been applied, so MEMORY.md says "completed iteration N+1" when it means iteration N.

---

## 6. Parallel Execution

With `--parallel N`:
1. Planner is prompted to generate N structurally different approaches.
2. `asyncio.gather(*[run_implementer(approach) for approach in approaches])` runs all N concurrently.
3. Successful results are collected; best OOF is chosen as `result`.

**Known bug**: successful results are appended to `state.experiments` in loop order, then `result = max(successful, key=lambda r: r["oof_score"])` picks the best. Validation agent then reads `state.experiments[-1]` (the last appended, not `result`). In parallel mode the last appended may not be the best. Fix: append `result` last, or store `result` separately.

---

## 7. Competition Configuration

Competition settings are read from YAML frontmatter in the competition's `README.md`:

```yaml
---
competition_id: my-competition
platform: kaggle           # kaggle | zindi | fake
metric: auc_roc
direction: maximize
data_dir: data
---
```

`utils/competition_config.py` parses this with string splitting (no `pyyaml` dependency). This is fragile on multi-line values or values with `:`; a `python-frontmatter` dependency would be safer.

---

## 8. Project Bootstrap — `.claude/` layout

On first run for a competition directory, `utils/project_setup.py` writes:

```
.claude/
  agents/
    planner.md            — AgentDefinition frontmatter (model, maxTurns, tools, permissionMode)
    implementer.md
  skills/
    ml-pipeline/SKILL.md
    submit-check/SKILL.md
    code-review/SKILL.md
    git-workflow/SKILL.md
    uv-venv/SKILL.md
  agent-memory/
    planner/MEMORY.md     — written blank, then managed by summarizer
  settings.json
scripts/
  after_edit.sh           — PostToolUse: py_compile on Write/Edit
  validate_bash.sh        — PreToolUse: block rm -rf /
CLAUDE.md                 — live context (rewritten every iteration by orchestrator)
.mcp.json                 — MCP server registrations
```

### Known issues

- `settings.json` is **always overwritten** on run. User customisations to `settings.json` are lost on restart.
- Agent `.md` files use `permissionMode: acceptEdits` (written by `project_setup.py`). If the planner invokes these via `Task` tool (subagent delegation), they crash with `CLIConnectionError`. Should be `bypassPermissions`.
- `_write_mcp_json` hardcodes `.venv/bin/python` relative to the gladius package root. Will break if the venv has a different name or path. Should use `sys.executable`.

---

## 9. The SDK MCP Race Condition

This is worth documenting because it is a subtle SDK bug that will recur if MCP tools are re-added.

**Symptom**: `BrokenPipeError` / `CLIConnectionError: ProcessTransport is not ready for writing` when an agent that has SDK MCP servers returns its result.

**Root cause**: When Claude finishes generating a result, the SDK calls `end_input()` which closes the subprocess stdin. If — between the last model token and `end_input()` — the subprocess is also processing a `_handle_control_request` to write back an MCP tool response, it attempts to write to the now-closed stdin. The write raises `BrokenPipeError`.

**This affects agents whose last action before returning is an MCP call.** Agents that use MCP tools mid-session (not as the very last action) are fine.

**Fix in this codebase**: `mcp_servers: dict = {}` in `planner.py`. The fake platform scoring is done by the orchestrator calling `_score_submission()` directly, not via MCP.

**If you want to re-add MCP tools**: the safest options are:
1. Register them in `CLAUDE.md`-referenced `.mcp.json` (file-based MCP, not SDK MCP) — these run as a persistent sidecar process, not through the SDK stdin/stdout channel.
2. Implement a `can_use_tool` callback that forces a delay between the last MCP response and `end_input()`.

---

## 10. Dependencies

```toml
[project.dependencies]
claude-agent-sdk = ">=0.1.44"
numpy = "*"
scikit-learn = "*"
scipy = "*"
psutil = "*"
kaggle = "*"
requests = "*"
python-dotenv = "*"
zindi = "*"
```

Notable omissions:
- No `langgraph`, `langchain*` (removed)
- No `chromadb` (memory tools not built)
- No `pyyaml` or `python-frontmatter` (competition config parsed manually — fragile)
- No test framework in main dependencies (`pytest` only in `[project.optional-dependencies].test`)

---

## 11. CLI

```bash
gladius --competition-dir PATH [--iterations N] [--no-resume] [--no-submit] [--parallel N]
```

Entry point: `gladius.orchestrator:main` (defined in `pyproject.toml`).

The v1.0 design specified `--competition`, `--data-dir`, `--metric`, `--direction` as separate flags. All of these are now read from the competition `README.md` frontmatter. The only required CLI argument is `--competition-dir`.

---

## 12. Design Principles

1. **Agents are complete autonomous workers** — not thin LLM wrappers. Tell the agent the goal; let it figure out the steps. `max_turns=80` for implementer.

2. **Orchestrator owns all routing and state mutation** — agents return structured JSON; the orchestrator acts on it. Prevents the class of bug in the original LangGraph version where validation was mutating `best_oof_score` directly.

3. **One persistent agent, one fresh agent** — planner resumes its session across iterations (accumulates deep understanding); implementer starts fresh every time (clean context for one plan).

4. **`bypassPermissions` for headless operation** — `acceptEdits` still sends `can_use_tool` control requests for Bash/WebSearch/Task, which require a registered callback. In a headless server `bypassPermissions` is the correct mode.

5. **CLAUDE.md as shared live context** — instead of injecting full state into every agent prompt, the orchestrator writes `CLAUDE.md` once per iteration. Every agent automatically reads it at session start.

6. **Structured output via `output_format`** — every agent specifies a `json_schema`. The SDK validates before returning. No `json.loads()` + exception handling.

7. **Parallelism via `asyncio.gather()`** — not LangGraph `Send()` primitives. Clean, easy to debug, no framework overhead.
