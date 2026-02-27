# Gladius Agent — Code Review

**Revision**: 4
**Reviewed**: 2025-02-27
**Scope**: Fresh review of the Claude Agent SDK implementation. Revisions 1-3 covered the LangGraph codebase, now deleted.

---

## Summary

The SDK migration is structurally sound. The core loop is clear, the state design is solid, and the agent boundary decisions (persistent planner / fresh implementer / stateless validation) are well chosen. The code is readable and largely free of the multi-layer, disconnected-edge bugs that plagued the LangGraph version.

However there are two **critical bugs** that will cause silent wrong behaviour in the default `--parallel` mode, and several high-priority issues that affect reliability.

---

## Priority Table

| # | Severity | File | Issue | Effort |
|---|---|---|---|---|
| 1 | 🔴 Critical | `orchestrator.py` | Parallel mode validates wrong experiment | S |
| 2 | 🔴 Critical | `utils/project_setup.py` | `permissionMode: acceptEdits` crashes subagent Task calls | S |
| 3 | 🟠 High | `agents/planner.py` | `_build_platform_mcp` is dead code | XS |
| 4 | 🟠 High | `orchestrator.py` | `store.close()` not in `finally` — connection leak on error | XS |
| 5 | 🟠 High | `agents/validation.py` | `notes` missing from OUTPUT_SCHEMA — always empty in orchestrator | S |
| 6 | 🟠 High | `utils/project_setup.py` | `_write_mcp_json` hardcodes `.venv/bin/python` | XS |
| 7 | 🟡 Medium | `agents/summarizer.py` | Off-by-one: iteration incremented before summarizer runs | XS |
| 8 | 🟡 Medium | `utils/competition_config.py` | Manual YAML frontmatter parsing — fragile on special chars | S |
| 9 | 🟡 Medium | `tests/` | Directory is empty — zero test coverage | L |
| 10 | 🟡 Medium | `state.py` | List tables: DELETE + INSERT every save — O(N) at 100+ iterations | M |
| 11 | 🟡 Medium | `agents/planner.py` | `planner_session_id` never cleared — stale session not handled | S |
| 12 | 🟢 Low | `scripts/validate_bash.sh` | Uses `python3` not venv Python | XS |
| 13 | 🟢 Low | `utils/project_setup.py` | `settings.json` overwritten every run — user customisations lost | S |

---

## Critical Issues

### 1. Parallel mode validates wrong experiment

**File**: `orchestrator.py`
**Severity**: 🔴 Critical — silent wrong behaviour

In `--parallel N` mode:
```python
results = await asyncio.gather(*[run_implementer(...) for ...])
successful = [r for r in results if r["status"] == "success"]

for r in successful:
    state.experiments.append(r)          # appended in gather() completion order

result = max(successful, key=lambda r: r["oof_score"])  # best OOF
```

Later, validation reads:
```python
latest = state.experiments[-1]           # last appended — NOT result
```

`asyncio.gather()` preserves input order, not completion order. `result` (best OOF) may not be `experiments[-1]`. The validation agent then evaluates the wrong experiment, and if `should_submit` comes back True, the wrong submission file is used.

**Fix**: Append `result` last, or store `result` separately in state (e.g. `state.current_experiment`) and have validation use that instead of `experiments[-1]`.

```python
# after selecting result
for r in successful:
    if r is not result:
        state.experiments.append(r)
state.experiments.append(result)   # result is always last
```

---

### 2. `permissionMode: acceptEdits` in agent definitions crashes subagent Task calls

**File**: `utils/project_setup.py` — `_write_agent_definitions()`
**Severity**: 🔴 Critical — CLIConnectionError crash

The generated `.claude/agents/planner.md` and `implementer.md` have:
```yaml
---
permissionMode: acceptEdits
...
---
```

The planner has access to the `Task` tool. If it delegates to a subagent defined in `.claude/agents/`, that subagent is launched with `permissionMode: acceptEdits`, which still sends `can_use_tool` control requests for `Bash`, `WebSearch`, and `Task`. These control requests require a registered `can_use_tool` callback in the SDK. Without one, the SDK raises:
```
CLIConnectionError: ProcessTransport is not ready for writing
```
This is the same race condition that was fixed for the planner's MCP servers.

**Fix**: Change the generated agent definitions to `permissionMode: bypassPermissions`.

```python
# in _write_agent_definitions():
frontmatter = {
    "permissionMode": "bypassPermissions",  # was: "acceptEdits"
    ...
}
```

---

## High Priority Issues

### 3. `_build_platform_mcp` is dead code

**File**: `agents/planner.py` — lines ~144-164
**Severity**: 🟠 High — misleading, confusing to maintainers

`_build_platform_mcp()` is defined but never called. After the MCP race condition fix (`mcp_servers: dict = {}`), nothing calls it. The function imports `create_sdk_mcp_server` and builds the fake platform server — but it is invisible to any agent.

**Fix**: Delete the function. If MCP is re-introduced later, it should be via `.mcp.json` (file-based sidecar), not SDK MCP. (See `agentic-framework.md §9` for rationale.)

---

### 4. `store.close()` not in `finally` — SQLite connection leak

**File**: `orchestrator.py` — `run_competition()`
**Severity**: 🟠 High — resource leak on any unhandled exception

```python
async def run_competition(...):
    store = StateStore(state_dir / "state.db")
    state = store.load() or CompetitionState(...)

    while iteration < max_iterations:
        ...

    store.close()    # only reached if loop exits normally
```

Any unhandled exception (network timeout, model API error after max retries, keyboard interrupt) skips `store.close()`. SQLite connections are file-locked on Linux; a leaked connection can block a subsequent resume attempt until the OS reclaims the file descriptor.

**Fix**:
```python
store = StateStore(state_dir / "state.db")
try:
    state = store.load() or CompetitionState(...)
    while iteration < max_iterations:
        ...
finally:
    store.close()
```

---

### 5. `notes` always empty from validation — schema mismatch

**File**: `agents/validation.py` — `OUTPUT_SCHEMA`; `orchestrator.py`
**Severity**: 🟠 High — silent data loss

`orchestrator.py` passes notes from the validation agent to the summarizer:
```python
notes = validation.get("notes", "")
await run_summarizer(state, result, notes=notes)
```

But `validation.py` defines:
```python
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "should_submit": {...},
        "reasoning": {...},
        "confidence": {...},
    },
    "additionalProperties": False,
    "required": [...]
}
```

`notes` is not in `properties`. Because `additionalProperties: False`, Claude **cannot** add a `notes` field. The orchestrator always receives `{}` with no `notes` key; `get("notes", "")` always returns `""`. The summarizer never receives the validation agent's qualitative observations.

**Fix** (option A — add `notes` to schema):
```python
"properties": {
    "should_submit": {...},
    "reasoning": {...},
    "confidence": {...},
    "notes": {"type": "string"},   # add this
},
```

**Fix** (option B — use `reasoning` as notes):
```python
notes = validation.get("reasoning", "")
```

---

### 6. `_write_mcp_json` hardcodes `.venv/bin/python`

**File**: `utils/project_setup.py` — `_write_mcp_json()`
**Severity**: 🟠 High — breaks on any non-standard environment

```python
python_path = gladius_root / ".venv" / "bin" / "python"
```

This path is constructed relative to the gladius package installation root. It will be wrong if:
- The package is installed in a conda env
- The venv is named differently (`.env`, `venv`, etc.)
- The package is installed system-wide (no `.venv` at all)
- Running on Windows

**Fix**:
```python
import sys
python_path = sys.executable
```

---

## Medium Priority Issues

### 7. Summarizer off-by-one in iteration number

**File**: `agents/summarizer.py`, `orchestrator.py`
**Severity**: 🟡 Medium — incorrect MEMORY.md history

The orchestrator increments `state.iteration` before calling `run_summarizer()`:
```python
state.iteration += 1
store.save(state)
await run_summarizer(state, ...)
```

The summarizer writes to MEMORY.md: "After completing iteration {state.iteration}, best OOF = ...". But `state.iteration` was already incremented, so MEMORY.md says "iteration 2" for what was actually iteration 1. The planner reads MEMORY.md and may be confused about which iteration it's on.

**Fix**: pass the pre-increment iteration number to the summarizer, or move `state.iteration += 1` to after the summarizer call.

---

### 8. Manual YAML frontmatter parsing is fragile

**File**: `utils/competition_config.py`
**Severity**: 🟡 Medium — breaks silently on edge-case inputs

The parser uses string splitting to extract frontmatter:
```python
content = readme_path.read_text()
if content.startswith("---"):
    end = content.index("---", 3)
    frontmatter = content[3:end]
    for line in frontmatter.splitlines():
        key, _, value = line.partition(":")
```

This breaks on:
- Multi-line YAML values
- Values containing `:` (e.g., `competition_id: my-comp:2025`)
- Quoted strings
- YAML anchors

**Fix**: add `python-frontmatter` or `pyyaml` as a dependency. Both are lightweight and handle all edge cases.

```python
import frontmatter
post = frontmatter.load(readme_path)
config = post.metadata  # dict of frontmatter keys
```

---

### 9. Zero test coverage

**File**: `tests/` — directory is empty
**Severity**: 🟡 Medium — every refactor is a regression risk

The test directory contains only `__init__.py`. There are no tests for:
- `StateStore` round-trip (save → load → verify fields match)
- `CompetitionConfig` parsing (valid frontmatter, missing fields, special chars)  
- Orchestrator logic (submit gate, stagnation detection, parallel result selection)
- Competition config defaults

**Minimum viable test suite**:
```
tests/
  test_state.py             — StateStore round-trip, resume after partial write
  test_competition_config.py — frontmatter parsing, missing fields, edge cases
  test_orchestrator.py       — submit gate, best-result selection, stagnation
```

---

### 10. State list tables: O(N) save at 100+ iterations

**File**: `state.py` — `StateStore.save()`
**Severity**: 🟡 Medium — performance only; correctness unaffected

```python
def save(self, state: CompetitionState) -> None:
    ...
    self.conn.execute("DELETE FROM experiments")
    for exp in state.experiments:
        self.conn.execute("INSERT INTO experiments ...", ...)
```

Every save re-inserts all N experiments, failed runs, etc. At iteration 100+ with N=100 experiments, this is 100 deletes + 100 inserts per save. Still fast (SQLite is fast for small tables), but wasteful.

**Fix**: use `INSERT OR IGNORE` with a stable `id` column, only inserting rows that don't already exist. Or use `INSERT OR REPLACE`. This makes save O(1) additions regardless of history length.

---

### 11. Stale `planner_session_id` not handled

**File**: `agents/planner.py`
**Severity**: 🟡 Medium — silent failure on long-running competitions

When resuming a competition (`--resume` default), `state.planner_session_id` is loaded from the database and passed to `run_agent(resume=session_id)`. Claude sessions expire. If the session has expired:
- The SDK raises an error (likely `APIError` or `CLIConnectionError`)
- The retry logic in `_base.py` will retry up to 3 times, all of which fail with the same expired session
- After 3 errors, `run_competition` raises `RuntimeError("Agent failed after max retries")`

**Fix**: catch the specific session-expired error in `run_planner`, clear `state.planner_session_id`, and retry with a fresh session:
```python
try:
    result = await run_agent(prompt, options)
except SessionExpiredError:
    state.planner_session_id = None
    options = dataclasses.replace(options, resume=None)
    result = await run_agent(prompt, options)
```

---

## Low Priority Issues

### 12. `validate_bash.sh` uses `python3`, not venv Python

**File**: `scripts/validate_bash.sh`
**Severity**: 🟢 Low

The bash validation hook runs `python3 -c "..."` to check commands. On systems where `python3` resolves to a different interpreter than the venv Python, this could behave unexpectedly. Should use the same Python that gladius is installed in.

**Fix**: write the executable path into the script at bootstrap time in `project_setup.py`, or use `#!/usr/bin/env python3` with explicit venv activation.

---

### 13. `settings.json` overwritten on every run

**File**: `utils/project_setup.py` — `_write_claude_settings()`
**Severity**: 🟢 Low

`settings.json` is written unconditionally on every `setup_project_dir()` call, which is called every time the orchestrator starts. Any user customisation to `settings.json` (e.g., adding allowed tools, changing the model) is silently overwritten.

**Fix**: write `settings.json` only if it doesn't exist, or deep-merge user settings with defaults:
```python
if not settings_path.exists():
    settings_path.write_text(json.dumps(DEFAULT_SETTINGS, indent=2))
```

---

## What's Working Well

**Architecture decisions that are correct and should be kept:**

- **`bypassPermissions` everywhere** — the only viable mode for headless multi-turn agents that use Bash. Discovered the hard way; now correctly applied.
- **Normalised SQLite state** — 7 tables, no JSON blobs (except `current_plan`). Survives crash/resume correctly. The old LangGraph SqliteSaver stored `GraphState` as a single JSON blob per checkpoint.
- **CLAUDE.md as live context** — elegant; avoids injecting the full state into every agent prompt.
- **Persistent planner / fresh implementer** — the right boundary. A planner that accumulates understanding is valuable; an implementer that carries stale context from a previous (failed) run is harmful.
- **`output_format` JSON schema** — the SDK validates before returning. Clean separation between agent output and orchestrator decision-making.
- **Structured error handling with max_errors retry** — `_base.py` retries transparently; the orchestrator only sees success or RuntimeError. Keeps the main loop clean.
- **Skills system** — `ml-pipeline/SKILL.md`, `submit-check/SKILL.md`, etc. are an elegant way to persist domain knowledge that every agent can reference without being wired into each prompt individually.
- **Stagnation detection in CLAUDE.md** — warning injected automatically when last 3 iterations moved the metric by < 0.001 is a simple, proven mechanism to redirect agent strategy.

---

## Recommended Fix Order

1. **Fix §1 (parallel experiment selection)** — one-line fix, silent wrong behaviour in default parallel mode
2. **Fix §2 (`permissionMode`)** — one-line fix in `project_setup.py`, prevents crash if planner ever uses `Task`
3. **Fix §4 (`store.close()` finally)** — two-line fix, prevents connection leak
4. **Fix §5 (`notes` schema)** — add one field to `OUTPUT_SCHEMA`, fixes silent data loss to summarizer
5. **Fix §6 (`sys.executable`)** — one-line fix, prevents MCP sidecar breakage
6. **Delete §3 (`_build_platform_mcp`)** — remove dead code to avoid future confusion
7. **Fix §7 (summarizer off-by-one)** — cosmetic but confusing for human-readable MEMORY.md
8. **Fix §8 (yaml parsing)** — add `python-frontmatter` dep, replace manual parser
9. **Write §9 (tests)** — highest-value medium-term investment; start with `test_state.py`
