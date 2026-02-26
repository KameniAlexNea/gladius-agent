# Code Review: Implementation vs. Design (`agentic-framework.md`)

> Reviewed: 2026-02-24 (revision 2 — post update)
> Based on `agentic-framework.md` v2 and the current codebase (`gladius/`).

---

## What changed since the last review

| Item | Was | Now |
|---|---|---|
| Checkpointer | `MemorySaver` — state lost on restart | ✅ `SqliteSaver` — crash recovery works |
| Router pattern | `router_node` used as both node and edge selector | ✅ Fixed — `lambda s: s.get("next_node", "strategy")` |
| `best_oof` in state | Missing | ✅ Added to `GraphState`, initialised in `create_initial_state` |
| Submission Decider OOF gate | `MIN_OOF_IMPROVEMENT` defined but unused | ✅ `oof_score <= baseline + MIN_OOF_IMPROVEMENT` enforced |
| Submission Decider 2h cooldown | Missing | ✅ `SUBMISSION_COOLDOWN_SECS = 7200` enforced |
| `best_oof` update on submit | Never updated | ✅ Set to `oof_score` in the approve return path |
| Code reviewer retry routing | All failures → `hypothesis` | ✅ retry 0 → `code_generator`, retry 1 → `hypothesis`, retry 2 → `strategy` |
| Code reviewer import check | Missing | ✅ AST-based `_check_imports` added |
| Code Generator correction loop | Ignored `reviewer_feedback` | ✅ `_apply_llm_correction` called when feedback present |
| Code Generator staging path | Wrote using stale `run_id` | ✅ Writes to `state/scripts/pending.py` |
| `graph.py` edge for `code_generator` retry | Missing edge | ✅ `"code_generator": "code_generator"` added to code_reviewer conditional edges |
| `utils/code_reader.py` | Did not exist | ✅ Added: `read_function`, `read_lines`, `list_functions`, `validate_syntax`, `check_imports` |
| ContextBuilder strategy context | 6 fields, thin | ✅ Now 13 fields: `top10_by_oof`, `last5_chronological`, `experiment_type_distribution`, `exploration_budget`, `best_oof` |
| ContextBuilder hypothesis context | No parent script | ✅ `parent_script_source` added |
| `session_summary` population | Never written | ✅ `_maybe_refresh_summary` every 10 findings in Knowledge Extractor |
| Experiment archive | Never written | ✅ `_archive_experiment` writes `state/experiments/{run_id}.json` |
| `pylint` in Code Reviewer | Present | Removed (replaced by AST import check — no subprocess dependency) |

---

## Current status table

| Area | Status |
|---|---|
| Graph structure & routing | ✅ Complete |
| State definition | ✅ Complete |
| SqliteSaver checkpointing | ✅ Complete |
| Strategy Agent (LLM + exploration logic) | ✅ Complete |
| Hypothesis Generator (LLM + spec schema) | ✅ Complete |
| Code Generator (targeted patching) | ⚠️ Partial — patching is still regex/string, no LibCST |
| Code Generator (LLM correction loop) | ✅ Complete |
| Code Generator (staging path) | ✅ Complete |
| Code Reviewer (static analysis) | ⚠️ Partial — sandbox run still missing |
| Code Reviewer (import check) | ✅ Complete |
| Code Reviewer (retry routing) | ✅ Complete |
| Versioning Agent | ✅ Complete |
| Executor | ✅ Complete |
| Watchdog | ✅ Complete |
| Resource Manager | ⚠️ Partial — no GPU monitoring |
| Validation Agent | ⚠️ Partial — missing leakage + metric consistency checks |
| Submission Decider | ✅ Complete (all 4 enforced gates) |
| Submission Agent | ✅ Complete |
| LB Tracker | ✅ Complete |
| Ensemble Agent | ⚠️ Partial — no weight optimisation |
| Knowledge Extractor | ✅ Complete |
| Notifier | ✅ Complete |
| Error Handler | ⚠️ `_notify_telegram` still a no-op |
| code_reader utility | ✅ Complete |
| ContextBuilder (strategy context) | ✅ Complete |
| ContextBuilder (hypothesis context) | ⚠️ Partial — parent script loaded by heuristic, not by version |
| session_summary | ✅ Populated every 10 findings |
| Conductor | ❌ Stub — not wired into graph |
| State Manager | ❌ Not implemented |
| Supervisor | ❌ Not implemented |
| Vector database / code index | ❌ Not implemented |
| Diff-based patching | ❌ Not implemented |
| Tool/dependency manifest | ❌ Not implemented |
| Parallel execution (Send API) | ❌ Not implemented |

---

## Remaining issues

### ❌ Still critical — Conductor, State Manager, Supervisor absent

No change since the first review. Not critical for a single sequential run, but required before the system can be left unattended.

- **Conductor** (`orchestration/conductor.py`): still 8 lines that forward `next_node` unchanged. Not wired into the graph.
- **State Manager**: not implemented. Needed as a prerequisite for parallel `Send` branches.
- **Supervisor**: no PID-based agent monitoring, no restart-on-crash, no `degraded` state.

---

### ❌ Code patching is still regex/string — correctness not guaranteed

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

All three patch functions carry the same failure modes as in the first review:

**`_apply_param_change`**: global `re.sub` matches any line containing `param =`. On a real LightGBM script with `learning_rate` mentioned in a comment, a docstring, and inside a `params = {}` dict, this rewrites every occurrence:

```python
# This will rewrite ALL of these:
# learning_rate = 0.05          ← the target
# # old learning_rate = 0.03    ← comment — now broken
# print(f"lr={learning_rate}")  ← reference, not assignment
```

**`_apply_feature_remove`**: `if feature_name not in line` deletes any line containing the feature name as a substring. `feature_name = "age"` removes lines with `"average"`, `"message"`, `"stage"`, `"page"`. Silent corruption.

**`_apply_feature_add`**: falls back to `script + f"\n{snippet}"` when the marker is absent, appending feature code after `if __name__ == "__main__":`. Syntactically valid, logically dead.

**The fix is LibCST** — a concrete syntax tree library (pip-installable, no native deps) that gives structure-aware transformers preserving formatting and failing loudly on ambiguous targets. On real competition scripts the regex patcher will corrupt code on the first non-trivial change.

---

### ❌ No sandbox run in Code Reviewer

**File:** [gladius/nodes/code/code_reviewer.py](../gladius/nodes/code/code_reviewer.py)

The import check (now present) catches missing packages. It does not catch:
- Shape mismatches (`df['col']` where `col` doesn't exist in the data)
- Runtime errors in feature transforms (division by zero, wrong dtypes)
- Missing function arguments, wrong return types
- Logic errors that produce NaN silently

A 1%-sample sandbox run catches all of these before a 30-minute full training run. The spec requires it. What's needed:
1. `tempfile.NamedTemporaryFile(suffix=".py", delete=True)` — ephemeral, cleaned up automatically
2. Training script must respect `GLADIUS_SAMPLE_FRACTION=0.01` envvar — an interface contract that must be in the default template
3. `subprocess.run([sys.executable, tmpfile.name], timeout=60, env={…, "GLADIUS_SAMPLE_FRACTION": "0.01"})`
4. Parse output for `Traceback` / `Error` patterns

This is the highest-value unimplemented check.

---

### ⚠️ `_apply_llm_correction` sends the entire script and expects a full re-emit in JSON

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

The correction loop is now wired correctly. But the prompt sends the entire script as a JSON string value:

```python
f"Current script:\n```python\n{script}\n```\n\n"
# asks for: {"fixed_script": "<entire corrected script>"}
```

Two problems:
1. For a 400-line script, the LLM must read ~300 tokens of code to fix a single-line import error. Wasteful.
2. Asking the LLM to return a full script as a JSON string value will break `json.loads` on any script containing double quotes, backslashes, or triple-quoted strings — unless the LLM perfectly escapes them (it won't, reliably).

**Better approach**: use `code_reader.read_lines` to send only ±20 lines around each reported issue, and ask for a targeted patch (before/after the specific lines), not a full re-emit.

---

### ⚠️ `_load_parent_script_source` uses a content heuristic, not version lookup

**File:** [gladius/utils/context_builder.py](../gladius/utils/context_builder.py)

```python
for f in sorted(SCRIPTS_DIR.glob("v*.py"), reverse=True):
    content = f.read_text()
    if target_model in content.lower():
        return content
```

Problems:
1. Returns the **most recently modified** matching script, not the **best-performing** one. The most recent version may be a failed experiment or a script mid-correction.
2. `"lgbm"` won't match a script that uses `import lightgbm as lgb` or `lgb.Dataset`. Brittle heuristic.
3. The Hypothesis Generator spec already contains `parent_version` (e.g. `"v41"`). The correct lookup is `SCRIPTS_DIR / f"{parent_version}.py"` — already implemented in `code_generator._load_parent_script`. ContextBuilder should use the same path, but there's a sequencing problem: `build_hypothesis_context` is called with the directive (before the spec exists). The parent script should be loaded in Code Generator (which has the spec) and passed to `_apply_llm_correction` there, not assembled separately in ContextBuilder.

---

### ⚠️ `_days_elapsed` is a constant-factor estimate, not real elapsed time

**File:** [gladius/utils/context_builder.py](../gladius/utils/context_builder.py)

```python
_DAYS_PER_EXPERIMENT_ESTIMATE = 0.1

def _days_elapsed(experiments: list) -> float:
    # TODO: use actual experiment timestamps for accurate elapsed-days calculation
    return len(experiments) * _DAYS_PER_EXPERIMENT_ESTIMATE
```

After 100 experiments this claims 10 days elapsed regardless of reality. The experiment archive now records `finding.timestamp` (written by `_archive_experiment`). Use it:

```python
def _days_elapsed(experiments: list) -> float:
    timestamps = [e.get("finding", {}).get("timestamp") for e in experiments
                  if e.get("finding", {}).get("timestamp")]
    if len(timestamps) < 2:
        return 0.0
    from datetime import datetime
    t0 = datetime.fromisoformat(timestamps[0])
    t1 = datetime.fromisoformat(timestamps[-1])
    return (t1 - t0).total_seconds() / 86400
```

---

### ⚠️ `session_summary` is a pipe-joined data dump, not a synthesised summary

**File:** [gladius/nodes/strategy/knowledge_extractor.py](../gladius/nodes/strategy/knowledge_extractor.py)

```python
return " | ".join([
    f"[{f['experiment_id']}] {f['finding_type']}: {f['conclusion']}"
    for f in recent
])
```

Two problems:
1. This is a flat string of 10 findings — not a distilled strategic insight. The LLM receives a data dump, not a summary.
2. It refreshes every 10 findings with a **fixed window** (the last 10). After 20 experiments, findings 1–10 are completely discarded.

The right approach: call the LLM to produce a 2–3 sentence synthesis ("We've tried X, Y failed because Z, current best is A, main open direction is B"). Prepend to the previous summary and truncate to a token budget (sliding window compression).

---

### ⚠️ Validation Agent: leakage check and metric consistency still missing

**File:** [gladius/nodes/validation/validation_agent.py](../gladius/nodes/validation/validation_agent.py)

No change since the first review. The spec requires two additional checks:

**Metric consistency**: recompute the metric from the OOF array and compare to `state["oof_score"]`. Distance > 1e-4 means the logged score is from a different run, fold indexing is wrong, or the metric function is different. `competition["metric"]` is available in state.

**Leakage check**: per-fold OOF scores on train folds must not significantly exceed the holdout score (threshold 0.01 AUC). Requires per-fold arrays: `{run_id}_fold_{k}_oof.npy`. This interface contract must be enforced in the default template and documented.

---

### ⚠️ Ensemble Agent: still no blend weight optimisation

**File:** [gladius/nodes/strategy/ensemble_agent.py](../gladius/nodes/strategy/ensemble_agent.py)

No change since the first review. The directive dispatched to Hypothesis Generator contains `base_model_paths` but no `weights`. The spec calls for Nelder-Mead or Optuna over OOF weights. The labels array (ground truth for OOF metric computation) must come from `competition.json`.

---

### ⚠️ `_notify_telegram` in Error Handler is still a no-op

**File:** [gladius/nodes/error_handler.py](../gladius/nodes/error_handler.py)

After 3 failures on a node, the human is never notified. `notifier._send_telegram` is already importable:

```python
from gladius.nodes.validation.notifier import _send_telegram

def _notify_telegram(msg: str, node: str):
    _send_telegram(f"❌ Node '{node}' failed 3× — escalating to strategy. Error: {msg}")
```

---

### ⚠️ Watchdog `_get_exit_code` will raise on non-child processes

**File:** [gladius/nodes/execution/watchdog.py](../gladius/nodes/execution/watchdog.py)

```python
def _get_exit_code(pid: int) -> int:
    try:
        proc = psutil.Process(pid)
        return proc.wait(timeout=5)   # raises AccessDenied on non-child PIDs
    except Exception:
        return -1
```

`psutil.Process.wait()` raises `psutil.AccessDenied` when the caller is not the parent of that PID — which happens when the graph resumes from a SQLite checkpoint (the process that launched the subprocess was a different Python runtime). The `except Exception: return -1` masks this as a failure exit code. A successfully completed training run will be misrouted to `knowledge_extractor` instead of `validation_agent`.

**Fix**: write the exit code to a sidecar file from the Executor subprocess:

```python
# In executor_node, wrap the subprocess call:
# After proc exits, write: (LOGS_DIR / f"{run_id}.exitcode").write_text(str(proc.returncode))
# Watchdog reads the file instead of calling psutil.wait()
```

---

### ⚠️ `code_retry_count` not reset in Strategy Agent

**File:** [gladius/nodes/strategy/strategy_agent.py](../gladius/nodes/strategy/strategy_agent.py)

Code Generator resets `code_retry_count` to `0` when it successfully writes a new script. But there is no reset when a **new directive** is issued by Strategy Agent. If the previous experiment exhausted 3 retries, `code_retry_count = 3` persists. The Code Reviewer then routes immediately to Strategy on the first failure of the new experiment, skipping the 2 correction attempts the new spec deserves.

**Fix**: add `"code_retry_count": 0` to `strategy_node`'s return dict.

---

### ⚠️ `best_oof` is updated in Submission Decider, not Validation Agent

**File:** [gladius/nodes/validation/submission_decider.py](../gladius/nodes/validation/submission_decider.py)

```python
return {
    "experiment_status": "submitted",
    "best_oof": oof_score,  # only updated on submission approval
}
```

`best_oof` is only updated when the Submission Decider approves a submission. A run with the best OOF ever that is held for the 2h cooldown will not update `best_oof`. The next experiment's OOF will therefore be compared to an outdated baseline, potentially triggering a premature submission.

`best_oof` should track the best OOF across all **validated** runs, regardless of whether they were submitted. Update it in Validation Agent on a clean pass, or in Knowledge Extractor after archiving the run.

---

### ⚠️ LB Tracker parses Kaggle CLI output by hardcoded column index

**File:** [gladius/nodes/strategy/lb_tracker.py](../gladius/nodes/strategy/lb_tracker.py)

```python
return float(parts[4])  # assumes publicScore is column index 4
```

Kaggle CLI output format has changed before. Use the Kaggle Python API (`kaggle.api.competitions_submissions_list(competition)`) which returns structured objects with a `.publicScore` attribute, or at minimum parse the header row to find the `publicScore` column index.

---

### ❌ No vector database / code index

No change since the first review. Without it:
- ContextBuilder sends entire 400-line scripts to the LLM
- There is no way to answer "what parameters has LightGBM used across all versions?"
- There is no deduplication of proposed features against the code history
- `_apply_llm_correction` sends the full script to fix a one-line import error

ChromaDB (on-disk, no server, Python-native) chunked by AST block (one chunk = one function body or dict literal) is the minimum viable implementation.

---

### ❌ No diff / patch representation

No change since the first review. Versioning Agent stores the full modified script but not a diff. No rollback path, no deduplication, no human-readable Telegram summary of what changed.

```python
import difflib
diff = "".join(difflib.unified_diff(
    parent_source.splitlines(keepends=True),
    modified_source.splitlines(keepends=True),
    fromfile=parent_version, tofile=new_version,
))
(VERSIONS_DIR / f"{version_tag}.diff").write_text(diff)
```

---

### ❌ No tool / dependency manifest in generated scripts

No change since the first review. Generated scripts can add arbitrary imports. There is no structured manifest, no approved-package whitelist, and no auto-install path. A `# GLADIUS_DEPS: lightgbm>=4.0` comment block parsed by Code Reviewer is the minimum viable layer.

---

## Priority list (updated)

| # | Item | Effort | Unblocks |
|---|---|---|---|
| 1 | `code_retry_count` reset in `strategy_node` | 5 min | Correct retry behaviour for all future experiments |
| 2 | `best_oof` updated in Validation Agent (not only Submission Decider) | 20 min | Accurate OOF gating |
| 3 | `_notify_telegram` wired to `notifier._send_telegram` | 10 min | Human visibility on node failures |
| 4 | Sandbox run in Code Reviewer (1% sample, 60s, temp file) | 2 h | Catches all runtime errors before full training |
| 5 | `_apply_llm_correction`: targeted context via `code_reader.read_lines`, return a patch not a full script | 1 h | Reliable LLM corrections, no JSON escaping issues |
| 6 | Fix `_load_parent_script_source` to use `parent_version` from spec directly | 30 min | Correct parent script in Hypothesis context |
| 7 | `_days_elapsed` using real timestamps from experiment archive | 30 min | Accurate `exploration_budget` |
| 8 | `session_summary`: LLM-synthesised rolling summary, not pipe-join | 1 h | Useful strategic memory for LLM |
| 9 | Watchdog: sidecar `.exitcode` file instead of `psutil.wait()` | 1 h | Correct success/failure routing after checkpoint resume |
| 10 | Replace regex patching with LibCST in Code Generator | 3 h | Structurally correct script modification |
| 11 | Validation Agent: metric consistency check | 1 h | Catches score logging bugs |
| 12 | Validation Agent: per-fold leakage check (requires training script contract) | 2 h | Catches overfitting before submission |
| 13 | LB Tracker: use Kaggle Python API instead of column-index CSV parsing | 1 h | Stable LB score retrieval |
| 14 | Ensemble Agent: Nelder-Mead weight optimisation | 2 h | Actually useful blend proposals |
| 15 | Diff in Versioning Agent (`difflib.unified_diff`) | 1 h | Rollback, deduplication, Telegram summaries |
| 16 | Feature deduplication check in Code Generator before `_apply_feature_add` | 1 h | Prevents LLM re-proposing existing features |
| 17 | Tool/dependency manifest + whitelist (`# GLADIUS_DEPS`) | 2 h | Controlled dependency management |
| 18 | Vector index (ChromaDB, AST-chunked) | 4 h | Semantic code search, targeted LLM context |
| 19 | Conductor wired + Supervisor implemented | ~1 week | Full unattended operation |
