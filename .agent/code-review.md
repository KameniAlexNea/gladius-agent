# Code Review: Implementation vs. Design (`agentic-framework.md`)

> Reviewed: 2026-02-26 (revision 3 — strict re-audit post implementation session)
> Based on the live codebase in `gladius/`. All files read from disk. 55 tests pass.

---

## What changed since revision 2

| Item | Was | Now |
|---|---|---|
| `best_oof` in Validation Agent | Updated only on submission approval | ✅ Updated on every validated run |
| Metric consistency check | Missing | ✅ Independent OOF recomputation vs script-reported score |
| `_days_elapsed` | Constant estimate (0.1 days/experiment) | ✅ Parses real ISO timestamps from experiment archive |
| `_load_parent_script_source` | Content heuristic, returns most-recent matching script | ✅ Version-exact lookup → best-OOF-per-model fallback |
| `.exitcode` sidecar file | Watchdog used `psutil.wait()` (fails after resume) | ✅ Executor writes sidecar; Watchdog reads it |
| `_apply_llm_correction` context | Sent entire script, expected full re-emit | ✅ Targeted ±20-line context; returns line-range replacements |
| Feature deduplication | Not checked | ✅ AST-based function/variable name dedup before insert |
| Diff in Versioning Agent | Not stored | ✅ `difflib.unified_diff` → `{version}.diff` alongside metadata |
| Sandbox run | Missing | ✅ `GLADIUS_SANDBOX=1` subprocess, 60s timeout |
| `session_summary` | Pipe-joined data dump | ✅ LLM-synthesised rolling summary with fallback |
| Ensemble weights | Uniform / none | ✅ Scipy Nelder-Mead on OOF metric; labels from `state/labels.npy` |
| `scipy` dependency | Not in pyproject.toml | ✅ Added `scipy>=1.10.0` |

---

## Current status table

| Area | Status |
|---|---|
| Graph structure & routing | ⚠️ Two wiring bugs (see below) |
| State definition | ✅ Complete |
| SqliteSaver checkpointing | ✅ Complete |
| Strategy Agent | ✅ Complete |
| Hypothesis Generator | ⚠️ SPEC_SCHEMA example `"v41"` can be copied literally by LLM |
| Code Generator — applies changes | ⚠️ Regex patching still fragile |
| Code Generator — LLM correction | ✅ Targeted ±20-line ctx + line-range replacements |
| Code Generator — feature dedup | ✅ AST-based |
| Code Reviewer — static analysis | ✅ Syntax + imports + hardcoded paths |
| Code Reviewer — sandbox run | ✅ Present; no resource limits |
| Code Reviewer — import dedup | ⚠️ `_check_imports` duplicates `utils/code_reader.check_imports` |
| Versioning Agent | ✅ Diff stored |
| Executor — sidecar exitcode | ✅ Daemon thread writes `.exitcode` |
| Executor — log mode | ⚠️ Opens log with `"w"` — overwrites on re-run |
| Watchdog — exitcode | ✅ Reads sidecar; psutil fallback |
| Watchdog — race condition | ⚠️ Small window between process exit and file write |
| Resource Manager | ❌ Orphan node — never called from graph |
| Validation Agent — OOF checks | ✅ Shape, NaN, range, metric consistency |
| Validation Agent — OOF recomputation | ✅ AUC/logloss/RMSE/MAE |
| Validation Agent — `best_oof` update | ✅ Updated on validated runs |
| Validation Agent — `_auc_roc` memory | ❌ O(n²) outer product — OOM on real data |
| Validation Agent — leakage check | ❌ Not implemented |
| Validation Agent — edge to graph | ❌ `add_edge` unconditional — failure path broken |
| Submission Decider — OOF gate | ❌ **REGRESSION**: gate always holds after validation_agent sets best_oof |
| Submission Decider — 2h cooldown | ✅ |
| Submission Decider — gap-widening | ✅ |
| Submission Decider — budget | ✅ |
| Submission Agent | ✅ |
| LB Tracker — score retrieval | ⚠️ Hardcoded column index `parts[4]` |
| LB Tracker — polling loop | ⚠️ Blocks for up to 3h in a tight loop |
| Knowledge Extractor | ✅ |
| Knowledge Extractor — finding classifier | ⚠️ Success classified as `model_failure` |
| Ensemble Agent — correlation filter | ✅ |
| Ensemble Agent — Nelder-Mead weights | ✅ |
| Ensemble Agent — edge to graph | ❌ `add_edge` unconditional — routes to hypothesis even when no models |
| Ensemble Agent — blend in codegen | ❌ Blend directive not handled by hypothesis/code_generator |
| Notifier | ✅ `_send_telegram` fully implemented |
| Error Handler — `_notify_telegram` | ❌ Still a no-op (intentionally deferred) |
| context_builder — `parent_version` | ⚠️ First branch is dead code — directive never carries parent_version |
| code_reader | ✅ Complete |
| Conductor | ❌ Stub — not wired into graph |
| State Manager | ❌ Not implemented |
| Supervisor | ❌ Not implemented |
| Vector database / code index | ❌ Not implemented |
| Tool/dependency manifest | ❌ Not implemented |
| Parallel execution (Send API) | ❌ Not implemented |

---

## Critical regressions introduced in last session

### ❌ REGRESSION: Submission gate permanently broken

**File:** [gladius/nodes/validation/validation_agent.py](../gladius/nodes/validation/validation_agent.py) + [gladius/nodes/validation/submission_decider.py](../gladius/nodes/validation/submission_decider.py)

**Root cause:** Two agents now both write to `best_oof`. Validation Agent sets `best_oof = oof_score` when the run is a new best. Submission Decider then reads `best_oof` and checks:

```python
baseline = best_oof if best_oof is not None else 0.0
if oof_score <= baseline + MIN_OOF_IMPROVEMENT:
    return {"experiment_status": "held", ...}
```

Because Validation Agent already set `best_oof = oof_score`, the check becomes:

```
oof_score <= oof_score + 1e-4  →  True  →  ALWAYS HELD
```

**No run will ever be submitted.** Every experiment that achieves a new best OOF will be held by the gate it just tripped.

**Fix:** Use two separate fields.
- `best_oof`: best OOF across all validated runs — owned by Validation Agent, for tracking and strategy context only.
- `best_submitted_oof`: OOF of the last approved submission — owned by Submission Decider, used as the gate baseline.

In state.py add `best_submitted_oof: Optional[float]`. Change Submission Decider gate to:

```python
baseline = state.get("best_submitted_oof") or 0.0
if oof_score <= baseline + MIN_OOF_IMPROVEMENT:
    return {"experiment_status": "held", ...}
# on approve:
return {"experiment_status": "submitted", ..., "best_submitted_oof": oof_score}
```

Remove `"best_oof": oof_score` from Submission Decider's return. Validation Agent keeps its `best_oof` update unchanged.

---

### ❌ REGRESSION: `graph.add_edge("validation_agent", "submission_decider")` is unconditional

**File:** [gladius/graph.py](../gladius/graph.py) — line 104

Validation Agent returns `"next_node": "knowledge_extractor"` on failure and `"next_node": "submission_decider"` on success. But the graph has:

```python
graph.add_edge("validation_agent", "submission_decider")  # unconditional
```

LangGraph's plain `add_edge` ignores the state completely. **All validation failures bypass knowledge_extractor and go directly to submission_decider**, which then holds (because `oof_score is None`) and routes to the router, which routes to strategy. Knowledge of the failure is never extracted.

**Fix:** Replace with a conditional edge:

```python
graph.add_conditional_edges(
    "validation_agent",
    lambda s: s.get("next_node", "knowledge_extractor"),
    {
        "submission_decider": "submission_decider",
        "knowledge_extractor": "knowledge_extractor",
    },
)
```

---

### ❌ REGRESSION: `graph.add_edge("ensemble_agent", "hypothesis")` is unconditional

**File:** [gladius/graph.py](../gladius/graph.py) — line 139

When Ensemble Agent finds fewer than `MIN_BASE_MODELS` uncorrelated OOF files, it returns `{"next_node": "strategy"}`. The unconditional edge ignores this and always routes to `hypothesis`. Hypothesis Generator then has no directive and returns an error → error_handler → strategy (correct destination, but via error path).

**Fix:** Replace with a conditional edge:

```python
graph.add_conditional_edges(
    "ensemble_agent",
    lambda s: s.get("next_node", "strategy"),
    {
        "hypothesis": "hypothesis",
        "strategy": "strategy",
    },
)
```

---

## High priority bugs

### ❌ `_auc_roc` creates O(n²) matrix — OOM on real datasets

**File:** [gladius/nodes/validation/validation_agent.py](../gladius/nodes/validation/validation_agent.py)

```python
return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
```

This outer product allocates a boolean matrix of shape `(len(pos), len(neg))`. For a competition with 50K positive and 50K negative examples, that is a 50K × 50K = 2.5 billion element matrix — ~2.5 GB per call. For class-imbalanced datasets (90% negative), 10K pos × 90K neg = 900M elements = ~900 MB. This will OOM on any production machine during the first real validation.

**Fix:** Use the rank-based formula which is O(n log n):

```python
def _auc_roc(labels: np.ndarray, preds: np.ndarray) -> float:
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(preds)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(preds) + 1)
    # Handle ties: average rank for tied predictions
    # For simplicity, np.argsort gives stable ranks; this is a good approximation
    rank_sum = np.sum(ranks[labels == 1])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

The same `_auc_roc` function is duplicated in `ensemble_agent.py` (`_evaluate_metric`) and should be fixed there too, or extracted to a shared utility.

---

### ❌ Resource Manager is an orphan node — never called

**File:** [gladius/graph.py](../gladius/graph.py) + [gladius/nodes/execution/resource_manager.py](../gladius/nodes/execution/resource_manager.py)

`resource_manager_node` is registered (`graph.add_node`) but has no incoming edges. No node ever routes to it. Disk cleanup never runs.

**Fix:** Wire it as a pre-executor gate. After versioning and before execution, check disk space:

```python
graph.add_edge("versioning_agent", "resource_manager")
# resource_manager sets next_node = "executor"
graph.add_edge("resource_manager", "executor")
```

Remove `graph.add_edge("versioning_agent", "executor")`.

---

### ❌ Ensemble blend directive has no handler in hypothesis/code_generator

**File:** [gladius/nodes/strategy/hypothesis_generator.py](../gladius/nodes/strategy/hypothesis_generator.py) + [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

When `ensemble_agent` produces `directive_type = "ensemble"`, it passes `base_model_paths` and `blend_weights` in the directive. But Hypothesis Generator has no special branch for this — it sends the directive verbatim to the LLM and asks it to produce changes as if it were a normal experiment. Code Generator then applies regex patches to a template. The blend weights are never written into any script.

An ensemble run requires a completely different code path: generate a new script that loads the OOF files at `base_model_paths`, applies `blend_weights`, and writes a submission CSV. Without this, `directive_type = "ensemble"` produces a broken script every time.

**Fix:** Add a branch in `hypothesis_node` (or `code_generator_node`) that detects `directive_type == "ensemble"` and generates a blending script directly (either from a template with weights injected, or via a dedicated LLM prompt that explains it's a blend task).

---

## Medium priority issues

### ⚠️ `_apply_param_change` global regex corrupts comments and references

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

```python
pattern = rf"({re.escape(param)}\s*=\s*)[^\n,\)]*"
return re.sub(pattern, replacement, script)
```

This is a global substitution — it rewrites every occurrence of `param = ...` in the file, including:
- Comments: `# old learning_rate = 0.03` → silently corrupted
- Dict entries: `params = {"learning_rate": 0.03}` — the pattern won't match exactly but adjacent assignments will
- Print statements: `print(f"lr={learning_rate}")` — won't match `=\s*` but any `lr = ...` reference will

For a real LightGBM script with 200+ lines and a hyperparameter mentioned multiple times, this will produce corrupted code. The minimum fix is to restrict to assignment statements only (start of line or after indent):

```python
pattern = rf"^(\s*{re.escape(param)}\s*=\s*)[^\n,\)]*"
re.sub(pattern, replacement, script, flags=re.MULTILINE)
```

LibCST is the correct long-term fix (see open item below).

---

### ⚠️ `_apply_feature_remove` uses raw substring match

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

```python
filtered = [line for line in lines if feature_name not in line]
```

`feature_name = "age"` removes lines containing `"average"`, `"message"`, `"stage"`, `"page_views"`. Silent, irreversible data corruption on any script with words containing the feature name as a substring. Fix: use `\b{feature_name}\b` word-boundary regex, or AST-based removal.

---

### ⚠️ `_apply_feature_add` falls back to appending after `__main__`

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

When the `# --- FEATURES END ---` marker is absent (e.g., in a real competition script not generated from the template), new feature code is appended after the last line:

```python
return script + f"\n{snippet}"
```

If the script ends with `if __name__ == "__main__": train()`, any new feature function defined after this block is syntactically valid Python but will never be called — the feature is silently discarded. For top-level assignments (e.g., `new_col = df["a"] / df["b"]`), they will execute at import time but after `train()` has already run.

The marker is part of the default template, but the spec says the system should work on user-provided base scripts too.

---

### ⚠️ LLM correction fallback still sends full script as JSON value

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

When feedback contains no line numbers (e.g., `"import not found: nonexistent_pkg"`), the fallback path sends the entire script inside a JSON string value and expects `{"fixed_script": "<entire script>"}` back. Any script with double quotes, backslashes, or triple-quoted strings will break `json.loads` unless the LLM perfectly escapes them. The import-error case is the most common reviewer feedback and it never contains line numbers.

**Fix:** In the fallback path, also get a targeted context by finding the import line via AST, then send only the import block (first ~20 lines of the script) rather than the full script.

---

### ⚠️ `_check_imports` is duplicated between code_reviewer and code_reader

**File:** [gladius/nodes/code/code_reviewer.py](../gladius/nodes/code/code_reviewer.py) + [gladius/utils/code_reader.py](../gladius/utils/code_reader.py)

`code_reviewer._check_imports(content)` and `code_reader.check_imports(code)` are byte-for-byte identical logic. They will diverge over time. Code Reviewer should import and call `from gladius.utils.code_reader import check_imports`.

---

### ⚠️ Executor opens log file with `"w"` — overwrites on checkpoint resume

**File:** [gladius/nodes/execution/executor.py](../gladius/nodes/execution/executor.py)

```python
with open(log_path, "w") as log_file:
```

If LangGraph resumes from a checkpoint at the `executor` node (e.g., after a host reboot), it will re-execute executor_node for the same `run_id`. The log from the previous run is overwritten before the new process produces any output. Use `"a"` (append) mode with a separator line instead.

---

### ⚠️ Race condition: `.exitcode` file may not exist when watchdog polls

**File:** [gladius/nodes/execution/watchdog.py](../gladius/nodes/execution/watchdog.py) + [gladius/nodes/execution/executor.py](../gladius/nodes/execution/executor.py)

The sequence is:
1. Process exits.
2. `_is_running(pid)` → False (process not in ptable).
3. Watchdog calls `_get_exit_code(pid, run_id)`.
4. Daemon thread calls `exitcode_path.write_text(str(rc))`.

Steps 2 and 4 are concurrent. There is a window between the process exiting and the daemon thread writing the file. In that window, `_get_exit_code` reads an absent file, falls through to `psutil.wait()` (which raises `NoSuchProcess`), and returns -1. A successful run (exit 0) is routed to `knowledge_extractor` as a failure.

**Fix:** In `_get_exit_code`, poll the sidecar file up to ~2s before falling back:

```python
import itertools, time as _time
for _ in range(20):  # up to 2s
    if exitcode_path.exists():
        break
    _time.sleep(0.1)
```

---

### ⚠️ Hypothesis Generator SPEC_SCHEMA contains a literal example version `"v41"`

**File:** [gladius/nodes/strategy/hypothesis_generator.py](../gladius/nodes/strategy/hypothesis_generator.py)

```python
SPEC_SCHEMA = {
    "parent_version": "v41",
    ...
}
```

The LLM is instructed to respond matching this schema. It may treat `"v41"` as a valid value rather than a placeholder. This would cause Code Generator to try to load `state/scripts/v41.py` on every new run, fall back to the default template, and produce an experiment that ignores all prior work.

**Fix:** Use `"<version_tag>"` as the placeholder, and include currently available versions in the context:

```python
SPEC_SCHEMA = {
    "parent_version": "<version_tag e.g. v3>",
    ...
}
```

---

### ⚠️ Successful experiments classified as `model_failure`

**File:** [gladius/nodes/strategy/knowledge_extractor.py](../gladius/nodes/strategy/knowledge_extractor.py)

`_classify_finding` returns `"model_failure"` as the default fallback — this is what a successful run with a small OOF-LB gap lands on:

```python
def _classify_finding(status, error, oof, lb, state):
    if status in ("killed", "failed"):
        ...
        return "model_failure"
    if oof is not None and lb is not None and (oof - lb) > 0.02:
        return "overfitting_signal"
    if status == "failed":
        return "feature_failure"
    return "model_failure"  ← hit by every successful run
```

Strategy Agent reads finding_type to understand what worked vs. what failed. A successful run with OOF=0.89 is recorded as `model_failure`, poisoning the knowledge base. Add a positive classification:

```python
if status in ("validated", "submitted", "complete", "done"):
    return "model_improvement" if (best_oof is None or oof > best_oof) else "model_result"
```

---

### ⚠️ `_load_parent_script_source` first branch is dead code

**File:** [gladius/utils/context_builder.py](../gladius/utils/context_builder.py)

```python
parent_version = directive.get("parent_version", "")
if parent_version:
    versioned_script = SCRIPTS_DIR / f"{parent_version}.py"
    ...
```

Strategy Agent sets the `directive` with fields: `directive_type`, `target_model`, `rationale`, `exploration_flag`, `priority`. It never sets `parent_version` — that lives in the `ExperimentSpec` produced by Hypothesis Generator. `build_hypothesis_context` is called with the directive (before the spec exists), so `directive.get("parent_version")` is always `""`. The first branch is dead code. Either remove it or add `parent_version` to the directive schema.

---

### ⚠️ LB Tracker parses Kaggle CSV by hardcoded column index

**File:** [gladius/nodes/strategy/lb_tracker.py](../gladius/nodes/strategy/lb_tracker.py)

```python
return float(parts[4])  # assumes publicScore is column index 4
```

The Kaggle CLI `competitions submissions -v` CSV header is:
`fileName,date,description,status,publicScore,privateScore`

`publicScore` is currently at index 4, but the Kaggle CLI has changed its output format before. Use the Kaggle Python API (`kaggle.api.competitions_submissions_list`) for structured access, or at minimum parse the header row to find the `publicScore` column index.

---

## Low priority / known open items

### `_apply_feature_remove` whole-line deletion is also missing comment lines

If a feature was added as a code block (function + docstring + multiple lines), `_apply_feature_remove` deletes only lines containing the feature name string, leaving orphan docstrings and blank lines. An AST-based approach that deletes entire FunctionDef nodes would be correct.

### `tempfile` imported but unused in code_reviewer

`import tempfile` at the top of `code_reviewer.py` — the sandbox runs the actual script, not a temp copy. Remove the import.

### Versioning Agent version numbering is count-based, not max-based

```python
version_num = len(existing) + 1
```

If any `v*.json` file is deleted manually, the numbering collides with existing files. Use `max + 1` instead:

```python
version_num = max((int(f.stem[1:]) for f in existing if f.stem[1:].isdigit()), default=0) + 1
```

### GPU monitoring absent in Resource Manager

`resource_manager_node` monitors only disk (no RAM, no GPU). The watchdog monitors only RAM. GPU VRAM OOM is a common cause of training script failure in Kaggle competitions and produces a confusing CUDA error rather than a clean exit code.

### Leakage check still missing in Validation Agent

Per-fold OOF score comparison requires a `{run_id}_fold_{k}_oof.npy` interface contract. This must also be enforced in the default script template. Not yet documented as a training script contract.

### Tool/dependency manifest not implemented

Generated scripts can introduce arbitrary imports. No approved-package whitelist, no `# GLADIUS_DEPS:` parsing, no auto-install. Related: `_check_imports` will flag any package not installed in the venv at review time, which is the right behavior only if the venv is the runtime environment.

### Vector database / code index not implemented

Without semantic code search, ContextBuilder sends entire scripts to the LLM. `_apply_llm_correction` in the no-line-number branch sends the full script for every import fix. ChromaDB (on-disk, no server) with AST-chunked functions is the minimum.

### Conductor / State Manager / Supervisor not implemented

Conductor is a pass-through stub. No agent restart-on-crash capability. Required for unattended multi-day runs.

### `_notify_telegram` in Error Handler is a no-op

After 3 consecutive failures on a node, `_notify_telegram(msg, node)` is called but does nothing. `notifier._send_telegram` is fully implemented and available. Intentionally deferred.

---

## Corrected priority list (revision 3)

| # | Item | Effort | Severity |
|---|---|---|---|
| 1 | **Fix OOF submission gate**: add `best_submitted_oof` field, split semantics | 30 min | 🔴 Critical — nothing ever gets submitted |
| 2 | **Fix `graph.add_edge("validation_agent", …)`**: replace with conditional edge | 10 min | 🔴 Critical — validation failures never extracted |
| 3 | **Fix `graph.add_edge("ensemble_agent", …)`**: replace with conditional edge | 10 min | 🔴 Critical — ensemble routing broken |
| 4 | **Fix `_auc_roc` O(n²) memory**: rank-based implementation | 20 min | 🔴 Critical — OOM on any real dataset |
| 5 | **Wire Resource Manager** into graph (versioning → resource_manager → executor) | 20 min | 🟠 High — disk cleanup never runs |
| 6 | **Fix `_apply_param_change`** regex to use `^(\s*param\s*=)` multiline | 20 min | 🟠 High — corrupts real scripts |
| 7 | **Fix `_apply_feature_remove`** to use word-boundary regex | 15 min | 🟠 High — silent substring corruption |
| 8 | **Handle `ensemble` directive in hypothesis/code_generator** | 2 h | 🟠 High — ensemble always produces broken scripts |
| 9 | **Fix SPEC_SCHEMA `"v41"` placeholder** | 5 min | 🟠 High — LLM copies literal version |
| 10 | **Fix `_classify_finding`** to not return `model_failure` for successful runs | 15 min | 🟡 Medium — poisons knowledge base |
| 11 | **Fix `.exitcode` race**: poll with ~2s retry before fallback | 15 min | 🟡 Medium — OOM crash misrouted as failure |
| 12 | **Fix executor log `"w"` → `"a"`** | 5 min | 🟡 Medium — loses log on resume |
| 13 | **Fix LLM correction fallback** to also use targeted context for import errors | 1 h | 🟡 Medium — full-script JSON escaping issues |
| 14 | **Deduplicate `_check_imports`**: use `code_reader.check_imports` in reviewer | 10 min | 🟡 Medium — divergent implementations |
| 15 | **Fix `_load_parent_script_source` dead branch** / document directive schema | 15 min | 🟡 Medium — misleading code |
| 16 | **Fix versioning count-based numbering** → max-based | 10 min | 🟡 Medium — collision on deletion |
| 17 | **Remove unused `import tempfile`** from code_reviewer | 2 min | 🟢 Low — lint noise |
| 18 | **LB Tracker**: parse CSV header instead of hardcoded `parts[4]` | 30 min | 🟡 Medium — breaks on CLI update |
| 19 | **Add sandbox resource limits** (ulimit / `resource.setrlimit`) | 1 h | 🟡 Medium — OOM before timeout |
| 20 | **Leakage check + per-fold OOF contract** | 2 h | 🟡 Medium — overfitting risk |
| 21 | **LibCST for all code patching** | 3 h | 🟡 Medium — correct long-term fix for items 6/7/8 |
| 22 | **GPU monitoring in Resource Manager / Watchdog** | 2 h | 🟢 Low |
| 23 | **Tool/dependency manifest** | 2 h | 🟢 Low |
| 24 | **Vector index (ChromaDB, AST-chunked)** | 4 h | 🟢 Low |
| 25 | **Conductor / Supervisor** | ~1 week | 🟢 Low |


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
