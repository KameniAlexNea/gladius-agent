# Code Review: Implementation vs. Design (`agentic-framework.md`)

> Reviewed: 2026-02-24  
> Based on `agentic-framework.md` v2 and the current codebase (`gladius/`).

---

## Summary

The codebase is a solid scaffold. The graph wiring, state definition, and most agent logic are faithful to the design. The critical path (strategy → hypothesis → code generation → review → execute → validate → submit → LB track) is end-to-end connected and non-trivially implemented. However, several production-critical requirements from the spec are either stubs or absent.

| Area | Status |
|---|---|
| Graph structure & routing | ✅ Complete |
| State definition | ✅ Complete |
| Strategy Agent (LLM + exploration logic) | ✅ Complete |
| Hypothesis Generator (LLM + spec schema) | ✅ Complete |
| Code Generator (targeted patching) | ⚠️ Partial — no LLM for complex changes |
| Code Reviewer (static analysis) | ⚠️ Partial — missing import check + sandbox |
| Versioning Agent | ✅ Complete |
| Executor | ✅ Complete |
| Watchdog | ✅ Complete |
| Resource Manager | ⚠️ Partial — no GPU, no running-job protection |
| Validation Agent | ⚠️ Partial — missing leakage + metric consistency |
| Submission Decider | ⚠️ Partial — missing `best_oof` compare + 2h buffer |
| Submission Agent | ✅ Complete |
| LB Tracker | ✅ Complete |
| Ensemble Agent | ⚠️ Partial — no weight optimisation |
| Knowledge Extractor | ✅ Complete |
| Notifier | ✅ Complete |
| Error Handler | ✅ Complete |
| Conductor | ❌ Stub only |
| State Manager | ❌ Not implemented |
| Supervisor | ❌ Not implemented |
| ContextBuilder (LLM context quality) | ⚠️ Partial — thin context |
| Checkpointing (SqliteSaver) | ❌ MemorySaver used instead |
| Parallel execution (Send API) | ❌ Not implemented |

---

## Issue Detail

### ❌ CRITICAL — Checkpointer is MemorySaver, not SqliteSaver

**File:** [gladius/graph.py](../gladius/graph.py)

The design spec is explicit:
> "State persistence is handled by LangGraph's `SqliteSaver` checkpointer. If the process crashes, the graph resumes from the last checkpoint — no custom recovery code needed."

The implementation uses `MemorySaver`, which loses all state on process exit. The `main()` function does `Path("state").mkdir(exist_ok=True)` (suggesting awareness of the state dir) but never uses it for checkpointing.

**Fix:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("state/checkpoint.db")
```

---

### ❌ CRITICAL — State Manager is absent

**Spec:** Agent #2 — all reads/writes route through State Manager with TTL-gated locking. No agent writes directly.

**Reality:** Every node returns a partial dict and LangGraph merges it. This is fine for concurrency-safe single-threaded graph execution, but there is no write validation, no TTL locking, and no stale-lock detection.

**Impact:** Low for sequential graph execution. High if parallel `Send`-based execution is added later, which the spec anticipates. The missing State Manager is a prerequisite for parallel slots.

---

### ❌ CRITICAL — Supervisor is absent

**Spec:** Agent #3 — separate process, monitors agent PIDs, restarts crashed agents, marks agents `degraded` after 3 crashes in 10 min, notifies human, Conductor reroutes around degraded agents.

**Reality:** Not implemented at all. The Error Handler provides node-level retry (3 attempts), but there is no PID tracking of agent processes or automatic restart logic.

---

### ❌ Conductor is a stub

**File:** [gladius/nodes/orchestration/conductor.py](../gladius/nodes/orchestration/conductor.py)

```python
def conductor_node(state: GraphState) -> GraphState:
    """High-level conductor that monitors overall competition progress."""
    return {"next_node": state.get("next_node", "strategy")}
```

The spec requires: per-tick health checks via Supervisor, running-job status checks, LB poll status, queue state inspection. None of this is implemented. Additionally, `conductor_node` is not wired into the graph (no edges lead to it in `graph.py`).

---

### ⚠️ Code Generator — no LLM for complex changes

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

The design draws a sharp line: simple changes (param swap, feature add/remove) use regex/AST; **complex changes** (joins, new data sources, multi-step transforms) call the LLM with ±20 lines of context.

The implementation handles all three change types without an LLM. For `feature_add`, it naively appends a `code_snippet` string — but that snippet has to come from somewhere (the Hypothesis Generator schema doesn't include `code_snippet` generation via LLM either). There is no fallback LLM path, so complex feature engineering will silently produce broken scripts.

**Specific risk:** `_apply_feature_add` splices a raw `code_snippet` string at a marker with no validation. If the snippet references columns or imports that don't exist, Code Reviewer's regex-level checks won't catch it.

---

### ⚠️ Code Reviewer — missing import check and sandbox run

**File:** [gladius/nodes/code/code_reviewer.py](../gladius/nodes/code/code_reviewer.py)

The spec lists four checks:
1. `py_compile` ✅
2. `pylint --errors-only` ✅
3. Import check against installed packages ❌
4. Sandbox run on 1% data with 60s timeout ❌
5. No hardcoded paths ✅

Missing #3: `importlib.util.find_spec` on every top-level import in the script would catch missing dependencies before execution. This is a common failure mode when Code Generator adds a new library.

Missing #4: A 1%-sample sandbox run is the only reliable way to catch shape errors, missing columns, and runtime exceptions before a full training run. Without it, the first real failure mode surfaces in the Watchdog 10–60 min later.

---

### ⚠️ Submission Decider — incomplete gating logic

**File:** [gladius/nodes/validation/submission_decider.py](../gladius/nodes/validation/submission_decider.py)

The spec defines 5 conditions (truncated in the doc, but condition 1 is clearly stated):
1. `new_oof > best_oof + min_improvement` — **missing**: there is no `best_oof` field in `GraphState` and no comparison
2. Daily budget ✅
3. OOF-LB gap not widening ✅
4. (truncated in spec)
5. `time_since_last_submission > 2h` — **missing**

The `MIN_OOF_IMPROVEMENT = 1e-4` constant is defined but never used. The decider will submit any run that passes budget and gap checks regardless of whether OOF actually improved.

**Fix:**
```python
# GraphState needs:  best_oof: Optional[float]
best_oof = state.get("best_oof") or 0.0
if oof_score <= best_oof + MIN_OOF_IMPROVEMENT:
    return {"experiment_status": "held", "next_node": "router"}

import time
last_sub = state.get("last_submission_time") or 0
if time.time() - last_sub < 7200:
    return {"experiment_status": "held", "next_node": "router"}
```

---

### ⚠️ Validation Agent — missing leakage check and metric consistency

**File:** [gladius/nodes/validation/validation_agent.py](../gladius/nodes/validation/validation_agent.py)

Spec requirements:
1. Shape, no NaN, values in `[0,1]` ✅
2. Metric matches logged score ±1e-4 ❌ — `oof_score` in state is never verified against recomputing the metric from the OOF array
3. Leakage check: OOF score on train folds must not significantly exceed holdout (threshold 0.01 AUC) ❌

The leakage check requires per-fold OOF arrays, which means the training script must save them separately. This is an interface contract that doesn't yet exist anywhere in the codebase.

---

### ⚠️ Ensemble Agent — missing blend weight optimisation

**File:** [gladius/nodes/strategy/ensemble_agent.py](../gladius/nodes/strategy/ensemble_agent.py)

The spec says: "Proposes blends using **Nelder-Mead or optuna** on OOF weights."

The implementation selects uncorrelated models correctly but dispatches to Hypothesis Generator with equal-weight intent. No optimisation is performed. The blend directive contains `base_model_paths` but no `weights` field, so the Hypothesis Generator would have to invent weights — which it's not prompted to do.

---

### ⚠️ ContextBuilder — thin LLM context

**File:** [gladius/utils/context_builder.py](../gladius/utils/context_builder.py)

The spec (Section 8) is specific about what context the Strategy Agent receives:
- Top 10 experiments by OOF score (version, model_type, key_params, OOF, LB, gap) ❌
- Last 5 experiments in chronological order ❌
- knowledge.json (last 20 entries) ✅ (last 20 is an approximation)
- session_summary ✅ (present but never populated — `session_summary` is always `None`)
- Current best submission score and gap to estimated LB top ✅ (partially — gap to LB top is not computed)
- Experiment type distribution: how many CatBoost/LGB/XGB/NN/ensemble ❌
- exploration_budget (fraction of remaining time) ❌

The Hypothesis Generator context is similarly thin — it receives the directive and relevant knowledge but not the parent script source, which the spec says is required ("full source code of the best current script of the target model type").

**Impact:** The LLM is operating with less signal than the design intended. Strategy decisions will be less grounded, and Hypothesis Generator will not know what it's modifying.

---

### ⚠️ Resource Manager — no GPU monitoring, weak running-job protection

**File:** [gladius/nodes/execution/resource_manager.py](../gladius/nodes/execution/resource_manager.py)

The spec: "Monitors GPU mem, disk, all PIDs."

- GPU memory: not monitored (no `nvidia-smi` or `pynvml` call)
- Running-job protection: uses `state.get("run_id")` as the only protected key. This protects the _current_ versioned run but not other running processes if parallel slots are ever used. The spec says "never evict caches referenced by current best run or any RUNNING job."

---

### ⚠️ `session_summary` never populated

**File:** [gladius/state.py](../gladius/state.py) / [gladius/utils/context_builder.py](../gladius/utils/context_builder.py)

`session_summary` exists in `GraphState` and is included in the strategy context dict, but no node ever writes to it. The design says it's a "compressed strategy history" — presumably updated by Knowledge Extractor or Strategy Agent after each cycle. Without it, the LLM has no memory of the session's strategic direction beyond `knowledge.json`.

---

### ⚠️ `graph.py` — router used as both node and routing function

**File:** [gladius/graph.py](../gladius/graph.py)

```python
graph.add_conditional_edges(
    "router",
    router_node,          # <-- routing function
    { ... }
)
```

`router_node` is registered as both a node (`graph.add_node("router", router_node)`) and the conditional edge selector for that same node. In LangGraph this means the node function is called once (as a node, updating state) and then called again (as the edge selector, returning a string). Because `router_node` returns a string (the target node name) — not an updated state dict — the node invocation returns `None` and LangGraph merges nothing, which is harmless but unintentional. The correct pattern is: the node updates `state["next_node"]`, and the edge selector reads `state["next_node"]`.

**Fix:**
```python
# router_node should remain a plain conditional function (returns str)
# Remove it from add_node, or make it return a state update dict
# and use a separate lambda as the edge selector:
graph.add_conditional_edges(
    "router",
    lambda s: s.get("next_node", "strategy"),
    { ... }
)
```

---

### Minor issues

**`_notify_telegram` in error_handler.py is a no-op**  
[gladius/nodes/error_handler.py](../gladius/nodes/error_handler.py) — `_notify_telegram` is defined as `pass`. Escalation events will be silent. Should call `notifier_node` or directly call `_send_telegram` from `notifier.py`.

**`best_oof` not tracked in GraphState**  
There is no `best_oof` field in `GraphState`. The Submission Decider needs it but falls back to always treating the new OOF as an improvement. Add `best_oof: Optional[float]` to `GraphState` and update it in Knowledge Extractor or Submission Agent on confirmed improvement.

**Watchdog uses `_get_exit_code(pid)` which calls `psutil.Process(pid).wait()`**  
[gladius/nodes/execution/watchdog.py](../gladius/nodes/execution/watchdog.py) — `wait()` is only valid for child processes. If the executor and watchdog run in separate threads/processes (as the design intends), `wait()` will raise `psutil.AccessDenied`. Use `proc.returncode` via subprocess tracking or a shared state file instead.

**`code_retry_count` is incremented but never reset on a new directive**  
[gladius/nodes/code/code_reviewer.py](../gladius/nodes/code/code_reviewer.py) — If a new directive is issued after a code failure, `code_retry_count` carries over from the previous attempt. Strategy Agent or Hypothesis Generator should reset it.

**LB Tracker parses Kaggle CSV by hardcoded column index**  
[gladius/nodes/strategy/lb_tracker.py](../gladius/nodes/strategy/lb_tracker.py) — `parts[4]` assumes the Kaggle CLI output format is stable and that `run_id` matches a field in the submission list. Kaggle CLI output format has changed before. Prefer using the `kaggle` Python API or parsing the header row.

---

## What to prioritise next

1. **SqliteSaver** — one-line fix, un-blocks crash recovery.
2. **`best_oof` in GraphState + Submission Decider OOF comparison** — without this, the system submits every validated run regardless of quality.
3. **ContextBuilder: load experiment archive + parent script** — the LLM agents are operating blind without prior run results and the code they're supposed to modify.
4. **Code Reviewer: import check + sandbox run** — catches the most common Code Generator failures cheaply.
5. **Router pattern fix** — low effort, avoids a subtle LangGraph anti-pattern.
6. **`session_summary` population** — populate in Knowledge Extractor as a rolling compressed string.

---

## Deep Review: Code Handling Layer

This section covers the most structurally weak area of the codebase: how scripts are searched, modified, validated, and executed. The current implementation will fail on any real competition script.

---

### ❌ Code patching is string manipulation on an unstructured source

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py)

The three patch functions fail in predictable ways on real ML scripts:

#### `_apply_param_change` — regex on assignment is wrong

```python
pattern = rf"({re.escape(param)}\s*=\s*)[^\n,\)]*"
```

This regex will match **any** assignment to that variable name anywhere in the file — including commented-out lines, print statements, local variables in different functions, and dict keys. Given a CatBoost script with:

```python
params = {
    "learning_rate": 0.05,       # will match
    ...
}
# learning_rate = 0.1            # will also match (commented out)
def set_learning_rate(learning_rate=0.05): ...   # will also match
```

The regex rewriter produces broken code silently. The right tool is `libcst` (LibCST — a concrete syntax tree library that preserves formatting) or `ast` + `astor`. LibCST allows:

```python
import libcst as cst

class ParamRewriter(cst.CSTTransformer):
    def leave_Assign(self, node, updated):
        # target-aware, scope-aware replacement
        ...
```

This preserves formatting, handles nested dicts, and raises on ambiguous matches instead of silently corrupting code.

#### `_apply_feature_add` — depends on a magic marker that won't exist

```python
marker = "# --- FEATURES END ---"
if marker in script:
    return script.replace(marker, f"{snippet}\n{marker}")
return script + f"\n{snippet}"   # fallback: append to end of file
```

Real competition scripts won't have `# --- FEATURES END ---`. The fallback — appending to the end of the file — will produce code with feature engineering after `if __name__ == "__main__":`, which is syntactically valid but logically dead code. There is no check that the appended snippet actually compiles in context, or that it references only columns and variables that exist at the insertion point.

The correct approach: identify the `train()` function's feature-building block via AST, find the last statement before the model call, and insert there. LibCST lets you do this structurally without string hacking.

#### `_apply_feature_remove` — deletes by substring match on every line

```python
filtered = [line for line in lines if feature_name not in line]
```

If `feature_name = "age"`, this deletes every line containing the string `"age"` — including `"message"`, `"average"`, `"stage"`, and `"page_views"`. In a 300-line script this will silently corrupt the file. It also has no concept of scope: removing a feature that is referenced in five places downstream produces `NameError` at runtime, not at generation time.

Correct approach: AST-walk to find all references to the feature name, remove the assignment, and either remove or replace all downstream references (or reject the change if it's too entangled).

---

### ❌ No vector database — code search is impossible

The design in Section 7 says: "the LLM receives the exact lines around the insertion point (±20 lines)." There is no mechanism to locate that insertion point.

For a 500-line LightGBM training script, how does the system answer: "Where is the feature engineering block?", "Where are the model hyperparameters defined?", "What features are currently in the feature list?", "Has a similar feature been added before in a previous version?"

Without a code index, the system has two bad options:
1. Send the entire script to the LLM (expensive, hits context limits, noise-heavy).
2. Guess by string search (unreliable on real scripts).

**What's needed: a chunked vector index over the script corpus.**

Concretely:

```
state/
  code_index/            # vector database (e.g., ChromaDB on-disk)
    chunks/              # per-function, per-class, per-block embeddings
```

Index structure:
- Each chunk = one logical block (function body, dict literal, import block) extracted by AST
- Each chunk embedded with a code embedding model (e.g., `text-embedding-3-small` or a local `nomic-embed-code`)
- Metadata: `{version, file_path, start_line, end_line, chunk_type, function_name}`

Usage in Code Generator:
```python
# Before: send entire 500-line file
# After: query by intent
chunks = code_index.query("hyperparameter dict for LightGBM", top_k=3)
context_lines = load_lines(chunks[0].file_path, chunks[0].start_line - 20, chunks[0].end_line + 20)
```

Usage in ContextBuilder (for Hypothesis Generator):
```python
# Find the best-performing feature similar to what's proposed
similar_features = code_index.query(f"feature: {proposed_feature_name}", top_k=5, filter={"chunk_type": "feature_block"})
```

Usage in Knowledge Extractor:
```python
# Deduplicate: has this feature been tried before?
existing = code_index.query(proposed_snippet, top_k=1)
if existing[0].score > 0.95:
    finding["conclusion"] += " (duplicate of prior attempt)"
```

Recommended library: **ChromaDB** (`chromadb`) — embeds locally, persists to disk, no server needed, has Python API. Alternatively **FAISS** + manual metadata store for lower overhead.

---

### ❌ No mechanism to read a specific block or function from a script

There is no utility in `utils/` to answer: "give me the body of the `train()` function from `v41.py`", or "give me lines 120–160 from `v41.py`".

`ContextBuilder.build_hypothesis_context` is supposed to include the parent script source, but it doesn't. Even if it did, it would send the entire file to the LLM. For a real competition script this is wasteful and unreliable.

**What's needed: a code reader utility using AST.**

```python
import ast
from pathlib import Path

def read_function(path: str, func_name: str) -> str:
    """Extract source lines for a named function."""
    source = Path(path).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise KeyError(f"{func_name} not found in {path}")

def read_lines(path: str, start: int, end: int) -> str:
    lines = Path(path).read_text().splitlines()
    return "\n".join(lines[start - 1 : end])

def list_functions(path: str) -> list[str]:
    tree = ast.parse(Path(path).read_text())
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
```

This belongs in `utils/code_reader.py` and should be the primary tool Code Generator and ContextBuilder use to extract context.

---

### ❌ No in-memory code execution — every validation requires a file

The Code Reviewer currently writes the script to `state/scripts/{run_id}.py` and runs `pylint` on it. The sandbox run (missing, but required) would also use that same file. There is no way to:
- Run a quick smoke check without writing to disk
- Execute a snippet to verify it compiles and runs in the current Python environment
- Test a single function in isolation

**For syntax validation, no file is needed:**

```python
import ast

def validate_syntax(code: str) -> list[str]:
    try:
        ast.parse(code)
        return []
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]
```

**For import validation, no file is needed:**

```python
import importlib.util

def check_imports(code: str) -> list[str]:
    tree = ast.parse(code)
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if importlib.util.find_spec(top) is None:
                    missing.append(f"import not found: {top}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if importlib.util.find_spec(top) is None:
                missing.append(f"import not found: {top}")
    return missing
```

**For the 1%-data sandbox, a temporary file is unavoidable — but it should be ephemeral:**

```python
import tempfile, subprocess, sys

def sandbox_run(code: str, timeout: int = 60) -> tuple[bool, str]:
    """Run code string in a subprocess using a temp file. File is deleted after."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=True) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            [sys.executable, f.name],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "GLADIUS_SAMPLE_FRACTION": "0.01"},
        )
    return result.returncode == 0, (result.stdout + result.stderr).strip()
```

The training script needs to respect `GLADIUS_SAMPLE_FRACTION` — this is an interface contract that must be enforced in the Code Generator's default template and in every generated script.

**For running a single function in isolation (import check + shape validation):**

```python
def exec_isolated(code: str, entry: str = "main") -> tuple[bool, str]:
    """Execute code in an isolated namespace, call entry()."""
    namespace = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        if entry in namespace:
            namespace[entry]()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
```

`exec()` in an isolated namespace (not `globals()`) is safe for shape checks and import validation. It is **not** safe for untrusted code — but in this system, the code is LLM-generated for a known ML pipeline, so the threat model is controlled.

---

### ❌ No tool / dependency management layer

The system has no concept of which Python packages a generated script requires, and no mechanism to ensure they are installed before execution.

Failure scenario:
1. Hypothesis Generator proposes adding a `polars`-based feature (reasonable for large tabular data)
2. Code Generator adds `import polars as pl` to the script
3. Code Reviewer's import check is missing (already noted), so it passes
4. Executor launches the script
5. `ModuleNotFoundError: No module named 'polars'` surfaces in the Watchdog log 5 minutes later
6. Knowledge Extractor records a `model_failure` finding — which is wrong, it's a dependency failure

**What's needed: a manifest + auto-install layer.**

Each generated script should emit a machine-readable requirements comment at the top (or a sidecar `.deps` file):

```python
# GLADIUS_DEPS: lightgbm>=4.0, polars>=0.20, scikit-learn>=1.3
```

The Code Reviewer reads this block and checks each package against `importlib.util.find_spec`. If a package is missing, two paths:
- **Auto-install**: `pip install {package}` in a subprocess, then re-run the import check. Only allowed for a whitelist of safe packages.
- **Reject + reclassify**: return `next_node=hypothesis` with `reviewer_feedback="missing dependency: polars"`, so the Hypothesis Generator is told to use an available alternative.

The whitelist approach is safer and avoids the system silently installing arbitrary packages during competition runs. The whitelist should live in `competition.json` or a `tools.json` config.

---

### ❌ No diff / patch format — changes are opaque strings

When the Hypothesis Generator produces a `changes` array like:

```json
{"type": "feature_add", "feature_name": "feat_age_x_income", "code_snippet": "df['feat_age_x_income'] = df['age'] * df['income']"}
```

There is no way to:
- Verify the diff is minimal (didn't accidentally overwrite other features)
- Reverse the change if the run fails (rollback to parent)
- Display a human-readable diff in the Telegram notification
- Detect that the same change has been applied before (deduplication)

**What's needed: a unified diff representation.**

Instead of a free-form `code_snippet`, the Hypothesis Generator should produce a unified diff against the parent script:

```python
import difflib

def apply_diff(original: str, diff_str: str) -> str:
    """Apply a unified diff string to the original source."""
    ...

def compute_diff(original: str, modified: str, from_version: str, to_version: str) -> str:
    return "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=from_version,
        tofile=to_version,
    ))
```

With this, Versioning Agent stores the diff alongside the full script, Knowledge Extractor can describe what changed precisely, and the Notifier can include a short diff summary in the Telegram message.

---

### ❌ `run_id` is reused between code generation and versioning

**File:** [gladius/nodes/code/code_generator.py](../gladius/nodes/code/code_generator.py) + [gladius/nodes/code/versioning_agent.py](../gladius/nodes/code/versioning_agent.py)

Code Generator writes to `state/scripts/{run_id}.py` using `state.get("run_id", "run_001")`. But `run_id` at generation time is whatever was left over from the previous experiment — it's not set to the new version until Versioning Agent runs. This means:

1. Code Generator writes `state/scripts/v41.py` (the old `run_id`)
2. Versioning Agent assigns `v42`
3. The script path in state still points to `v41.py`
4. A second generation attempt overwrites `v41.py` and corrupts the previous version

Versioning Agent does copy the script to a new versioned path, but the Code Generator shouldn't use the old `run_id` at all. It should write to a staging path (`state/scripts/pending.py`) and let Versioning Agent be the one to assign the version tag and rename.

```python
STAGING_PATH = Path("state/scripts/pending.py")

def code_generator_node(state):
    ...
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGING_PATH.write_text(modified)
    return {"generated_script_path": str(STAGING_PATH), "next_node": "code_reviewer", "code_retry_count": 0}
```

---

### ❌ Code Generator has no LLM correction loop

When Code Reviewer returns `reviewer_feedback`, the graph routes back to `hypothesis_node`. But the Hypothesis Generator re-generates a new spec from scratch — it does **not** receive the reviewer feedback as a correction prompt.

The spec says:
> "The rejection message (specific: 'line 47: hardcoded path', 'import X not installed') is fed back to the LLM as a correction prompt. Max 2 retries."

The correction should go back to **Code Generator** (fix the same script), not Hypothesis Generator (which generates a new spec). The flow should be:

```
code_reviewer FAIL (retry < 2) → code_generator (with reviewer_feedback in state)
code_reviewer FAIL (retry == 2) → hypothesis (simplify spec)
code_reviewer FAIL (retry == 3) → strategy (give up on this directive)
```

Currently there is no "send feedback to code_generator" path at all. The Code Generator node ignores `reviewer_feedback` entirely.

---

### ❌ No deduplication of generated features

There is no check that the feature proposed by Hypothesis Generator doesn't already exist in the parent script, in a prior failed experiment, or in the current script after patching.

Common LLM failure mode: the model proposes `df['ratio_a_b'] = df['a'] / df['b']` repeatedly across different directives because it has no memory of having generated it before — especially once the `session_summary` gap is left unfilled.

**Fix**: before `_apply_feature_add`, check whether `feature_name` already appears in the parent script via `ast.walk` on the assignment targets. If it does, return a `no-op` or flag it as a duplicate in `reviewer_feedback`.

---

## Revised Priority List

| Priority | Fix | Effort |
|---|---|---|
| 1 | SqliteSaver checkpointer | 5 min |
| 2 | `best_oof` in state + Submission Decider OOF gate | 30 min |
| 3 | Code Generator: staging path (not `run_id`) | 15 min |
| 4 | Code Generator: reviewer feedback correction loop | 1 h |
| 5 | Code Reviewer: AST-based import check (no temp file) | 1 h |
| 6 | Code Reviewer: sandbox run via temp file + `GLADIUS_SAMPLE_FRACTION` | 2 h |
| 7 | Code Generator: replace regex patching with LibCST | 2–3 h |
| 8 | `utils/code_reader.py`: AST function/block extractor | 1 h |
| 9 | ContextBuilder: load top-10 experiment archive + parent script body | 2 h |
| 10 | ContextBuilder: experiment type distribution + exploration_budget | 1 h |
| 11 | Tool whitelist + `GLADIUS_DEPS` manifest in generated scripts | 2 h |
| 12 | Diff-based patching + diff stored in Versioning Agent | 2 h |
| 13 | Vector index (ChromaDB) for code chunk retrieval | 3–4 h |
| 14 | Feature deduplication check before `_apply_feature_add` | 1 h |
| 15 | `session_summary` population in Knowledge Extractor | 1 h |
| 16 | Ensemble Agent: Nelder-Mead weight optimisation on OOF | 2 h |
| 17 | Router pattern fix (node vs. edge selector) | 30 min |
