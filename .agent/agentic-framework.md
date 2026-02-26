# Gladius Agent — Claude Agent SDK Framework

**Revision**: 1.0  
**Status**: Design / Migration Plan  
**Replaces**: LangGraph StateGraph architecture (see `code-review.md` for why it was discarded)

---

## 1. Why Replace LangGraph

The LangGraph approach required:
- A hand-written Python function for every agent node
- A hand-written routing function for every conditional edge
- A TypedDict schema that had to be kept in sync with every node
- Custom JSON-schema validation inside `call_llm()` wrappers

Every refactor touched all four layers. Critical bugs were introduced in each session because the routing edges were disconnected from the nodes (see review v3: `validation_agent→submission_decider` and `ensemble_agent→hypothesis` both used `add_edge` instead of `add_conditional_edges`). The `best_oof` gate was permanently broken, the `_auc_roc` was O(n²), and the `resource_manager` node was never wired.

The core issue: a Kaggle competition pipeline is inherently complex, multi-step, and tool-heavy. Every LangGraph node was basically just wrapping an LLM call that should also do file I/O, run subprocess commands, and make decisions. These are exactly the things Claude Code is built to handle autonomously.

**Claude Agent SDK gives us:**
- Each "agent" is a full autonomous agent with built-in tools (Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch)
- No hand-written tool loop — Claude handles multi-step execution internally
- Structured JSON output via `output_format={"type":"json_schema","schema":{...}}`
- Subagents via `AgentDefinition` in `ClaudeAgentOptions.agents`
- Parallel execution via `asyncio.gather()`
- Session continuity via `resume=session_id`
- Custom tools via `@tool` decorator + `create_sdk_mcp_server()`
- Hooks for pre/post tool auditing, blocking, and permission management

---

## 2. Architecture Overview

```
gladius/
├── orchestrator.py          # Main coroutine: drives the competition loop
├── state.py                 # SQLite-backed state (replaces TypedDict + SqliteSaver)
├── agents/
│   ├── strategy.py          # StrategyAgent, HypothesisGenerator, KnowledgeExtractor
│   ├── code.py              # CodeAgent (generate + review + version in one agent)
│   ├── execution.py         # ExecutionAgent (run experiments, manage resources)
│   ├── validation.py        # ValidationAgent, SubmissionDecider
│   └── ensemble.py          # EnsembleAgent
├── tools/
│   ├── kaggle_tools.py      # @tool wrappers for Kaggle API (submit, leaderboard)
│   ├── metric_tools.py      # @tool wrappers for OOF scoring, AUC-ROC, etc.
│   └── memory_tools.py      # @tool wrappers for ChromaDB similarity search
└── config.py                # AgentConfig dataclass: model, budget, paths, etc.
```

The `orchestrator.py` is the only place that knows about the competition loop flow. Agents are pure `async def` functions that call `query()` or use `ClaudeSDKClient`. The orchestrator calls them, reads their structured JSON output, and decides what to do next — it is the "graph" but written as clean Python `if/elif/await` logic instead of LangGraph edge declarations.

---

## 3. State Management

Replace `GraphState` TypedDict + SqliteSaver with a `StateStore` backed by SQLite.

```python
# gladius/state.py
import sqlite3
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class CompetitionState:
    # Competition context
    competition_id: str
    data_dir: str
    output_dir: str
    target_metric: str          # "auc_roc", "rmse", etc.
    metric_direction: str       # "maximize" / "minimize"

    # Progress tracking
    iteration: int = 0
    max_iterations: int = 20
    phase: str = "strategy"     # strategy | coding | execution | validation | ensemble | done

    # Best known performance
    best_oof_score: float = -1.0
    best_submission_score: float = -1.0
    best_submission_path: Optional[str] = None
    submission_count: int = 0
    max_submissions_per_day: int = 5

    # Hypothesis and strategy memory
    current_hypothesis: Optional[str] = None
    completed_hypotheses: list = field(default_factory=list)
    failed_hypotheses: list = field(default_factory=list)

    # Experiment registry
    experiments: list = field(default_factory=list)  # [{path, oof, params, notes}]

    # Session IDs for resumable agents
    strategy_session_id: Optional[str] = None
    code_session_id: Optional[str] = None

    # Error tracking
    consecutive_errors: int = 0
    error_log: list = field(default_factory=list)

    # LB tracking
    lb_scores: list = field(default_factory=list)  # [{score, timestamp, public_lb}]


class StateStore:
    def __init__(self, db_path: str = ".gladius/state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save(self, state: CompetitionState):
        data = json.dumps(asdict(state))
        self.conn.execute(
            "INSERT OR REPLACE INTO state(key, value) VALUES ('current', ?)", (data,)
        )
        self.conn.commit()

    def load(self) -> Optional[CompetitionState]:
        row = self.conn.execute(
            "SELECT value FROM state WHERE key='current'"
        ).fetchone()
        if row:
            return CompetitionState(**json.loads(row[0]))
        return None
```

---

## 4. Agent Definitions

### 4.1 Core Pattern

Every agent follows this pattern:

```python
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
import asyncio

async def run_agent(
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    output_schema: dict,
    cwd: str,
    resume: str | None = None,
    mcp_servers: dict | None = None,
) -> tuple[dict, str]:
    """Returns (structured_output, session_id)."""
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        output_format={"type": "json_schema", "schema": output_schema},
        cwd=cwd,
        resume=resume,
        mcp_servers=mcp_servers or {},
    )
    result_msg = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result_msg = message
    if result_msg is None or result_msg.is_error:
        raise RuntimeError(f"Agent failed: {result_msg}")
    return result_msg.structured_output, result_msg.session_id
```

### 4.2 Strategy Agent

Replaces: `strategy_agent`, `hypothesis_generator`, `lb_tracker` (collapsed into one)

```python
# gladius/agents/strategy.py

STRATEGY_SYSTEM_PROMPT = """\
You are an elite Kaggle Grandmaster working on a machine learning competition.
Your role is to analyze the competition, study the leaderboard, examine existing
experiments, and generate the most promising next hypothesis to try.

You have access to:
- The competition data directory (use Read, Glob to explore)
- The experiments log (read .gladius/experiments.json)
- Web search for relevant papers and winning solutions

Always reason about:
1. What has been tried and what worked / failed
2. The current leaderboard gap
3. The most impactful next improvement (data, features, model, ensemble)
"""

STRATEGY_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "changes", "expected_improvement", "rationale", "priority"],
    "properties": {
        "hypothesis": {"type": "string", "description": "One-line hypothesis title"},
        "changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete code/config changes to make"
        },
        "expected_improvement": {"type": "number"},
        "rationale": {"type": "string"},
        "priority": {"enum": ["critical", "high", "medium", "low"]},
        "requires_new_features": {"type": "boolean"},
        "suggested_models": {"type": "array", "items": {"type": "string"}},
    }
}

async def run_strategy_agent(state: CompetitionState, data_dir: str) -> tuple[dict, str]:
    prompt = f"""\
Competition: {state.competition_id}
Target metric: {state.target_metric} ({state.metric_direction})
Current best OOF: {state.best_oof_score}
Current best LB:  {state.best_submission_score}
Iteration: {state.iteration}/{state.max_iterations}

Completed hypotheses: {json.dumps(state.completed_hypotheses[-5:], indent=2)}
Failed hypotheses: {json.dumps(state.failed_hypotheses[-3:], indent=2)}

Analyze the competition data, study what has been tried, and propose the single
most impactful next hypothesis. Read the experiments log at .gladius/experiments.json
and the competition description at {data_dir}/competition_description.md.
"""
    return await run_agent(
        prompt=prompt,
        system_prompt=STRATEGY_SYSTEM_PROMPT,
        allowed_tools=["Read", "Glob", "Grep", "WebSearch"],
        output_schema=STRATEGY_OUTPUT_SCHEMA,
        cwd=data_dir,
        resume=state.strategy_session_id,  # maintains memory across iterations
    )
```

### 4.3 Code Agent

Replaces: `code_generator`, `code_reviewer`, `versioning_agent` (collapsed into one)

The key insight: **Claude Code can generate, review, and version code in a single agent invocation** because it can autonomously read the existing code, write new code, run the linter, fix issues, and commit — all within one `query()` call. Three separate LangGraph nodes become one.

```python
# gladius/agents/code.py

CODE_SYSTEM_PROMPT = """\
You are an expert ML engineer implementing Kaggle competition solutions.
Given a hypothesis, you will:
1. Read the existing solution code (explore the src/ directory)
2. Implement the required changes
3. Write clean, well-documented Python code
4. Run a quick syntax check (python -m py_compile)
5. Save the new solution to a versioned file (e.g., src/solution_v{N}.py)
6. Update .gladius/experiments.json with the new experiment entry

You have full filesystem access and can run bash commands.
Always create atomic, testable experiments. Do not break existing working solutions.
"""

CODE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["solution_path", "changes_made", "estimated_runtime_minutes"],
    "properties": {
        "solution_path": {"type": "string"},
        "changes_made": {"type": "array", "items": {"type": "string"}},
        "estimated_runtime_minutes": {"type": "number"},
        "requires_gpu": {"type": "boolean"},
        "new_dependencies": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    }
}

async def run_code_agent(
    hypothesis: dict,
    state: CompetitionState,
    project_dir: str,
) -> tuple[dict, str]:
    prompt = f"""\
Hypothesis to implement:
{json.dumps(hypothesis, indent=2)}

Project directory: {project_dir}
Current iteration: {state.iteration}

Explore the existing solution (src/ directory), then implement this hypothesis
as a new versioned solution file. The solution must:
- Be self-contained and runnable: python solution_vN.py
- Accept no CLI args; read data from {state.data_dir}
- Write OOF predictions to .gladius/oof_vN.npy
- Write test predictions to .gladius/sub_vN.csv
- Print the OOF score on the last line as: OOF_SCORE: {{score:.6f}}
"""
    return await run_agent(
        prompt=prompt,
        system_prompt=CODE_SYSTEM_PROMPT,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        output_schema=CODE_OUTPUT_SCHEMA,
        cwd=project_dir,
        resume=state.code_session_id,
    )
```

### 4.4 Execution Agent

Replaces: `executor`, `resource_manager`, `watchdog` (collapsed into one)

Claude Code has background bash process support built-in (`run_in_background: True` in Bash tool input). The agent can launch the training script, poll it with `BashOutput`, and terminate it with `KillBash` if the watchdog threshold is exceeded — all autonomously.

```python
# gladius/agents/execution.py

EXECUTION_SYSTEM_PROMPT = """\
You are responsible for executing ML training runs on this machine.
Given a solution script path, you will:
1. Check available GPU/CPU resources before starting (nvidia-smi, free -h)
2. Launch the training script as a background bash process
3. Monitor it: check output every 60 seconds using BashOutput
4. If it exceeds the time budget or produces NaN loss, kill it with KillBash
5. When it completes, extract the OOF score from stdout (format: OOF_SCORE: X.XXXXXX)
6. Report resource usage and any errors
"""

EXECUTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "oof_score", "runtime_seconds"],
    "properties": {
        "status": {"enum": ["success", "timeout", "error", "oom"]},
        "oof_score": {"type": ["number", "null"]},
        "runtime_seconds": {"type": "number"},
        "stdout_tail": {"type": "string"},
        "error_message": {"type": ["string", "null"]},
        "peak_memory_gb": {"type": ["number", "null"]},
    }
}

async def run_execution_agent(
    solution_path: str,
    max_runtime_minutes: int,
    state: CompetitionState,
    project_dir: str,
) -> dict:
    prompt = f"""\
Execute training: {solution_path}
Time budget: {max_runtime_minutes} minutes
Data directory: {state.data_dir}

1. Check GPU memory: nvidia-smi (if available)
2. Launch: python {solution_path} as a background process
3. Monitor every 60 seconds — check for NaN, OOM, or completion
4. Kill if it exceeds {max_runtime_minutes} minutes
5. Report the OOF score printed on the last line (format: OOF_SCORE: X.XXXXXX)
"""
    output, _ = await run_agent(
        prompt=prompt,
        system_prompt=EXECUTION_SYSTEM_PROMPT,
        allowed_tools=["Bash", "Read"],
        output_schema=EXECUTION_OUTPUT_SCHEMA,
        cwd=project_dir,
    )
    return output
```

### 4.5 Validation Agent

Replaces: `validation_agent`, `submission_decider`, `notifier`

**Important**: The agent reports whether improvement was seen and whether to submit. The *orchestrator* is the only thing that updates `best_oof_score` in state. This was the core regression in the LangGraph version (the node was mutating state it shouldn't own).

```python
# gladius/agents/validation.py

VALIDATION_SYSTEM_PROMPT = """\
You are responsible for validating ML experiment results and deciding on Kaggle submissions.
Given the OOF score of a new experiment, you will:
1. Compare against the provided best known OOF score
2. Decide whether to submit (score improvement > threshold AND daily-limit not exceeded)
3. If submitting, verify the submission file format is correct (read first 3 lines)
4. Report the decision with clear reasoning

You do NOT update any state files. You only report what should happen.
"""

VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["oof_score", "is_improvement", "submit", "reasoning"],
    "properties": {
        "oof_score": {"type": "number"},
        "is_improvement": {"type": "boolean"},
        "improvement_delta": {"type": "number"},
        "submit": {"type": "boolean"},
        "submission_path": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
    }
}

async def run_validation_agent(
    solution_path: str,
    oof_score: float,
    submission_path: str,
    state: CompetitionState,
    project_dir: str,
) -> dict:
    prompt = f"""\
New experiment results:
- Solution: {solution_path}
- OOF score: {oof_score:.6f}
- Current best OOF: {state.best_oof_score:.6f}
- Metric: {state.target_metric} ({state.metric_direction})
- Submissions today: {state.submission_count} / {state.max_submissions_per_day}
- Submission file: {submission_path}

Validate: Is {oof_score:.6f} a meaningful improvement over {state.best_oof_score:.6f}?
Threshold: improvement must be > 0.0001 to submit.
Check that the submission file exists and has the correct format (read first 3 lines).
"""
    output, _ = await run_agent(
        prompt=prompt,
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        allowed_tools=["Read", "Bash"],
        output_schema=VALIDATION_OUTPUT_SCHEMA,
        cwd=project_dir,
    )
    return output
```

### 4.6 Ensemble Agent

Replaces: `ensemble_agent`, `knowledge_extractor`

```python
# gladius/agents/ensemble.py

ENSEMBLE_SYSTEM_PROMPT = """\
You are an expert at combining ML models for Kaggle competitions.
You have access to all OOF predictions and can compute optimal blend weights.
You will:
1. Identify successful experiments (OOF better than baseline)
2. Load their OOF prediction arrays
3. Compute pairwise correlations — prefer diverse, low-correlation models
4. Find optimal blend weights using scipy.optimize.minimize (Nelder-Mead)
5. Generate the blended submission at .gladius/ensemble_submission.csv
6. Save a reproducible ensemble script at src/ensemble.py
"""

ENSEMBLE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["ensemble_type", "components", "oof_score", "submission_path"],
    "properties": {
        "ensemble_type": {"enum": ["simple_average", "weighted_blend", "stacking", "rank_average"]},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "weight": {"type": "number"},
                    "oof_score": {"type": "number"}
                }
            }
        },
        "oof_score": {"type": "number"},
        "submission_path": {"type": "string"},
        "notes": {"type": "string"},
    }
}

async def run_ensemble_agent(state: CompetitionState, project_dir: str) -> dict:
    good_experiments = [
        e for e in state.experiments
        if e.get("oof_score", -1) > state.best_oof_score * 0.99
    ]
    prompt = f"""\
Ensemble task for competition: {state.competition_id}
Metric: {state.target_metric} ({state.metric_direction})

Good experiments (OOF within 1% of best):
{json.dumps(good_experiments, indent=2)}

OOF predictions are in .gladius/oof_vN.npy
Test predictions are in .gladius/sub_vN.csv

Tasks:
1. Read the OOF arrays and compute pairwise correlations
2. Find the optimal blend weights using scipy.optimize.minimize (Nelder-Mead)
3. Create the blended submission at .gladius/ensemble_submission.csv
4. Save an ensemble script at src/ensemble.py

Prefer diversity over raw score — low-correlation models blend better.
"""
    output, _ = await run_agent(
        prompt=prompt,
        system_prompt=ENSEMBLE_SYSTEM_PROMPT,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob"],
        output_schema=ENSEMBLE_OUTPUT_SCHEMA,
        cwd=project_dir,
    )
    return output
```

---

## 5. Custom Tools (MCP)

Custom tool servers are registered via `ClaudeAgentOptions.mcp_servers`. Agents that need Kaggle API access or metric utilities receive them.

```python
# gladius/tools/kaggle_tools.py
from claude_agent_sdk import tool, create_sdk_mcp_server
from typing import Any
import subprocess


@tool(
    "kaggle_submit",
    "Submit a CSV file to the Kaggle competition leaderboard",
    {"competition": str, "file_path": str, "message": str}
)
async def kaggle_submit(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", args["competition"],
        "-f", args["file_path"],
        "-m", args["message"],
    ], capture_output=True, text=True)
    return {"content": [{"type": "text", "text": result.stdout + result.stderr}]}


@tool(
    "kaggle_leaderboard",
    "Fetch the current public leaderboard for a competition",
    {"competition": str, "top_n": int}
)
async def kaggle_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run([
        "kaggle", "competitions", "leaderboard",
        "-c", args["competition"],
        "--show", "--csv",
    ], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")[:args.get("top_n", 20) + 1]
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "compute_oof_metric",
    "Compute OOF metric score (AUC-ROC, RMSE, etc.) from numpy arrays on disk",
    {"metric": str, "oof_path": str, "labels_path": str}
)
async def compute_oof_metric(args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import roc_auc_score, mean_squared_error

    oof = np.load(args["oof_path"])
    y = np.load(args["labels_path"])

    metric = args["metric"].lower()
    if metric == "auc_roc":
        # Multiclass: macro OVR — memory-safe, no outer product matrix
        if oof.ndim == 2 and oof.shape[1] > 2:
            score = roc_auc_score(y, oof, multi_class="ovr", average="macro")
        else:
            oof_1d = oof[:, 1] if oof.ndim == 2 else oof
            score = roc_auc_score(y, oof_1d)
    elif metric == "rmse":
        score = float(np.sqrt(mean_squared_error(y, oof)))
    else:
        return {"content": [{"type": "text", "text": f"Unknown metric: {metric}"}], "is_error": True}

    return {"content": [{"type": "text", "text": f"METRIC_SCORE: {score:.6f}"}]}


kaggle_server = create_sdk_mcp_server(
    name="kaggle",
    version="1.0.0",
    tools=[kaggle_submit, kaggle_leaderboard, compute_oof_metric],
)
```

---

## 6. Parallel Execution

When the strategic plan generates multiple independent hypotheses, run them concurrently:

```python
# In orchestrator.py

async def run_parallel_experiments(
    hypotheses: list[dict],
    state: CompetitionState,
    project_dir: str,
    max_parallel: int = 3,
) -> list[dict]:
    """Run up to max_parallel experiments concurrently."""
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_one(hypothesis: dict) -> dict:
        async with semaphore:
            code_output, session_id = await run_code_agent(hypothesis, state, project_dir)
            exec_output = await run_execution_agent(
                code_output["solution_path"],
                max_runtime_minutes=60,
                state=state,
                project_dir=project_dir,
            )
            return {
                "hypothesis": hypothesis,
                "code": code_output,
                "execution": exec_output,
                "session_id": session_id,
            }

    tasks = [run_one(h) for h in hypotheses[:max_parallel]]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 7. Subagents (Native Claude)

For complex analysis tasks (knowledge extraction from papers, winning solutions), use native Claude subagents via `AgentDefinition`. These run as sub-sessions inside the parent agent — launched via the `Task` tool.

```python
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition

knowledge_extractor_def = AgentDefinition(
    description="Extracts actionable insights from papers, notebooks, and winning solutions",
    prompt="""\
You are a knowledge extraction specialist. Given a document or URL, extract:
1. Model architectures used
2. Feature engineering techniques
3. Training tricks (augmentation, regularization, etc.)
4. Post-processing methods
5. Ensemble strategies

Return a structured summary focused on what is directly applicable to the current competition.
""",
    tools=["Read", "WebFetch", "WebSearch"],
    model="haiku",  # fast + cheap for extraction
)

# The parent strategy agent gets this subagent registered
options = ClaudeAgentOptions(
    agents={"knowledge_extractor": knowledge_extractor_def},
    allowed_tools=["Read", "WebSearch", "Task"],  # Task invokes the subagent
    # ...
)
# The parent agent's prompt can say:
# "Use the knowledge_extractor agent to analyze https://arxiv.org/abs/XXXX
#  and return the key findings applicable to tabular classification."
```

---

## 8. Main Orchestrator

```python
# gladius/orchestrator.py
"""
Main competition loop. This is the only routing logic in the system.
No graph edges, no conditional routing functions — just Python control flow.
"""
import asyncio
import json
import logging
from typing import Optional

from .state import CompetitionState, StateStore
from .agents.strategy import run_strategy_agent
from .agents.code import run_code_agent
from .agents.execution import run_execution_agent
from .agents.validation import run_validation_agent
from .agents.ensemble import run_ensemble_agent

logger = logging.getLogger(__name__)


async def run_competition(
    competition_id: str,
    data_dir: str,
    project_dir: str,
    target_metric: str = "auc_roc",
    metric_direction: str = "maximize",
    max_iterations: int = 20,
    resume_from_db: bool = True,
) -> CompetitionState:

    store = StateStore(f"{project_dir}/.gladius/state.db")

    # Resume or initialize state
    state = store.load() if resume_from_db else None
    if state is None:
        state = CompetitionState(
            competition_id=competition_id,
            data_dir=data_dir,
            output_dir=f"{project_dir}/.gladius",
            target_metric=target_metric,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
        )

    while state.iteration < state.max_iterations and state.phase != "done":
        logger.info(f"[Iteration {state.iteration}] Phase: {state.phase}")

        try:
            if state.phase == "strategy":
                hypothesis, session_id = await run_strategy_agent(state, data_dir)
                state.current_hypothesis = hypothesis
                state.strategy_session_id = session_id  # persist for next iteration
                state.phase = "coding"

            elif state.phase == "coding":
                code_result, session_id = await run_code_agent(
                    state.current_hypothesis, state, project_dir
                )
                state.code_session_id = session_id
                state.current_hypothesis["solution_path"] = code_result["solution_path"]
                state.phase = "execution"

            elif state.phase == "execution":
                solution_path = state.current_hypothesis["solution_path"]
                exec_result = await run_execution_agent(
                    solution_path, max_runtime_minutes=90, state=state, project_dir=project_dir
                )

                if exec_result["status"] != "success":
                    logger.warning(f"Execution failed: {exec_result.get('error_message')}")
                    state.failed_hypotheses.append({
                        **state.current_hypothesis,
                        "reason": exec_result["status"],
                        "error": exec_result.get("error_message"),
                    })
                    state.consecutive_errors += 1
                    state.phase = "strategy"
                else:
                    state.consecutive_errors = 0
                    state.current_hypothesis["oof_score"] = exec_result["oof_score"]
                    state.current_hypothesis["runtime_seconds"] = exec_result["runtime_seconds"]
                    state.phase = "validation"

            elif state.phase == "validation":
                solution_path = state.current_hypothesis["solution_path"]
                oof_score = state.current_hypothesis["oof_score"]
                submission_path = solution_path.replace(".py", "_sub.csv")

                validation = await run_validation_agent(
                    solution_path, oof_score, submission_path, state, project_dir
                )

                # ORCHESTRATOR owns the state mutation — not the agent
                if validation["is_improvement"]:
                    state.best_oof_score = oof_score
                    state.experiments.append({
                        "solution_path": solution_path,
                        "oof_score": oof_score,
                        "iteration": state.iteration,
                        "hypothesis": state.current_hypothesis.get("hypothesis"),
                    })

                if validation["submit"] and state.submission_count < state.max_submissions_per_day:
                    state.submission_count += 1
                    state.best_submission_path = submission_path
                    logger.info(f"Submitting: {submission_path}")

                state.completed_hypotheses.append(state.current_hypothesis)
                state.iteration += 1

                # Trigger ensemble every 5 iterations if enough experiments
                if state.iteration % 5 == 0 and len(state.experiments) >= 3:
                    state.phase = "ensemble"
                else:
                    state.phase = "strategy"

            elif state.phase == "ensemble":
                ensemble_result = await run_ensemble_agent(state, project_dir)
                if ensemble_result["oof_score"] > state.best_oof_score:
                    state.best_oof_score = ensemble_result["oof_score"]
                    state.best_submission_path = ensemble_result["submission_path"]
                    logger.info(f"Ensemble improved OOF: {ensemble_result['oof_score']:.6f}")
                state.phase = "strategy"

        except Exception as e:
            logger.error(f"Error in phase {state.phase}: {e}", exc_info=True)
            state.error_log.append({"phase": state.phase, "iteration": state.iteration, "error": str(e)})
            state.consecutive_errors += 1
            if state.consecutive_errors >= 3:
                logger.critical("3 consecutive errors — stopping")
                state.phase = "done"
            else:
                state.phase = "strategy"

        finally:
            store.save(state)

    logger.info(f"Competition run complete. Best OOF: {state.best_oof_score:.6f}")
    return state


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--metric", default="auc_roc")
    parser.add_argument("--direction", default="maximize")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    await run_competition(
        competition_id=args.competition,
        data_dir=args.data_dir,
        project_dir=args.project_dir,
        target_metric=args.metric,
        metric_direction=args.direction,
        max_iterations=args.iterations,
        resume_from_db=not args.no_resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. Session Continuity

Agent SDK sessions are resumed via `resume=session_id` in `ClaudeAgentOptions`. Session IDs come from `ResultMessage.session_id`.

| Agent | Session Strategy | Rationale |
|---|---|---|
| Strategy Agent | Resume every iteration | Accumulates competition context, LB history, reasoning chain across all iterations — replaces context-stuffing |
| Code Agent | Resume every iteration | "Remembers" the codebase structure, naming conventions, what it already wrote |
| Execution Agent | No resume (stateless) | Each run is independent |
| Validation Agent | No resume (stateless) | Each validation is independent |
| Ensemble Agent | No resume (stateless) | Triggered on-demand, reads state from disk |

```python
# Resume a prior session — the agent re-enters its prior conversation context
options = ClaudeAgentOptions(
    resume=state.strategy_session_id,
    # ...
)

# Fork a session — new session_id branching from the resumed one
options = ClaudeAgentOptions(
    resume=state.strategy_session_id,
    fork_session=True,
)
```

---

## 10. Error Handling

```python
from claude_agent_sdk import CLINotFoundError, ProcessError, CLIJSONDecodeError

async def run_agent_with_retry(
    prompt: str,
    system_prompt: str,
    allowed_tools: list[str],
    output_schema: dict,
    cwd: str,
    max_retries: int = 3,
) -> tuple[dict, str]:
    for attempt in range(max_retries):
        try:
            return await run_agent(prompt, system_prompt, allowed_tools, output_schema, cwd)
        except CLINotFoundError:
            raise  # Fatal — Claude Code CLI not installed
        except ProcessError as e:
            if e.exit_code == 1 and "rate limit" in (e.stderr or ""):
                await asyncio.sleep(60 * (2 ** attempt))  # exponential backoff
            elif attempt == max_retries - 1:
                raise
        except CLIJSONDecodeError:
            if attempt == max_retries - 1:
                raise
        except Exception:
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("Max retries exceeded")
```

---

## 11. Migration Map

| LangGraph Component | SDK Replacement |
|---|---|
| `GraphState` TypedDict | `CompetitionState` dataclass |
| `SqliteSaver` checkpointer | `StateStore` (SQLite) |
| `graph.add_node()` | `async def run_X_agent()` function |
| `graph.add_conditional_edges()` | `if/elif` in `orchestrator.py` |
| `call_llm(prompt, schema=...)` | `output_format={"type":"json_schema",...}` in `ClaudeAgentOptions` |
| `code_generator` + `code_reviewer` + `versioning_agent` | Single `run_code_agent()` |
| `executor` + `resource_manager` + `watchdog` | Single `run_execution_agent()` |
| `validation_agent` + `submission_decider` + `notifier` | Single `run_validation_agent()` |
| `knowledge_extractor` | `AgentDefinition` subagent within strategy agent |
| `ensemble_agent` + `lb_tracker` | Single `run_ensemble_agent()` |
| `Send()` for parallel branches | `asyncio.gather()` with `asyncio.Semaphore` |
| `router.py` routing functions | Deleted — `if/elif` in orchestrator |
| `utils/llm.py` `call_llm()` | Deleted — SDK handles it |

**Implementation order:**
1. `gladius/state.py` — `CompetitionState` + `StateStore`
2. `gladius/tools/kaggle_tools.py` + `gladius/tools/metric_tools.py`
3. `gladius/agents/execution.py` — validates the pattern end-to-end
4. `gladius/agents/code.py` — core value
5. `gladius/agents/validation.py`
6. `gladius/agents/strategy.py`
7. `gladius/agents/ensemble.py`
8. `gladius/orchestrator.py` — wire everything
9. Delete: `gladius/nodes/`, `gladius/graph.py`, `gladius/utils/llm.py`

---

## 12. Dependencies

```toml
# pyproject.toml
[project.dependencies]
claude-agent-sdk = ">=0.1.0"
numpy = ">=1.26"
scikit-learn = ">=1.4"
scipy = ">=1.11"
kaggle = ">=1.6.0"

# Remove: langgraph, langchain-core, langchain-anthropic
```

```bash
# Requires Claude Code CLI (installed via npm)
npm install -g @anthropic-ai/claude-code

pip install claude-agent-sdk
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 13. Key Design Principles

1. **Agents are complete, autonomous workers** — not thin wrappers around a single LLM call. A code agent reads, writes, runs, and verifies. Tell it the goal; let it figure out the steps.

2. **The orchestrator owns all routing and state mutation** — agents output structured JSON, the orchestrator decides what to do with it. Agents never update `best_oof_score` directly; the orchestrator does after reading their output. This was the fatal regression in the LangGraph version.

3. **Sessions are long-lived for agents that accumulate context** — strategy and code agents resume their sessions across iterations, building up understanding of the competition and codebase without needing to reconstruct context from scratch each time.

4. **Structured output via `output_format`** — every agent specifies a `json_schema`. The SDK guarantees the output conforms before returning. No `json.loads()` + exception handling in orchestrator code.

5. **Parallelism via `asyncio.gather()`** — not LangGraph `Send()` primitives. Clean, debuggable, and trivially resource-bounded with `asyncio.Semaphore`.

6. **Tools are the permission boundary** — `allowed_tools` in `ClaudeAgentOptions` specifies exactly what each agent can do. The execution agent gets `Bash`; the strategy agent does not. This replaces LangGraph's implicit "everything can do anything" node design.

7. **Three LangGraph nodes → one SDK agent** — the code/review/version triplet collapses because Claude Code natively handles multi-step file operations. Same for executor/watchdog/resource-manager. This cuts the agent count from 18 to 5.
