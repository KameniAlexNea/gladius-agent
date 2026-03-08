# Implementer Redesign: From Monolith to Subagent Coordinator

*Lead decision — March 2026*

---

## 1. Problem Statement

The current `implementer` is a 80-turn monolithic session that does everything in one context window:

- Read the plan  
- Set up project structure  
- Write feature engineering code  
- Write the CV/training pipeline  
- Run and iterate until it works  
- Fetch skills on-demand (`Skill({...})`)  
- Evaluate OOF  
- Run code review  
- Prepare submission CSV  
- Report results  

This is not wrong — it works — but it carries two real costs:

**Context contamination.** Every debug loop, every failed Bash run, every `pip install` output lands in the same context that later needs to reason about leakage and OOF correctness. By the 60th turn, the context is a wall of noise.

**Skill pull overhead.** Each `Skill({"name": "..."})` call consumes a turn and loads a potentially large markdown document into an already crowded context. Skills are fetched reactively, so the model decides mid-stream whether to call adversarial-validation, mostly after the code is written.

**Implicit sequencing.** The phases (scaffold → code → debug → evaluate → review → submit) have no enforcement. The model must remember not to write evaluation code before the pipeline runs. Code review is "required before reporting" by prompt instruction — which is only as reliable as the model following instructions under token pressure.

The deeper issue: **autonomy should be given to the right scope.** The implementer session should be autonomous at the experiment level — "build something that scores well" — not autonomous at the line-execution level ("try to run this command, fail, patch, retry"). Those are different jobs.

---

## 2. What Claude Code Now Offers

Reading both the [subagents docs](https://code.claude.com/docs/en/sub-agents) and [agent teams docs](https://code.claude.com/docs/en/agent-teams), three primitives are relevant to this decision:

### 2.1 Custom Subagents (`.claude/agents/*.md`)

Subagents are Markdown files with YAML frontmatter placed at `.claude/agents/`. When an agent session uses the `Agent` tool, it spawns a fresh session using the named subagent definition — custom system prompt, restricted tools, scoped permissions, and skills preloaded at spawn time.

Key capabilities:
- **`skills:` frontmatter field** — skills are *injected* into the subagent's context at spawn, not fetched on demand. The full skill content is there from turn 1.
- **`tools:` allowlist** — a debugger gets `Read, Edit, Bash`; a reviewer gets `Read, Grep, Glob`. No write access leaks to the reviewer.
- **`permissionMode: bypassPermissions`** — the subagent can act without prompts, inheriting the parent session's bypass setting.
- **`Agent(subagent-name)` in tools** — an allowlist restricting which subagents the session can spawn. This is how we build a controlled coordinator.

### 2.2 Skills Preloading vs. Skill Tool Pull

Current design: the implementer *requests* skills at runtime with `Skill({"name": "ml-pipeline"})`. This is a pull model — one turn consumed, full skill content added to an already large context.

Subagent design: skills are listed in the subagent's `skills:` field. They are injected before the first turn. The coder subagent already has `ml-pipeline`, `feature-engineering`, and `hpo` in context when it starts — no fetching, no turn cost. This is the right model for focused workers.

### 2.3 The Recursion Stop

The current "NEVER spawn Task subagents" rule exists to prevent infinite recursion: if the implementer can spawn subagents, and those subagents could spawn further agents, the chain could be unbounded.

The docs clarify the actual constraint:

> **Subagents cannot spawn other subagents.**

This is enforced at the platform level, not by prompt instruction. The recursion stop is structural. Only the main session (the one launched by the Python SDK) can spawn subagents. Subagents launched *by* the implementer-lead cannot go deeper. The "NEVER spawn Task subagents" rule in the current implementer prompt was protecting against a risk that the platform now handles by construction.

---

## 3. Options Considered

### Option A: Keep the monolith, improve it

Continue with one implementer session, but increase `max_turns`, add forced checkpoints, and improve the system prompt. Low effort, backwards-compatible.

**Why this is wrong:** The problem isn't prompt quality. The prompt is already dense with structure. The problem is that one context window is doing too many orthogonal things. Longer prompts or more turns don't fix context contamination.

### Option B: Python-level phase splitting

Split `run_implementer()` in Python into `run_coder()`, `run_debugger()`, `run_evaluator()`, `run_reviewer()` — four separate `run_agent()` calls in the Python orchestrator.

This gives clean contexts per phase. But it moves coordination logic into Python, where it becomes if/elif routing of YAML state. The Python orchestrator already does macroscopic routing (plan → implement → validate). Having it also route micro-phases (scaffold → code → debug → evaluate) is wrong — that coordination requires reading intermediate results and making judgment calls (is debugging done? did the OOF improve?). That is an LLM job.

Also, phases are not strictly sequential. The coder may produce code that needs multiple debug rounds before evaluation is even possible. A Python for loop can't manage that adaptively.

### Option C: Claude Code Agent Teams

Use `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` — one lead, teammates for coding/debugging/evaluating that share a task list.

This is the wrong fit here for a concrete reason: agent teams are designed for independent agents that *communicate with each other*. In the ML implementation workflow, the phases are sequential with hard dependencies. The evaluator cannot start until the coder finishes. Teammates sharing a task list and messaging each other is overhead with no benefit when the work is a pipeline, not parallel exploration. Agent teams also carry the experimental flag and known limitations around session resumption.

### Option D: Implementer as Coordinator (Recommended)

Transform the implementer session into a **thin coordinator** that delegates to focused subagents. The Python SDK still calls `run_implementer()` as before. The implementer-lead uses the `Agent` tool to spawn subagents defined in `.claude/agents/`. The Python orchestrator is untouched.

This is the right design. Coordination requiring judgment lives in the implementer-lead (an LLM), not in Python if/elif. Each worker has focused context with preloaded skills. The recursion stop is structural. The Python API surface doesn't change.

---

## 4. The New Architecture

### 4.1 Overview

```
Python Orchestrator (unchanged)
  │
  ├── planning
  │    └── planner session (resumed, read-only)         [unchanged]
  │
  ├── implementing
  │    └── implementer-lead session (fresh each iter)  [CHANGED: coordinator]
  │         │  Tools: Agent(ml-scaffolder, ml-developer, ml-scientist,
  │         │               ml-evaluator, code-reviewer, submission-builder),
  │         │               Read, Write, Glob, TodoWrite
  │         │  Writes: .claude/EXPERIMENT_STATE.json   [NEW artifact handshake]
  │         │
  │         ├── [phase 1] ml-scaffolder subagent       [NEW]
  │         │    Skills: ml-setup (preloaded)
  │         │    Tools:  Read, Write, Bash, Glob
  │         │    Model:  haiku
  │         │
  │         ├── [phase 2] ml-developer subagent        [NEW — merged coder+debugger]
  │         │    Skills: ml-pipeline, feature-engineering,
  │         │             polars, hpo, ensembling (preloaded)
  │         │    Tools:  Read, Write, Edit, Bash, Glob, Grep, TodoWrite
  │         │    Hook:   PreToolUse → validate_bash.sh
  │         │
  │         ├── [phase 3] ml-evaluator subagent        [NEW]
  │         │    Skills: ml-pipeline (OOF formulas, preloaded)
  │         │    Tools:  Read, Bash, Glob, Grep
  │         │    Model:  haiku
  │         │
  │         ├── [phase 4] code-reviewer subagent       [NEW]
  │         │    Skills: code-review (preloaded)
  │         │    Tools:  Read, Grep, Glob
  │         │    permissionMode: plan
  │         │
  │         ├── [phase 5] ml-scientist subagent        [NEW — logical bug fixer]
  │         │    Skills: ml-pipeline, feature-engineering, code-review (preloaded)
  │         │    Tools:  Read, Write, Edit, Bash, Glob, Grep
  │         │    Spawned only when code-reviewer reports CRITICAL logical issues
  │         │
  │         └── [phase 6] submission-builder subagent  [NEW]
  │              Skills: submit-check (preloaded)
  │              Tools:  Read, Write, Bash, Glob
  │
  ├── validation
  │    └── validation session (stateless)              [unchanged]
  │
  └── summarization
       └── summarizer session (stateless)              [unchanged]
```

**Routing is a directed graph, not a waterfall.** The coordinator uses back-edges:

```
SCAFFOLD ──► DEVELOP ──► EVALUATE ──► REVIEW
               ▲              │            │
               │    re-run    │  CRITICAL  │
               └──────────────┘  logical   ▼
                                  issues  SCIENCE ──► DEVELOP (patch) ──► EVALUATE ──► REVIEW
                                                                                            │
                                                                                            ▼
                                                                                         SUBMIT
```

A CRITICAL execution error inside `ml-developer` is handled internally (the developer loops on its own). Only logical ML bugs discovered by `code-reviewer` warrant spawning `ml-scientist`.

### 4.2 The Implementer-Lead's Job

The implementer-lead **no longer writes code or runs commands**. It:

1. Reads `CLAUDE.md` and the plan.
2. Assesses what's already done (existing `src/` structure, previous score).
3. Initialises (or reads) `.claude/EXPERIMENT_STATE.json` — the shared artifact file.
4. Creates a `TodoWrite` task list for the phases.
5. Delegates each phase by spawning the appropriate subagent with a precise prompt.
6. After each subagent completes, reads `.claude/EXPERIMENT_STATE.json` to determine the next action. **Does not parse free text from subagent messages to make routing decisions.**
7. Returns the final `OUTPUT_SCHEMA` JSON to the Python SDK.

Its tool list:
```
Agent(ml-scaffolder, ml-developer, ml-scientist, ml-evaluator, code-reviewer, submission-builder)
Read
Write   ← only for EXPERIMENT_STATE.json
Glob
TodoWrite
```

No `Bash`, no `Edit`. The coordinator cannot run commands or patch files — only read state, write the state file, and spawn workers.

### 4.2.1 EXPERIMENT_STATE.json — The Artifact Handshake

Every subagent writes its output to `.claude/EXPERIMENT_STATE.json` before returning. The coordinator reads this file, not the subagent's conversational output, to decide the next phase. This eliminates lossy free-text parsing at the routing layer.

```json
{
  "phase": "reviewed",
  "developer": {
    "solution_files": ["src/train.py", "src/features.py"],
    "preliminary_metric": 0.847,
    "status": "success"
  },
  "evaluator": {
    "oof_score": 0.851,
    "metric": "auc",
    "status": "success"
  },
  "reviewer": {
    "critical_issues": ["target encoding fitted on full train — data leakage"],
    "warnings": ["random seed not fixed in fold loop"],
    "suggestions": [],
    "status": "complete"
  },
  "scientist": null,
  "submission": null
}
```

Each subagent receives the current `EXPERIMENT_STATE.json` contents as part of its spawn prompt, so it knows what prior phases have produced. The coordinator's routing logic is then: read the JSON, check `reviewer.critical_issues`, if non-empty spawn `ml-scientist`, else proceed to `submission-builder`.

### 4.3 Subagent Definitions

Each subagent lives in the competition directory at `.claude/agents/<name>.md`, bootstrapped by `project_setup.py` from templates in `gladius/utils/templates/agents/`.

#### `ml-scaffolder.md`
```yaml
---
name: ml-scaffolder
description: Bootstraps ML project structure. Use only once per competition to create src/ layout.
tools: Read, Write, Bash, Glob
model: haiku
maxTurns: 15
permissionMode: bypassPermissions
skills:
  - ml-setup
---
```
Cheap (Haiku), fast, one purpose: run the init script from the ml-setup skill, create `src/`, install deps. Skip if `src/` already exists.

#### `ml-developer.md`
```yaml
---
name: ml-developer
description: >-
  Writes ML pipeline code, runs it, and fixes execution errors until the script
  exits cleanly and produces an initial metric. Handles the full write-run-fix-loop.
  Does NOT fix logical ML bugs (data leakage, wrong metric formulas) — those go to ml-scientist.
tools: Read, Write, Edit, Bash, Glob, Grep, TodoWrite
model: inherit
maxTurns: 80
permissionMode: bypassPermissions
skills:
  - ml-pipeline
  - feature-engineering
  - polars
  - hpo
  - ensembling
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate_bash.sh"
---
```
The heavy lifter. Gets all ML skills preloaded at turn 1. Writes code, runs it, and loops on execution errors (import failures, OOM, syntax errors) until the script exits with code 0 and prints a metric line. Before returning, appends its output to `.claude/EXPERIMENT_STATE.json`.

**Why merged:** Writing code and fixing its execution errors share the same mental model. Separating them forces the debugger to reconstruct the full pipeline context cold, based on a coordinator summary. The write-run-fix cycle must stay in one context window.

#### `ml-scientist.md`
```yaml
---
name: ml-scientist
description: >-
  Fixes logical ML bugs: data leakage, wrong validation schemes, impossible metrics,
  and flawed feature formulations. Spawned by the coordinator only when code-reviewer
  reports CRITICAL issues that are logical (not execution) errors.
tools: Read, Write, Edit, Bash, Glob, Grep, TodoWrite
model: inherit
maxTurns: 40
permissionMode: bypassPermissions
skills:
  - ml-pipeline
  - feature-engineering
  - code-review
---
```
Receives the full code-reviewer output (from `EXPERIMENT_STATE.json`) in its spawn prompt, alongside the flagged files. Its mandate is to fix *logical* ML correctness issues — target leakage, metric computation bugs, CV contamination — not to rewrite the boilerplate architecture. Reports which files it changed back to the state file.

#### `ml-evaluator.md`
```yaml
---
name: ml-evaluator
description: Computes OOF score. Runs the evaluation script and extracts the metric value.
tools: Read, Bash, Glob, Grep
model: haiku
maxTurns: 15
permissionMode: acceptEdits
skills:
  - ml-pipeline
---
```
Read-only after the fact. Runs the existing training/eval script or post-processes saved OOF arrays. Cannot write new code. Reports the exact OOF score. Haiku is sufficient for this deterministic task.

#### `code-reviewer.md`
```yaml
---
name: code-reviewer
description: Reviews code for data leakage, metric bugs, and correctness. Always invoke before reporting results.
tools: Read, Grep, Glob
model: inherit
maxTurns: 20
permissionMode: plan
skills:
  - code-review
---
```
Strictly read-only. `permissionMode: plan` enforced at the platform level — no accidental edits. Gets the full `code-review` skill preloaded. Reports a structured list of CRITICAL, WARNING, and SUGGESTION issues. The implementer-lead decides whether to spawn another `ml-debugger` round to fix CRITICAL items.

#### `submission-builder.md`
```yaml
---
name: submission-builder
description: Generates test set predictions and formats submission CSV. Validates format before reporting.
tools: Read, Write, Bash, Glob
model: inherit
maxTurns: 20
permissionMode: bypassPermissions
skills:
  - submit-check
---
```
Focused purely on the final artifact. Gets `submit-check` preloaded so it validates format *before* reporting the file path to the lead.

### 4.4 The Implementer-Lead System Prompt (excerpt)

```
You are the ML experiment coordinator for a competition.
Your job is to run a complete experiment by coordinating specialized subagents.
You do NOT write code or run commands directly.

Artifact file: .claude/EXPERIMENT_STATE.json
  - Initialise it at the start of every run.
  - Each subagent appends its output section to this file when done.
  - YOU read this file — not the subagent's messages — to decide the next phase.
  - Pass the current file contents in every subagent spawn prompt.

Routing logic (not a strict sequence — use back-edges as needed):

1. SCAFFOLD   → spawn ml-scaffolder      (skip if src/ already exists)
2. DEVELOP    → spawn ml-developer       (write code, run it, fix execution errors)
3. EVALUATE   → spawn ml-evaluator       (compute OOF from existing artifacts)
4. REVIEW     → spawn code-reviewer      (read-only, logical analysis)
   After REVIEW:
     - If reviewer.critical_issues contains logical ML bugs (leakage, metric errors):
         → spawn ml-scientist to fix them
         → re-spawn ml-evaluator
         → re-spawn code-reviewer
     - If reviewer.critical_issues contains only execution issues:
         → re-spawn ml-developer
         → loop back to EVALUATE
     - If no critical issues: proceed to SUBMIT
5. SUBMIT     → spawn submission-builder

Track progress with TodoWrite. When all phases complete, report:
  - status: success | error | timeout | oom
  - oof_score: from evaluator.oof_score in EXPERIMENT_STATE.json
  - quality_score: your assessment of the work
  - solution_files: from developer + scientist + submission-builder file lists
  - submission_file: from submission.path in EXPERIMENT_STATE.json
  - notes: what was built and any key observations

NEVER modify CLAUDE.md.
```

---

## 5. What Changes in the Codebase

### 5.1 Files to Create (templates)

New subagent template files in `gladius/utils/templates/agents/`:

```
gladius/utils/templates/agents/
  implementer.md          ← MODIFIED (becomes coordinator prompt)
  planner.md              ← unchanged
  ml-scaffolder.md        ← NEW
  ml-developer.md         ← NEW (merged coder + debugger)
  ml-scientist.md         ← NEW (logical ML bug fixer)
  ml-evaluator.md         ← NEW
  code-reviewer.md        ← NEW
  submission-builder.md   ← NEW
```

### 5.2 `project_setup.py`

The agent template copy loop currently copies only `planner.md` and `implementer.md`. Extend it to copy all `.md` files found in `templates/agents/` — no hardcoding.

```python
# Current (brittle):
for name in ["planner", "implementer"]:
    shutil.copy(templates / "agents" / f"{name}.md", agents_dir / f"{name}.md")

# Proposed (generic):
for src in (templates / "agents").glob("*.md"):
    dest = agents_dir / src.name
    if not dest.exists():
        shutil.copy(src, dest)
```

### 5.3 `_base.py` — `_IMPLEMENTER_AGENT_DEF`

Update the implementer `AgentDefinition`:

```python
_IMPLEMENTER_AGENT_DEF = AgentDefinition(
    description="ML experiment coordinator. Delegates to specialized subagents for coding, evaluation, review, and submission. Coordinates via EXPERIMENT_STATE.json.",
    prompt="...(new coordinator prompt)...",
    tools=[
        "Agent(ml-scaffolder, ml-developer, ml-scientist, ml-evaluator, code-reviewer, submission-builder)",
        "Read",
        "Write",   # Only for .claude/EXPERIMENT_STATE.json
        "Glob",
        "TodoWrite",
    ],
    model=_model,
)
```

Remove `Edit`, `Bash`, `Grep`, `Skill` from the implementer-lead tools. `Write` is kept for `EXPERIMENT_STATE.json` only — the coordinator cannot execute code.

### 5.4 `implementer_spec.py`

Update `IMPLEMENTER_SYSTEM_PROMPT` accordingly. The new prompt is a coordinator prompt (see 4.4 above), not a "do everything" prompt.

`build_implementer_prompt()` can stay largely the same — it still builds the plan prompt that becomes the implementer-lead's first message.

### 5.5 `implementer.py` — `run_implementer()`

The `allowed_tools` list passed to `run_agent()` needs updating:

```python
allowed_tools=[
    "Agent(ml-scaffolder,ml-developer,ml-scientist,ml-evaluator,code-reviewer,submission-builder)",
    "Read",
    "Write",
    "Glob",
    "TodoWrite",
],
max_turns=30,  # reduced from 80 — coordinator turns, not execution turns
```

The total work turns now live inside subagents (80 for ml-developer, 40 for ml-scientist, etc.). The coordinator needs far fewer turns.

### 5.6 `OUTPUT_SCHEMA` — No Change

The Python orchestrator's contract with the implementer is unchanged. `status`, `oof_score`, `quality_score`, `solution_files`, `submission_file`, `notes` — all the same fields. The orchestrator never sees the intermediate subagent structure.

---

## 6. Skill Distribution (Push vs Pull)

The current model has skills loaded on-demand by the monolithic implementer via `Skill({...})`. Each fetch costs a turn and grows the context. The new model preloads skills into the subagent that actually uses them:

| Current (pull) | New (push) |
|---|---|
| Implementer fetches `adversarial-validation` mid-run | `ml-developer`: `skills: [adversarial-validation]` |
| Implementer fetches `feature-engineering` | `ml-developer`: `skills: [feature-engineering]` |
| Implementer fetches `ml-pipeline` | `ml-developer` + `ml-evaluator`: `skills: [ml-pipeline]` |
| Implementer fetches `hpo` | `ml-developer`: `skills: [hpo]` |
| Implementer fetches `ensembling` | `ml-developer`: `skills: [ensembling]` |
| Implementer fetches `code-review` (REQUIRED before results) | `code-reviewer` + `ml-scientist`: `skills: [code-review]` (preloaded) |
| Implementer fetches `submit-check` | `submission-builder`: `skills: [submit-check]` (preloaded) |

**Important:** do not preload all 5 skills into every subagent. `ml-developer` gets ML skills; `ml-scientist` gets ML + code-review; `ml-evaluator` gets only ml-pipeline. Preloading irrelevant skills into a subagent adds cold-start token cost with no benefit — the cold-start tax is real (Gemini's valid point) and selectivity is the mitigation.

The `Skill` tool remains available for the planner (which uses it to read skills and reference them in plans). For subagents, `skills:` frontmatter is the right mechanism — the content is injected before turn 1 so the model has it immediately without consuming a turn.

---

## 7. Parallel Experiments (`--parallel N`)

The current parallel path: the Python orchestrator runs `N` `run_implementer()` calls via `asyncio.gather`. Each is an independent Claude Code session.

This is unchanged. Each of the `N` implementer-leads is still a separate Python SDK invocation. Each lead independently spawns its own subagents. There is no cross-contamination between parallel experiments. The `asyncio.gather` call doesn't change.

---

## 8. Trade-offs

### Gains

**Context separation.** The coder context never sees debug traces from the evaluator. The reviewer context never sees the full training run output. Each subagent gets domain-appropriate context from turn 1.

**Enforced phase sequencing.** The implementer-lead orchestrates phases explicitly. Code review can't be skipped because it's a distinct subagent spawn, not an instruction in a busy context. The coordinator *must* spawn `code-reviewer` to get REVIEWER output; it can't shortcut to StructuredOutput.

**Preloaded skills = fewer turns, cleaner context.** The coder gets `feature-engineering` immediately. No turn spent fetching. No large skill markdown appended mid-context.

**Tighter permission scoping.** The reviewer can't accidentally write files (it's `plan` mode). The evaluator can't write new model code (`acceptEdits` on Bash only). The scaffolder uses Haiku — cheap, fast, appropriate for a templating task.

**Lower coordinator turn count.** The implementer-lead goes from 80 turns to ~30 turns. The work turns are distributed across subagents. Total turns per experiment are similar or slightly higher (executor subagents get their own 60/30/20/20 budgets), but each context is cleaner.

### Costs

**More files to maintain.** 6 new template files in `templates/agents/`. Each needs to be correct and up to date when skills evolve.

**Coordination failures.** If the implementer-lead misroutes based on bad state-file content, the experiment fails. Mitigated significantly by `EXPERIMENT_STATE.json` — routing reads structured JSON, not free text. The residual risk is: a subagent writes incorrect values to the state file.

**Token cold-start tax per subagent spawn.** Every subagent spawn injects: system prompt + skills + workspace context. If an experiment requires 3 loops through `ml-developer` (which it shouldn't, but might), you pay this tax 3 times. Mitigated by: (a) merging coder + debugger so the loop stays internal to one agent, (b) being selective about skill preloading, (c) keeping ml-evaluator and ml-scaffolder on Haiku.

**Experimental: `Agent` tool behavior.** The `Agent(ml-coder, ...)` allowlist syntax is newer platform functionality. Test carefully on the first iteration.

---

## 9. What Stays the Same

- **Python orchestrator.** The macro loop, state machine, SQLite persistence, platform submission logic, improvement check — all unchanged.
- **`OUTPUT_SCHEMA`.** The contract between Python and the implementer is identical.
- **Planner, validation, summarizer.** These agents are already well-scoped. They don't need restructuring.
- **CLAUDE.md mechanism.** Written by the orchestrator, read by all agents automatically. Still the shared context source.
- **MEMORY.md + summarizer.** The planner's persistent memory loop is untouched.
- **MCP servers.** Jupyter MCP, platform tools — unchanged.
- **`--parallel N` logic.** Multiple asyncio-gathered implementer-leads, each with their own subagents.

---

## 10. Implementation Order

Do not implement all of this at once. The sequence that minimizes risk:

1. **Add `EXPERIMENT_STATE.json` contract first.** Define the JSON schema. Add `Write` to the current monolithic implementer so it writes the file at the end of its run. Validate that the coordinator can read it. This is infrastructure with no agent changes.

2. **Implement `code-reviewer` subagent.** It's read-only and safe. Add `code-reviewer.md` template, update `project_setup.py` to copy all agent templates generically, add `Agent(code-reviewer)` to the current monolithic implementer's tools. The implementer optionally spawns it instead of using `Skill`.

3. **Implement `ml-evaluator` subagent.** Read-mostly. Validate that it correctly extracts OOF from run logs and writes to `EXPERIMENT_STATE.json`.

4. **Implement `ml-developer` subagent.** Core refactor — the merged coder+debugger. The implementer-lead transitions to coordinator role. Keep the monolithic implementer as a fallback for one iteration.

5. **Implement `ml-scientist` subagent.** Add after `code-reviewer` + `ml-developer` are proven. Test with a seeded leakage bug.

6. **Implement `ml-scaffolder` + `submission-builder`.** Simple, bounded. Last because they are the least risky parts of the monolith.

7. **Remove `Edit`, `Bash`, `Grep`, `Skill` from the implementer-lead.** Only after steps 1–6 are validated in production.

---

## 11. Response to External Review (Gemini)

After an external architectural review, three criticisms were accepted and incorporated into this document:

**Accepted — Coder-debugger should not be split.** The write-run-fix loop is a tight feedback cycle sharing the same mental model of the codebase. Splitting it adds a cold-start penalty and lossy coordinator-summary in the middle of the most critical loop. `ml-coder` + `ml-debugger` → `ml-developer`. This is the single largest change.

**Accepted — Artifact handshake via `EXPERIMENT_STATE.json`.** The original design had the coordinator parsing subagent conversational output to decide routing. This is lossy. Subagents write structured JSON; the coordinator reads it. Routing becomes deterministic and auditable.

**Accepted — `ml-scientist` for logical ML bugs.** The `validate_bash.sh` mechanics of a debugger cannot fix data leakage. A code-review CRITICAL that says "target encoding fitted on full train" requires ML reasoning to resolve, not bash retry loops. A distinct logical-reasoning agent with `code-review` + `feature-engineering` skills is the right tool.

**Rejected — "Thin coordinator is a bottleneck."** This was correct as stated for the original design (free-text parsing). With `EXPERIMENT_STATE.json` in place, the coordinator reads a JSON file to make routing decisions. The concern dissolves once the artifact handshake is in place.

**Partial — Waterfall illusion.** The original text described phases as a sequence 1→6. The routing is actually a directed graph with back-edges (REVIEW can send back to SCIENCE → DEVELOP → EVALUATE). The architecture diagram and lead prompt now make this explicit.

---

## 12. Verdict

The monolithic implementer was the right first step — get end-to-end working before decomposing. The decomposition is now the right next step.

The final design has three clean separations:
- **Write-run-fix** stays in one context (`ml-developer`). Context coherence over phase purity.
- **Execution errors** vs **logical ML bugs** are handled by different agents (`ml-developer` vs `ml-scientist`). The distinction matters: a bash-retry loop cannot fix data leakage.
- **Routing decisions** are made by reading structured JSON, not parsing free text. The coordinator is thin in complexity, not thin in capability.

The Python orchestrator remains the macro-level routing logic. The implementer-lead handles micro-level routing within an experiment. Each layer does what it's best suited for: deterministic state management in Python, adaptive judgment in the LLM coordinator, focused execution in bounded worker subagents.
