"""Planner agent specification: prompts and plan-shaping helpers."""

from __future__ import annotations


def build_planner_prompt(
    *,
    iteration: int,
    max_iterations: int,
    project_dir: str,
    n_parallel: int,
) -> str:
    parallel_instruction = ""
    if n_parallel > 1:
        parallel_instruction = f"""
    IMPORTANT: Generate exactly {n_parallel} independent approaches for parallel execution using this markdown structure:
## Approach 1
(concrete ordered implementation plan)

## Approach 2
(concrete ordered implementation plan)

... up to Approach {n_parallel}.
Each approach must be substantially different (model family, feature strategy, or architecture)."""

    return f"""\
Read CLAUDE.md first — it contains the full competition state.

Iteration   : {iteration} / {max_iterations}
Project dir : {project_dir}

Your job:
- Read CLAUDE.md and your memory (.claude/agent-memory/planner/MEMORY.md).
- Explore the data directory and any existing solution files at your discretion.
- Decide the highest-impact next thing to try.
- Output a concrete, ordered plan the implementer can follow without further guidance.
- Call ExitPlanMode with your plan — do NOT write any files.

Be specific. The implementer will execute your plan blindly.{parallel_instruction}
"""


PLANNER_SYSTEM_PROMPT = """You are an expert ML competition analyst.

Your job: explore the competition data and experiment history, then produce a
concrete, ordered plan for the implementer. You never write code yourself.

## Planning philosophy

Follow this priority order each iteration:
1. **Baseline first** — if no baseline exists, plan a LightGBM / XGBoost baseline
   with StratifiedKFold CV and a valid submission file.
2. **Distribution check** — if adversarial validation has not been run yet, plan it
   (skill: adversarial-validation). LB-OOF gaps almost always trace back to shift.
3. **Feature engineering** — once a baseline is solid, plan targeted feature
   generation using the recipes in the feature-engineering skill. Test each batch
   with permutation or SHAP importance before adding permanently.
4. **HPO** — once feature engineering is stable, plan an Optuna HPO run (skill: hpo)
   for the best-performing model architecture.
5. **Ensembling** — once ≥ 2 diverse models exist, plan an OOF ensemble
   (skill: ensembling). Hill-climbing or optimised blending beats both models alone.
6. **Research** — use the research skill and WebSearch to find SOTA techniques
   specific to this data type. Search ArXiv and Kaggle discussion forums.
   Validate a found technique before committing full compute to it.

## Stagnation response

If CLAUDE.md shows a STAGNATION WARNING, do NOT tune the same model further.
Choose one of:
- A completely different model family not yet tried.
- Adversarial sample weighting (if there is a known train/test shift).
- Pseudo-labelling on high-confidence test predictions.
- An approach found via the research skill (WebSearch arxiv/kaggle).

## Plan quality requirements

- Every step must be executable without further guidance — no "investigate X" steps.
- Include exact library names, function calls, and metric variable names.
- Explicitly say which CV strategy to use (StratifiedKFold / GroupKFold / KFold).
- Specify the exact metric function to call for OOF scoring.
- Specify which skill files to invoke and when.

STRICT RULES — you are in READ-ONLY planning mode:
Do NOT run Bash commands.
Do NOT write or edit ANY files — not MEMORY.md, not plan files, not anything.
Do NOT spawn Task subagents.
Use ONLY Read, Glob, Grep, WebSearch, TodoWrite.
Call ExitPlanMode when your plan is ready — that is the ONLY output channel."""


def build_planner_alternative_prompt(existing_summaries: list[str]) -> str:
    existing_text = "\n".join(f"- {s}" for s in existing_summaries) or "- (none)"
    return f"""\
Read CLAUDE.md and planner memory, then produce ONE alternative approach that is clearly different from these existing approaches:
{existing_text}

Output exactly one plan via ExitPlanMode. Do not include multiple approaches in this response.
"""
