from gladius.agents.specs.implementer_spec import (
    IMPLEMENTER_OUTPUT_SCHEMA,
    build_implementer_prompt,
)
from gladius.agents.specs.planner_spec import build_planner_prompt
from gladius.agents.specs.validation_spec import build_validation_prompt


def test_build_planner_prompt_includes_parallel_instruction_without_leading_blank_block():
    prompt = build_planner_prompt(
        iteration=2,
        max_iterations=10,
        project_dir="/tmp/project",
        n_parallel=2,
    )

    assert "IMPORTANT: Generate exactly 2 independent approaches" in prompt
    assert "## Approach 1" in prompt
    assert "## Approach 2" in prompt


def test_build_implementer_prompt_includes_plan_and_metric():
    prompt = build_implementer_prompt(
        plan={
            "approach_summary": "Try catboost baseline",
            "plan_text": "1. Load data\n2. Train model",
        },
        target_metric="auc_roc",
    )

    assert "Try catboost baseline" in prompt
    assert "1. Load data" in prompt
    assert "auc_roc" in prompt


def test_validation_prompt_metric_mode_contains_submission_checks():
    prompt = build_validation_prompt(
        solution_path="solution.py",
        oof_score=0.8123,
        quality_score=0,
        submission_path="submission.csv",
        target_metric="auc_roc",
        metric_direction="maximize",
        best_oof_score=0.801,
        best_quality_score=None,
        submission_count=1,
        max_submissions_per_day=5,
        quota_instruction="",
    )

    assert "OOF score     : 0.812300" in prompt
    assert "Use Read to open submission.csv" in prompt


def test_implementer_schema_required_keys_stable():
    required = IMPLEMENTER_OUTPUT_SCHEMA["required"]
    assert "status" in required
    assert "oof_score" in required
    assert "quality_score" in required
