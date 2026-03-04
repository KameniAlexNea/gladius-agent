import asyncio

from gladius.agents import planner as planner_module
from gladius.state import CompetitionState


def _state() -> CompetitionState:
    return CompetitionState(
        competition_id="comp-1",
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        target_metric="auc_roc",
        metric_direction="maximize",
    )


def test_extract_parallel_plans_from_structured_markdown():
    plan_text = """
## Approach 1
Use LightGBM with target encoding.
1. Build features
2. Train CV

## Approach 2
Use CatBoost with categorical features.
1. Minimal preprocessing
2. Train CV
""".strip()

    plans = planner_module._extract_parallel_plans(plan_text, n_parallel=2)

    assert len(plans) == 2
    assert "Approach 1" in plans[0]["plan_text"]
    assert "Approach 2" in plans[1]["plan_text"]
    assert plans[0]["approach_summary"]
    assert plans[1]["approach_summary"]


def test_run_planner_generates_alternative_when_structured_sections_missing(
    monkeypatch,
):
    calls = []

    async def fake_run_planning_agent(**kwargs):
        calls.append(kwargs)
        # First call: primary plan only (no Approach headings)
        if len(calls) == 1:
            return (
                "Primary baseline plan\n1. train xgboost\n2. submit",
                "planner-session-1",
            )
        # Second call: explicit alternative response
        return (
            "Alternative plan with neural net\n1. prepare embeddings\n2. train model",
            "planner-session-1",
        )

    monkeypatch.setattr(planner_module, "run_planning_agent", fake_run_planning_agent)

    plan_dict, session_id = asyncio.run(
        planner_module.run_planner(
            state=_state(),
            data_dir="/tmp/data",
            project_dir="/tmp/project",
            platform="fake",
            n_parallel=2,
        )
    )

    assert session_id == "planner-session-1"
    assert plan_dict["plan_text"].startswith("Primary baseline plan")
    assert len(plan_dict["plans"]) == 2
    assert "Primary baseline plan" in plan_dict["plans"][0]["plan_text"]
    assert "Alternative plan" in plan_dict["plans"][1]["plan_text"]

    # Ensure the alternative call resumed from the primary planner session.
    assert calls[1]["resume"] == "planner-session-1"
