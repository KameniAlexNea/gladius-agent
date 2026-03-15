"""Tests for parallel plan extraction (autonomous topology)."""

from gladius.agents.topologies.autonomous import _extract_parallel_plans


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

    plans = _extract_parallel_plans(plan_text, n_parallel=2)

    assert len(plans) == 2
    assert "Approach 1" in plans[0]
    assert "Approach 2" in plans[1]
    assert "LightGBM" in plans[0]
    assert "CatBoost" in plans[1]


def test_extract_parallel_plans_falls_back_to_single_when_no_headings():
    plan_text = "Just a single plan with no Approach headings."
    plans = _extract_parallel_plans(plan_text, n_parallel=2)
    assert len(plans) == 1
    assert plans[0] == plan_text


def test_extract_parallel_plans_caps_at_n_parallel():
    plan_text = """
## Approach 1
First plan

## Approach 2
Second plan

## Approach 3
Third plan
""".strip()

    plans = _extract_parallel_plans(plan_text, n_parallel=2)
    assert len(plans) == 2


def test_extract_parallel_plans_deduplicates_identical_sections():
    plan_text = """
## Approach 1
Identical content here

## Approach 1
Identical content here
""".strip()

    plans = _extract_parallel_plans(plan_text, n_parallel=3)
    assert len(plans) == 1


def test_extract_parallel_plans_handles_case_insensitive_headings():
    plan_text = """
## APPROACH 1
Upper case heading

## approach 2
Lower case heading
""".strip()

    plans = _extract_parallel_plans(plan_text, n_parallel=2)
    assert len(plans) == 2
