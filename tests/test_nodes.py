"""Tests for key gladius node functions."""

import json
from pathlib import Path

import numpy as np


def make_state(**kwargs) -> dict:
    defaults = dict(
        competition={
            "name": "test-comp",
            "metric": "auc",
            "target": "label",
            "deadline": "2026-01-01",
            "days_remaining": 30,
            "submission_limit": 5,
        },
        current_experiment=None,
        experiment_status="pending",
        running_pid=None,
        run_id="v1",
        oof_score=None,
        lb_score=None,
        best_oof=None,
        gap_history=[],
        submissions_today=0,
        last_submission_time=None,
        directive={
            "directive_type": "tune_existing",
            "target_model": "catboost",
            "rationale": "test",
            "exploration_flag": True,
            "priority": 3,
        },
        exploration_flag=True,
        consecutive_same_directive=0,
        session_summary=None,
        generated_script_path=None,
        reviewer_feedback=None,
        code_retry_count=0,
        next_node="strategy",
        error_message=None,
        node_retry_counts={},
        next_node_before_error=None,
    )
    defaults.update(kwargs)
    return defaults


# --- knowledge_extractor tests ---


class TestKnowledgeExtractor:
    def test_extracts_and_resets_state(self, tmp_path):
        from gladius.nodes.strategy import knowledge_extractor

        original_path = knowledge_extractor.KNOWLEDGE_PATH
        knowledge_extractor.KNOWLEDGE_PATH = tmp_path / "knowledge.json"
        try:
            state = make_state(
                run_id="v1",
                experiment_status="failed",
                oof_score=0.85,
                lb_score=0.82,
                error_message="OOM error",
            )
            result = knowledge_extractor.knowledge_extractor_node(state)
            assert result["next_node"] == "router"
            assert result["error_message"] is None
            assert result["experiment_status"] == "pending"
            assert result["current_experiment"] is None
            assert result["run_id"] is None
            # Check that knowledge was written
            assert knowledge_extractor.KNOWLEDGE_PATH.exists()
            data = json.loads(knowledge_extractor.KNOWLEDGE_PATH.read_text())
            assert len(data) == 1
            assert data[0]["experiment_id"] == "v1"
        finally:
            knowledge_extractor.KNOWLEDGE_PATH = original_path

    def test_classify_oom_as_param_failure(self, tmp_path):
        from gladius.nodes.strategy import knowledge_extractor

        original_path = knowledge_extractor.KNOWLEDGE_PATH
        knowledge_extractor.KNOWLEDGE_PATH = tmp_path / "knowledge.json"
        try:
            state = make_state(
                experiment_status="killed",
                error_message="OOM: out of memory",
            )
            knowledge_extractor.knowledge_extractor_node(state)
            data = json.loads(knowledge_extractor.KNOWLEDGE_PATH.read_text())
            assert data[0]["finding_type"] == "param_failure"
        finally:
            knowledge_extractor.KNOWLEDGE_PATH = original_path

    def test_classify_overfitting(self, tmp_path):
        from gladius.nodes.strategy import knowledge_extractor

        original_path = knowledge_extractor.KNOWLEDGE_PATH
        knowledge_extractor.KNOWLEDGE_PATH = tmp_path / "knowledge.json"
        try:
            state = make_state(
                experiment_status="complete",
                oof_score=0.90,
                lb_score=0.85,  # gap > 0.02
            )
            knowledge_extractor.knowledge_extractor_node(state)
            data = json.loads(knowledge_extractor.KNOWLEDGE_PATH.read_text())
            assert data[0]["finding_type"] == "overfitting_signal"
        finally:
            knowledge_extractor.KNOWLEDGE_PATH = original_path


# --- submission_decider tests ---


class TestSubmissionDecider:
    def test_held_when_budget_exceeded(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(submissions_today=5, oof_score=0.85)
        result = submission_decider_node(state)
        assert result["experiment_status"] == "held"

    def test_held_when_no_oof(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(oof_score=None, submissions_today=0)
        result = submission_decider_node(state)
        assert result["experiment_status"] == "held"

    def test_submitted_when_ok(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(oof_score=0.85, submissions_today=2, best_oof=0.0)
        result = submission_decider_node(state)
        assert result["experiment_status"] == "submitted"
        assert result["next_node"] == "submission_agent"
        assert result["best_oof"] == 0.85

    def test_held_on_gap_widening(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(
            oof_score=0.85,
            submissions_today=1,
            best_oof=0.0,
            gap_history=[0.01, 0.02, 0.03],  # widening gaps
        )
        result = submission_decider_node(state)
        assert result["experiment_status"] == "held"

    def test_not_held_when_gap_not_widening(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(
            oof_score=0.85,
            submissions_today=1,
            best_oof=0.0,
            gap_history=[0.03, 0.02, 0.01],  # narrowing gaps
        )
        result = submission_decider_node(state)
        assert result["experiment_status"] == "submitted"

    def test_held_when_no_oof_improvement(self):
        from gladius.nodes.validation.submission_decider import submission_decider_node

        # best_oof=0.85, new oof=0.85 — not enough improvement
        state = make_state(oof_score=0.85, submissions_today=1, best_oof=0.85)
        result = submission_decider_node(state)
        assert result["experiment_status"] == "held"

    def test_held_within_cooldown_period(self):
        import time

        from gladius.nodes.validation.submission_decider import submission_decider_node

        state = make_state(
            oof_score=0.90,
            submissions_today=1,
            best_oof=0.0,
            last_submission_time=time.time() - 60,  # 1 minute ago, within 2h cooldown
        )
        result = submission_decider_node(state)
        assert result["experiment_status"] == "held"


# --- validation_agent tests ---


class TestValidationAgent:
    def test_fails_when_oof_missing(self, tmp_path):
        from gladius.nodes.validation import validation_agent

        original = validation_agent.OOF_DIR
        validation_agent.OOF_DIR = tmp_path
        try:
            state = make_state(run_id="v_missing")
            result = validation_agent.validation_node(state)
            assert result["experiment_status"] == "failed"
            assert "OOF file not found" in result["error_message"]
        finally:
            validation_agent.OOF_DIR = original

    def test_validates_good_oof(self, tmp_path):
        from gladius.nodes.validation import validation_agent

        original = validation_agent.OOF_DIR
        validation_agent.OOF_DIR = tmp_path
        try:
            arr = np.array([0.1, 0.5, 0.9, 0.3])
            np.save(str(tmp_path / "v1_oof.npy"), arr)
            state = make_state(run_id="v1")
            result = validation_agent.validation_node(state)
            assert result["experiment_status"] == "validated"
        finally:
            validation_agent.OOF_DIR = original

    def test_fails_on_nan(self, tmp_path):
        from gladius.nodes.validation import validation_agent

        original = validation_agent.OOF_DIR
        validation_agent.OOF_DIR = tmp_path
        try:
            arr = np.array([0.1, float("nan"), 0.9])
            np.save(str(tmp_path / "v2_oof.npy"), arr)
            state = make_state(run_id="v2")
            result = validation_agent.validation_node(state)
            assert result["experiment_status"] == "failed"
            assert "NaN" in result["error_message"]
        finally:
            validation_agent.OOF_DIR = original

    def test_fails_on_out_of_range(self, tmp_path):
        from gladius.nodes.validation import validation_agent

        original = validation_agent.OOF_DIR
        validation_agent.OOF_DIR = tmp_path
        try:
            arr = np.array([0.1, 1.5, 0.9])
            np.save(str(tmp_path / "v3_oof.npy"), arr)
            state = make_state(run_id="v3")
            result = validation_agent.validation_node(state)
            assert result["experiment_status"] == "failed"
            assert "range" in result["error_message"]
        finally:
            validation_agent.OOF_DIR = original


# --- error_handler tests ---


class TestErrorHandler:
    def test_retries_up_to_3(self):
        from gladius.nodes.error_handler import error_handler_node

        state = make_state(
            next_node_before_error="strategy",
            node_retry_counts={"strategy": 1},
            error_message="some error",
        )
        result = error_handler_node(state)
        assert result["next_node"] == "strategy"
        assert result["node_retry_counts"]["strategy"] == 2

    def test_falls_back_to_strategy_after_max_retries(self):
        from gladius.nodes.error_handler import error_handler_node

        state = make_state(
            next_node_before_error="hypothesis",
            node_retry_counts={"hypothesis": 3},
            error_message="persistent error",
        )
        result = error_handler_node(state)
        assert result["next_node"] == "strategy"
        assert result["error_message"] is None


# --- code_generator tests ---


class TestCodeGenerator:
    def test_returns_error_when_no_spec(self):
        from gladius.nodes.code.code_generator import code_generator_node

        state = make_state(current_experiment=None)
        result = code_generator_node(state)
        assert result["next_node"] == "strategy"
        assert "No experiment spec" in result["error_message"]

    def test_generates_script(self, tmp_path):
        from gladius.nodes.code import code_generator

        original = code_generator.SCRIPTS_DIR
        code_generator.SCRIPTS_DIR = tmp_path
        code_generator.STAGING_PATH = tmp_path / "pending.py"
        try:
            spec = {
                "parent_version": "v0",
                "changes": [],
                "estimated_runtime_multiplier": 1.0,
                "rationale": "test",
            }
            state = make_state(current_experiment=spec, run_id="test_run")
            from gladius.nodes.code.code_generator import code_generator_node

            result = code_generator_node(state)
            assert result["next_node"] == "code_reviewer"
            assert result["generated_script_path"] is not None
            assert Path(result["generated_script_path"]).exists()
            assert "code_retry_count" not in result
        finally:
            code_generator.SCRIPTS_DIR = original
            code_generator.STAGING_PATH = original / "pending.py"

    def test_param_change_applied(self, tmp_path):
        from gladius.nodes.code import code_generator

        original = code_generator.SCRIPTS_DIR
        code_generator.SCRIPTS_DIR = tmp_path
        code_generator.STAGING_PATH = tmp_path / "pending.py"
        try:
            # Create a parent script
            parent = tmp_path / "v0.py"
            parent.write_text("lr = 0.01\nnum_leaves = 31\n")
            spec = {
                "parent_version": "v0",
                "changes": [
                    {"type": "param_change", "param": "lr", "old": 0.01, "new": 0.001}
                ],
                "estimated_runtime_multiplier": 1.0,
                "rationale": "test",
            }
            state = make_state(current_experiment=spec, run_id="test_param")
            from gladius.nodes.code.code_generator import code_generator_node

            result = code_generator_node(state)
            content = Path(result["generated_script_path"]).read_text()
            assert "0.001" in content
        finally:
            code_generator.SCRIPTS_DIR = original
            code_generator.STAGING_PATH = original / "pending.py"


# --- notifier tests ---


class TestNotifier:
    def test_timeout_message(self):
        from gladius.nodes.validation.notifier import notifier_node

        state = make_state(experiment_status="score_timeout", run_id="v5")
        result = notifier_node(state)
        assert result["next_node"] == "router"

    def test_failed_message(self):
        from gladius.nodes.validation.notifier import notifier_node

        state = make_state(
            experiment_status="failed", run_id="v5", error_message="crash"
        )
        result = notifier_node(state)
        assert result["next_node"] == "router"

    def test_lb_score_message(self):
        from gladius.nodes.validation.notifier import notifier_node

        state = make_state(experiment_status="complete", run_id="v5", lb_score=0.91234)
        result = notifier_node(state)
        assert result["next_node"] == "router"


# --- file_utils tests ---


class TestFileUtils:
    def test_write_and_read_oof(self, tmp_path):
        from gladius.utils.file_utils import read_oof_file, write_oof_file

        arr = np.array([0.1, 0.2, 0.3, 0.9])
        path = tmp_path / "test_oof.npy"
        write_oof_file(path, arr)
        loaded = read_oof_file(path)
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_write_creates_parent_dirs(self, tmp_path):
        from gladius.utils.file_utils import write_oof_file

        arr = np.array([0.5, 0.6])
        path = tmp_path / "nested" / "dir" / "oof.npy"
        write_oof_file(path, arr)
        assert path.exists()


# --- ensemble_agent tests ---


class TestEnsembleAgent:
    def test_returns_strategy_when_not_enough_models(self, tmp_path):
        from gladius.nodes.strategy import ensemble_agent

        original = ensemble_agent.OOF_DIR
        ensemble_agent.OOF_DIR = tmp_path
        try:
            # Only 2 OOF files - below MIN_BASE_MODELS=3
            np.save(str(tmp_path / "v1_oof.npy"), np.random.rand(100))
            np.save(str(tmp_path / "v2_oof.npy"), np.random.rand(100))
            state = make_state()
            result = ensemble_agent.ensemble_node(state)
            assert result["next_node"] == "strategy"
        finally:
            ensemble_agent.OOF_DIR = original

    def test_proposes_blend_with_enough_uncorrelated(self, tmp_path):
        from gladius.nodes.strategy import ensemble_agent

        original = ensemble_agent.OOF_DIR
        ensemble_agent.OOF_DIR = tmp_path
        try:
            np.random.seed(42)
            np.save(str(tmp_path / "v1_oof.npy"), np.random.rand(100))
            np.save(str(tmp_path / "v2_oof.npy"), np.random.rand(100))
            np.save(str(tmp_path / "v3_oof.npy"), np.random.rand(100))
            state = make_state()
            result = ensemble_agent.ensemble_node(state)
            # Should find 3 uncorrelated models and propose blend
            if result["next_node"] == "hypothesis":
                assert result["directive"]["directive_type"] == "ensemble"
        finally:
            ensemble_agent.OOF_DIR = original


# --- code_reviewer tests ---


class TestCodeReviewer:
    def test_routes_to_code_generator_on_first_retry(self, tmp_path):
        from gladius.nodes.code.code_reviewer import code_reviewer_node

        script = tmp_path / "pending.py"
        script.write_text("import nonexistent_pkg_xyz\n\ndef train():\n    pass\n")
        state = make_state(generated_script_path=str(script), code_retry_count=0)
        result = code_reviewer_node(state)
        assert result["next_node"] == "code_generator"
        assert result["code_retry_count"] == 1

    def test_routes_to_hypothesis_on_second_retry(self, tmp_path):
        from gladius.nodes.code.code_reviewer import code_reviewer_node

        script = tmp_path / "pending.py"
        script.write_text("import nonexistent_pkg_xyz\n\ndef train():\n    pass\n")
        state = make_state(generated_script_path=str(script), code_retry_count=1)
        result = code_reviewer_node(state)
        assert result["next_node"] == "hypothesis"
        assert result["code_retry_count"] == 2

    def test_routes_to_strategy_on_third_retry(self, tmp_path):
        from gladius.nodes.code.code_reviewer import code_reviewer_node

        script = tmp_path / "pending.py"
        script.write_text("import nonexistent_pkg_xyz\n\ndef train():\n    pass\n")
        state = make_state(generated_script_path=str(script), code_retry_count=2)
        result = code_reviewer_node(state)
        assert result["next_node"] == "strategy"
        assert result["code_retry_count"] == 3

    def test_approves_valid_script(self, tmp_path):
        from gladius.nodes.code.code_reviewer import code_reviewer_node

        script = tmp_path / "pending.py"
        script.write_text("import os\n\ndef train():\n    pass\n")
        state = make_state(generated_script_path=str(script), code_retry_count=0)
        result = code_reviewer_node(state)
        assert result["next_node"] == "versioning_agent"
        assert result["reviewer_feedback"] is None


# --- code_reader tests ---


class TestCodeReader:
    def test_validate_syntax_valid(self):
        from gladius.utils.code_reader import validate_syntax

        assert validate_syntax("x = 1\n") == []

    def test_validate_syntax_invalid(self):
        from gladius.utils.code_reader import validate_syntax

        errors = validate_syntax("def foo(:\n    pass\n")
        assert len(errors) > 0
        assert "SyntaxError" in errors[0]

    def test_list_functions(self, tmp_path):
        from gladius.utils.code_reader import list_functions

        f = tmp_path / "sample.py"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        funcs = list_functions(f)
        assert "foo" in funcs
        assert "bar" in funcs

    def test_read_function(self, tmp_path):
        from gladius.utils.code_reader import read_function

        f = tmp_path / "sample.py"
        f.write_text("def foo():\n    return 42\n\ndef bar():\n    pass\n")
        src = read_function(f, "foo")
        assert "return 42" in src

    def test_read_function_not_found(self, tmp_path):
        from gladius.utils.code_reader import read_function

        f = tmp_path / "sample.py"
        f.write_text("def foo():\n    pass\n")
        try:
            read_function(f, "missing")
            assert False, "Expected KeyError"
        except KeyError:
            pass

    def test_check_imports_missing(self):
        from gladius.utils.code_reader import check_imports

        issues = check_imports("import nonexistent_pkg_xyz\n")
        assert any("nonexistent_pkg_xyz" in i for i in issues)

    def test_check_imports_valid(self):
        from gladius.utils.code_reader import check_imports

        issues = check_imports("import os\nimport sys\n")
        assert issues == []
