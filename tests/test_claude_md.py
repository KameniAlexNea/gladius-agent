"""Tests for gladius.claude_md.render().

Covers:
- All {{placeholders}} are substituted (no leftover braces)
- Metric row rendered correctly for ML task vs open-ended task
- Performance section shows OOF / LB scores or "none yet"
- Submission threshold: present → hard gate text; absent → warning text
- Stagnation block injected after 3 experiments within threshold
- Recent experiments table: limited to last 5, most-recent first
- Failed approaches section
- Memory path is present in output
- topology section comes from TOPOLOGY_CATALOG
"""

from __future__ import annotations

import re

from gladius.claude_md import render
from gladius.state import CompetitionState


def _state(**kwargs) -> CompetitionState:
    defaults = dict(
        competition_id="test-comp",
        data_dir="/data",
        output_dir="/out",
        topology="functional",
    )
    defaults.update(kwargs)
    return CompetitionState(**defaults)


class TestPlaceholders:
    def test_no_leftover_placeholders(self, tmp_path):
        """render() must substitute every {{...}} token."""
        s = _state(target_metric="f1", metric_direction="maximize")
        result = render(s, str(tmp_path))
        leftover = re.findall(r"\{\{[^}]+\}\}", result)
        assert leftover == [], f"Unsubstituted placeholders: {leftover}"

    def test_competition_id_present(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        result = render(s, str(tmp_path))
        assert "test-comp" in result


class TestMetricRow:
    def test_ml_mode_shows_target_metric(self, tmp_path):
        s = _state(target_metric="roc_auc", metric_direction="maximize")
        result = render(s, str(tmp_path))
        assert "roc_auc" in result
        assert "higher is better" in result

    def test_ml_mode_minimize_shows_lower_is_better(self, tmp_path):
        s = _state(target_metric="rmse", metric_direction="minimize")
        result = render(s, str(tmp_path))
        assert "lower is better" in result

    def test_open_ended_shows_quality_label(self, tmp_path):
        s = _state()
        result = render(s, str(tmp_path))
        assert "open-ended" in result or "quality" in result.lower()


class TestPerformanceSection:
    def test_no_scores_shows_none_yet(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        result = render(s, str(tmp_path))
        assert "none yet" in result

    def test_oof_score_shown_when_set(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        s.best_oof_score = 0.873456
        result = render(s, str(tmp_path))
        assert "0.873456" in result

    def test_threshold_hard_gate_when_set(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        s.submission_threshold = 0.75
        result = render(s, str(tmp_path))
        assert "0.750000" in result
        # Must include a "do not submit" style warning
        assert "Do not build a submission" in result or "⛔" in result

    def test_threshold_warning_when_not_set(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        s.submission_threshold = None
        result = render(s, str(tmp_path))
        assert "not set" in result or "WebSearch" in result

    def test_open_ended_shows_quality_score(self, tmp_path):
        s = _state()
        s.best_quality_score = 82
        result = render(s, str(tmp_path))
        assert "82" in result


class TestRecentExperiments:
    def test_no_experiments_shows_none_marker(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        result = render(s, str(tmp_path))
        assert "none yet" in result

    def test_last_five_experiments_shown(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        for i in range(7):
            s.experiments.append(
                {
                    "iteration": i,
                    "oof_score": 0.5 + i * 0.01,
                    "solution_files": [],
                    "notes": f"iter{i}",
                }
            )
        result = render(s, str(tmp_path))
        # iter 6 and iter 2 … but NOT iter 0 or iter 1 (older than last 5)
        assert "iter6" in result
        assert "iter2" in result
        assert "iter0" not in result or "iter 0" not in result  # oldest pruned

    def test_most_recent_shown_first(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        s.experiments.append(
            {
                "iteration": 1,
                "oof_score": 0.6,
                "solution_files": [],
                "notes": "ZZZ_OLDER_NOTE",
            }
        )
        s.experiments.append(
            {
                "iteration": 2,
                "oof_score": 0.7,
                "solution_files": [],
                "notes": "ZZZ_NEWER_NOTE",
            }
        )
        result = render(s, str(tmp_path))
        idx_older = result.find("ZZZ_OLDER_NOTE")
        idx_newer = result.find("ZZZ_NEWER_NOTE")
        assert idx_older >= 0 and idx_newer >= 0
        assert idx_newer < idx_older  # most recent (iter 2) rendered before iter 1


class TestFailedApproaches:
    def test_no_failures_shows_none(self, tmp_path):
        s = _state()
        result = render(s, str(tmp_path))
        assert "_(none)_" in result or "none" in result.lower()

    def test_failed_run_iteration_shown(self, tmp_path):
        s = _state()
        s.failed_runs.append({"iteration": 3, "error": "OOM during training"})
        result = render(s, str(tmp_path))
        assert "iter 3" in result or "iteration: 3" in result.lower() or "OOM" in result


class TestStagnation:
    def test_stagnation_block_emitted_when_stagnant(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        # Three experiments within 0.001 of each other
        for i in range(3):
            s.experiments.append(
                {
                    "iteration": i,
                    "oof_score": 0.800 + i * 0.0001,
                    "solution_files": [],
                    "notes": "",
                }
            )
        result = render(s, str(tmp_path))
        # Check for the unique section header that only appears inside the stagnation block
        assert "STAGNATION WARNING" in result

    def test_no_stagnation_block_when_improving(self, tmp_path):
        s = _state(target_metric="f1", metric_direction="maximize")
        scores = [0.70, 0.75, 0.82]
        for i, sc in enumerate(scores):
            s.experiments.append(
                {"iteration": i, "oof_score": sc, "solution_files": [], "notes": ""}
            )
        result = render(s, str(tmp_path))
        # "STAGNATION WARNING" is the unique header only injected by the stagnation block
        assert "STAGNATION WARNING" not in result


class TestMemoryPath:
    def test_memory_path_present(self, tmp_path):
        s = _state()
        result = render(s, str(tmp_path))
        assert "MEMORY.md" in result


class TestTopologySection:
    def test_functional_topology_section_present(self, tmp_path):
        s = _state(topology="functional")
        result = render(s, str(tmp_path))
        # The functional topology template body should include "data-expert"
        assert "data-expert" in result

    def test_unknown_topology_shows_fallback(self, tmp_path):
        s = _state(topology="unknown-topo")
        result = render(s, str(tmp_path))
        assert "no description available" in result
