"""Tests for the unified three-phase environment."""

import pytest

from sre_env.models import SREAction
from sre_env.server.unified_environment import UnifiedSREEnvironment


@pytest.fixture
def env() -> UnifiedSREEnvironment:
    return UnifiedSREEnvironment()


def test_easy_uses_15_ticks(env: UnifiedSREEnvironment) -> None:
    obs = env.reset(difficulty="easy", scenario_id="easy_001")
    assert obs.max_ticks == 15


def test_medium_uses_20_ticks(env: UnifiedSREEnvironment) -> None:
    obs = env.reset(difficulty="medium", scenario_id="medium_001")
    assert obs.max_ticks == 20


def test_hard_uses_25_ticks(env: UnifiedSREEnvironment) -> None:
    obs = env.reset(difficulty="hard", scenario_id="hard_001")
    assert obs.max_ticks == 25


def test_phase2_unlocks_on_correct_log(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.phase2_unlocked is True
    assert obs.security_sub_quest is not None
    assert obs.security_sub_quest["task_id"] == "sqli_login"


def test_phase2_does_not_unlock_on_wrong_log(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="api-gateway"))
    assert obs.phase2_unlocked is False
    assert obs.security_sub_quest is None


def test_classify_vuln_before_phase2_blocked(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(
        SREAction(tool="classify_vuln", vulnerability_type="sql_injection")
    )
    assert "not active" in obs.last_action_result.lower() or obs.reward < 0


def test_postmortem_blocked_before_tick_5(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="post_mortem", root_cause="test"))
    assert "not yet available" in obs.last_action_result.lower()


def test_postmortem_available_at_tick_5(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    for _ in range(5):
        env.step(SREAction(tool="get_metrics", service="cache", metric="cpu"))
    obs = env.step(
        SREAction(
            tool="post_mortem",
            root_cause="database OOM from SQL injection attack",
            attack_vector="sql injection in login handler",
            fix_sequence=["restart database", "parameterized query patch"],
            prevention="use parameterized queries",
        )
    )
    assert env._ep["postmortem_submitted"] is True
    assert env._ep["postmortem_score"] >= 0.0
    assert "post-mortem submitted" in obs.last_action_result.lower()


def test_full_easy_optimal_solve(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    env.step(SREAction(tool="get_logs", service="database"))
    env.step(SREAction(tool="classify_vuln", vulnerability_type="sql_injection"))
    env.step(SREAction(tool="apply_patch", patch_id="parameterized_query"))
    env.step(SREAction(tool="verify_patch"))
    env.step(SREAction(tool="restart", service="database"))
    obs = env.step(
        SREAction(
            tool="post_mortem",
            root_cause="database OOM caused by SQL injection dropping users table",
            attack_vector="sql injection in login handler via unsanitized concatenation",
            fix_sequence=["parameterized query patch", "database restart"],
            prevention="always use parameterized queries, add WAF rules",
        )
    )
    assert env._ep["success"] is True
    assert obs.done is True
    assert env._ep["final_score"] > 0.7


def test_final_score_components_all_present(env: UnifiedSREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    env.step(SREAction(tool="get_logs", service="database"))
    env.step(SREAction(tool="restart", service="database"))
    for _ in range(4):
        env.step(SREAction(tool="get_metrics", service="cache", metric="cpu"))
    obs = env.step(SREAction(tool="post_mortem", root_cause="db crash"))
    assert "final_score" in env._ep
    assert 0.0 <= env._ep["final_score"] <= 1.0
    assert set(obs.phase_scores) == {"infrastructure", "security", "postmortem"}
