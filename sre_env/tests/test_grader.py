"""Reward and grading tests."""

from sre_env.server.challenge import grade_episode
from sre_env.server.grader import SREGrader


def test_reward_root_cause_investigation_positive() -> None:
    reward = SREGrader().compute(
        action_tool="get_logs",
        action_service="database",
        root_cause_service="database",
        investigated_root_cause_before=False,
        was_correct_fix=False,
        was_trap_action=False,
        alerts_cleared=0,
        all_healthy=False,
    )
    assert reward > 0


def test_grade_episode_perfect_run_scores_high() -> None:
    report = grade_episode(
        {
            "scenario_id": "medium_001",
            "investigated_root_cause_service": True,
            "all_services_healthy": True,
            "current_tick": 4,
            "max_ticks": 20,
            "wrong_fix_attempts": 0,
            "correct_fixes_applied": 2,
        }
    )
    assert report.passed is True
    assert report.score >= 0.8
