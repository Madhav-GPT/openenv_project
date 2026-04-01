"""Environment behavior tests."""

import pytest

from sre_env.models import SREAction
from sre_env.server.environment import SREEnvironment


@pytest.fixture
def env() -> SREEnvironment:
    return SREEnvironment()


def test_reset_returns_observation(env: SREEnvironment) -> None:
    obs = env.reset(difficulty="easy", scenario_id="easy_001")
    assert obs.tick == 0
    assert len(obs.active_alerts) > 0
    assert obs.difficulty == "easy"
    assert not obs.episode_complete


def test_step_increments_tick(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.tick == 1


def test_investigation_returns_tool_output(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.tool_output is not None
    assert "OutOfMemoryError" in obs.tool_output


def test_correct_fix_resolves_easy(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.episode_complete is True
    assert obs.success is True


def test_trap_action_negative_reward(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="restart", service="api-gateway"))
    assert obs.reward < 0
    assert obs.episode_complete is False


def test_investigating_root_cause_positive_reward(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = env.step(SREAction(tool="get_logs", service="database"))
    assert obs.reward > 0


def test_medium_trap_mentions_cache(env: SREEnvironment) -> None:
    env.reset(difficulty="medium", scenario_id="medium_001")
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.episode_complete is False
    assert "cache" in obs.last_action_result.lower() or "root" in obs.last_action_result.lower()


def test_medium_correct_sequence(env: SREEnvironment) -> None:
    env.reset(difficulty="medium", scenario_id="medium_001")
    env.step(SREAction(tool="restart", service="cache"))
    obs = env.step(SREAction(tool="restart", service="database"))
    assert obs.success is True
    assert obs.episode_complete is True


def test_hard_rollback_first(env: SREEnvironment) -> None:
    env.reset(difficulty="hard", scenario_id="hard_001")
    env.step(SREAction(tool="rollback", service="worker", version="previous"))
    env.step(SREAction(tool="restart", service="database"))
    obs = env.step(SREAction(tool="restart", service="api-gateway"))
    assert obs.success is True


def test_state_metadata(env: SREEnvironment) -> None:
    env.reset(difficulty="hard", scenario_id="hard_001")
    state = env.state
    assert state.difficulty == "hard"
    assert state.step_count == 0


def test_timeout(env: SREEnvironment) -> None:
    env.reset(difficulty="easy", scenario_id="easy_001")
    obs = None
    for _ in range(21):
        obs = env.step(SREAction(tool="get_metrics", service="cache", metric="cpu"))
        if obs.episode_complete:
            break
    assert obs is not None
    assert obs.episode_complete is True


def test_grader_score_perfect_easy(env: SREEnvironment) -> None:
    from sre_env.server.challenge import grade_episode

    env.reset(difficulty="easy", scenario_id="easy_001")
    env.step(SREAction(tool="get_logs", service="database"))
    env.step(SREAction(tool="restart", service="database"))
    score_data = grade_episode(env._state_dict())
    assert score_data.score >= 0.7
