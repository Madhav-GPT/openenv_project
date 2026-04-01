"""Model validation tests."""

from sre_env.models import SREAction, SREObservation, ServiceInfo


def test_action_requires_metric_for_metrics_tool_payload_shape() -> None:
    action = SREAction(tool="get_metrics", service="cache", metric="cpu")
    assert action.metric == "cpu"
    assert action.replicas is None


def test_observation_contains_reward_and_done() -> None:
    observation = SREObservation(
        tick=1,
        max_ticks=20,
        difficulty="easy",
        services={
            "database": ServiceInfo(
                name="database",
                status="crashed",
                cpu_pct=0,
                memory_pct=0,
                error_rate_pct=100,
                latency_ms=0,
                replicas=1,
            )
        },
        active_alerts=[],
        last_action_result="noop",
        tool_output=None,
        episode_complete=False,
        success=False,
        failure_reason=None,
        reward=-0.05,
        done=False,
    )
    assert observation.reward == -0.05
    assert observation.done is False
