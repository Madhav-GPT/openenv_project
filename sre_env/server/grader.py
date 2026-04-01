"""Per-step reward computation for dense reward shaping."""

from __future__ import annotations


class SREGrader:
    """Reward shaping tuned for investigation-first incident response."""

    def compute(
        self,
        action_tool: str,
        action_service: str,
        root_cause_service: str,
        investigated_root_cause_before: bool,
        was_correct_fix: bool,
        was_trap_action: bool,
        alerts_cleared: int,
        all_healthy: bool,
    ) -> float:
        reward = -0.05

        if action_tool in {"get_logs", "get_metrics", "get_dependencies"}:
            if action_service == root_cause_service and not investigated_root_cause_before:
                reward += 0.15
            elif action_service != root_cause_service:
                reward += 0.08

        if was_correct_fix:
            reward += 0.35
        elif was_trap_action:
            reward -= 0.20

        reward += alerts_cleared * 0.20

        if all_healthy:
            reward += 1.00

        return round(max(-1.0, min(2.0, reward)), 4)
