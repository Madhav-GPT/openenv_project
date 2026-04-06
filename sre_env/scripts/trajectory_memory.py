"""Trajectory memory for few-shot in-context learning.

Stores (observation, action, reward) trajectories from past episodes and
provides the top-K highest-reward trajectories as few-shot examples in the
system prompt. This enables genuine learning: the agent improves because
it sees more examples of what worked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryStep:
    """One step in a trajectory."""

    tick: int
    observation_summary: str
    action: dict[str, Any]
    reward: float


@dataclass
class Trajectory:
    """A complete episode trajectory with metadata."""

    scenario_id: str
    difficulty: str
    total_reward: float
    final_score: float
    success: bool
    steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class TrajectoryMemory:
    """Stores and retrieves high-reward trajectories for few-shot prompting.

    The core idea: after each episode, save the full trajectory. Before the
    next episode, include the top-K highest-reward trajectories as examples
    in the system prompt. The agent genuinely improves because it sees
    increasingly good examples of incident resolution.
    """

    max_trajectories_per_scenario: int = 20
    top_k: int = 3
    _store: dict[str, list[Trajectory]] = field(default_factory=dict)

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory, keeping only the best ones."""
        key = trajectory.scenario_id
        if key not in self._store:
            self._store[key] = []

        self._store[key].append(trajectory)

        # Sort by total reward descending, keep only top N
        self._store[key].sort(key=lambda t: t.total_reward, reverse=True)
        self._store[key] = self._store[key][: self.max_trajectories_per_scenario]

    def get_best(self, scenario_id: str, k: int | None = None) -> list[Trajectory]:
        """Return the top-K highest-reward trajectories for a scenario."""
        k = k or self.top_k
        return self._store.get(scenario_id, [])[:k]

    def get_fewshot_prompt(self, scenario_id: str, k: int | None = None) -> str:
        """Build a few-shot examples string from the best trajectories."""
        best = self.get_best(scenario_id, k)
        if not best:
            return ""

        lines = [
            "Below are examples of successful incident resolution trajectories, "
            "ranked from best to worst. Learn from these patterns:\n"
        ]
        for idx, trajectory in enumerate(best, 1):
            score_label = "✓ SUCCESS" if trajectory.success else "✗ PARTIAL"
            lines.append(
                f"--- Example {idx} ({score_label}, reward={trajectory.total_reward:+.2f}, "
                f"score={trajectory.final_score:.2f}) ---"
            )
            for step in trajectory.steps:
                action_str = json.dumps(step.action, separators=(",", ":"))
                lines.append(
                    f"  Tick {step.tick}: {action_str} → reward={step.reward:+.2f}"
                )
            lines.append("")

        lines.append(
            "Use these examples to guide your actions. Investigate before fixing. "
            "Respond with JSON only."
        )
        return "\n".join(lines)

    def build_system_prompt(self, base_prompt: str, scenario_id: str) -> str:
        """Augment the base system prompt with few-shot examples."""
        fewshot = self.get_fewshot_prompt(scenario_id)
        if not fewshot:
            return base_prompt
        return f"{base_prompt}\n\n{fewshot}"

    @property
    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        return {
            scenario_id: {
                "count": len(trajectories),
                "best_reward": trajectories[0].total_reward if trajectories else 0.0,
                "best_score": trajectories[0].final_score if trajectories else 0.0,
                "avg_reward": (
                    sum(t.total_reward for t in trajectories) / len(trajectories)
                    if trajectories
                    else 0.0
                ),
            }
            for scenario_id, trajectories in self._store.items()
        }

    def save(self, path: Path) -> None:
        """Persist the memory to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for scenario_id, trajectories in self._store.items():
            data[scenario_id] = [
                {
                    "scenario_id": t.scenario_id,
                    "difficulty": t.difficulty,
                    "total_reward": t.total_reward,
                    "final_score": t.final_score,
                    "success": t.success,
                    "steps": [
                        {
                            "tick": s.tick,
                            "observation_summary": s.observation_summary,
                            "action": s.action,
                            "reward": s.reward,
                        }
                        for s in t.steps
                    ],
                }
                for t in trajectories
            ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "TrajectoryMemory":
        """Load memory from disk."""
        memory = cls(**kwargs)
        if not path.exists():
            return memory

        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario_id, trajectory_list in data.items():
            for t_data in trajectory_list:
                trajectory = Trajectory(
                    scenario_id=t_data["scenario_id"],
                    difficulty=t_data["difficulty"],
                    total_reward=t_data["total_reward"],
                    final_score=t_data["final_score"],
                    success=t_data["success"],
                    steps=[
                        TrajectoryStep(
                            tick=s["tick"],
                            observation_summary=s["observation_summary"],
                            action=s["action"],
                            reward=s["reward"],
                        )
                        for s in t_data["steps"]
                    ],
                )
                memory.add(trajectory)

        return memory


def summarize_observation(obs: dict[str, Any]) -> str:
    """Create a compact text summary of an observation for storage."""
    parts = [f"tick={obs.get('tick', '?')}/{obs.get('max_ticks', '?')}"]
    parts.append(f"phase={obs.get('phase', '?')}")

    services = obs.get("services", {})
    unhealthy = [
        name
        for name, svc in services.items()
        if isinstance(svc, dict) and svc.get("status") != "healthy"
    ]
    if unhealthy:
        parts.append(f"unhealthy=[{','.join(unhealthy)}]")
    else:
        parts.append("all_healthy")

    alerts = obs.get("active_alerts", [])
    if alerts:
        parts.append(f"alerts={len(alerts)}")

    if obs.get("phase2_unlocked"):
        parts.append("sec_unlocked")
    if obs.get("phase2_complete"):
        parts.append("sec_complete")
    if obs.get("postmortem_submitted"):
        parts.append("pm_done")

    return " | ".join(parts)
