#!/usr/bin/env python3
"""Live dashboard for the unified three-phase environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from sre_env.client import SREEnv
from sre_env.models import SREAction
from sre_env.scripts.benchmark_policies import (
    OPTIMAL_ACTIONS,
    SCENARIO_FOR_DIFFICULTY,
    naive_action,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class DashboardState:
    provider: str
    student_model: str
    episodes: int
    difficulties: list[str]
    current_difficulty: str = "-"
    current_episode: int = 0
    baseline_reward_history: list[float] = field(default_factory=list)
    reference_reward_history: list[float] = field(default_factory=list)
    baseline_phase: str = "Phase 1: Infrastructure Triage"
    baseline_phase2_unlocked: bool = False
    baseline_phase2_complete: bool = False
    baseline_postmortem_submitted: bool = False
    baseline_final_score: float = 0.0
    baseline_phase_scores: dict[str, float] = field(default_factory=dict)
    baseline_websec_state: dict[str, Any] = field(default_factory=dict)
    baseline_task: dict[str, Any] | None = None
    reference_phase: str = "Phase 1: Infrastructure Triage"
    reference_phase2_unlocked: bool = False
    reference_phase2_complete: bool = False
    reference_postmortem_submitted: bool = False
    reference_final_score: float = 0.0
    reference_phase_scores: dict[str, float] = field(default_factory=dict)
    reference_websec_state: dict[str, Any] = field(default_factory=dict)
    reference_task: dict[str, Any] | None = None
    baseline_last_action: str = "-"
    baseline_last_result: str = "-"
    reference_last_action: str = "-"
    reference_last_result: str = "-"
    log_lines: list[str] = field(default_factory=list)


def render_phase_progress(
    phase2_unlocked: bool,
    phase2_complete: bool,
    postmortem_submitted: bool,
    final_score: float,
    phase_scores: dict[str, float],
) -> Panel:
    p1_done = phase_scores.get("infrastructure", 0.0) > 0.25
    p2_done = phase2_complete
    p3_done = postmortem_submitted

    p1_bar = "█" * 16 if p1_done else "█" * 8 + "·" * 8
    p2_bar = "█" * 16 if p2_done else ("█" * 4 + "·" * 12 if phase2_unlocked else "·" * 16)
    p3_bar = "█" * 16 if p3_done else "·" * 16
    border = "green" if final_score > 0.7 else ("yellow" if final_score > 0.4 else "dim")

    lines = [
        f"Phase 1 [{p1_bar}] {'DONE' if p1_done else 'ACTIVE'}",
        f"Phase 2 [{p2_bar}] {'DONE' if p2_done else ('ACTIVE' if phase2_unlocked else 'LOCKED')}",
        f"Phase 3 [{p3_bar}] {'DONE' if p3_done else 'PENDING'}",
        "─" * 48,
        (
            f"Infra: {phase_scores.get('infrastructure', 0.0):.2f}/0.50  "
            f"Security: {phase_scores.get('security', 0.0):.2f}/0.40  "
            f"PostMortem: {phase_scores.get('postmortem', 0.0):.2f}/0.30"
        ),
        f"TOTAL: {final_score:.3f}/1.000",
    ]
    return Panel("\n".join(lines), title="Phase Progress", border_style=border)


def render_websec_panel(
    phase2_unlocked: bool,
    websec_state: dict[str, Any],
    sub_quest: dict[str, Any] | None = None,
) -> Panel:
    if not phase2_unlocked:
        return Panel(
            "Security sub-quest locked. Investigate the trigger logs to unlock it.",
            title="Security Sub-Quest",
            border_style="dim",
        )

    task_id = (sub_quest or {}).get("task_id", "unknown")
    classified = websec_state.get("classified", False)
    vuln_type = websec_state.get("vulnerability_type", "")
    patch = websec_state.get("patch_applied", "")
    exploit = websec_state.get("exploit_blocked", False)
    functionality = websec_state.get("functionality_ok", False)
    all_done = classified and bool(patch) and exploit and functionality
    lines = [
        f"Task: {task_id}",
        f"Classified: {'YES ' + vuln_type if classified else 'PENDING'}",
        f"Patch: {'YES ' + patch if patch else 'PENDING'}",
        f"Exploit blocked: {'YES' if exploit else 'NO'}",
        f"Functionality OK: {'YES' if functionality else 'NO'}",
    ]
    return Panel(
        "\n".join(lines),
        title="Security Sub-Quest",
        border_style="green" if all_done else "yellow",
    )


def reward_table(state: DashboardState) -> Table:
    table = Table(show_header=True, expand=True)
    table.add_column("Agent")
    table.add_column("Episodes", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Best Score", justify="right")
    baseline_avg = (
        sum(state.baseline_reward_history) / len(state.baseline_reward_history)
        if state.baseline_reward_history
        else 0.0
    )
    reference_avg = (
        sum(state.reference_reward_history) / len(state.reference_reward_history)
        if state.reference_reward_history
        else 0.0
    )
    table.add_row(
        "Baseline",
        str(len(state.baseline_reward_history)),
        f"{baseline_avg:+.3f}",
        f"{state.baseline_final_score:.3f}",
    )
    table.add_row(
        "Reference",
        str(len(state.reference_reward_history)),
        f"{reference_avg:+.3f}",
        f"{state.reference_final_score:.3f}",
    )
    return table


def render_dashboard(state: DashboardState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=10),
    )
    layout["body"].split_row(Layout(name="baseline"), Layout(name="reference"))
    layout["footer"].split_row(Layout(name="scores"), Layout(name="log"))
    layout["header"].update(
        Panel(
            (
                f"SRE-Env Unified Dashboard | provider={state.provider} "
                f"model={state.student_model} difficulty={state.current_difficulty} "
                f"episode={state.current_episode}/{state.episodes}"
            ),
            border_style="bright_white",
        )
    )

    baseline_group = Group(
        render_phase_progress(
            state.baseline_phase2_unlocked,
            state.baseline_phase2_complete,
            state.baseline_postmortem_submitted,
            state.baseline_final_score,
            state.baseline_phase_scores,
        ),
        render_websec_panel(
            state.baseline_phase2_unlocked,
            state.baseline_websec_state,
            state.baseline_task,
        ),
        Panel(
            f"phase={state.baseline_phase}\naction={state.baseline_last_action}\nresult={state.baseline_last_result}",
            title="Baseline Latest Step",
            border_style="cyan",
        ),
    )
    reference_group = Group(
        render_phase_progress(
            state.reference_phase2_unlocked,
            state.reference_phase2_complete,
            state.reference_postmortem_submitted,
            state.reference_final_score,
            state.reference_phase_scores,
        ),
        render_websec_panel(
            state.reference_phase2_unlocked,
            state.reference_websec_state,
            state.reference_task,
        ),
        Panel(
            f"phase={state.reference_phase}\naction={state.reference_last_action}\nresult={state.reference_last_result}",
            title="Reference Latest Step",
            border_style="green",
        ),
    )

    layout["baseline"].update(Panel(baseline_group, title="Baseline Agent", border_style="cyan"))
    layout["reference"].update(Panel(reference_group, title="Reference Agent", border_style="green"))
    layout["scores"].update(Panel(reward_table(state), title="Episode Summary", border_style="magenta"))
    layout["log"].update(
        Panel("\n".join(state.log_lines[-8:]) if state.log_lines else "Waiting...", title="Event Log", border_style="yellow")
    )
    return layout


def make_baseline_policy() -> Callable[[dict[str, Any], int, str], SREAction]:
    def chooser(obs_dict: dict[str, Any], _step_idx: int, _scenario_id: str) -> SREAction:
        return naive_action(obs_dict)

    return chooser


def make_reference_policy() -> Callable[[dict[str, Any], int, str], SREAction]:
    def chooser(_obs_dict: dict[str, Any], step_idx: int, scenario_id: str) -> SREAction:
        actions = OPTIMAL_ACTIONS[scenario_id]
        if step_idx < len(actions):
            return actions[step_idx]
        return actions[-1]

    return chooser


def run_agent_episode(
    *,
    base_url: str,
    difficulty: str,
    scenario_id: str,
    agent_name: str,
    chooser: Callable[[dict[str, Any], int, str], SREAction],
    state: DashboardState,
    live: Live,
) -> tuple[float, dict[str, Any]]:
    total_reward = 0.0
    final_observation: dict[str, Any] = {}
    with SREEnv(base_url=base_url).sync() as env:
        observation = env.reset(difficulty=difficulty, scenario_id=scenario_id).observation
        step_idx = 0
        while not observation.done and not observation.episode_complete:
            obs_dict = observation.model_dump()
            action = chooser(obs_dict, step_idx, scenario_id)
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward
            final_observation = observation.model_dump()
            key = "baseline" if agent_name == "baseline" else "reference"
            setattr(state, f"{key}_phase", observation.phase)
            setattr(state, f"{key}_phase2_unlocked", observation.phase2_unlocked)
            setattr(state, f"{key}_phase2_complete", observation.phase2_complete)
            setattr(state, f"{key}_postmortem_submitted", observation.postmortem_submitted)
            setattr(state, f"{key}_final_score", observation.final_score)
            setattr(state, f"{key}_phase_scores", dict(observation.phase_scores))
            setattr(state, f"{key}_websec_state", dict(observation.websec_state))
            setattr(state, f"{key}_task", observation.security_sub_quest)
            setattr(
                state,
                f"{key}_last_action",
                str(action.model_dump(exclude_none=True)),
            )
            setattr(state, f"{key}_last_result", observation.last_action_result)
            state.log_lines.append(
                f"{difficulty} {agent_name} tick={observation.tick} reward={result.reward:+.2f}"
            )
            live.update(render_dashboard(state), refresh=True)
            step_idx += 1
            if step_idx > observation.max_ticks + 5:
                break
    return total_reward, final_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--student-model", default="qwen2.5:1.5b")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    args = parser.parse_args()

    state = DashboardState(
        provider=args.provider,
        student_model=args.student_model,
        episodes=args.episodes,
        difficulties=args.difficulties,
    )

    baseline_policy = make_baseline_policy()
    reference_policy = make_reference_policy()

    with Live(render_dashboard(state), refresh_per_second=4) as live:
        for difficulty in args.difficulties:
            scenario_id = SCENARIO_FOR_DIFFICULTY[difficulty]
            for episode_idx in range(1, args.episodes + 1):
                state.current_difficulty = difficulty
                state.current_episode = episode_idx
                baseline_reward, _ = run_agent_episode(
                    base_url=args.base_url,
                    difficulty=difficulty,
                    scenario_id=scenario_id,
                    agent_name="baseline",
                    chooser=baseline_policy,
                    state=state,
                    live=live,
                )
                state.baseline_reward_history.append(baseline_reward)
                reference_reward, _ = run_agent_episode(
                    base_url=args.base_url,
                    difficulty=difficulty,
                    scenario_id=scenario_id,
                    agent_name="reference",
                    chooser=reference_policy,
                    state=state,
                    live=live,
                )
                state.reference_reward_history.append(reference_reward)
                state.log_lines.append(
                    f"completed difficulty={difficulty} episode={episode_idx} baseline={baseline_reward:+.2f} reference={reference_reward:+.2f}"
                )
                live.update(render_dashboard(state), refresh=True)


if __name__ == "__main__":
    main()
