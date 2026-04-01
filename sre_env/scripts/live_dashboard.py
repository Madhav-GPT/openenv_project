"""Rich live dashboard renderers for SRE execution scripts."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sre_env.scripts.llm_backends import LLMCallMetrics

DIFFICULTY_TITLES = {
    "easy": "Easy - DB Crash",
    "medium": "Medium - Cache Cascade",
    "hard": "Hard - Bad Deploy",
}
SPARK_BLOCKS = " .:-=+*#%@"


def _sparkline(values: list[float], width: int = 18) -> str:
    if not values:
        return "-" * min(width, 8)
    tail = values[-width:]
    lo = min(tail)
    hi = max(tail)
    if hi == lo:
        return SPARK_BLOCKS[-2] * len(tail)
    chars = []
    for value in tail:
        ratio = (value - lo) / (hi - lo)
        idx = min(len(SPARK_BLOCKS) - 1, int(ratio * (len(SPARK_BLOCKS) - 1)))
        chars.append(SPARK_BLOCKS[idx])
    return "".join(chars)


def _shorten(value: str | None, limit: int = 140) -> str:
    if not value:
        return "-"
    text = value.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _status_text(value: str) -> str:
    colors = {
        "pending": "yellow",
        "running": "cyan",
        "thinking": "magenta",
        "success": "green",
        "failed": "red",
        "idle": "white",
        "complete": "green",
    }
    color = colors.get(value, "white")
    return f"[{color}]{value}[/{color}]"


@dataclass
class DifficultyState:
    difficulty: str
    title: str
    status: str = "pending"
    current_episode: int = 0
    total_episodes: int = 0
    current_mode: str = "-"
    rewards: list[float] = field(default_factory=list)
    baseline_rewards: list[float] = field(default_factory=list)
    fewshot_rewards: list[float] = field(default_factory=list)


@dataclass
class RunState:
    label: str
    mode: str = "-"
    difficulty: str = "-"
    title: str = "-"
    status: str = "idle"
    episode_index: int = 0
    total_episodes: int = 0
    tick: int = 0
    max_ticks: int = 20
    step_count: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    last_action: str = "-"
    last_result: str = "-"
    last_tool_output: str = "-"
    reward_history: list[float] = field(default_factory=list)
    cumulative_history: list[float] = field(default_factory=list)
    tps_history: list[float] = field(default_factory=list)
    latency_history: list[float] = field(default_factory=list)
    fallback_count: int = 0
    success: str = "pending"
    services: dict[str, Any] = field(default_factory=dict)
    active_alerts: list[Any] = field(default_factory=list)
    metrics: LLMCallMetrics = field(
        default_factory=lambda: LLMCallMetrics(provider="-", model="-")
    )


def _reward_table() -> Table:
    table = Table(show_header=True, box=None, expand=True, padding=(0, 1))
    table.add_column("Reward Event", style="bold")
    table.add_column("Value", justify="right")
    rows = [
        ("Tick penalty", "-0.05"),
        ("Root-cause investigation", "+0.15"),
        ("Other investigation", "+0.08"),
        ("Correct fix", "+0.35"),
        ("Alert cleared", "+0.20"),
        ("Trap action", "-0.20"),
        ("All healthy", "+1.00"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


def _render_metrics(metrics: LLMCallMetrics) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("k", style="cyan", ratio=1)
    table.add_column("v", ratio=2)
    table.add_row("provider", f"{metrics.provider} / {metrics.model}")
    table.add_row("latency", f"{metrics.latency_s:.2f}s")
    table.add_row(
        "tokens",
        f"prompt={metrics.prompt_tokens} output={metrics.completion_tokens}",
    )
    table.add_row(
        "token/sec",
        (
            f"prompt={metrics.prompt_tps:.1f} out={metrics.completion_tps:.1f} "
            f"total={metrics.total_tps:.1f}"
        ),
    )
    table.add_row("load", f"{metrics.load_s:.2f}s")
    table.add_row("done", metrics.done_reason or "-")
    fallback = "yes" if metrics.used_fallback else "no"
    table.add_row("fallback", fallback)
    if metrics.fallback_reason:
        table.add_row("reason", _shorten(metrics.fallback_reason, 80))
    return table


def _render_services(services: dict[str, Any]) -> Table:
    table = Table(show_header=True, box=None, expand=True, padding=(0, 1))
    table.add_column("service", style="bold")
    table.add_column("status")
    table.add_column("cpu", justify="right")
    table.add_column("err", justify="right")
    for name in ("api-gateway", "cache", "database", "worker"):
        service = services.get(name)
        if not service:
            continue
        table.add_row(
            name,
            str(service.get("status", "-")),
            f"{service.get('cpu_pct', 0):.0f}%",
            f"{service.get('error_rate_pct', 0):.0f}%",
        )
    return table


def _render_run_panel(state: RunState) -> Panel:
    meta = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    meta.add_column("k", style="cyan", ratio=1)
    meta.add_column("v", ratio=2)
    meta.add_row("status", _status_text(state.status))
    meta.add_row("difficulty", state.title)
    meta.add_row(
        "episode",
        f"{state.episode_index}/{state.total_episodes} ({state.mode})",
    )
    meta.add_row("tick", f"{state.tick}/{state.max_ticks}")
    meta.add_row("steps", str(state.step_count))
    meta.add_row("reward", f"last={state.last_reward:+.2f} total={state.total_reward:+.2f}")
    meta.add_row("success", state.success)
    meta.add_row("fallbacks", str(state.fallback_count))

    graphs = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    graphs.add_column("k", style="cyan", ratio=1)
    graphs.add_column("v", ratio=2)
    graphs.add_row("reward graph", _sparkline(state.reward_history))
    graphs.add_row("total graph", _sparkline(state.cumulative_history))
    graphs.add_row("token/sec", _sparkline(state.tps_history))
    graphs.add_row("latency", _sparkline(state.latency_history))

    content = Group(
        meta,
        graphs,
        Panel(_render_metrics(state.metrics), title="Model Telemetry", border_style="blue"),
        Panel(_render_services(state.services), title="Services", border_style="green"),
        Panel(
            Text.from_markup(
                f"[bold]action[/bold] {_shorten(state.last_action, 100)}\n"
                f"[bold]result[/bold] {_shorten(state.last_result, 150)}\n"
                f"[bold]tool[/bold] {_shorten(state.last_tool_output, 150)}\n"
                f"[bold]alerts[/bold] {_shorten('; '.join(state.active_alerts), 150)}"
            ),
            title="Latest Step",
            border_style="magenta",
        ),
    )
    return Panel(content, title=state.label, border_style="bright_blue")


def _render_learning_curve_difficulty(state: DifficultyState) -> Panel:
    lines = [
        f"status: {state.status}",
        f"episode: {state.current_episode}/{state.total_episodes}",
        f"mode: {state.current_mode}",
        (
            f"baseline avg={_avg(state.baseline_rewards):+.2f} "
            f"latest={_latest(state.baseline_rewards):+.2f}"
        ),
        f"  {_sparkline(state.baseline_rewards)}",
        (
            f"fewshot avg={_avg(state.fewshot_rewards):+.2f} "
            f"latest={_latest(state.fewshot_rewards):+.2f}"
        ),
        f"  {_sparkline(state.fewshot_rewards)}",
        f"delta={_latest(state.fewshot_rewards) - _latest(state.baseline_rewards):+.2f}",
    ]
    return Panel("\n".join(lines), title=state.title, border_style="cyan")


def _render_baseline_difficulty_table(states: dict[str, DifficultyState]) -> Table:
    table = Table(show_header=True, expand=True)
    table.add_column("Difficulty", style="bold")
    table.add_column("Status")
    table.add_column("Reward", justify="right")
    table.add_column("Curve")
    for difficulty in ("easy", "medium", "hard"):
        state = states[difficulty]
        table.add_row(
            state.title,
            _status_text(state.status),
            f"{_latest(state.rewards):+.2f}" if state.rewards else "-",
            _sparkline(state.rewards),
        )
    return table


def _render_log(logs: list[str]) -> Panel:
    body = "\n".join(logs[-10:]) if logs else "Waiting for first event..."
    return Panel(body, title="Event Log", border_style="yellow")


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _latest(values: list[float]) -> float:
    if not values:
        return 0.0
    return values[-1]


class BaselineDashboard:
    def __init__(self, provider: str, model: str) -> None:
        self.console = Console()
        self.provider = provider
        self.model = model
        self.logs: list[str] = []
        self.current = RunState(label="Live Episode")
        self.states = {
            difficulty: DifficultyState(
                difficulty=difficulty,
                title=DIFFICULTY_TITLES[difficulty],
                total_episodes=1,
            )
            for difficulty in ("easy", "medium", "hard")
        }
        self.live = Live(self._render(), console=self.console, refresh_per_second=4)

    def __enter__(self) -> "BaselineDashboard":
        self.live.start()
        self.refresh()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.refresh()
        self.live.stop()

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.refresh()

    def start_episode(self, difficulty: str) -> None:
        state = self.states[difficulty]
        state.status = "running"
        state.current_episode = 1
        self.current = RunState(
            label="Live Episode",
            difficulty=difficulty,
            title=state.title,
            status="running",
            mode="baseline",
            episode_index=1,
            total_episodes=1,
        )
        self.log(f"started {state.title}")

    def record_step(
        self,
        difficulty: str,
        action: str,
        observation: dict[str, Any],
        reward: float,
        total_reward: float,
        metrics: LLMCallMetrics,
    ) -> None:
        self.current.tick = observation.get("tick", 0)
        self.current.max_ticks = observation.get("max_ticks", 20)
        self.current.step_count += 1
        self.current.last_action = action
        self.current.last_result = observation.get("last_action_result", "")
        self.current.last_tool_output = observation.get("tool_output") or "-"
        self.current.last_reward = reward
        self.current.total_reward = total_reward
        self.current.reward_history.append(reward)
        self.current.cumulative_history.append(total_reward)
        self.current.metrics = metrics
        self.current.services = observation.get("services", {})
        self.current.active_alerts = [
            f"{item['service']}:{item['severity']}"
            for item in observation.get("active_alerts", [])
        ]
        if metrics.total_tps:
            self.current.tps_history.append(metrics.total_tps)
        if metrics.latency_s:
            self.current.latency_history.append(metrics.latency_s)
        if metrics.used_fallback:
            self.current.fallback_count += 1
        self.log(
            f"{DIFFICULTY_TITLES[difficulty]} tick={self.current.tick} "
            f"reward={reward:+.2f} total={total_reward:+.2f}"
        )

    def finish_episode(
        self,
        difficulty: str,
        total_reward: float,
        success: bool,
        failure_reason: str | None,
    ) -> None:
        state = self.states[difficulty]
        state.rewards.append(total_reward)
        state.status = "success" if success else "failed"
        self.current.status = "success" if success else "failed"
        self.current.success = "yes" if success else "no"
        self.current.last_result = failure_reason or self.current.last_result
        self.log(
            f"completed {state.title} reward={total_reward:+.2f} "
            f"success={success}"
        )

    def refresh(self) -> None:
        self.live.update(self._render(), refresh=True)

    def _render(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12),
        )
        layout["body"].split_row(Layout(name="summary"), Layout(name="current"))
        layout["footer"].split_row(Layout(name="reward"), Layout(name="log"))
        layout["header"].update(
            Panel(
                f"SRE-Env Baseline Live Dashboard | provider={self.provider} model={self.model}",
                border_style="bright_white",
            )
        )
        layout["summary"].update(
            Panel(
                _render_baseline_difficulty_table(self.states),
                title="Difficulty Summary",
                border_style="cyan",
            )
        )
        layout["current"].update(_render_run_panel(self.current))
        layout["reward"].update(Panel(_reward_table(), title="Reward System", border_style="green"))
        layout["log"].update(_render_log(self.logs))
        return layout


LC_SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"
LC_INVESTIGATION_TOOLS = {"get_logs", "get_metrics", "get_dependencies"}
LC_AGENT_COLORS = {"baseline": "orange1", "fewshot": "green"}
LC_AGENT_LABELS = {"baseline": "BASELINE AGENT", "fewshot": "FEW-SHOT AGENT"}
LC_DIFFICULTY_SHORT = {"easy": "E", "medium": "M", "hard": "H"}
LC_DIFFICULTY_COLORS = {"easy": "green", "medium": "yellow", "hard": "red"}
LC_STATUS_STYLES = {
    "healthy": "bold green",
    "degraded": "bold yellow",
    "overloaded": "bold magenta",
    "crashed": "bold red",
}
LC_BLOCKS = "▁▂▃▄▅▆▇█"


def _load_curve_meta() -> dict[str, dict[str, Any]]:
    with LC_SCENARIOS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)

    meta: dict[str, dict[str, Any]] = {}
    for scenario in data["scenarios"]:
        difficulty = scenario["difficulty"]
        meta[difficulty] = {
            "root_cause": scenario["root_cause_service"],
            "fix_sequence": list(scenario["correct_fix_sequence"]),
            "trap_actions": [item["action"] for item in scenario.get("trap_actions", [])],
        }
    return meta


LC_META = _load_curve_meta()


@dataclass
class LCRewardBreakdown:
    tick_penalty: float = 0.0
    root_invest: float = 0.0
    other_invest: float = 0.0
    correct_fix: float = 0.0
    trap_action: float = 0.0
    alert_cleared: float = 0.0
    all_healthy: float = 0.0


@dataclass
class LCActionSummary:
    tool: str = "-"
    service: str = "-"
    kind: str = "neutral"
    display: str = "-"
    detail: str = "-"


@dataclass
class LCEvent:
    tick: int
    difficulty: str
    agent: str
    kind: str
    action: str
    reward: float


@dataclass
class LCDifficultyState:
    difficulty: str
    title: str
    total_episodes: int
    baseline_rewards: list[float] = field(default_factory=list)
    fewshot_rewards: list[float] = field(default_factory=list)
    baseline_running_history: list[float] = field(default_factory=list)
    fewshot_running_history: list[float] = field(default_factory=list)
    baseline_last_history: list[float] = field(default_factory=list)
    fewshot_last_history: list[float] = field(default_factory=list)
    status: str = "pending"
    current_episode: int = 0
    running_mode: str = ""
    current_tick: int = 0
    baseline_running_reward: float | None = None
    fewshot_running_reward: float | None = None


@dataclass
class LCAgentState:
    agent: str
    difficulty: str = "-"
    difficulty_title: str = "-"
    episode_num: int = 0
    total_episodes: int = 0
    tick: int = 0
    max_ticks: int = 20
    total_reward: float = 0.0
    last_reward: float = 0.0
    success: bool = False
    status: str = "idle"
    traps: int = 0
    correct_fixes: int = 0
    total_fixes_needed: int = 0
    investigated_root: bool = False
    services: dict[str, Any] = field(default_factory=dict)
    last_action: LCActionSummary = field(default_factory=LCActionSummary)
    last_reward_breakdown: LCRewardBreakdown = field(default_factory=LCRewardBreakdown)
    latency_s: float = 0.0
    total_tps: float = 0.0
    model_name: str = "-"
    reward_history: list[float] = field(default_factory=list)

    def reset_for_run(
        self,
        difficulty: str,
        episode_num: int,
        total_episodes: int,
        model_name: str,
    ) -> None:
        meta = LC_META[difficulty]
        self.difficulty = difficulty
        self.difficulty_title = DIFFICULTY_TITLES[difficulty]
        self.episode_num = episode_num
        self.total_episodes = total_episodes
        self.tick = 0
        self.max_ticks = 20
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.success = False
        self.status = "running"
        self.traps = 0
        self.correct_fixes = 0
        self.total_fixes_needed = len(meta["fix_sequence"])
        self.investigated_root = False
        self.services = {}
        self.last_action = LCActionSummary()
        self.last_reward_breakdown = LCRewardBreakdown()
        self.latency_s = 0.0
        self.total_tps = 0.0
        self.model_name = model_name
        self.reward_history = []


@dataclass
class LCDashboardState:
    provider: str
    model_name: str
    total_episodes: int
    started_at: float = field(default_factory=time.monotonic)
    finished: bool = False
    note: str = ""
    events: deque[LCEvent] = field(default_factory=lambda: deque(maxlen=12))
    difficulties: dict[str, LCDifficultyState] = field(default_factory=dict)
    baseline: LCAgentState = field(default_factory=lambda: LCAgentState(agent="baseline"))
    fewshot: LCAgentState = field(default_factory=lambda: LCAgentState(agent="fewshot"))

    def __post_init__(self) -> None:
        if not self.difficulties:
            self.difficulties = {
                difficulty: LCDifficultyState(
                    difficulty=difficulty,
                    title=DIFFICULTY_TITLES[difficulty],
                    total_episodes=self.total_episodes,
                )
                for difficulty in ("easy", "medium", "hard")
            }
        self.baseline.model_name = self.model_name
        self.fewshot.model_name = self.model_name

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    @property
    def current_episode(self) -> int:
        current = 0
        for difficulty in self.difficulties.values():
            current = max(current, difficulty.current_episode)
        if self.finished:
            return self.total_episodes
        return current


def _lc_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _lc_format_reward(value: float) -> str:
    return f"{value:+.2f}"


def _lc_format_time(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _lc_shorten(value: str | None, limit: int = 120) -> str:
    if not value:
        return "-"
    text = value.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _lc_action_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for field_name in ("tool", "service", "metric", "replicas", "version"):
        if field_name in expected and actual.get(field_name) != expected[field_name]:
            return False
    return True


def _lc_action_text(action: dict[str, Any]) -> str:
    parts = [str(action.get("tool", "-")), str(action.get("service", "-"))]
    if action.get("metric"):
        parts.append(str(action["metric"]))
    if action.get("version"):
        parts.append(str(action["version"]))
    if action.get("replicas") is not None:
        parts.append(f"x{action['replicas']}")
    return " ".join(parts)


def _lc_classify_action(
    difficulty: str,
    action: dict[str, Any],
    agent_state: LCAgentState,
) -> str:
    meta = LC_META[difficulty]
    tool = action.get("tool", "")
    service = action.get("service", "")

    if tool in LC_INVESTIGATION_TOOLS:
        if service == meta["root_cause"] and not agent_state.investigated_root:
            return "root_investigation"
        return "investigation"

    if (
        agent_state.correct_fixes < len(meta["fix_sequence"])
        and _lc_action_matches(meta["fix_sequence"][agent_state.correct_fixes], action)
    ):
        return "correct_fix"

    if any(_lc_action_matches(trap, action) for trap in meta["trap_actions"]):
        return "trap"

    return "neutral"


def _lc_reward_breakdown(
    action_kind: str,
    alerts_cleared: int,
    all_healthy: bool,
) -> LCRewardBreakdown:
    breakdown = LCRewardBreakdown(tick_penalty=-0.05)
    if action_kind == "root_investigation":
        breakdown.root_invest = 0.15
    elif action_kind == "investigation":
        breakdown.other_invest = 0.08
    elif action_kind == "correct_fix":
        breakdown.correct_fix = 0.35
    elif action_kind == "trap":
        breakdown.trap_action = -0.20
    breakdown.alert_cleared = round(alerts_cleared * 0.20, 2)
    if all_healthy:
        breakdown.all_healthy = 1.00
    return breakdown


def render_tick_bar(current: int, maximum: int, width: int, color: str) -> Text:
    maximum = max(1, maximum)
    width = max(8, width)
    filled = min(width, max(0, round((current / maximum) * width)))
    if current > 0 and filled == 0:
        filled = 1
    empty = width - filled
    text = Text()
    text.append("tick ", style="dim white")
    text.append("[", style="dim white")
    text.append("█" * filled, style=f"bold {color}")
    text.append("·" * empty, style="grey35")
    text.append("]", style="dim white")
    text.append(f" {current}/{maximum}", style="bold white")
    return text


def render_cpu_bar(cpu_pct: float) -> str:
    filled = min(10, max(0, round(cpu_pct / 10)))
    return f"{'▓' * filled}{'░' * (10 - filled)}"


def render_sparkline(rewards: list[float], width: int) -> str:
    width = max(1, width)
    if not rewards:
        return "─" * width
    tail = rewards[-width:]
    max_abs = max(abs(value) for value in tail) or 1.0
    output: list[str] = []
    for value in tail:
        if abs(value) < 1e-9:
            output.append("─")
        elif value < 0:
            output.append("▁")
        else:
            idx = max(0, min(len(LC_BLOCKS) - 1, round((value / max_abs) * (len(LC_BLOCKS) - 1))))
            output.append(LC_BLOCKS[idx])
    if len(output) < width:
        output = (["─"] * (width - len(output))) + output
    return "".join(output[-width:])


def _lc_make_text(lines: list[Text]) -> Text:
    text = Text()
    for idx, line in enumerate(lines):
        text.append_text(line)
        if idx < len(lines) - 1:
            text.append("\n")
    return text


def _lc_center_lines(message: str, width: int, height: int, label_width: int) -> Text:
    lines: list[Text] = []
    mid_row = max(0, height // 2)
    for row in range(height):
        line = Text()
        line.append(" " * label_width, style="dim white")
        line.append("│", style="dim white")
        if row == mid_row:
            line.append(message.center(width), style="dim white")
        else:
            line.append("·" * width, style="grey23")
        lines.append(line)
    axis = Text(" " * (label_width + 1))
    axis.append(" " * width, style="dim white")
    lines.append(axis)
    return _lc_make_text(lines)


def render_ascii_graph(
    baseline_rewards: list[float],
    fewshot_rewards: list[float],
    width: int,
    height: int,
    running: bool,
    current_tick: int,
    *,
    total_episodes: int,
    running_mode: str = "",
    running_reward: float | None = None,
    baseline_step_history: list[float] | None = None,
    fewshot_step_history: list[float] | None = None,
    baseline_last_history: list[float] | None = None,
    fewshot_last_history: list[float] | None = None,
    max_ticks: int = 20,
) -> Text:
    label_width = 5
    plot_width = max(12, width - label_width - 1)
    plot_height = max(5, height - 1)

    baseline_step_history = baseline_step_history or []
    fewshot_step_history = fewshot_step_history or []
    baseline_last_history = baseline_last_history or []
    fewshot_last_history = fewshot_last_history or []

    use_tick_axis = False
    baseline_plot = baseline_rewards
    fewshot_plot = fewshot_rewards
    baseline_char = "━"
    fewshot_char = "━"
    baseline_style = "orange1"
    fewshot_style = "green"

    if len(baseline_rewards) < 2:
        if running_mode == "baseline" and baseline_step_history:
            baseline_plot = baseline_step_history
            baseline_char = "╌"
            baseline_style = "orange1"
            use_tick_axis = True
        elif baseline_last_history:
            baseline_plot = baseline_last_history
            use_tick_axis = True

    if len(fewshot_rewards) < 2:
        if running_mode == "fewshot" and fewshot_step_history:
            fewshot_plot = fewshot_step_history
            fewshot_char = "╌"
            fewshot_style = "green"
            use_tick_axis = True
        elif fewshot_last_history:
            fewshot_plot = fewshot_last_history
            use_tick_axis = True

    if not use_tick_axis and running and running_reward is not None and current_tick > 0:
        use_tick_axis = total_episodes <= 1

    all_values = list(baseline_plot) + list(fewshot_plot)
    if running and running_reward is not None and not all_values:
        all_values.append(running_reward)
    if not all_values:
        return _lc_center_lines("waiting...", plot_width, plot_height, label_width)

    lo = min(all_values)
    hi = max(all_values)
    if hi == lo:
        hi += 1.0
        lo -= 1.0

    chars = [["·" for _ in range(plot_width)] for _ in range(plot_height)]
    styles = [["grey23" for _ in range(plot_width)] for _ in range(plot_height)]

    def x_for_domain(point_num: int, domain_max: int) -> int:
        if domain_max <= 1:
            return plot_width - 1
        return round((point_num - 1) * (plot_width - 1) / (domain_max - 1))

    def y_for_value(value: float) -> int:
        ratio = (value - lo) / (hi - lo)
        return max(0, min(plot_height - 1, plot_height - 1 - round(ratio * (plot_height - 1))))

    def draw_line(points: list[tuple[int, int]], char: str, style: str) -> None:
        for idx in range(1, len(points)):
            x0, y0 = points[idx - 1]
            x1, y1 = points[idx]
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                x = round(x0 + ((x1 - x0) * step / steps))
                y = round(y0 + ((y1 - y0) * step / steps))
                chars[y][x] = char
                styles[y][x] = style

    if use_tick_axis:
        domain_max = max(
            2,
            current_tick,
            len(baseline_plot),
            len(fewshot_plot),
        )
    else:
        domain_max = total_episodes
    baseline_points = [
        (x_for_domain(index + 1, domain_max), y_for_value(value))
        for index, value in enumerate(baseline_plot)
    ]
    fewshot_points = [
        (x_for_domain(index + 1, domain_max), y_for_value(value))
        for index, value in enumerate(fewshot_plot)
    ]

    draw_line(baseline_points, baseline_char, baseline_style)
    draw_line(fewshot_points, fewshot_char, fewshot_style)

    if baseline_points:
        bx, by = baseline_points[-1]
        chars[by][bx] = "●" if running_mode == "baseline" and baseline_char == "╌" else chars[by][bx]
        styles[by][bx] = f"bold {baseline_style}" if running_mode == "baseline" and baseline_char == "╌" else styles[by][bx]
    if fewshot_points:
        fx, fy = fewshot_points[-1]
        chars[fy][fx] = "●" if running_mode == "fewshot" and fewshot_char == "╌" else chars[fy][fx]
        styles[fy][fx] = f"bold {fewshot_style}" if running_mode == "fewshot" and fewshot_char == "╌" else styles[fy][fx]

    max_label = f"{hi:+.1f}".rjust(label_width)
    min_label = f"{lo:+.1f}".rjust(label_width)
    lines: list[Text] = []
    for row in range(plot_height):
        line = Text()
        label = max_label if row == 0 else min_label if row == plot_height - 1 else (" " * label_width)
        line.append(label, style="dim white")
        line.append("│", style="dim white")
        for col in range(plot_width):
            line.append(chars[row][col], style=styles[row][col])
        lines.append(line)

    axis_chars = [" " for _ in range(plot_width)]
    axis_max = domain_max if use_tick_axis else total_episodes
    tick_step = max(1, axis_max // 4)
    tick_marks = list(range(1, axis_max + 1, tick_step))
    if axis_max not in tick_marks:
        tick_marks.append(axis_max)
    for tick_mark in tick_marks:
        label = str(tick_mark)
        pos = min(plot_width - len(label), x_for_domain(tick_mark, axis_max))
        for index, char in enumerate(label):
            axis_chars[pos + index] = char
    axis = Text(" " * (label_width + 1))
    axis.append("".join(axis_chars), style="dim white")
    lines.append(axis)
    return _lc_make_text(lines)


def _lc_reward_chart(values: list[float], width: int, color: str) -> Text:
    width = max(12, width)
    height = 5
    tail = values[-width:]
    if not tail:
        lines = [Text("·" * width, style="grey23") for _ in range(height)]
        lines[height // 2] = Text("waiting...".center(width), style="dim white")
        return _lc_make_text(lines)

    lo = min(min(tail), 0.0)
    hi = max(max(tail), 0.0)
    if hi == lo:
        hi += 1.0
        lo -= 1.0

    chars = [["·" for _ in range(width)] for _ in range(height)]
    styles = [["grey23" for _ in range(width)] for _ in range(height)]
    zero_row = max(0, min(height - 1, height - 1 - round(((0.0 - lo) / (hi - lo)) * (height - 1))))
    for col in range(width):
        chars[zero_row][col] = "─"
        styles[zero_row][col] = "grey35"

    points = []
    for idx, value in enumerate(tail):
        x = idx
        ratio = (value - lo) / (hi - lo)
        y = max(0, min(height - 1, height - 1 - round(ratio * (height - 1))))
        points.append((x, y))
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + ((x1 - x0) * step / steps))
            y = round(y0 + ((y1 - y0) * step / steps))
            chars[y][x] = "━"
            styles[y][x] = color

    chars[points[-1][1]][points[-1][0]] = "●"
    styles[points[-1][1]][points[-1][0]] = f"bold {color}"

def _lc_reward_chart(values: list[float], width: int, color: str) -> Text:
    width = max(12, width)
    height = 5
    tail = values[-width:]
    if not tail:
        lines = [Text("·" * width, style="grey23") for _ in range(height)]
        lines[height // 2] = Text("waiting...".center(width), style="dim white")
        return _lc_make_text(lines)

    lo = min(min(tail), 0.0)
    hi = max(max(tail), 0.0)
    if hi == lo:
        hi += 1.0
        lo -= 1.0

    chars = [["·" for _ in range(width)] for _ in range(height)]
    styles = [["grey23" for _ in range(width)] for _ in range(height)]
    zero_row = max(0, min(height - 1, height - 1 - round(((0.0 - lo) / (hi - lo)) * (height - 1))))
    for col in range(width):
        chars[zero_row][col] = "─"
        styles[zero_row][col] = "grey35"

    points = []
    for idx, value in enumerate(tail):
        ratio = (value - lo) / (hi - lo)
        y = max(0, min(height - 1, height - 1 - round(ratio * (height - 1))))
        points.append((idx, y))

    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + ((x1 - x0) * step / steps))
            y = round(y0 + ((y1 - y0) * step / steps))
            chars[y][x] = "━"
            styles[y][x] = color

    end_x, end_y = points[-1]
    chars[end_y][end_x] = "●"
    styles[end_y][end_x] = f"bold {color}"

    lines: list[Text] = []
    for row in range(height):
        line = Text()
        for col in range(width):
            line.append(chars[row][col], style=styles[row][col])
        lines.append(line)
    return _lc_make_text(lines)


def _lc_status_badge(state: LCDifficultyState) -> tuple[str, str]:
    if len(state.baseline_rewards) >= state.total_episodes and len(state.fewshot_rewards) >= state.total_episodes:
        return "DONE", "bold green"
    if state.status == "running":
        return "RUNNING", "bold yellow"
    return "PENDING", "dim white"


def _lc_delta_style(delta: float) -> str:
    if delta > 0:
        return "green"
    if delta < 0:
        return "red"
    return "dim white"


def render_difficulty_panel(
    name: str,
    subtitle: str,
    baseline_rewards: list[float],
    fewshot_rewards: list[float],
    running: bool,
    current_tick: int,
    total_episodes: int,
    panel_width: int,
    *,
    current_episode: int,
    running_mode: str,
    running_reward: float | None,
    baseline_step_history: list[float] | None = None,
    fewshot_step_history: list[float] | None = None,
    baseline_last_history: list[float] | None = None,
    fewshot_last_history: list[float] | None = None,
    max_ticks: int = 20,
) -> Panel:
    del subtitle
    avg_base = _lc_avg(baseline_rewards)
    avg_few = _lc_avg(fewshot_rewards)
    delta = avg_few - avg_base
    completed = max(len(baseline_rewards), len(fewshot_rewards), current_episode)
    has_progress = completed > 0
    done = has_progress and not running
    if done and avg_few > avg_base:
        border_style = "green"
    elif running:
        border_style = "yellow"
    else:
        border_style = "dim white"

    badge_text = "DONE" if done else ("RUNNING" if running else "PENDING")
    badge_style = "bold green" if done else ("bold yellow" if running else "dim white")
    legend = Text()
    legend.append("● baseline   ", style="orange1")
    legend.append("● few-shot", style="green")

    stats = Table.grid(expand=True)
    for _ in range(4):
        stats.add_column(ratio=1)
    episodes_style = "green" if done else ("yellow" if running else "dim white")
    stats.add_row(
        Text.assemble((_lc_format_reward(avg_base), "bold orange1"), ("\nbase avg", "dim white")),
        Text.assemble((_lc_format_reward(avg_few), "bold green"), ("\nfew-shot", "dim white")),
        Text.assemble((_lc_format_reward(delta), _lc_delta_style(delta)), ("\ndelta", "dim white")),
        Text.assemble((f"{completed}/{total_episodes}", episodes_style), ("\nepisodes", "dim white")),
    )

    graph = render_ascii_graph(
        baseline_rewards,
        fewshot_rewards,
        width=max(24, panel_width - 4),
        height=8,
        running=running,
        current_tick=current_tick,
        total_episodes=total_episodes,
        running_mode=running_mode,
        running_reward=running_reward,
        baseline_step_history=baseline_step_history,
        fewshot_step_history=fewshot_step_history,
        baseline_last_history=baseline_last_history,
        fewshot_last_history=fewshot_last_history,
        max_ticks=max_ticks,
    )

    content = Group(legend, graph, stats)
    return Panel(
        content,
        title=name.upper(),
        subtitle=Text(badge_text, style=badge_style),
        border_style=border_style,
        padding=(0, 1),
    )


def _lc_service_status(status: str) -> Text:
    labels = {
        "healthy": "[healthy ]",
        "degraded": "[degraded]",
        "overloaded": "[overload]",
        "crashed": "[crashed ]",
    }
    return Text(labels.get(status, f"[{status[:8]:<8}]"), style=LC_STATUS_STYLES.get(status, "dim white"))


def _lc_agent_subtitle(agent_state: LCAgentState) -> Text:
    color = LC_AGENT_COLORS[agent_state.agent]
    status_word = "SUCCESS" if agent_state.success else ("FAILED" if agent_state.status == "failed" else "TICK")
    style = "bold green" if agent_state.success else ("bold red" if agent_state.status == "failed" else f"bold {color}")
    return Text(f"{status_word} {agent_state.tick}/{agent_state.max_ticks}", style=style)


def render_agent_panel(agent_type: str, state: LCDashboardState) -> Panel:
    agent_state = state.baseline if agent_type == "baseline" else state.fewshot
    color = LC_AGENT_COLORS[agent_type]

    progress = render_tick_bar(agent_state.tick, agent_state.max_ticks, 22, color)
    info = Table.grid(expand=True, padding=(0, 1))
    info.add_column(style="dim white", ratio=1)
    info.add_column(ratio=2)
    info.add_column(style="dim white", ratio=1)
    info.add_column(ratio=2)
    info.add_row(
        "difficulty",
        Text(
            agent_state.difficulty_title or "-",
            style=LC_DIFFICULTY_COLORS.get(agent_state.difficulty, "yellow"),
        ),
        "model",
        Text(agent_state.model_name, style="white"),
    )
    info.add_row(
        "episode",
        f"{agent_state.episode_num} / {agent_state.total_episodes}",
        "latency",
        Text(
            f"{agent_state.latency_s:.1f}s/step  {agent_state.total_tps:.1f} tok/s",
            style="white",
        ),
    )
    info.add_row(
        "reward",
        Text.assemble(
            ("last ", "dim white"),
            (_lc_format_reward(agent_state.last_reward), "white"),
            ("  total ", "dim white"),
            (_lc_format_reward(agent_state.total_reward), "green" if agent_state.total_reward >= 0 else "red"),
        ),
        "root cause",
        Text(
            "YES ✓" if agent_state.investigated_root else "NO ✗",
            style="green" if agent_state.investigated_root else "red",
        ),
    )

    stats_strip = Table.grid(expand=True)
    stats_strip.add_column(ratio=1)
    stats_strip.add_column(ratio=1)
    stats_strip.add_column(ratio=1)
    stats_strip.add_row(
        Text.assemble(("success ", "dim white"), ("yes" if agent_state.success else "-", "green" if agent_state.success else "dim white")),
        Text.assemble(("traps ", "dim white"), (str(agent_state.traps), "red" if agent_state.traps else "dim white")),
        Text.assemble(
            ("fixes ", "dim white"),
            (
                f"{agent_state.correct_fixes}/{agent_state.total_fixes_needed}",
                "green" if agent_state.correct_fixes else "dim white",
            ),
        ),
    )

    services = Table.grid(expand=True, padding=(0, 1))
    services.add_column(ratio=2)
    services.add_column(ratio=2)
    services.add_column(justify="right")
    services.add_column(justify="right")
    services.add_column(ratio=2)
    services.add_row(
        Text("service", style="dim white"),
        Text("status", style="dim white"),
        Text("cpu", style="dim white"),
        Text("err", style="dim white"),
        Text("trend", style="dim white"),
    )
    for service_name in ("api-gateway", "cache", "database", "worker"):
        service = agent_state.services.get(service_name, {})
        services.add_row(
            Text(service_name, style="white"),
            _lc_service_status(str(service.get("status", "healthy"))),
            Text(f"{service.get('cpu_pct', 0):.0f}%", style="white"),
            Text(
                f"{service.get('error_rate_pct', 0):.0f}%",
                style="red" if float(service.get("error_rate_pct", 0)) > 0 else "green",
            ),
            Text(render_cpu_bar(float(service.get("cpu_pct", 0))), style=color),
        )

    action_border = {
        "correct_fix": "green",
        "trap": "red",
        "investigation": "blue",
        "root_investigation": "blue",
    }.get(agent_state.last_action.kind, "yellow")
    action_box = Panel(
        Text.assemble(
            ("► ", f"bold {color}"),
            (agent_state.last_action.display, f"bold {color}"),
            ("\n", "white"),
            (_lc_shorten(agent_state.last_action.detail, 120), "white"),
        ),
        border_style=action_border,
        padding=(0, 1),
    )

    return Panel(
        Group(progress, info, stats_strip, services, action_box),
        title=LC_AGENT_LABELS[agent_type],
        subtitle=_lc_agent_subtitle(agent_state),
        border_style=color,
        padding=(0, 1),
    )


def _lc_reward_rows(breakdown: LCRewardBreakdown) -> list[tuple[str, float, str, str]]:
    return [
        ("tick penalty", breakdown.tick_penalty, "red", ""),
        ("root inv.", breakdown.root_invest, "bold green", "✓" if breakdown.root_invest > 0 else ""),
        ("other inv.", breakdown.other_invest, "green", "✓" if breakdown.other_invest > 0 else ""),
        ("correct fix", breakdown.correct_fix, "bold green", "✓" if breakdown.correct_fix > 0 else ""),
        ("trap action", breakdown.trap_action, "bold red", "✗" if breakdown.trap_action < 0 else ""),
        ("alert clear", breakdown.alert_cleared, "green", "✓" if breakdown.alert_cleared > 0 else ""),
        ("all healthy", breakdown.all_healthy, "green", "✓" if breakdown.all_healthy > 0 else ""),
    ]


def _lc_reward_breakdown_table(breakdown: LCRewardBreakdown) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=2)
    table.add_column(justify="right")
    table.add_column(width=1)
    for label, value, style, marker in _lc_reward_rows(breakdown):
        display = _lc_format_reward(value) if value else "+0.00"
        table.add_row(
            Text(label, style="dim white"),
            Text(display, style=style if value else "dim white"),
            Text(marker, style=style),
        )
    return table


def render_reward_stream(state: LCDashboardState) -> Panel:
    baseline = state.baseline
    fewshot = state.fewshot

    baseline_block = Group(
        Text("Baseline (cumulative)", style="bold orange1"),
        _lc_reward_chart(baseline.reward_history, 30, "orange1"),
        _lc_reward_breakdown_table(baseline.last_reward_breakdown),
    )
    fewshot_block = Group(
        Text("Few-shot (cumulative)", style="bold green"),
        _lc_reward_chart(fewshot.reward_history, 30, "green"),
        _lc_reward_breakdown_table(fewshot.last_reward_breakdown),
    )
    return Panel(
        Columns([baseline_block, fewshot_block], equal=True, expand=True),
        title="LIVE REWARD STREAM - BOTH AGENTS",
        border_style="cyan",
        padding=(0, 1),
    )


def render_event_log(events: list[LCEvent]) -> Panel:
    lines: list[Text] = []
    for event in events[:12]:
        difficulty_text = Text(f"[{LC_DIFFICULTY_SHORT[event.difficulty]}]", style=LC_DIFFICULTY_COLORS[event.difficulty])
        agent_text = Text(f"[{'B' if event.agent == 'baseline' else 'F'}]", style=LC_AGENT_COLORS[event.agent])
        kind_styles = {
            "trap": ("TRAP  ", "bold red"),
            "correct_fix": ("FIX✓  ", "bold green"),
            "root_investigation": ("INV★  ", "bold blue"),
            "investigation": ("INV   ", "blue"),
            "neutral": ("ACT   ", "yellow"),
        }
        kind_label, kind_style = kind_styles.get(event.kind, ("ACT   ", "yellow"))
        line = Text()
        line.append(f"t={event.tick:02d} ", style="dim white")
        line.append_text(difficulty_text)
        line.append(" ", style="dim white")
        line.append_text(agent_text)
        line.append(" ", style="dim white")
        line.append(kind_label, style=kind_style)
        line.append(_lc_shorten(event.action, 26).ljust(26), style="white")
        line.append(f"{event.reward:+.2f}".rjust(7), style="green" if event.reward >= 0 else "red")
        lines.append(line)

    legend = Text.assemble(
        ("[E]", "green"),
        ("=Easy ", "dim white"),
        ("[M]", "yellow"),
        ("=Medium ", "dim white"),
        ("[H]", "red"),
        ("=Hard   ", "dim white"),
        ("[B]", "orange1"),
        ("=Baseline ", "dim white"),
        ("[F]", "green"),
        ("=Few-shot   ", "dim white"),
        ("INV★", "bold blue"),
        ("=root cause found", "dim white"),
    )
    if not lines:
        lines.append(Text("waiting for first event...", style="dim white"))
    lines.append(Text(""))
    lines.append(legend)
    return Panel(Group(*lines), title="EVENT LOG", subtitle=Text("LAST 12 EVENTS", style="dim white"), border_style="cyan", padding=(0, 1))


def _lc_title_bar(state: LCDashboardState) -> Panel:
    running_style = "bold green" if int(state.elapsed_seconds * 2) % 2 == 0 else "green"
    status_label = "done" if state.finished else "running"
    status_style = "bold green" if state.finished else running_style
    text = Text()
    text.append("SRE-Env Learning Curve Dashboard", style="bold white")
    text.append(" | ", style="dim white")
    text.append(f"provider={state.provider} ", style="white")
    text.append(f"model={state.model_name}", style="cyan")
    text.append(" | ", style="dim white")
    text.append(f"episode {state.current_episode}/{state.total_episodes}", style="yellow")
    text.append(" | ", style="dim white")
    text.append(f"elapsed {_lc_format_time(state.elapsed_seconds)}", style="white")
    text.append(" | ", style="dim white")
    text.append(status_label, style=status_style)
    return Panel(text, border_style="grey35", padding=(0, 1), style="on #111820")


def build_layout(state: LCDashboardState, console: Console) -> Layout:
    width = console.size.width
    height = console.size.height
    top_size = 14 if height >= 40 else 12
    middle_size = 19 if height >= 40 else 17

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=top_size),
        Layout(name="middle", size=middle_size),
        Layout(name="bottom"),
    )
    layout["top"].split_row(Layout(name="easy"), Layout(name="medium"), Layout(name="hard"))
    layout["middle"].split_row(Layout(name="baseline"), Layout(name="fewshot"))
    layout["bottom"].split_row(Layout(name="reward", ratio=3), Layout(name="events", ratio=2))

    panel_width = max(32, (width - 10) // 3)
    layout["header"].update(_lc_title_bar(state))
    for difficulty in ("easy", "medium", "hard"):
        diff_state = state.difficulties[difficulty]
        running_reward = (
            diff_state.baseline_running_reward
            if diff_state.running_mode == "baseline"
            else diff_state.fewshot_running_reward
        )
        layout["top"][difficulty].update(
            render_difficulty_panel(
                diff_state.title,
                diff_state.status,
                diff_state.baseline_rewards,
                diff_state.fewshot_rewards,
                diff_state.status == "running",
                diff_state.current_tick,
                diff_state.total_episodes,
                panel_width,
                current_episode=diff_state.current_episode,
                running_mode=diff_state.running_mode,
                running_reward=running_reward,
                baseline_step_history=diff_state.baseline_running_history,
                fewshot_step_history=diff_state.fewshot_running_history,
                baseline_last_history=diff_state.baseline_last_history,
                fewshot_last_history=diff_state.fewshot_last_history,
                max_ticks=20,
            )
        )

    layout["middle"]["baseline"].update(render_agent_panel("baseline", state))
    layout["middle"]["fewshot"].update(render_agent_panel("fewshot", state))
    layout["bottom"]["reward"].update(render_reward_stream(state))
    layout["bottom"]["events"].update(render_event_log(list(state.events)))
    return layout


class LearningCurveDashboard:
    def __init__(self, provider: str, model: str, total_episodes: int) -> None:
        self.console = Console()
        self.state = LCDashboardState(
            provider=provider,
            model_name=model,
            total_episodes=total_episodes,
        )
        self.live = Live(
            build_layout(self.state, self.console),
            console=self.console,
            refresh_per_second=2,
            screen=True,
        )

    def __enter__(self) -> "LearningCurveDashboard":
        self.live.start()
        self.refresh()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.refresh()
        self.live.stop()

    def log(self, message: str) -> None:
        self.state.note = message
        self.refresh()

    def start_run(self, difficulty: str, mode: str, episode_index: int) -> None:
        target = self.state.baseline if mode == "baseline" else self.state.fewshot
        target.reset_for_run(
            difficulty=difficulty,
            episode_num=episode_index,
            total_episodes=self.state.total_episodes,
            model_name=self.state.model_name,
        )
        diff_state = self.state.difficulties[difficulty]
        diff_state.status = "running"
        diff_state.current_episode = episode_index
        diff_state.running_mode = mode
        diff_state.current_tick = 0
        if mode == "baseline":
            diff_state.baseline_running_reward = 0.0
            diff_state.baseline_running_history = []
        else:
            diff_state.fewshot_running_reward = 0.0
            diff_state.fewshot_running_history = []
        self.refresh()

    def record_step(
        self,
        difficulty: str,
        mode: str,
        action: dict[str, Any],
        pre_observation: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        total_reward: float,
        metrics: LLMCallMetrics,
    ) -> None:
        target = self.state.baseline if mode == "baseline" else self.state.fewshot
        diff_state = self.state.difficulties[difficulty]

        action_kind = _lc_classify_action(difficulty, action, target)
        alerts_before = len(pre_observation.get("active_alerts", []))
        alerts_after = len(observation.get("active_alerts", []))
        breakdown = _lc_reward_breakdown(
            action_kind=action_kind,
            alerts_cleared=max(0, alerts_before - alerts_after),
            all_healthy=bool(observation.get("success")),
        )

        if action_kind == "root_investigation":
            target.investigated_root = True
        elif action_kind == "correct_fix":
            target.correct_fixes += 1
        elif action_kind == "trap":
            target.traps += 1

        detail = observation.get("tool_output") or observation.get("last_action_result") or "-"
        target.tick = int(observation.get("tick", 0))
        target.max_ticks = int(observation.get("max_ticks", 20))
        target.total_reward = total_reward
        target.last_reward = reward
        target.success = bool(observation.get("success", False))
        target.status = "running"
        target.services = observation.get("services", {})
        target.last_action = LCActionSummary(
            tool=str(action.get("tool", "-")),
            service=str(action.get("service", "-")),
            kind=action_kind,
            display=_lc_action_text(action),
            detail=detail,
        )
        target.last_reward_breakdown = breakdown
        target.latency_s = metrics.latency_s
        target.total_tps = metrics.total_tps
        target.reward_history.append(total_reward)

        diff_state.status = "running"
        diff_state.current_episode = target.episode_num
        diff_state.current_tick = target.tick
        if mode == "baseline":
            diff_state.baseline_running_reward = total_reward
            diff_state.baseline_running_history = list(target.reward_history)
        else:
            diff_state.fewshot_running_reward = total_reward
            diff_state.fewshot_running_history = list(target.reward_history)

        self.state.events.appendleft(
            LCEvent(
                tick=target.tick,
                difficulty=difficulty,
                agent=mode,
                kind=action_kind,
                action=_lc_action_text(action),
                reward=reward,
            )
        )
        self.refresh()

    def finish_episode(
        self,
        difficulty: str,
        mode: str,
        reward: float,
        success: bool,
        failure_reason: str | None,
    ) -> None:
        target = self.state.baseline if mode == "baseline" else self.state.fewshot
        diff_state = self.state.difficulties[difficulty]
        target.status = "complete" if success else "failed"
        target.success = success
        if failure_reason:
            target.last_action.detail = failure_reason

        if mode == "baseline":
            diff_state.baseline_rewards.append(reward)
            diff_state.baseline_running_reward = None
            diff_state.baseline_last_history = list(target.reward_history)
            diff_state.baseline_running_history = []
        else:
            diff_state.fewshot_rewards.append(reward)
            diff_state.fewshot_running_reward = None
            diff_state.fewshot_last_history = list(target.reward_history)
            diff_state.fewshot_running_history = []

        if len(diff_state.baseline_rewards) >= self.state.total_episodes and len(
            diff_state.fewshot_rewards
        ) >= self.state.total_episodes:
            diff_state.status = "complete"
        elif diff_state.baseline_rewards or diff_state.fewshot_rewards:
            diff_state.status = "running"
        else:
            diff_state.status = "pending"

        self.state.finished = all(
            len(item.baseline_rewards) >= self.state.total_episodes
            and len(item.fewshot_rewards) >= self.state.total_episodes
            for item in self.state.difficulties.values()
        )
        self.refresh()

    def refresh(self) -> None:
        self.live.update(build_layout(self.state, self.console), refresh=True)
