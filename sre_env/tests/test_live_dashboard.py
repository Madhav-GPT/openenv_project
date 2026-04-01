from sre_env.scripts.live_dashboard import (
    render_ascii_graph,
    render_cpu_bar,
    render_sparkline,
    render_tick_bar,
)


def test_render_cpu_bar_width_and_fill() -> None:
    assert render_cpu_bar(52) == "▓▓▓▓▓░░░░░"


def test_render_tick_bar_plain_text() -> None:
    assert render_tick_bar(10, 20, 10, "green").plain == "tick [█████·····] 10/20"


def test_render_sparkline_contains_negative_zero_and_positive_marks() -> None:
    output = render_sparkline([-1.0, 0.0, 0.5, 1.0], 4)
    assert len(output) == 4
    assert "▁" in output
    assert "─" in output
    assert any(char in output for char in "▂▃▄▅▆▇█")


def test_render_ascii_graph_waiting_state() -> None:
    graph = render_ascii_graph(
        baseline_rewards=[],
        fewshot_rewards=[],
        width=24,
        height=8,
        running=False,
        current_tick=0,
        total_episodes=20,
    )
    assert "waiting..." in graph.plain
