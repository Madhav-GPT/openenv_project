#!/usr/bin/env python3
"""Unified demo pipeline for SRE-Env.

Runs the complete story in one pass:
  1. Preflight checks (Ollama, judge server)
  2. Baseline agent exam
  3. Reward-signal comparison
  4. Live Phase 2 dashboard
  5. Submission inference
  6. Pre-submission validation
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.environ.get("MODEL_NAME", "qwen2.5:7b")

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def banner(step: int, total: int, title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'━' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  [{step}/{total}] {title}{RESET}")
    print(f"{BOLD}{CYAN}{'━' * 60}{RESET}\n")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")
    sys.exit(1)


def check_ollama() -> None:
    """Check that Ollama is running and the model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        ok(f"Ollama is running at {OLLAMA_URL}")

        model_base = MODEL.split(":")[0]
        matching = [m for m in models if m.startswith(model_base)]
        if matching:
            ok(f"Model available: {matching[0]}")
        else:
            warn(f"Model '{MODEL}' not found. Available: {models}")
            warn(f"Run: ollama pull {MODEL}")
    except Exception:
        fail(
            f"Ollama is not running at {OLLAMA_URL}.\n"
            "       Start it in a separate terminal:\n"
            "         ollama serve"
        )


def check_judge() -> None:
    """Check that the SRE judge server is running."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        resp.raise_for_status()
        health = resp.json()
        ok(f"Judge server is healthy at {BASE_URL}")
        ok(f"Environment: {health.get('environment', '?')} v{health.get('version', '?')}")
        ok(f"Phases: {', '.join(health.get('phases', []))}")
        difficulties = health.get("difficulties", {})
        ok(f"Difficulties: {', '.join(f'{k}({v} ticks)' for k, v in difficulties.items())}")
    except Exception:
        fail(
            f"Judge server is not running at {BASE_URL}.\n"
            "       Start it in a separate terminal:\n"
            "         source .venv/bin/activate && make dev"
        )


def run_step(cmd: list[str], *, timeout: int = 600, env: dict | None = None) -> None:
    """Run a command, streaming output to the terminal."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            timeout=timeout,
            env=merged_env,
        )
    except subprocess.TimeoutExpired:
        warn("Step timed out but continuing...")
    except subprocess.CalledProcessError as exc:
        fail(f"Step failed with exit code {exc.returncode}")
    except KeyboardInterrupt:
        warn("Interrupted by user, skipping to next step...")


def main() -> None:
    parser = argparse.ArgumentParser(description="SRE-Env unified demo pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run shorter versions of baseline and dashboard (fewer episodes).",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip the interactive live dashboard step.",
    )
    args = parser.parse_args()

    total = 6
    if args.skip_dashboard:
        total = 5

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  SRE-Env: Unified Demo Pipeline{RESET}")
    print(f"{BOLD}  Mode: {'quick' if args.quick else 'full'}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")

    # ── Step 1: Preflight ──────────────────────────────────────
    step = 1
    banner(step, total, "Preflight Checks")
    check_ollama()
    check_judge()

    # ── Step 2: Baseline Agent ─────────────────────────────────
    step = 2
    banner(step, total, "Baseline Agent Exam (LLM takes the test)")
    baseline_cmd = [
        PYTHON, "-m", "sre_env.scripts.baseline_agent",
        "--provider", "ollama",
        "--model", MODEL,
        "--base-url", BASE_URL,
    ]
    run_step(baseline_cmd, timeout=600)
    ok("Baseline agent exam complete")

    # ── Step 3: Reward Signal Comparison ───────────────────────
    step = 3
    banner(step, total, "Reward Signal Proof (random vs naive vs optimal)")
    episodes = "3" if args.quick else "10"
    compare_cmd = [
        PYTHON, "-m", "sre_env.scripts.three_agent_comparison",
        "--base-url", BASE_URL,
        "--n-episodes", episodes,
    ]
    run_step(compare_cmd, timeout=600)
    if (ROOT / "three_agent_comparison.png").exists():
        ok("Saved: three_agent_comparison.png")
    else:
        warn("three_agent_comparison.png not found")

    # ── Step 4: Live Dashboard ─────────────────────────────────
    if not args.skip_dashboard:
        step = 4
        banner(step, total, "Live Phase 2 Dashboard (baseline vs reference)")
        dashboard_episodes = "2" if args.quick else "5"
        dashboard_cmd = [
            PYTHON, "-m", "sre_env.scripts.learning_curve_dashboard",
            "--provider", "ollama",
            "--student-model", MODEL,
            "--episodes", dashboard_episodes,
            "--difficulties", "easy", "medium", "hard",
            "--base-url", BASE_URL,
        ]
        run_step(dashboard_cmd, timeout=900)
        ok("Dashboard complete")
        step_offset = 0
    else:
        step_offset = -1

    # ── Step 5: Submission Inference ───────────────────────────
    step = 5 + step_offset
    banner(step, total, "Submission Inference (official exam run)")
    inference_env = {
        "API_BASE_URL": f"{OLLAMA_URL}/v1",
        "MODEL_NAME": os.environ.get("MODEL_NAME", "qwen2.5:1.5b"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", "local"),
        "ENV_BASE_URL": BASE_URL,
    }
    run_step([PYTHON, "inference.py"], timeout=300, env=inference_env)
    if (ROOT / "baseline_scores.json").exists():
        ok("Saved: baseline_scores.json")
    else:
        warn("baseline_scores.json not found")

    # ── Step 6: Pre-Submission Validation ──────────────────────
    step = 6 + step_offset
    banner(step, total, "Pre-Submission Validation (final checklist)")
    run_step([PYTHON, "pre_submission_validate.py"], timeout=600)
    ok("Validation complete")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  ✓ Demo pipeline finished successfully{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}")
    print()
    print("  Artifacts produced:")
    for artifact in ["three_agent_comparison.png", "learning_curve.png", "baseline_scores.json"]:
        path = ROOT / artifact
        status = f"{GREEN}✓{RESET}" if path.exists() else f"{YELLOW}─{RESET}"
        print(f"    {status} {artifact}")
    print()


if __name__ == "__main__":
    main()
