#!/usr/bin/env python3
"""Local pre-submission validation for the SRE-Env repo."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
OPENENV = str(ROOT / ".venv" / "bin" / "openenv")
BASE_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")
SPACE_URL = os.environ.get("SPACE_URL")
RUN_DOCKER_VALIDATION = os.environ.get("RUN_DOCKER_VALIDATION", "0") == "1"


def run(cmd: list[str], *, timeout: int = 300, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        timeout=timeout,
        env=env,
        text=True,
        capture_output=True,
    )


def wait_for_server(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{url}/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
    raise RuntimeError(f"Server did not become healthy at {url}")


def server_is_healthy(url: str) -> bool:
    try:
        response = httpx.get(f"{url}/health", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def main() -> None:
    print("pre-submit: pytest")
    run([PYTHON, "-m", "pytest", "sre_env/tests", "-q"])
    print("pre-submit: local openenv validate")
    run([OPENENV, "validate", "."])

    server = None
    started_server = False
    if server_is_healthy(BASE_URL):
        print("pre-submit: reusing existing local server")
    else:
        print("pre-submit: start local server")
        server = subprocess.Popen(
            [PYTHON, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started_server = True
    try:
        wait_for_server(BASE_URL)
        print("pre-submit: runtime openenv validate")
        run([OPENENV, "validate", "--url", BASE_URL], timeout=60)

        with httpx.Client(timeout=10.0) as client:
            print("pre-submit: endpoint checks")
            health = client.get(f"{BASE_URL}/health").json()
            assert health["status"] == "healthy"

            tasks = client.get(f"{BASE_URL}/unified-tasks").json()["scenarios"]
            assert len(tasks) >= 3

            for task in tasks[:3]:
                grader = client.get(f"{BASE_URL}/grader", params={"scenario_id": task["id"]}).json()
                score = float(grader["score"])
                assert 0.0 <= score <= 1.0

            reset_response = client.post(
                f"{BASE_URL}/reset",
                json={"difficulty": "easy", "scenario_id": "easy_001"},
            )
            reset_response.raise_for_status()
            assert "observation" in reset_response.json()

        env = os.environ.copy()
        env.setdefault("API_BASE_URL", "http://127.0.0.1:11434/v1")
        env.setdefault("MODEL_NAME", "qwen2.5:1.5b")
        env.setdefault("HF_TOKEN", "local")
        print("pre-submit: inference smoke")
        inference = subprocess.run(
            [PYTHON, "inference.py"],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=180,
            text=True,
            capture_output=True,
        )
        stdout = inference.stdout
        assert "[START]" in stdout and "[STEP]" in stdout and "[END]" in stdout

        print("pre-submit: baseline smoke")
        run(
            [
                PYTHON,
                "-m",
                "sre_env.scripts.baseline_agent",
                "--provider",
                "heuristic",
                "--model",
                "heuristic",
                "--base-url",
                BASE_URL,
            ],
            timeout=300,
            env=env,
        )

        print("pre-submit: learning-curve smoke")
        lc_env = env.copy()
        lc_env["MPLCONFIGDIR"] = "/tmp/matplotlib"
        run(
            [
                PYTHON,
                "-m",
                "sre_env.scripts.learning_curve",
                "--provider",
                "heuristic",
                "--model",
                "heuristic",
                "--episodes",
                "1",
                "--base-url",
                BASE_URL,
                "--save-plot",
            ],
            timeout=240,
            env=lc_env,
        )
        assert (ROOT / "learning_curve.png").exists()

        print("pre-submit: stored-policy training smoke")
        run(
            [
                PYTHON,
                "-m",
                "sre_env.scripts.grpo_train",
                "--provider",
                "heuristic",
                "--difficulty",
                "easy",
                "--steps",
                "1",
                "--group-size",
                "2",
                "--base-url",
                BASE_URL,
                "--output-dir",
                "outputs/grpo_sre_smoke",
            ],
            timeout=300,
            env=env,
        )
        assert (ROOT / "outputs" / "grpo_sre_smoke" / "trained_policy.json").exists()

        print("pre-submit: stored-policy comparison smoke")
        run(
            [
                PYTHON,
                "-m",
                "sre_env.scripts.three_agent_comparison",
                "--base-url",
                BASE_URL,
                "--n-episodes",
                "1",
                "--use-trained-weights",
                "--weights-path",
                "outputs/grpo_sre_smoke/best_weights.npz",
            ],
            timeout=300,
            env=lc_env,
        )

        if RUN_DOCKER_VALIDATION and shutil.which("docker"):
            print("pre-submit: docker build")
            run(["docker", "build", "-t", "sre-env:presubmit", "."], timeout=1800)

        if SPACE_URL:
            print("pre-submit: remote space check")
            with httpx.Client(timeout=20.0) as client:
                health = client.get(f"{SPACE_URL.rstrip('/')}/health")
                health.raise_for_status()
                reset = client.post(
                    f"{SPACE_URL.rstrip('/')}/reset",
                    json={"difficulty": "easy", "scenario_id": "easy_001"},
                )
                reset.raise_for_status()
        print("pre-submit: ok")
    finally:
        if started_server and server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
