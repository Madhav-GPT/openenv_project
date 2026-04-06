"""HTTP route tests."""

from fastapi.testclient import TestClient

from sre_env.server.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["environment"] == "sre_env"
    assert response.json()["status"] == "healthy"


def test_tasks_all() -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "sre_env"
    assert len(data["scenarios"]) == 3
    assert sorted(data["available_difficulties"]) == ["easy", "hard", "medium"]


def test_tasks_filtered_easy() -> None:
    response = client.get("/tasks", params={"difficulty": "easy"})
    assert response.status_code == 200
    data = response.json()
    assert all(item["difficulty"] == "easy" for item in data["scenarios"])


def test_tasks_invalid_difficulty() -> None:
    response = client.get("/tasks", params={"difficulty": "impossible"})
    assert response.status_code == 404


def test_baseline_all() -> None:
    response = client.get("/baseline")
    assert response.status_code == 200
    assert len(response.json()["baselines"]) == 3


def test_baseline_filtered() -> None:
    response = client.get("/baseline", params={"scenario_id": "easy_001"})
    assert response.status_code == 200
    assert response.json()["baselines"][0]["scenario_id"] == "easy_001"


def test_grader_returns_report() -> None:
    response = client.get("/grader")
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "checks" in data


def test_status() -> None:
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "sre_env"
    assert "progress" in data
    assert "grader" in data


def test_unified_tasks() -> None:
    response = client.get("/unified-tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "sre_env"
    assert len(data["scenarios"]) == 3


def test_phase2_baseline() -> None:
    response = client.get("/phase2-baseline")
    assert response.status_code == 200
    data = response.json()
    assert len(data["baselines"]) == 3
