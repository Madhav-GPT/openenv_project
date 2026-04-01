"""Typed HTTP client for the SRE incident-response environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import SREAction, SREObservation, SREState


class SREEnv(EnvClient[SREAction, SREObservation, SREState]):
    """OpenEnv client with typed payload parsing."""

    DEFAULT_BASE_URL = "http://localhost:8000"

    def _step_payload(self, action: SREAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[SREObservation]:
        observation_data = dict(payload.get("observation", {}))
        observation_data.setdefault("reward", payload.get("reward", 0.0))
        observation_data.setdefault("done", payload.get("done", False))
        observation_data.setdefault(
            "episode_complete", observation_data.get("done", payload.get("done", False))
        )
        observation = SREObservation.model_validate(observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> SREState:
        return SREState.model_validate(payload)
