"""Shared LLM backend helpers for local Ollama and optional Groq."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

try:
    from groq import Groq
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    Groq = None

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    ollama_host: str = DEFAULT_OLLAMA_HOST
    groq_client: Any | None = None


@dataclass
class LLMCallMetrics:
    provider: str
    model: str
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    completion_tps: float = 0.0
    total_tps: float = 0.0
    load_s: float = 0.0
    done_reason: str = ""
    raw_preview: str = ""
    used_fallback: bool = False
    fallback_reason: str = ""

    def mark_fallback(self, reason: str) -> "LLMCallMetrics":
        self.used_fallback = True
        self.fallback_reason = reason
        return self


@dataclass
class LLMCallResult:
    action: dict[str, Any]
    metrics: LLMCallMetrics


def ollama_has_model(model: str, host: str = DEFAULT_OLLAMA_HOST) -> bool:
    """Return whether the running Ollama server exposes ``model``."""

    try:
        response = httpx.get(f"{host}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception:
        return False

    names = {item.get("name") for item in response.json().get("models", [])}
    return model in names


def resolve_provider(
    provider: str,
    model: str,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> str:
    """Resolve ``auto`` to a concrete provider, preferring local Ollama."""

    if provider == "auto":
        if ollama_has_model(model, ollama_host):
            return "ollama"
        if os.environ.get("GROQ_API_KEY") and Groq is not None:
            return "groq"
        return "heuristic"

    if provider == "ollama" and not ollama_has_model(model, ollama_host):
        raise RuntimeError(
            f"Ollama model {model!r} was not found at {ollama_host}. "
            "Start `ollama serve` and ensure the model is pulled."
        )

    if provider == "groq":
        if Groq is None or not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for provider=groq.")

    return provider


def build_llm_config(
    provider: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> LLMConfig:
    """Build a resolved LLM configuration."""

    resolved_provider = resolve_provider(provider, model, ollama_host)
    groq_client = None
    if resolved_provider == "groq":
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return LLMConfig(
        provider=resolved_provider,
        model=model,
        ollama_host=ollama_host,
        groq_client=groq_client,
    )


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""

    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    return json.loads(text.strip())


def _ns_to_s(value: Any) -> float:
    if not value:
        return 0.0
    return float(value) / 1_000_000_000.0


def _safe_tps(tokens: int, seconds: float) -> float:
    if tokens <= 0 or seconds <= 0:
        return 0.0
    return round(tokens / seconds, 2)


def call_action_model(
    config: LLMConfig,
    messages: list[dict[str, str]],
    fallback: dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
) -> LLMCallResult:
    """Call the configured model and return a parsed action JSON object."""

    if config.provider == "heuristic":
        return LLMCallResult(
            action=fallback,
            metrics=LLMCallMetrics(
                provider="heuristic",
                model=config.model,
                raw_preview="deterministic heuristic policy",
            ),
        )

    started = perf_counter()
    try:
        if config.provider == "ollama":
            response = httpx.post(
                f"{config.ollama_host}/api/chat",
                json={
                    "model": config.model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature},
                },
                timeout=90.0,
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload["message"]["content"]
            prompt_s = _ns_to_s(payload.get("prompt_eval_duration"))
            completion_s = _ns_to_s(payload.get("eval_duration"))
            total_s = _ns_to_s(payload.get("total_duration"))
            metrics = LLMCallMetrics(
                provider="ollama",
                model=config.model,
                latency_s=round(perf_counter() - started, 3),
                prompt_tokens=int(payload.get("prompt_eval_count") or 0),
                completion_tokens=int(payload.get("eval_count") or 0),
                prompt_tps=_safe_tps(
                    int(payload.get("prompt_eval_count") or 0),
                    prompt_s,
                ),
                completion_tps=_safe_tps(
                    int(payload.get("eval_count") or 0),
                    completion_s,
                ),
                total_tps=_safe_tps(
                    int(payload.get("prompt_eval_count") or 0)
                    + int(payload.get("eval_count") or 0),
                    total_s,
                ),
                load_s=round(_ns_to_s(payload.get("load_duration")), 3),
                done_reason=str(payload.get("done_reason") or ""),
                raw_preview=raw.strip().replace("\n", " ")[:180],
            )
        else:
            response = config.groq_client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            latency_s = round(perf_counter() - started, 3)
            metrics = LLMCallMetrics(
                provider="groq",
                model=config.model,
                latency_s=latency_s,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                completion_tps=_safe_tps(completion_tokens, latency_s),
                total_tps=_safe_tps(prompt_tokens + completion_tokens, latency_s),
                raw_preview=raw.strip().replace("\n", " ")[:180],
            )
        action = extract_json_object(raw)
        return LLMCallResult(action=action, metrics=metrics)
    except Exception as exc:
        return LLMCallResult(
            action=fallback,
            metrics=LLMCallMetrics(
                provider=config.provider,
                model=config.model,
                latency_s=round(perf_counter() - started, 3),
                used_fallback=True,
                fallback_reason=str(exc),
            ),
        )
