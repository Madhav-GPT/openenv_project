"""Shared benchmark policies and scenario mappings for Phase 2 scripts."""

from __future__ import annotations

import random

from sre_env.models import SREAction

SCENARIO_FOR_DIFFICULTY = {"easy": "easy_001", "medium": "medium_001", "hard": "hard_001"}
ALL_SERVICES = ["api-gateway", "cache", "database", "worker"]
ALL_METRICS = ["cpu", "memory", "latency", "error_rate", "throughput"]

OPTIMAL_ACTIONS: dict[str, list[SREAction]] = {
    "easy_001": [
        SREAction(tool="get_logs", service="database"),
        SREAction(tool="classify_vuln", vulnerability_type="sql_injection"),
        SREAction(tool="apply_patch", patch_id="parameterized_query"),
        SREAction(tool="verify_patch"),
        SREAction(tool="restart", service="database"),
        SREAction(
            tool="post_mortem",
            root_cause="database OOM caused by SQL injection dropping the users table",
            attack_vector="sql injection in the login handler via string concatenation",
            fix_sequence=["parameterized query patch", "database restart"],
            prevention="use parameterized queries and add WAF coverage",
        ),
    ],
    "medium_001": [
        SREAction(tool="get_logs", service="cache"),
        SREAction(tool="get_logs", service="api-gateway"),
        SREAction(tool="classify_vuln", vulnerability_type="xss"),
        SREAction(tool="apply_patch", patch_id="html_escape"),
        SREAction(tool="verify_patch"),
        SREAction(tool="restart", service="cache"),
        SREAction(tool="restart", service="database"),
        SREAction(
            tool="post_mortem",
            root_cause="cache OOM after an XSS-stolen admin session changed cache config",
            attack_vector="xss in the comment renderer stole the admin session cookie",
            fix_sequence=["html_escape patch", "restart cache", "restart database"],
            prevention="escape HTML, set HttpOnly cookies, and add CSP headers",
        ),
    ],
    "hard_001": [
        SREAction(tool="get_metrics", service="worker", metric="memory"),
        SREAction(tool="get_logs", service="worker"),
        SREAction(tool="get_logs", service="database"),
        SREAction(tool="classify_vuln", vulnerability_type="broken_auth"),
        SREAction(tool="apply_patch", patch_id="require_admin_role"),
        SREAction(tool="verify_patch"),
        SREAction(tool="rollback", service="worker", version="previous"),
        SREAction(tool="restart", service="database"),
        SREAction(tool="restart", service="api-gateway"),
        SREAction(
            tool="post_mortem",
            root_cause="worker v2.4.1 memory leak deployed through broken admin auth",
            attack_vector="broken access control on /admin/deploy let a non-admin deploy v2.4.1",
            fix_sequence=[
                "require_admin_role patch",
                "rollback worker",
                "restart database",
                "restart api-gateway",
            ],
            prevention="enforce RBAC, require signed deploys, and add approval gates",
        ),
    ],
}

NAIVE_ACTIONS: dict[str, list[SREAction]] = {
    "easy_001": [
        SREAction(tool="restart", service="database"),
        SREAction(tool="get_logs", service="api-gateway"),
        SREAction(tool="get_dependencies", service="cache"),
    ],
    "medium_001": [
        SREAction(tool="restart", service="database"),
        SREAction(tool="get_logs", service="cache"),
        SREAction(tool="restart", service="cache"),
        SREAction(tool="restart", service="database"),
        SREAction(tool="get_dependencies", service="worker"),
    ],
    "hard_001": [
        SREAction(tool="restart", service="database"),
        SREAction(tool="get_metrics", service="worker", metric="memory"),
        SREAction(tool="rollback", service="worker", version="previous"),
        SREAction(tool="restart", service="database"),
        SREAction(tool="restart", service="api-gateway"),
        SREAction(tool="get_dependencies", service="cache"),
    ],
}


def random_action(obs_dict: dict, rng: random.Random) -> SREAction:
    del obs_dict
    tool = rng.choice(
        [
            "get_logs",
            "get_metrics",
            "get_dependencies",
            "restart",
            "scale",
            "rollback",
            "classify_vuln",
            "apply_patch",
            "verify_patch",
            "post_mortem",
        ]
    )
    if tool == "get_metrics":
        return SREAction(tool=tool, service=rng.choice(ALL_SERVICES), metric=rng.choice(ALL_METRICS))
    if tool == "rollback":
        return SREAction(tool=tool, service=rng.choice(ALL_SERVICES), version="previous")
    if tool == "scale":
        return SREAction(tool=tool, service=rng.choice(ALL_SERVICES), replicas=rng.randint(1, 3))
    if tool == "classify_vuln":
        return SREAction(
            tool=tool,
            vulnerability_type=rng.choice(["sql_injection", "xss", "broken_auth"]),
        )
    if tool == "apply_patch":
        return SREAction(
            tool=tool,
            patch_id=rng.choice(
                [
                    "parameterized_query",
                    "strip_quotes",
                    "disable_login",
                    "html_escape",
                    "remove_script_substring",
                    "disable_comments",
                    "require_admin_role",
                    "hide_admin_link",
                    "deny_all_admin",
                ]
            ),
        )
    if tool == "verify_patch":
        return SREAction(tool=tool)
    if tool == "post_mortem":
        return SREAction(
            tool=tool,
            root_cause="unknown",
            attack_vector="unknown",
            fix_sequence=["guessed action"],
            prevention="monitor",
        )
    return SREAction(tool=tool, service=rng.choice(ALL_SERVICES))


def naive_action(obs_dict: dict) -> SREAction:
    services = obs_dict.get("services", {})
    if not services:
        return SREAction(tool="get_logs", service="database")

    def score(item: tuple[str, dict]) -> tuple[float, float]:
        _, service = item
        return (
            float(service.get("error_rate_pct", 0)),
            float(service.get("cpu_pct", 0)),
        )

    worst_name, _ = max(services.items(), key=score)
    return SREAction(tool="restart", service=worst_name)
