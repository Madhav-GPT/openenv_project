"""Deterministic WebSec verification outcomes for unified scenarios."""

from __future__ import annotations

from typing import Tuple


VERIFY_OUTCOMES: dict[str, dict[str, tuple[bool, bool, str]]] = {
    "sqli_login": {
        "parameterized_query": (
            True,
            True,
            "Exploit blocked: parameterized query prevents SQL injection and preserves login behavior.",
        ),
        "strip_quotes": (
            False,
            True,
            "Exploit still possible: quote stripping is incomplete and bypassable.",
        ),
        "disable_login": (
            True,
            False,
            "Exploit blocked but legitimate users cannot log in - core functionality is broken.",
        ),
    },
    "xss_comments": {
        "html_escape": (
            True,
            True,
            "Exploit blocked: HTML escaping neutralizes script payloads while preserving comments.",
        ),
        "remove_script_substring": (
            False,
            True,
            "Exploit still possible: blacklisting the string 'script' is bypassable.",
        ),
        "disable_comments": (
            True,
            False,
            "Exploit blocked but the comment feature is disabled for legitimate users.",
        ),
    },
    "broken_auth_admin": {
        "require_admin_role": (
            True,
            True,
            "Access control fixed: non-admin users are blocked and admins still retain access.",
        ),
        "hide_admin_link": (
            False,
            True,
            "Security through obscurity failed: the endpoint is still reachable directly.",
        ),
        "deny_all_admin": (
            True,
            False,
            "Attack vector closed, but all legitimate admin functionality is also blocked.",
        ),
    },
}


class WebSecGrader:
    """Verify whether a proposed patch blocks the exploit and preserves behavior."""

    def verify(self, sub_quest: dict, patch_id: str) -> Tuple[bool, bool, str]:
        task_id = sub_quest.get("task_id", "")
        if task_id not in VERIFY_OUTCOMES:
            return False, False, f"Unknown task_id '{task_id}'."

        outcomes = VERIFY_OUTCOMES[task_id]
        if patch_id not in outcomes:
            valid = ", ".join(sorted(outcomes))
            return False, False, f"Unknown patch_id '{patch_id}'. Valid options: {valid}"

        return outcomes[patch_id]
