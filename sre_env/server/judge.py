"""Deterministic post-mortem judge for the unified environment.

Scores the agent's post-incident analysis using keyword matching against
expected root cause, attack vector, and prevention strategies. No external
LLM is needed — the environment is fully self-contained.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Score how many expected keywords appear in the text. Returns 0.0-1.0."""
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def _similarity_score(text: str, reference: str) -> float:
    """Fuzzy similarity between agent's text and a reference string."""
    if not text or not reference:
        return 0.0
    return SequenceMatcher(None, text.lower(), reference.lower()).ratio()


class PostMortemJudge:
    """Score free-text reasoning deterministically, returning a 0.0-0.30 score.

    Scoring breakdown (max 0.30 total):
    - root_cause keywords:      0.00 - 0.10
    - attack_vector keywords:   0.00 - 0.08
    - prevention keywords:      0.00 - 0.06
    - fix_sequence quality:     0.00 - 0.06
    """

    def score(
        self,
        postmortem: str,
        scenario: dict[str, Any],
    ) -> float:
        """Score a post-mortem submission deterministically."""
        try:
            pm = json.loads(postmortem)
        except (json.JSONDecodeError, TypeError):
            pm = {"root_cause": postmortem, "attack_vector": "", "fix_sequence": [], "prevention": ""}

        keywords = scenario.get("postmortem_correct_keywords", {})
        root_cause_text = str(pm.get("root_cause", "")).lower()
        attack_text = str(pm.get("attack_vector", "")).lower()
        prevention_text = str(pm.get("prevention", "")).lower()
        fix_sequence = pm.get("fix_sequence", [])

        total = 0.0

        # Root cause scoring (max 0.10)
        root_kw_score = _keyword_score(root_cause_text, keywords.get("root_cause", []))
        root_service = scenario.get("root_cause_service", "")
        root_service_bonus = 0.03 if root_service and root_service in root_cause_text else 0.0
        total += min(0.10, root_kw_score * 0.07 + root_service_bonus)

        # Attack vector scoring (max 0.08)
        attack_kw_score = _keyword_score(attack_text, keywords.get("attack_vector", []))
        total += min(0.08, attack_kw_score * 0.08)

        # Prevention scoring (max 0.06)
        prevention_kw_score = _keyword_score(prevention_text, keywords.get("prevention", []))
        total += min(0.06, prevention_kw_score * 0.06)

        # Fix sequence scoring (max 0.06)
        expected_fix = scenario.get("correct_fix_sequence", [])
        if fix_sequence and expected_fix:
            fix_text = " ".join(str(f) for f in fix_sequence).lower()
            expected_text = " ".join(
                f.get("tool", "") + " " + f.get("service", "")
                for f in expected_fix
            ).lower()
            fix_sim = _similarity_score(fix_text, expected_text)
            total += min(0.06, fix_sim * 0.06)

        return round(total, 4)
