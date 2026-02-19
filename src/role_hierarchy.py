"""
Role Hierarchy — Keyword-based fallback authority scorer.

Used ONLY when the LLM is unavailable or returns invalid JSON.
The primary weight source is ParticipantStore (LLM-inferred).
"""

from typing import Dict

# ─── Keyword authority map ────────────────────────────────────────────────────
# Each entry: (keyword, authority_score)
# Ordered from most-specific to least-specific; first match wins.
_KEYWORD_AUTHORITY: list[tuple[str, float]] = [
    # C-suite acronyms (must be checked before generic keywords)
    ("ceo",           0.95),
    ("cto",           0.93),
    ("cfo",           0.92),
    ("coo",           0.92),
    ("cpo",           0.90),
    # C-suite / Founders
    ("founder",       0.95),
    ("chief",         0.92),
    ("president",     0.90),
    # VP / Director
    ("vp ",           0.85),
    ("vice president",0.85),
    ("director",      0.80),
    ("head of",       0.78),
    # Senior IC / Manager
    ("principal",     0.75),
    ("engineering manager", 0.73),
    ("manager",       0.70),
    # Mid-level IC
    ("lead",          0.65),
    ("senior",        0.62),
    ("tech lead",     0.64),
    ("product manager", 0.60),
    ("architect",     0.63),
    # External / Stakeholder
    ("investor",      0.68),
    ("board",         0.72),
    ("client",        0.65),
    ("customer",      0.63),
    ("vendor",        0.60),
    ("partner",       0.62),
    ("stakeholder",   0.62),
    ("external",      0.60),
    ("guest",         0.55),
    ("advisor",       0.65),
    # Junior IC
    ("developer",     0.40),
    ("engineer",      0.42),
    ("designer",      0.38),
    ("analyst",       0.40),
    ("qa",            0.35),
    ("tester",        0.35),
    # Support
    ("intern",        0.20),
    ("trainee",       0.18),
    ("assistant",     0.22),
    ("coordinator",   0.28),
]

_NEUTRAL_SCORE = 0.50


def get_role_authority(role: str) -> float:
    """
    Infer authority score (0–1) from a role string via keyword scan.

    Precedence:
      1. First keyword match in _KEYWORD_AUTHORITY list (most-specific wins).
      2. Neutral fallback of 0.50 for unrecognised roles.

    Args:
        role: Free-text role string, e.g. "Senior External Stakeholder"

    Returns:
        Authority score between 0.0 and 1.0.
    """
    lowered = role.lower().strip()
    for keyword, score in _KEYWORD_AUTHORITY:
        if keyword in lowered:
            return score
    return _NEUTRAL_SCORE


def get_fallback_weights(role: str, is_external: bool = False) -> Dict[str, float]:
    """
    Derive fusion weights from authority score when LLM is unavailable.

    Formula:
        role_w     = clip(0.15 + 0.20 * authority + 0.05 * is_external, 0.10, 0.40)
        tonal_w    = clip(0.20 - 0.10 * authority - 0.03 * is_external, 0.05, 0.25)
        temporal_w = clip(0.10 + 0.15 * authority - 0.05 * is_external, 0.05, 0.30)
        semantic_w = 1.0 - role_w - tonal_w - temporal_w

    All four weights are normalised to sum exactly to 1.0.

    Args:
        role: Free-text role string.
        is_external: True if participant is outside the organisation.

    Returns:
        Dict with keys: semantic, tonal, role, temporal.
    """
    authority = get_role_authority(role)
    ext = float(is_external)

    role_w     = _clip(0.15 + 0.20 * authority + 0.05 * ext,  0.10, 0.40)
    tonal_w    = _clip(0.20 - 0.10 * authority - 0.03 * ext,  0.05, 0.25)
    temporal_w = _clip(0.10 + 0.15 * authority - 0.05 * ext,  0.05, 0.30)
    semantic_w = max(0.10, 1.0 - role_w - tonal_w - temporal_w)

    weights = {
        "semantic":  semantic_w,
        "tonal":     tonal_w,
        "role":      role_w,
        "temporal":  temporal_w,
    }
    return _normalise(weights)


def get_role_description(role: str, is_external: bool = False) -> str:
    """
    Human-readable description of inferred authority for UI display.

    Args:
        role: Role string.
        is_external: External flag.

    Returns:
        Short description string.
    """
    authority = get_role_authority(role)
    if authority >= 0.85:
        tier = "Executive"
    elif authority >= 0.70:
        tier = "Senior / Manager"
    elif authority >= 0.55:
        tier = "Mid-level"
    elif authority >= 0.35:
        tier = "Junior IC"
    else:
        tier = "Support"

    ext_label = " (External)" if is_external else ""
    return f"{tier}{ext_label} — authority {authority:.2f}"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.25 for k in weights}
    return {k: round(v / total, 4) for k, v in weights.items()}
