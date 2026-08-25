"""Turn raw model outputs into the contract's labels.

Gender: the model gives a probability per class; we take the winner and use its
probability as confidence.

Age: the model regresses a single number of years, not a per-bracket
distribution. We bucket it, and derive a bracket confidence from how far the
estimate sits from the nearest bracket boundary - deep inside a bracket -> high
confidence; sitting on an edge -> low confidence. A model that output a real age
distribution would let us report a principled probability instead; this is a
documented, defensible approximation.
"""
from __future__ import annotations

# (label, lower_inclusive, upper_exclusive) - contract has no under-18 bracket.
_BRACKETS = [
    ("18-30", 18.0, 31.0),
    ("31-45", 31.0, 46.0),
    ("46-60", 46.0, 61.0),
    ("60+", 61.0, 200.0),
]


def gender_from_probs(probs: dict[str, float]) -> tuple[str, float]:
    """probs like {"female": 0.1, "male": 0.9}. Returns (label, confidence)."""
    if not probs:
        return "unknown", 0.0
    label = max(probs, key=probs.get)
    conf = float(probs[label])
    if label not in ("male", "female"):
        return "unknown", conf
    return label, conf


def age_bracket_from_years(age_years: float) -> tuple[str, float]:
    """Map a predicted age in years to a bracket label + derived confidence."""
    if age_years < 18.0:
        # Below the contract's supported range - be honest rather than force-fit.
        return "unknown", 0.5

    label, lo, hi = _find_bracket(age_years)
    confidence = _boundary_confidence(age_years, lo, hi)
    return label, confidence


def _find_bracket(age_years: float) -> tuple[str, float, float]:
    for label, lo, hi in _BRACKETS:
        if lo <= age_years < hi:
            return label, lo, hi
    return _BRACKETS[-1]  # 60+ catch-all


def _boundary_confidence(age: float, lo: float, hi: float) -> float:
    """Higher when `age` is near the bracket centre, lower near an edge.

    We map the distance to the nearest boundary (in years) onto [0.5, 0.95]:
    0 years from an edge -> 0.5, >= 6 years inside -> capped at 0.95.
    """
    dist_to_edge = min(age - lo, hi - age)
    scaled = 0.5 + 0.45 * min(dist_to_edge / 6.0, 1.0)
    return round(float(scaled), 3)
