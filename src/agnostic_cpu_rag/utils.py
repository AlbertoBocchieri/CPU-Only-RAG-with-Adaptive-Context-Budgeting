from __future__ import annotations

import math
import re
from statistics import median

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(float(v) for v in values)
    q = clamp(float(q), 0.0, 1.0)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    alpha = pos - lo
    return float((1.0 - alpha) * s[lo] + alpha * s[hi])


def mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = float(median(values))
    return float(median([abs(float(v) - med) for v in values]))


def minmax_normalize(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    vals = [float(v) for v in score_map.values()]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return {key: 1.0 for key in score_map}
    return {key: float((float(value) - lo) / (hi - lo)) for key, value in score_map.items()}


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter / max(1, union))


def entropy_concentration(values: list[float]) -> float:
    positives = [max(0.0, float(v)) for v in values if math.isfinite(float(v))]
    if not positives:
        return 0.0
    total = sum(positives)
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in positives]
    if len(probs) == 1:
        return 1.0
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0.0:
        return 1.0
    return float(1.0 - (entropy / max_entropy))


def head_words(text: str, max_words: int) -> str:
    words = str(text).split()
    return " ".join(words[: max(0, int(max_words))]).strip()
