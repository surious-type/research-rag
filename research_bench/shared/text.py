from __future__ import annotations

import math
import statistics
from typing import Any, Sequence


def word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def aggregate_numeric(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "median": None, "standard_deviation": None, "valid_count": 0, "failed_count": len(values)}
    return {
        "mean": float(statistics.fmean(clean)),
        "median": float(statistics.median(clean)),
        "standard_deviation": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
        "valid_count": len(clean),
        "failed_count": len(values) - len(clean),
    }


def latency_summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "questions_count": len(values),
            "successful_questions": 0,
            "failed_questions": len(values),
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "min_latency_seconds": None,
            "max_latency_seconds": None,
        }
    return {
        "questions_count": len(values),
        "successful_questions": len(clean),
        "failed_questions": len(values) - len(clean),
        "mean_latency_seconds": float(statistics.fmean(clean)),
        "median_latency_seconds": float(statistics.median(clean)),
        "p95_latency_seconds": percentile(clean, 0.95),
        "min_latency_seconds": min(clean),
        "max_latency_seconds": max(clean),
    }


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
