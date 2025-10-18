from __future__ import annotations
import math
from typing import Sequence


def beale(v: Sequence[float]) -> float:
    x, y = v[0], v[1]
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y * y) ** 2
        + (2.625 - x + x * y * y * y) ** 2
    )


def easom(v: Sequence[float]) -> float:
    x, y = v[0], v[1]
    return -math.cos(x) * math.cos(y) * math.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))


def rastrigin(v: Sequence[float], A: float = 10.0) -> float:
    n = len(v)
    return A * n + sum((x * x - A * math.cos(2 * math.pi * x)) for x in v)
