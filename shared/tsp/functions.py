from __future__ import annotations
import math
import random
from typing import List, Tuple


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def make_random_positions(n: int, width: float = 100.0, height: float = 100.0, seed: int | None = None) -> List[Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)
    return [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]


def make_grid_positions(n: int, spacing: float = 10.0) -> List[Tuple[float, float]]:
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError("n must be a perfect square for grid positions (e.g., 25, 100, 225)")
    coords: List[Tuple[float, float]] = []
    for r in range(side):
        for c in range(side):
            coords.append((c * spacing, r * spacing))
    return coords
