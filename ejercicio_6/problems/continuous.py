from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Sequence, Tuple, Optional

from abc_c.interfaces import ABCProblem


class ABCContinuousProblem(ABCProblem):
    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)
        assert len(lower_bounds) == len(upper_bounds)
        self.func = func
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self.dim = len(self.lower_bounds)

    def _clip(self, x: List[float]) -> List[float]:
        return [max(lb, min(ub, xi)) for xi, lb, ub in zip(x, self.lower_bounds, self.upper_bounds)]

    def initialize_population(self, pop_size: int) -> List[List[float]]:
        pop: List[List[float]] = []
        for _ in range(pop_size):
            x = [random.uniform(lb, ub) for lb, ub in zip(self.lower_bounds, self.upper_bounds)]
            pop.append(x)
        return pop

    def fitness(self, solution: Sequence[float]) -> float:
        return self.func(solution)

    def generate_candidate(self, current: Sequence[float], peer: Sequence[float]) -> List[float]:
        # Standard ABC neighbor: v_ij = x_ij + phi_ij * (x_ij - x_kj), phi in [-1,1]
        v = list(current)
        j = random.randrange(self.dim)
        phi = random.uniform(-1.0, 1.0)
        v[j] = current[j] + phi * (current[j] - peer[j])
        return self._clip(v)

    def stringify(self, solution: Sequence[float]) -> str:
        vec = '[' + ', '.join(f"{xi:.6f}" for xi in solution) + ']'
        return f"position={vec}"


# Standard benchmarks (re-exported to ease usage)

def beale_function(v: Sequence[float]) -> float:
    x, y = v[0], v[1]
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y * y) ** 2
        + (2.625 - x + x * y * y * y) ** 2
    )


def easom_function(v: Sequence[float]) -> float:
    x, y = v[0], v[1]
    return -math.cos(x) * math.cos(y) * math.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))
