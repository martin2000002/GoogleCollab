from __future__ import annotations

import random
from typing import Any, List, Sequence, Tuple, Optional

from abc_c.interfaces import ABCProblem
from shared.tsp.functions import euclidean


class ABCTSPProblem(ABCProblem):
    def __init__(self, positions: Sequence[Tuple[float, float]], seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.positions = list(positions)
        self.n = len(self.positions)
        # Precompute distances
        self.dist: List[List[float]] = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = euclidean(self.positions[i], self.positions[j])
                self.dist[i][j] = d
                self.dist[j][i] = d

    def _tour_length(self, tour: Sequence[int]) -> float:
        total = 0.0
        for i in range(self.n - 1):
            total += self.dist[tour[i]][tour[i + 1]]
        total += self.dist[tour[-1]][tour[0]]
        return total

    def initialize_population(self, pop_size: int) -> List[List[int]]:
        base = list(range(self.n))
        population: List[List[int]] = []
        for _ in range(pop_size):
            perm = base[1:]
            random.shuffle(perm)
            population.append([0] + perm)
        return population

    def fitness(self, solution: Sequence[int]) -> float:
        return self._tour_length(solution)

    def generate_candidate(self, current: Sequence[int], peer: Sequence[int]) -> List[int]:
        # Discrete neighbor guided by peer:
        # pick a random index j>=1; make the city at peer[j] be at position j in current via a swap
        n = self.n
        cand = list(current)
        j = random.randrange(1, n)
        city = peer[j]
        i = cand.index(city)
        if i != j:
            cand[i], cand[j] = cand[j], cand[i]
        else:
            # fallback: random 2-opt segment reversal
            a, b = sorted(random.sample(range(1, n), 2))
            cand[a:b + 1] = reversed(cand[a:b + 1])
        return cand

    def stringify(self, solution: Sequence[int]) -> str:
        tour = '[' + ', '.join(str(x) for x in solution) + ']'
        return f"tour={tour}"
