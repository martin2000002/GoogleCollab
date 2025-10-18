import math
import random
from typing import List, Sequence, Tuple, Optional
from ga.interfaces import Problem

from shared.tsp.functions import euclidean

class TSPProblem(Problem):
    def __init__(self, positions: Sequence[Tuple[float, float]], seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        self.positions = list(positions)
        self.n = len(self.positions)

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

    def fitness(self, individual: Sequence[int]) -> float:
        return self._tour_length(individual)

    def crossover(self, p1: Sequence[int], p2: Sequence[int]) -> Tuple[List[int], List[int]]:
        size = self.n
        a, b = sorted(random.sample(range(1, size), 2))
        def ox(pa: Sequence[int], pb: Sequence[int]) -> List[int]:
            child = [-1] * size
            child[0] = 0
            child[a:b] = pa[a:b]
            pb_items = [x for x in pb if x not in child]
            idx = b
            for x in pb_items:
                if idx >= size:
                    idx = 1
                child[idx] = x
                idx += 1
            return child
        return ox(p1, p2), ox(p2, p1)

    def mutate(self, individual: Sequence[int], mutation_prob: float) -> List[int]:
        mutated = list(individual)
        if random.random() < mutation_prob:
            i, j = sorted(random.sample(range(1, self.n), 2))
            mutated[i:j + 1] = reversed(mutated[i:j + 1])
        return mutated

    def stringify(self, individual: Sequence[int]) -> tuple[str, str]:
        chrom = '[' + ', '.join(str(x) for x in individual) + ']'
        return chrom, '-'
