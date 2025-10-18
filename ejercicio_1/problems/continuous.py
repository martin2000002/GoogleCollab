import math
import random
from typing import Callable, List, Sequence, Tuple, Optional
from ga.interfaces import Problem

class BinaryContinuousProblem(Problem):

    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        num_vars: int,
        bits_per_var: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)
        assert len(lower_bounds) == num_vars and len(upper_bounds) == num_vars
        self.func = func
        self.num_vars = num_vars
        self.bits_per_var = bits_per_var
        self.lower_bounds = list(lower_bounds)
        self.upper_bounds = list(upper_bounds)
        self._mask = (1 << bits_per_var) - 1
        self.total_bits = num_vars * bits_per_var

    def _decode(self, chromosome: Sequence[int]) -> List[float]:
        values: List[float] = []
        for i in range(self.num_vars):
            start = i * self.bits_per_var
            end = start + self.bits_per_var
            segment = chromosome[start:end]

            val = 0
            for b in segment:
                val = (val << 1) | (1 if b else 0)
            lb, ub = self.lower_bounds[i], self.upper_bounds[i]
            if self._mask == 0:
                scaled = lb
            else:
                scaled = lb + (val / self._mask) * (ub - lb)
            values.append(scaled)
        return values

    def initialize_population(self, pop_size: int) -> List[List[int]]:
        return [
            [random.randint(0, 1) for _ in range(self.total_bits)] for _ in range(pop_size)
        ]

    def fitness(self, individual: Sequence[int]) -> float:
        x = self._decode(individual)
        return self.func(x)

    def crossover(self, p1: Sequence[int], p2: Sequence[int]) -> Tuple[List[int], List[int]]:
        n = self.total_bits
        cut = random.randint(1, n - 1)
        c1 = list(p1[:cut]) + list(p2[cut:])
        c2 = list(p2[:cut]) + list(p1[cut:])
        return c1, c2

    def mutate(self, individual: Sequence[int], mutation_prob: float) -> List[int]:
        mutated = list(individual)
        for i in range(self.total_bits):
            if random.random() < mutation_prob:
                mutated[i] = 1 - mutated[i]
        return mutated

    def stringify(self, individual: Sequence[int]) -> tuple[str, str]:
        chrom = ''.join('1' if b else '0' for b in individual)
        decoded = self._decode(individual)
        decoded_str = '[' + ', '.join(f"{v:.6f}" for v in decoded) + ']'
        return chrom, decoded_str


# Functions
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