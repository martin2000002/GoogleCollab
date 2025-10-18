import random
from typing import Any, Sequence, Tuple
from ga.interfaces import SelectionStrategy

class TournamentSelection(SelectionStrategy):
    def __init__(self, k: int = 3) -> None:
        if k < 2:
            raise ValueError("Tournament k must be >= 2")
        self.k = k

    def select(self, ranked: Sequence[Tuple[Any, float]], maximize: bool) -> Any:
        if self.k > len(ranked):
            raise ValueError("Tournament k cannot exceed population size")
        contenders = random.sample(ranked, self.k)
        contenders.sort(key=lambda t: t[1], reverse=maximize)
        return contenders[0][0]