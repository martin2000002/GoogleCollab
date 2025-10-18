from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

class Problem(ABC):
    @abstractmethod
    def initialize_population(self, pop_size: int) -> List[Any]:
        ...

    @abstractmethod
    def fitness(self, individual: Any) -> float:
        ...

    @abstractmethod
    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        ...

    @abstractmethod
    def mutate(self, individual: Any, mutation_prob: float) -> Any:
        ...

    @abstractmethod
    def stringify(self, individual: Any) -> tuple[str, str]:
        """
        Return (chromosome_repr, decoded_values_repr) for logging.
        """
        ...


class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, ranked: Sequence[Tuple[Any, float]], maximize: bool) -> Any:
        """Return a selected individual from ranked population."""
        ...