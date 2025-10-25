from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple



class ABCProblem(ABC):
	@abstractmethod
	def initialize_population(self, pop_size: int) -> List[Any]:
		"""Return an initial list of candidate solutions (food sources)."""
		...

	@abstractmethod
	def fitness(self, solution: Any) -> float:
		"""Return objective value. Lower is better if the algorithm runs with maximize=False."""
		...

	@abstractmethod
	def generate_candidate(self, current: Any, peer: Any) -> Any:
		"""Generate a neighbor candidate from current, optionally guided by peer."""
		...

	@abstractmethod
	def stringify(self, solution: Any) -> str:
		"""Return a concise, problem-appropriate string for logging.

		Expected conventions:
		- Continuous problems should return: "position=[x1, x2, ...]"
		- TSP problems should return: "tour=[i0, i1, ..., iN]"

		This replaces GA-style (chromosome/decoded) since ABC works directly on
		positions and permutations, not bitstrings.
		"""
		...

