from __future__ import annotations
import random
import shutil
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

from pathlib import Path
from tqdm import tqdm as _tqdm
from concurrent.futures import ThreadPoolExecutor

from shared.tsp.functions import euclidean
from shared.plots import save_convergence
from shared.utils import fmt_best
from ejercicio_1.problems.tsp import TSPProblem
from ejercicio_1.ga.core import GeneticAlgorithm
from ejercicio_1.ga.selections.tournament import TournamentSelection


@dataclass
class Ant:
    tour: List[int]
    length: float


def _tour_length(tour: Sequence[int], dist: List[List[float]]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


class HybridGAACO_TSP:
    """GA→ACO Hybrid for TSP: seed pheromone using elite GA tours, then run ACO."""

    def __init__(
        self,
        positions: Sequence[Tuple[float, float]],
        # GA params
        ga_population_size: int = 200,
        ga_elite_ratio: float = 0.2,
        ga_mutation_prob: float = 0.3,
        ga_max_generations: int = 200,
        ga_k: int = 3,
        ga_runs: int = 3,
        # ACO params
        num_ants: int = 50,
        max_epochs: int = 100,
        alpha: float = 1.0,
        beta: float = 5.0,
        rho: float = 0.5,
        q: float = 1.0,
        # misc
        seed_with_ga: bool = True,
        random_seed: Optional[int] = None,
        log: bool = False,
        dir_name: Optional[str] = None,
        exe_root: str = "ejercicio_5",
        show_progress: bool = False,
        parallel_workers: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)
        self.pos = list(positions)
        self.n = len(self.pos)
        if self.n < 2:
            raise ValueError("positions must contain at least 2 nodes")

        self.ga_population_size = ga_population_size
        self.ga_elite_ratio = ga_elite_ratio
        self.ga_mutation_prob = ga_mutation_prob
        self.ga_max_generations = ga_max_generations
        self.ga_k = ga_k
        self.ga_runs = ga_runs

        self.num_ants = num_ants
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.seed_with_ga = seed_with_ga
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name or "hybrid_tsp"
        self.exe_root = exe_root
        self.show_progress = show_progress
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

        # distances
        self.dist: List[List[float]] = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = euclidean(self.pos[i], self.pos[j])
                self.dist[i][j] = d
                self.dist[j][i] = d

        # pheromone, initialized later with GA seeding
        self.tau: List[List[float]] = [[1.0] * self.n for _ in range(self.n)]

    # --- GA phase to collect elites ---
    def _collect_elite_tours(self) -> List[List[int]]:
        tsp = TSPProblem(self.pos, seed=self.random_seed)
        ga = GeneticAlgorithm(
            problem=tsp,
            population_size=self.ga_population_size,
            mutation_prob=self.ga_mutation_prob,
            elite_ratio=self.ga_elite_ratio,
            max_generations=self.ga_max_generations,
            selection=TournamentSelection(k=self.ga_k),
            maximize=False,
            random_seed=self.random_seed or 0,
            log=False,
            dir_name=f"{self.dir_name}_ga_hyb",
            exe_root=self.exe_root,
            save_plots=False,
            show_progress=False,
        )
        elites: List[List[int]] = []
        _best_overall, results = ga.run_multiple(runs=self.ga_runs)
        for (best_tour, _fit, _fit_str, _hist, _seed, _rt) in results:
            elites.append(best_tour)
        return elites

    def _seed_pheromone(self, elite_tours: List[List[int]]) -> None:
        # start from tau0=1.0
        self.tau = [[1.0] * self.n for _ in range(self.n)]
        for tour in elite_tours:
            L = _tour_length(tour, self.dist)
            d_tau = self.q / (L + 1e-12)
            for k in range(self.n):
                i, j = tour[k], tour[(k + 1) % self.n]
                self.tau[i][j] += d_tau
                self.tau[j][i] += d_tau

    # --- ACO constructs ---
    def _construct_tour(self, start: int) -> Ant:
        unvisited = set(range(self.n))
        tour = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            denom = 0.0
            probs: List[Tuple[int, float]] = []
            for j in unvisited:
                val = (self.tau[current][j] ** self.alpha) * ((1.0 / (self.dist[current][j] + 1e-12)) ** self.beta)
                probs.append((j, val))
                denom += val
            r = random.random() * denom if denom > 0 else 0.0
            acc = 0.0
            next_city = None
            for j, val in probs:
                acc += val
                if acc >= r:
                    next_city = j
                    break
            if next_city is None:
                next_city = random.choice(list(unvisited))
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        L = _tour_length(tour, self.dist)
        return Ant(tour=tour, length=L)

    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[List[int], float, List[float], List[Tuple[List[int], float]]]:
        # GA seeding (optional)
        elites: List[List[int]] = []
        if self.seed_with_ga:
            elites = self._collect_elite_tours()
            self._seed_pheromone(elites)

        # initial best
        init = elites[0][:] if elites else list(range(self.n))
        if not elites:
            random.shuffle(init)
        gbest_tour = init
        gbest_len = _tour_length(gbest_tour, self.dist)

        best_history: List[float] = [gbest_len]
        per_epoch_best: List[Tuple[List[int], float]] = [(gbest_tour[:], gbest_len)]

        pbar = _tqdm(total=self.max_epochs, desc=progress_label or "GA→ACO-TSP", leave=False) if self.show_progress else None
        executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=self.parallel_workers) if self.parallel_workers else None

        for epoch in range(self.max_epochs):
            starts = [random.randrange(self.n) for _ in range(self.num_ants)]
            if executor is not None:
                ants = list(executor.map(self._construct_tour, starts))  # type: ignore[arg-type]
            else:
                ants = [self._construct_tour(s) for s in starts]

            ants.sort(key=lambda a: a.length)
            if ants[0].length < gbest_len:
                gbest_tour, gbest_len = ants[0].tour[:], ants[0].length

            # evaporate
            for i in range(self.n):
                for j in range(self.n):
                    self.tau[i][j] *= (1.0 - self.rho)

            # deposit (all ants)
            for ant in ants:
                d_tau = self.q / (ant.length + 1e-12)
                tour = ant.tour
                for k in range(self.n):
                    i, j = tour[k], tour[(k + 1) % self.n]
                    self.tau[i][j] += d_tau
                    self.tau[j][i] += d_tau

            best_history.append(gbest_len)
            per_epoch_best.append((gbest_tour[:], gbest_len))

            if pbar is not None:
                pbar.set_postfix({"best": fmt_best(gbest_len)})
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return gbest_tour, gbest_len, best_history, per_epoch_best

    # --- infra ---
    def _resolve_log_dir(self) -> str:
        return str(Path(self.exe_root) / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path(self.exe_root) / 'visualization' / self.dir_name

    def _prepare_log_dir(self, dir_path: str | Path) -> None:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    def _write_log(self, filepath: str, per_epoch_best: List[Tuple[List[int], float]], seed: Optional[int]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            if seed is not None:
                f.write(f"seed={seed}\n")
            for epoch, (tour, fit) in enumerate(per_epoch_best):
                tour_str = '[' + ', '.join(str(x) for x in tour) + ']'
                f.write(f"gen={epoch}\tfitness={fit:.12f}\ttour={tour_str}\n")

    def run_multiple(self, runs: int) -> Tuple[Tuple[List[int], float, str, List[float], int, float], List[Tuple[List[int], float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")
        prev_state = random.getstate()
        results: List[Tuple[List[int], float, str, List[float], int, float]] = []
        best_overall: Optional[Tuple[List[int], float, str, List[float], int, float]] = None
        try:
            if self.log:
                self._prepare_log_dir(self._resolve_log_dir())
            for i in range(runs):
                run_seed = (self.random_seed or 0) + i
                random.seed(run_seed)
                t0 = time.perf_counter()
                best_tour, best_len, best_hist, per_epoch = self._run_body(progress_label=f"run {i+1}/{runs}")
                run_time = time.perf_counter() - t0
                packed = (best_tour, best_len, fmt_best(best_len), best_hist, run_seed, run_time)
                results.append(packed)
                if self.log:
                    self._write_log(Path(self._resolve_log_dir()) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
                if best_overall is None or best_len < best_overall[1]:
                    best_overall = packed
        finally:
            random.setstate(prev_state)

        assert best_overall is not None
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(best_overall[3], vis_dir / 'convergence_best.png', title=f"GA→ACO-TSP - {self.dir_name} (best)")
        return best_overall, results
