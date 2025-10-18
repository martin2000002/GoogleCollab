from __future__ import annotations
import shutil
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import math

# shared fmt
from shared.utils import fmt_best
from tqdm import tqdm as _tqdm
from shared.tsp.functions import euclidean
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from shared.plots import save_convergence


@dataclass
class Ant:
    tour: List[int]
    length: float


class AntSystem:
    def __init__(
        self,
        positions: Sequence[Tuple[float, float]],
        num_ants: int,
        max_epochs: int,
        alpha: float = 1.0,    # pheromone influence
        beta: float = 5.0,     # heuristic influence (1/d)
        rho: float = 0.5,      # evaporation rate
        q: float = 1.0,        # pheromone deposit constant
        random_seed: Optional[int] = None,
        log: bool = False,
        log_dir_name: Optional[str] = None,
        dir_name: Optional[str] = None,
        show_progress: bool = False,
        progress_interval: int = 10,
        parallel_workers: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)
        if num_ants < 1:
            raise ValueError("num_ants must be >= 1")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if rho < 0 or rho >= 1:
            raise ValueError("rho must be in [0,1)")
        self.pos = list(positions)
        self.n = len(self.pos)
        if self.n < 2:
            raise ValueError("positions must contain at least 2 nodes")

        self.num_ants = num_ants
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name if dir_name else (log_dir_name if log_dir_name else "aco")
        self.log_dir_name = self.dir_name
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

        # Distances and heuristic
        self.dist: List[List[float]] = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = euclidean(self.pos[i], self.pos[j])
                self.dist[i][j] = d
                self.dist[j][i] = d
        self.eta: List[List[float]] = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.eta[i][j] = 1.0 / (self.dist[i][j] + 1e-12)

        # Initialize pheromone
        tau0 = 1.0
        self.tau: List[List[float]] = [[tau0] * self.n for _ in range(self.n)]

    def _tour_length(self, tour: Sequence[int]) -> float:
        total = 0.0
        for i in range(self.n - 1):
            total += self.dist[tour[i]][tour[i + 1]]
        total += self.dist[tour[-1]][tour[0]]
        return total

    def _construct_tour(self, start: int) -> Ant:
        unvisited = set(range(self.n))
        tour = [start]
        unvisited.remove(start)
        current = start
        while unvisited:
            # transition probabilities
            denom = 0.0
            probs: List[Tuple[int, float]] = []
            for j in unvisited:
                val = (self.tau[current][j] ** self.alpha) * (self.eta[current][j] ** self.beta)
                probs.append((j, val))
                denom += val
            # roulette wheel
            r = random.random() * denom if denom > 0 else 0.0
            acc = 0.0
            next_city = None
            for j, val in probs:
                acc += val
                if acc >= r:
                    next_city = j
                    break
            if next_city is None:
                # fallback (uniform)
                next_city = random.choice(list(unvisited))
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        L = self._tour_length(tour)
        return Ant(tour=tour, length=L)

    def _resolve_log_dir(self) -> str:
        return str(Path('ejercicio_3') / 'log' / self.dir_name)

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

    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[List[int], float, List[float], List[Tuple[List[int], float]]]:
        # initial best (optional: nearest neighbor heuristic); here use random tour
        initial = list(range(self.n))
        random.shuffle(initial)
        gbest_tour = initial
        gbest_len = self._tour_length(gbest_tour)

        best_history: List[float] = [gbest_len]
        per_epoch_best: List[Tuple[List[int], float]] = [(gbest_tour[:], gbest_len)]

        pbar = None
        executor: Optional[ThreadPoolExecutor] = None
        start_time = time.perf_counter()
        if self.show_progress:
            pbar = _tqdm(total=self.max_epochs, desc=progress_label or "ACO-AS", leave=False)
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        for epoch in range(self.max_epochs):
            # Construct solutions
            # random starting positions for ants
            starts = [random.randrange(self.n) for _ in range(self.num_ants)]
            # Construct solutions (optionally in parallel)
            if executor is not None:
                ants = list(executor.map(self._construct_tour, starts))  # type: ignore[arg-type]
            else:
                ants = [self._construct_tour(s) for s in starts]

            # Find best this epoch
            ants.sort(key=lambda a: a.length)
            if ants[0].length < gbest_len:
                gbest_tour, gbest_len = ants[0].tour[:], ants[0].length

            # Evaporate
            for i in range(self.n):
                for j in range(self.n):
                    self.tau[i][j] *= (1.0 - self.rho)

            # Deposit pheromone (Ant System: all ants)
            for ant in ants:
                d_tau = self.q / (ant.length + 1e-12)
                tour = ant.tour
                for k in range(self.n):
                    i, j = tour[k], tour[(k + 1) % self.n]
                    self.tau[i][j] += d_tau
                    self.tau[j][i] += d_tau

            best_history.append(gbest_len)
            per_epoch_best.append((gbest_tour[:], gbest_len))

            # Progress
            if self.show_progress:
                if pbar is not None:
                    pbar.set_postfix({"best": fmt_best(gbest_len)})
                    pbar.update(1)
                else:
                    if (epoch + 1) % self.progress_interval == 0 or (epoch + 1) == self.max_epochs:
                        elapsed = time.perf_counter() - start_time
                        rate = elapsed / (epoch + 1)
                        remaining = rate * (self.max_epochs - (epoch + 1))
                        label = f"[{progress_label}] " if progress_label else ""
                        print(f"{label}gen {epoch+1}/{self.max_epochs} | best={fmt_best(gbest_len)} | eta ~ {remaining:.1f}s")

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return gbest_tour, gbest_len, best_history, per_epoch_best

    def run(self) -> Tuple[List[int], float, str, List[float], float]:
        _t0 = time.perf_counter()
        best_tour, best_len, best_hist, per_epoch = self._run_body(progress_label="run")
        run_time = time.perf_counter() - _t0
        if self.log:
            log_dir = self._resolve_log_dir()
            self._prepare_log_dir(log_dir)
            self._write_log(Path(log_dir) / 'run.txt', per_epoch, seed=None)
        # Save convergence plot
        vis_dir = Path('ejercicio_3') / 'visualization' / self.dir_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(best_hist, vis_dir / 'convergence_run.png', title=f"ACO-AS - {self.dir_name}")
        return best_tour, best_len, fmt_best(best_len), best_hist, run_time

    def run_multiple(self, runs: int) -> Tuple[Tuple[List[int], float, str, List[float], int, float], List[Tuple[List[int], float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")

        prev_state = random.getstate()
        results: List[Tuple[List[int], float, str, List[float], int, float]] = []
        best_overall: Optional[Tuple[List[int], float, str, List[float], int, float]] = None
        try:
            if self.log:
                batch_dir = self._resolve_log_dir()
                self._prepare_log_dir(batch_dir)
            for i in range(runs):
                run_seed = (self.random_seed or 0) + i
                random.seed(run_seed)
                label = f"run {i+1}/{runs}"
                _t0 = time.perf_counter()
                best_tour, best_len, best_hist, per_epoch = self._run_body(progress_label=label)
                run_time = time.perf_counter() - _t0
                results.append((best_tour, best_len, fmt_best(best_len), best_hist, run_seed, run_time))
                if self.log:
                    log_dir = self._resolve_log_dir()
                    self._write_log(Path(log_dir) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
                if best_overall is None:
                    best_overall = (best_tour, best_len, fmt_best(best_len), best_hist, run_seed, run_time)
                else:
                    if best_len < best_overall[1]:
                        best_overall = (best_tour, best_len, fmt_best(best_len), best_hist, run_seed, run_time)
        finally:
            random.setstate(prev_state)

        assert best_overall is not None
        # Save best convergence plot
        _best_hist = best_overall[3]
        vis_dir = Path('ejercicio_3') / 'visualization' / self.dir_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(_best_hist, vis_dir / 'convergence_best.png', title=f"ACO-AS - {self.dir_name} (best)")
        return best_overall, results
