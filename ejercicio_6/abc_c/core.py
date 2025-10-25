from __future__ import annotations

import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .interfaces import ABCProblem
from shared.plots import save_convergence
from shared.utils import fmt_best
from tqdm import tqdm as _tqdm


class ArtificialBeeColony:
    def __init__(
        self,
        problem: ABCProblem,
        colony_size: int,
        max_epochs: int,
        limit: int,
        maximize: bool = False,
        random_seed: Optional[int] = None,
        log: bool = False,
        dir_name: Optional[str] = None,
        exe_root: str = "ejercicio_6",
        save_plots: bool = True,
        show_progress: bool = False,
        progress_interval: int = 10,
        parallel_workers: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)
        if colony_size <= 1:
            raise ValueError("colony_size must be > 1")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if not isinstance(problem, ABCProblem):
            raise TypeError("problem must implement ABCProblem interface")

        self.problem = problem
        self.colony_size = colony_size
        self.max_epochs = max_epochs
        self.limit = limit
        self.maximize = maximize
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name if dir_name else type(self.problem).__name__
        self.exe_root = exe_root
        self.save_plots = save_plots
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

    # Directories
    def _resolve_log_dir(self) -> str:
        return str(Path(self.exe_root) / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path(self.exe_root) / 'visualization' / self.dir_name

    def _prepare_log_dir(self, dir_path: str | Path) -> None:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # Utilities
    def _evaluate(self, population: Sequence[Any], executor: ThreadPoolExecutor | None) -> List[float]:
        if executor is not None:
            return list(executor.map(self.problem.fitness, population))
        return [self.problem.fitness(ind) for ind in population]

    def _better(self, f_new: float, f_old: float) -> bool:
        return (f_new > f_old) if self.maximize else (f_new < f_old)

    def _quality(self, fitness: Sequence[float]) -> List[float]:
        # Convert objective values to selection weights (higher is better)
        eps = 1e-12
        if self.maximize:
            m = min(fitness)
            return [max(eps, f - m + eps) for f in fitness]
        else:
            m = min(fitness)
            # shift so min is small but positive denominator
            return [1.0 / (1.0 + (f - m)) for f in fitness]

    def _roulette_index(self, weights: Sequence[float]) -> int:
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                return i
        return len(weights) - 1

    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[Any, float, List[float], List[Tuple[Any, float]]]:
        # Initialize food sources (employed bees = colony_size)
        population = self.problem.initialize_population(self.colony_size)
        executor: Optional[ThreadPoolExecutor] = None
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        fitness = self._evaluate(population, executor)
        trials = [0] * self.colony_size

        # Best tracking
        if self.maximize:
            best_idx = max(range(self.colony_size), key=lambda i: fitness[i])
        else:
            best_idx = min(range(self.colony_size), key=lambda i: fitness[i])
        best = population[best_idx]
        best_fit = fitness[best_idx]
        best_hist: List[float] = [best_fit]
        per_epoch_best: List[Tuple[Any, float]] = [(best, best_fit)]

        # Progress bar
        pbar = None
        start_time = time.perf_counter()
        if self.show_progress:
            pbar = _tqdm(total=self.max_epochs, desc=progress_label or "ABC", leave=False)

        for epoch in range(self.max_epochs):
            # Employed bee phase
            for i in range(self.colony_size):
                k = i
                while k == i:
                    k = random.randrange(self.colony_size)
                candidate = self.problem.generate_candidate(population[i], population[k])
                f_cand = self.problem.fitness(candidate)
                if self._better(f_cand, fitness[i]):
                    population[i] = candidate
                    fitness[i] = f_cand
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bee phase (roulette selection)
            weights = self._quality(fitness)
            onlookers = self.colony_size
            for _ in range(onlookers):
                i = self._roulette_index(weights)
                k = i
                while k == i:
                    k = random.randrange(self.colony_size)
                candidate = self.problem.generate_candidate(population[i], population[k])
                f_cand = self.problem.fitness(candidate)
                if self._better(f_cand, fitness[i]):
                    population[i] = candidate
                    fitness[i] = f_cand
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout phase
            for i in range(self.colony_size):
                if trials[i] >= self.limit:
                    population[i] = self.problem.initialize_population(1)[0]
                    fitness[i] = self.problem.fitness(population[i])
                    trials[i] = 0

            # Update best
            if self.maximize:
                idx = max(range(self.colony_size), key=lambda j: fitness[j])
                if fitness[idx] > best_fit:
                    best, best_fit = population[idx], fitness[idx]
            else:
                idx = min(range(self.colony_size), key=lambda j: fitness[j])
                if fitness[idx] < best_fit:
                    best, best_fit = population[idx], fitness[idx]
            best_hist.append(best_fit)
            per_epoch_best.append((best, best_fit))

            # Update progress
            if self.show_progress:
                if pbar is not None:
                    pbar.set_postfix({"best": fmt_best(best_fit)})
                    pbar.update(1)
                else:
                    if (epoch + 1) % self.progress_interval == 0 or (epoch + 1) == self.max_epochs:
                        elapsed = time.perf_counter() - start_time
                        rate = elapsed / (epoch + 1)
                        remaining = rate * (self.max_epochs - (epoch + 1))
                        label = f"[{progress_label}] " if progress_label else ""
                        print(f"{label}epoch {epoch+1}/{self.max_epochs} | best={fmt_best(best_fit)} | eta ~ {remaining:.1f}s")

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return best, best_fit, best_hist, per_epoch_best

    def _write_log(self, filepath: str | Path, per_epoch_best: List[Tuple[Any, float]], seed: Optional[int]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            if seed is not None:
                f.write(f"seed={seed}\n")
            for ep_idx, (sol, fit) in enumerate(per_epoch_best):
                rep = self.problem.stringify(sol)
                # Write a compact, problem-aware representation (e.g., position=[...] or tour=[...])
                f.write(f"gen={ep_idx}\tfitness={fit:.12f}\t{rep}\n")

    def run_multiple(self, runs: int) -> Tuple[Tuple[Any, float, str, List[float], int, float], List[Tuple[Any, float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")

        prev_state = random.getstate()
        results: List[Tuple[Any, float, str, List[float], int, float]] = []
        best_overall: Optional[Tuple[Any, float, str, List[float], int, float]] = None
        try:
            if self.log:
                batch_dir = self._resolve_log_dir()
                self._prepare_log_dir(batch_dir)
            for i in range(runs):
                run_seed = (self.random_seed or 0) + i
                random.seed(run_seed)
                label = f"run {i+1}/{runs}"
                _t0 = time.perf_counter()
                best, fit, hist, per_epoch = self._run_body(progress_label=label)
                run_time = time.perf_counter() - _t0
                results.append((best, fit, fmt_best(fit), hist, run_seed, run_time))
                if self.log:
                    log_dir = self._resolve_log_dir()
                    log_path = Path(log_dir) / f'run_{i+1}.txt'
                    self._write_log(log_path, per_epoch, seed=run_seed)
                if best_overall is None:
                    best_overall = (best, fit, fmt_best(fit), hist, run_seed, run_time)
                else:
                    if self._better(fit, best_overall[1]):
                        best_overall = (best, fit, fmt_best(fit), hist, run_seed, run_time)
        finally:
            random.setstate(prev_state)

        if best_overall is not None and self.save_plots:
            _best_hist = best_overall[3]
            vis_dir = self._resolve_vis_dir()
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_convergence(_best_hist, vis_dir / 'convergence_best.png', title=f"ABC - {self.dir_name} (best)")
        return best_overall, results
