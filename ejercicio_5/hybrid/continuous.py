from __future__ import annotations
import random
import shutil
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from pathlib import Path
from tqdm import tqdm as _tqdm
from concurrent.futures import ThreadPoolExecutor

from shared.plots import save_convergence
from shared.utils import fmt_best
from ejercicio_1.problems.continuous import BinaryContinuousProblem
from ejercicio_1.ga.core import GeneticAlgorithm
from ejercicio_1.ga.selections.tournament import TournamentSelection


class HybridGAACOR:
    """GA→ACOR for continuous optimization: seed ACOR archive with elite GA solutions."""

    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        # GA params
        ga_bits_per_var: int = 16,
        ga_population_size: int = 200,
        ga_elite_ratio: float = 0.2,
        ga_mutation_prob: float = 0.2,
        ga_max_generations: int = 200,
        ga_k: int = 3,
        ga_runs: int = 3,
        # ACOR params
        archive_size: int = 60,
        samples_per_iter: int = 60,
        max_epochs: int = 150,
        q: float = 0.1,
        xi: float = 0.85,
        maximize: bool = False,
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
            np.random.seed(random_seed)
        self.func = func
        self.lb = np.array(lower_bounds, dtype=float)
        self.ub = np.array(upper_bounds, dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("lower_bounds and upper_bounds must have same shape")
        self.dim = self.lb.size

        self.ga_bits_per_var = ga_bits_per_var
        self.ga_population_size = ga_population_size
        self.ga_elite_ratio = ga_elite_ratio
        self.ga_mutation_prob = ga_mutation_prob
        self.ga_max_generations = ga_max_generations
        self.ga_k = ga_k
        self.ga_runs = ga_runs

        self.archive_size = archive_size
        self.samples_per_iter = samples_per_iter
        self.max_epochs = max_epochs
        self.q = q
        self.xi = xi
        self.maximize = maximize
        self.seed_with_ga = seed_with_ga
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name or "hybrid_acor"
        self.exe_root = exe_root
        self.show_progress = show_progress
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

    # --- helpers ---
    def _evaluate(self, x: Sequence[float]) -> float:
        return float(self.func(x))

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self.lb), self.ub)

    def _ga_elites(self) -> List[np.ndarray]:
        problem = BinaryContinuousProblem(
            func=self.func,
            num_vars=self.dim,
            bits_per_var=self.ga_bits_per_var,
            lower_bounds=self.lb.tolist(),
            upper_bounds=self.ub.tolist(),
            random_seed=self.random_seed,
        )
        ga = GeneticAlgorithm(
            problem=problem,
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
            save_plots=True,
            show_progress=True,
        )
        elites: List[np.ndarray] = []
        _best_overall, results = ga.run_multiple(runs=self.ga_runs)
        for (best_ind, _fit, _fit_str, _hist, _seed, _rt) in results:
            x = np.array(problem._decode(best_ind), dtype=float)
            elites.append(x)
        return elites

    def _init_archive_seeded(self, elites: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # start with elites (unique), then fill with uniform random up to archive_size
        uniq = []
        seen = set()
        for e in elites:
            key = tuple(np.round(e, 10))
            if key not in seen:
                uniq.append(e)
                seen.add(key)
        k = min(len(uniq), self.archive_size)
        X_list = [self._clip(uniq[i]) for i in range(k)]
        while len(X_list) < self.archive_size:
            X_list.append(np.random.uniform(self.lb, self.ub))
        X = np.vstack([x[None, :] for x in X_list])
        F = np.array([self._evaluate(x) for x in X])
        return X, F

    def _sort_archive(self, X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(-F) if self.maximize else np.argsort(F)
        return X[idx], F[idx]

    def _weights(self, n: int) -> np.ndarray:
        # In ACOR, q controls selection pressure. For ablation (no_pheromone) we set q=0.
        # When q <= 0, fall back to uniform weights to avoid division by zero and to remove bias.
        sigma = float(self.q) * float(n)
        if not np.isfinite(sigma) or sigma <= 1e-12:
            return np.full(n, 1.0 / n)
        idx = np.arange(1, n + 1, dtype=float)
        # Gaussian-like weighting centered at the best (rank 1)
        w = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((idx - 1.0) ** 2) / (2.0 * (sigma ** 2)))
        w = np.where(np.isfinite(w), w, 0.0)
        s = w.sum()
        return (w / s) if s > 0 else np.full(n, 1.0 / n)

    def _sigma_dim(self, X: np.ndarray, k: int, d: int) -> float:
        n = X.shape[0]
        diffs = np.abs(X[:, d] - X[k, d])
        return self.xi * (diffs.sum() / (n - 1) + 1e-12)

    def _sample_solution(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n, dim = X.shape
        k = int(np.random.choice(np.arange(n), p=weights))
        x_new = np.empty(dim, dtype=float)
        for d in range(dim):
            sigma_d = self._sigma_dim(X, k, d)
            x_new[d] = np.random.normal(loc=X[k, d], scale=sigma_d)
        return self._clip(x_new)

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

    def _write_log(self, filepath: str, per_epoch_best: List[Tuple[np.ndarray, float]], seed: Optional[int]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            if seed is not None:
                f.write(f"seed={seed}\n")
            for epoch, (x, fit) in enumerate(per_epoch_best):
                pos_str = '[' + ', '.join(f"{float(v):.6f}" for v in x) + ']'
                f.write(f"gen={epoch}\tfitness={fit:.12f}\tposition={pos_str}\n")

    # --- core ---
    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[np.ndarray, float, List[float], List[Tuple[np.ndarray, float]]]:
        if self.seed_with_ga:
            elites = self._ga_elites()
            X, F = self._init_archive_seeded(elites)
        else:
            # Initialize archive purely at random (no GA seeding)
            X = np.random.uniform(self.lb, self.ub, size=(self.archive_size, self.dim))
            F = np.array([self._evaluate(x) for x in X])
        X, F = self._sort_archive(X, F)
        best_x = X[0].copy()
        best_f = float(F[0])
        history: List[float] = [best_f]
        per_epoch_best: List[Tuple[np.ndarray, float]] = [(best_x.copy(), best_f)]

        pbar = _tqdm(total=self.max_epochs, desc=progress_label or "GA→ACOR", leave=False) if self.show_progress else None
        executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=self.parallel_workers) if self.parallel_workers else None

        for epoch in range(self.max_epochs):
            w = self._weights(self.archive_size)
            candidates: List[Tuple[np.ndarray, float]] = []
            if executor is not None:
                def _sample_eval(_):
                    x = self._sample_solution(X, w)
                    return x, self._evaluate(x)
                candidates = list(executor.map(_sample_eval, range(self.samples_per_iter)))
            else:
                for _ in range(self.samples_per_iter):
                    x = self._sample_solution(X, w)
                    candidates.append((x, self._evaluate(x)))

            X_all = np.vstack([X] + [c[0][None, :] for c in candidates])
            F_all = np.concatenate([F] + [np.array([c[1]]) for c in candidates])
            X, F = self._sort_archive(X_all, F_all)
            X = X[: self.archive_size]
            F = F[: self.archive_size]

            if (self.maximize and F[0] > best_f) or (not self.maximize and F[0] < best_f):
                best_x = X[0].copy()
                best_f = float(F[0])

            history.append(best_f)
            per_epoch_best.append((best_x.copy(), best_f))

            if pbar is not None:
                pbar.set_postfix({"best": fmt_best(best_f)})
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return best_x, best_f, history, per_epoch_best

    def run_multiple(self, runs: int) -> Tuple[Tuple[np.ndarray, float, str, List[float], int, float], List[Tuple[np.ndarray, float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")
        prev_np, prev_py = np.random.get_state(), random.getstate()
        results: List[Tuple[np.ndarray, float, str, List[float], int, float]] = []
        best_overall: Optional[Tuple[np.ndarray, float, str, List[float], int, float]] = None
        try:
            if self.log:
                self._prepare_log_dir(self._resolve_log_dir())
            for i in range(runs):
                run_seed = (self.random_seed or 0) + i
                random.seed(run_seed)
                np.random.seed(run_seed)
                t0 = time.perf_counter()
                best_x, best_f, hist, per_epoch = self._run_body(progress_label=f"run {i+1}/{runs}")
                run_time = time.perf_counter() - t0
                packed = (best_x, best_f, fmt_best(best_f), hist, run_seed, run_time)
                results.append(packed)
                if self.log:
                    self._write_log(Path(self._resolve_log_dir()) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
                if best_overall is None:
                    best_overall = packed
                else:
                    if (self.maximize and best_f > best_overall[1]) or (not self.maximize and best_f < best_overall[1]):
                        best_overall = packed
        finally:
            np.random.set_state(prev_np)
            random.setstate(prev_py)

        assert best_overall is not None
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(best_overall[3], vis_dir / 'convergence_best.png', title=f"GA→ACOR - {self.dir_name} (best)")
        return best_overall, results
