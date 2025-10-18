from __future__ import annotations
import shutil
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import math
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm as _tqdm

from shared.plots import save_convergence
from shared.utils import fmt_best


class ACOR:
    """Ant Colony Optimization for Continuous Domains (ACOR).

    Implementation inspired by Socha & Dorigo (2008). Minimization by default.
    """

    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        archive_size: int,
        samples_per_iter: int,
        max_epochs: int,
        q: float = 0.1,
        xi: float = 0.85,
        maximize: bool = False,
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
            np.random.seed(random_seed)
        if archive_size < 2:
            raise ValueError("archive_size must be >= 2")
        if samples_per_iter < 1:
            raise ValueError("samples_per_iter must be >= 1")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if not (0.0 < q <= 1.0):
            raise ValueError("q must be in (0,1]")
        if not (0.0 < xi):
            raise ValueError("xi must be > 0")

        self.func = func
        self._lb = np.array(lower_bounds, dtype=float)
        self._ub = np.array(upper_bounds, dtype=float)
        if self._lb.shape != self._ub.shape:
            raise ValueError("lower_bounds and upper_bounds must have same shape")
        self.dim = self._lb.size

        self.archive_size = archive_size
        self.samples_per_iter = samples_per_iter
        self.max_epochs = max_epochs
        self.q = q
        self.xi = xi
        self.maximize = maximize
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name if dir_name else (log_dir_name if log_dir_name else "acor")
        self.log_dir_name = self.dir_name
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

    # --- helpers ---
    def _evaluate(self, x: Sequence[float]) -> float:
        return float(self.func(x))

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self._lb), self._ub)

    def _init_archive(self) -> Tuple[np.ndarray, np.ndarray]:
        # random uniform initialization
        X = np.random.uniform(self._lb, self._ub, size=(self.archive_size, self.dim))
        F = np.array([self._evaluate(x) for x in X])
        return X, F

    def _sort_archive(self, X: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.maximize:
            idx = np.argsort(-F)
        else:
            idx = np.argsort(F)
        return X[idx], F[idx]

    def _weights(self, n: int) -> np.ndarray:
        # Gaussian-like weights based on position in sorted archive
        # sigma = q * n
        sigma = self.q * n
        idx = np.arange(1, n + 1)
        w = (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-((idx - 1) ** 2) / (2 * (sigma ** 2)))
        w_sum = w.sum()
        return w / w_sum if w_sum > 0 else np.full(n, 1.0 / n)

    def _sigma_dim(self, X: np.ndarray, k: int, d: int) -> float:
        # Standard deviation for dimension d around solution k
        n = X.shape[0]
        diffs = np.abs(X[:, d] - X[k, d])
        return self.xi * (diffs.sum() / (n - 1) + 1e-12)

    def _sample_solution(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n, dim = X.shape
        # choose center index according to weights
        k = int(np.random.choice(np.arange(n), p=weights))
        x_new = np.empty(dim, dtype=float)
        for d in range(dim):
            sigma_d = self._sigma_dim(X, k, d)
            x_new[d] = np.random.normal(loc=X[k, d], scale=sigma_d)
        return self._clip(x_new)

    def _resolve_log_dir(self) -> str:
        return str(Path('ejercicio_5') / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path('ejercicio_5') / 'visualization' / self.dir_name

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
        X, F = self._init_archive()
        X, F = self._sort_archive(X, F)
        best_x = X[0].copy()
        best_f = float(F[0])

        history: List[float] = [best_f]
        per_epoch_best: List[Tuple[np.ndarray, float]] = [(best_x.copy(), best_f)]

        pbar = None
        executor: Optional[ThreadPoolExecutor] = None
        if self.show_progress:
            pbar = _tqdm(total=self.max_epochs, desc=progress_label or "ACOR", leave=False)
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        for epoch in range(self.max_epochs):
            # compute weights
            w = self._weights(self.archive_size)

            # generate samples
            candidates = []
            if executor is not None:
                # parallel sampling and evaluation
                def _sample_and_eval(_):
                    x = self._sample_solution(X, w)
                    return x, self._evaluate(x)
                out = list(executor.map(_sample_and_eval, range(self.samples_per_iter)))
                candidates.extend(out)
            else:
                for _ in range(self.samples_per_iter):
                    x = self._sample_solution(X, w)
                    fx = self._evaluate(x)
                    candidates.append((x, fx))

            # merge and truncate to archive size
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

            if self.show_progress and pbar is not None:
                pbar.set_postfix({"best": fmt_best(best_f)})
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return best_x, best_f, history, per_epoch_best

    def run(self) -> Tuple[np.ndarray, float, str, List[float], float]:
        t0 = time.perf_counter()
        best_x, best_f, history, per_epoch = self._run_body(progress_label="run")
        run_time = time.perf_counter() - t0
        if self.log:
            log_dir = self._resolve_log_dir()
            self._prepare_log_dir(log_dir)
            self._write_log(Path(log_dir) / 'run.txt', per_epoch, seed=None)
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(history, vis_dir / 'convergence_run.png', title=f"ACOR - {self.dir_name}")
        return best_x, best_f, fmt_best(best_f), history, run_time

    def run_multiple(self, runs: int) -> Tuple[Tuple[np.ndarray, float, str, List[float], int, float], List[Tuple[np.ndarray, float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")
        prev_np = np.random.get_state()
        prev_py = random.getstate()
        results: List[Tuple[np.ndarray, float, str, List[float], int, float]] = []
        best_overall: Optional[Tuple[np.ndarray, float, str, List[float], int, float]] = None
        try:
            if self.log:
                batch_dir = self._resolve_log_dir()
                self._prepare_log_dir(batch_dir)
            for i in range(runs):
                run_seed = (self.random_seed or 0) + i
                random.seed(run_seed)
                np.random.seed(run_seed)
                label = f"run {i+1}/{runs}"
                t0 = time.perf_counter()
                best_x, best_f, history, per_epoch = self._run_body(progress_label=label)
                run_time = time.perf_counter() - t0
                packed = (best_x, best_f, fmt_best(best_f), history, run_seed, run_time)
                results.append(packed)
                if self.log:
                    log_dir = self._resolve_log_dir()
                    self._write_log(Path(log_dir) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
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
        save_convergence(best_overall[3], vis_dir / 'convergence_best.png', title=f"ACOR - {self.dir_name} (best)")
        return best_overall, results
