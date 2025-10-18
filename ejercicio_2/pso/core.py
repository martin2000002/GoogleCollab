from __future__ import annotations
import shutil
import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Callable
from shared.utils import fmt_best

import numpy as np

from tqdm import tqdm as _tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shared.plots import save_convergence


@dataclass
class Particle:
    x: np.ndarray
    v: np.ndarray
    pbest_x: np.ndarray
    pbest_f: float


class ParticleSwarm:
    def __init__(
        self,
        func: Callable[[Sequence[float]], float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        swarm_size: int,
        max_epochs: int,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        inertia: float = 1.0,
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
        if swarm_size <= 1:
            raise ValueError("swarm_size must be > 1")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if alpha1 < 0 or alpha2 < 0:
            raise ValueError("alpha1 and alpha2 must be >= 0")
        if inertia < 0:
            raise ValueError("inertia must be >= 0")

        self.func = func
        self.swarm_size = swarm_size
        self.max_epochs = max_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.inertia = inertia
        self.maximize = maximize  # PSO formulated for minimization by default; toggle if needed
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name if dir_name else (log_dir_name if log_dir_name else "pso")
        self.log_dir_name = self.dir_name
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

        # Bounds
        self._lb = np.array(lower_bounds, dtype=float)
        self._ub = np.array(upper_bounds, dtype=float)
        if self._lb.shape != self._ub.shape:
            raise ValueError("lower_bounds and upper_bounds must have same shape")
        if self._lb.size == 0:
            raise ValueError("bounds must be non-empty")

    # --- Internal helpers ---
    def _init_swarm(self) -> List[Particle]:
        lb, ub = self._lb, self._ub

        particles: List[Particle] = []
        for _ in range(self.swarm_size):
            x = np.random.uniform(lb, ub)
            # Initialize small random velocity within 10% of range
            v = np.random.uniform(-0.1 * np.abs(ub - lb), 0.1 * np.abs(ub - lb))
            f = self.func(x)
            particles.append(Particle(x=x, v=v, pbest_x=x.copy(), pbest_f=f))
        return particles

    def _evaluate(self, x: Sequence[float]) -> float:
        return float(self.func(x))

    def _rank_best(self, particles: List[Particle]) -> Tuple[np.ndarray, float]:
        # Determine global best using personal bests
        if not particles:
            raise ValueError("Empty swarm")
        if self.maximize:
            idx = max(range(len(particles)), key=lambda i: particles[i].pbest_f)
        else:
            idx = min(range(len(particles)), key=lambda i: particles[i].pbest_f)
        return particles[idx].pbest_x.copy(), particles[idx].pbest_f

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, self._lb), self._ub)

    def _resolve_log_dir(self) -> str:
        return str(Path('ejercicio_2') / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path('ejercicio_2') / 'visualization' / self.dir_name

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

    # --- Core run body ---
    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[np.ndarray, float, List[float], List[Tuple[np.ndarray, float]]]:
        particles = self._init_swarm()
        gbest_x, gbest_f = self._rank_best(particles)

        history: List[float] = [gbest_f]
        per_epoch_best: List[Tuple[np.ndarray, float]] = [(gbest_x.copy(), gbest_f)]

        pbar = None
        executor: Optional[ThreadPoolExecutor] = None
        start_time = time.perf_counter()
        if self.show_progress:
            pbar = _tqdm(total=self.max_epochs, desc=progress_label or "PSO", leave=False)
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        for epoch in range(self.max_epochs):
            # Random components (can be per-particle and per-dimension)
            phi1 = float(np.random.random())
            phi2 = float(np.random.random())

            # First compute proposed velocities and positions for all particles
            new_vs: List[np.ndarray] = []
            new_xs: List[np.ndarray] = []
            for p in particles:
                cognitive = self.alpha1 * phi1 * (p.pbest_x - p.x)
                social = self.alpha2 * phi2 * (gbest_x - p.x)
                new_v = self.inertia * p.v + cognitive + social
                new_x = p.x + new_v
                new_x = self._clip_to_bounds(new_x)
                new_vs.append(new_v)
                new_xs.append(new_x)

            # Evaluate all new positions (parallel if requested)
            if executor is not None:
                f_news = list(executor.map(self._evaluate, new_xs))
            else:
                f_news = [self._evaluate(nx) for nx in new_xs]

            # Update particles and bests
            for idx, p in enumerate(particles):
                new_v = new_vs[idx]
                new_x = new_xs[idx]
                f_new = f_news[idx]
                if self.maximize:
                    if f_new > p.pbest_f:
                        p.pbest_x = new_x.copy()
                        p.pbest_f = f_new
                    if f_new > gbest_f:
                        gbest_x = new_x.copy()
                        gbest_f = f_new
                else:
                    if f_new < p.pbest_f:
                        p.pbest_x = new_x.copy()
                        p.pbest_f = f_new
                    if f_new < gbest_f:
                        gbest_x = new_x.copy()
                        gbest_f = f_new

                p.v = new_v
                p.x = new_x

            history.append(gbest_f)
            per_epoch_best.append((gbest_x.copy(), gbest_f))

            # Update progress
            if self.show_progress:
                if pbar is not None:
                    pbar.set_postfix({"best": fmt_best(gbest_f)})
                    pbar.update(1)
                else:
                    if (epoch + 1) % self.progress_interval == 0 or (epoch + 1) == self.max_epochs:
                        elapsed = time.perf_counter() - start_time
                        rate = elapsed / (epoch + 1)
                        remaining = rate * (self.max_epochs - (epoch + 1))
                        label = f"[{progress_label}] " if progress_label else ""
                        print(f"{label}gen {epoch+1}/{self.max_epochs} | best={fmt_best(gbest_f)} | eta ~ {remaining:.1f}s")

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return gbest_x, gbest_f, history, per_epoch_best

    # --- Public API ---
    def run(self) -> Tuple[np.ndarray, float, str, List[float], float]:
        _t0 = time.perf_counter()
        best_x, best_f, history, per_epoch = self._run_body(progress_label="run")
        run_time = time.perf_counter() - _t0
        if self.log:
            log_dir = self._resolve_log_dir()
            self._prepare_log_dir(log_dir)
            self._write_log(Path(log_dir) / 'run.txt', per_epoch, seed=None)
        # Save convergence plot
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(history, vis_dir / 'convergence_run.png', title=f"PSO - {self.dir_name}")
        return best_x, best_f, fmt_best(best_f), history, run_time

    def run_multiple(self, runs: int) -> Tuple[Tuple[np.ndarray, float, str, List[float], int, float], List[Tuple[np.ndarray, float, str, List[float], int, float]]]:
        if runs <= 0:
            raise ValueError("runs must be > 0")

        prev_np_state = np.random.get_state()
        prev_py_state = random.getstate()
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
                _t0 = time.perf_counter()
                best_x, best_f, history, per_epoch = self._run_body(progress_label=label)
                run_time = time.perf_counter() - _t0
                results.append((best_x, best_f, fmt_best(best_f), history, run_seed, run_time))
                if self.log:
                    log_dir = self._resolve_log_dir()
                    self._write_log(Path(log_dir) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
                if best_overall is None:
                    best_overall = (best_x, best_f, fmt_best(best_f), history, run_seed, run_time)
                else:
                    if (self.maximize and best_f > best_overall[1]) or (not self.maximize and best_f < best_overall[1]):
                        best_overall = (best_x, best_f, fmt_best(best_f), history, run_seed, run_time)
        finally:
            # restore RNG states
            np.random.set_state(prev_np_state)
            random.setstate(prev_py_state)

        assert best_overall is not None
        # Save best convergence plot
        _best_hist = best_overall[3]
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(_best_hist, vis_dir / 'convergence_best.png', title=f"PSO - {self.dir_name} (best)")
        return best_overall, results
