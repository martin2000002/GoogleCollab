from __future__ import annotations
import shutil
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm as _tqdm

from shared.tsp.functions import euclidean
from shared.plots import save_convergence
from shared.utils import fmt_best


Swap = Tuple[int, int]


def _tour_length(tour: Sequence[int], dist: List[List[float]]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


def _swap_sequence(from_perm: List[int], to_perm: Sequence[int]) -> List[Swap]:
    """Return a sequence of swaps that transforms from_perm into to_perm.

    This produces a simple O(n) sequence by placing each target element in order.
    The input list from_perm will be modified (copy before calling if needed).
    """
    n = len(from_perm)
    pos = {val: i for i, val in enumerate(from_perm)}
    swaps: List[Swap] = []
    for i in range(n):
        desired = to_perm[i]
        j = pos[desired]
        if j != i:
            # swap positions i and j in from_perm
            ai, aj = from_perm[i], from_perm[j]
            from_perm[i], from_perm[j] = from_perm[j], from_perm[i]
            pos[ai], pos[aj] = j, i
            swaps.append((i, j))
    return swaps


def _apply_swaps(perm: List[int], velocity: Sequence[Swap]) -> List[int]:
    p = list(perm)
    for i, j in velocity:
        p[i], p[j] = p[j], p[i]
    return p


def _scale_swaps(swaps: Sequence[Swap], factor: float) -> List[Swap]:
    if factor <= 0 or not swaps:
        return []
    k = max(1, int(round(factor * len(swaps))))
    k = min(k, len(swaps))
    return list(swaps[:k])


def _truncate_velocity(velocity: List[Swap], vmax: int) -> List[Swap]:
    if vmax <= 0:
        return []
    if len(velocity) > vmax:
        return velocity[:vmax]
    return velocity


@dataclass
class Particle:
    perm: List[int]
    v: List[Swap]
    pbest_perm: List[int]
    pbest_len: float


class SwapSequencePSO:
    """PSO adapted to TSP using swap-sequence representation for velocity.

    Notes:
    - Minimization of tour length.
    - Velocity is a list of swaps; updates concatenate scaled components.
    - To control explosion, truncate velocity length by vmax_frac * n.
    """

    def __init__(
        self,
        positions: Sequence[Tuple[float, float]],
        swarm_size: int,
        max_epochs: int,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        inertia: float = 0.8,
        vmax_frac: float = 0.7,
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
        if swarm_size <= 1:
            raise ValueError("swarm_size must be > 1")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if alpha1 < 0 or alpha2 < 0 or inertia < 0:
            raise ValueError("alpha1, alpha2, inertia must be >= 0")
        if not (0.0 < vmax_frac <= 1.0):
            raise ValueError("vmax_frac must be in (0,1]")

        self.pos = list(positions)
        self.n = len(self.pos)
        if self.n < 2:
            raise ValueError("positions must have at least 2 nodes")

        self.swarm_size = swarm_size
        self.max_epochs = max_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.inertia = inertia
        self.vmax_frac = vmax_frac
        self.random_seed = random_seed
        self.log = log
        self.dir_name = dir_name if dir_name else (log_dir_name if log_dir_name else "pso_tsp")
        self.log_dir_name = self.dir_name
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

        # Distances
        self.dist: List[List[float]] = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = euclidean(self.pos[i], self.pos[j])
                self.dist[i][j] = d
                self.dist[j][i] = d

    def _init_swarm(self) -> List[Particle]:
        particles: List[Particle] = []
        base = list(range(self.n))
        for _ in range(self.swarm_size):
            perm = base[:]
            random.shuffle(perm)
            v: List[Swap] = []
            L = _tour_length(perm, self.dist)
            particles.append(Particle(perm=perm, v=v, pbest_perm=perm[:], pbest_len=L))
        return particles

    def _rank_best(self, particles: List[Particle]) -> Tuple[List[int], float]:
        idx = min(range(len(particles)), key=lambda i: particles[i].pbest_len)
        p = particles[idx]
        return p.pbest_perm[:], p.pbest_len

    def _resolve_log_dir(self) -> str:
        return str(Path('ejercicio_5') / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path('ejercicio_5') / 'visualization' / self.dir_name

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
                t_str = '[' + ', '.join(str(x) for x in tour) + ']'
                f.write(f"gen={epoch}\tfitness={fit:.12f}\ttour={t_str}\n")

    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[List[int], float, List[float], List[Tuple[List[int], float]]]:
        particles = self._init_swarm()
        gbest_perm, gbest_len = self._rank_best(particles)

        history: List[float] = [gbest_len]
        per_epoch_best: List[Tuple[List[int], float]] = [(gbest_perm[:], gbest_len)]

        pbar = None
        executor: Optional[ThreadPoolExecutor] = None
        if self.show_progress:
            pbar = _tqdm(total=self.max_epochs, desc=progress_label or "PSO-TSP", leave=False)
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        vmax = max(1, int(self.vmax_frac * self.n))

        for epoch in range(self.max_epochs):
            r1 = random.random()
            r2 = random.random()

            new_perms: List[List[int]] = []
            new_vs: List[List[Swap]] = []

            for p in particles:
                # current velocity (inertia)
                v_inertia = _scale_swaps(p.v, self.inertia)

                # cognitive component: towards personal best
                # compute swap sequence from current to pbest
                s_cog = _swap_sequence(p.perm[:], p.pbest_perm)
                v_cog = _scale_swaps(s_cog, self.alpha1 * r1)

                # social component: towards global best
                s_soc = _swap_sequence(p.perm[:], gbest_perm)
                v_soc = _scale_swaps(s_soc, self.alpha2 * r2)

                v_new = v_inertia + v_cog + v_soc
                v_new = _truncate_velocity(v_new, vmax)

                x_new = _apply_swaps(p.perm, v_new)
                new_vs.append(v_new)
                new_perms.append(x_new)

            # Evaluate new perms (optionally parallel)
            if executor is not None:
                lengths = list(executor.map(lambda tour: _tour_length(tour, self.dist), new_perms))
            else:
                lengths = [_tour_length(t, self.dist) for t in new_perms]

            # Update particles
            for idx, p in enumerate(particles):
                x_new = new_perms[idx]
                f_new = lengths[idx]
                if f_new < p.pbest_len:
                    p.pbest_perm = x_new[:]
                    p.pbest_len = f_new
                if f_new < gbest_len:
                    gbest_perm = x_new[:]
                    gbest_len = f_new
                p.perm = x_new
                p.v = new_vs[idx]

            history.append(gbest_len)
            per_epoch_best.append((gbest_perm[:], gbest_len))

            if self.show_progress and pbar is not None:
                pbar.set_postfix({"best": fmt_best(gbest_len)})
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        if executor is not None:
            executor.shutdown(wait=True)

        return gbest_perm, gbest_len, history, per_epoch_best

    def run(self) -> Tuple[List[int], float, str, List[float], float]:
        t0 = time.perf_counter()
        best_tour, best_len, history, per_epoch = self._run_body(progress_label="run")
        run_time = time.perf_counter() - t0
        if self.log:
            log_dir = self._resolve_log_dir()
            self._prepare_log_dir(log_dir)
            self._write_log(Path(log_dir) / 'run.txt', per_epoch, seed=None)
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(history, vis_dir / 'convergence_run.png', title=f"PSO-TSP - {self.dir_name}")
        return best_tour, best_len, fmt_best(best_len), history, run_time

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
                t0 = time.perf_counter()
                best_tour, best_len, hist, per_epoch = self._run_body(progress_label=label)
                run_time = time.perf_counter() - t0
                packed = (best_tour, best_len, fmt_best(best_len), hist, run_seed, run_time)
                results.append(packed)
                if self.log:
                    log_dir = self._resolve_log_dir()
                    self._write_log(Path(log_dir) / f"run_{i+1}.txt", per_epoch, seed=run_seed)
                if best_overall is None or best_len < best_overall[1]:
                    best_overall = packed
        finally:
            random.setstate(prev_state)

        assert best_overall is not None
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(best_overall[3], vis_dir / 'convergence_best.png', title=f"PSO-TSP - {self.dir_name} (best)")
        return best_overall, results
