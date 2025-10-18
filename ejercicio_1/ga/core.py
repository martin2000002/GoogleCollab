import shutil
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Sequence, Tuple, Optional
from shared.utils import fmt_best
from ga.interfaces import Problem, SelectionStrategy
from tqdm import tqdm as _tqdm
from pathlib import Path
from shared.plots import save_convergence

class GeneticAlgorithm:
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        mutation_prob: float,
        elite_ratio: float,
        max_generations: int,
        selection: SelectionStrategy,
        maximize: bool = True,
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
        if not (0.0 <= elite_ratio <= 1.0):
            raise ValueError("elite_ratio must be in [0.0, 1.0]")
        if not (0.0 <= mutation_prob <= 1.0):
            raise ValueError("mutation_prob must be in [0,1]")
        if max_generations <= 0:
            raise ValueError("max_generations must be > 0")
        if population_size <= 1:
            raise ValueError("population_size must be > 1")
        if not isinstance(problem, Problem):
            raise TypeError("problem must implement Problem interface")
        if not isinstance(selection, SelectionStrategy):
            raise TypeError("selection must implement SelectionStrategy interface")

        self.problem = problem
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elite_ratio = elite_ratio
        self.elite_count = int(round(self.elite_ratio * self.population_size))
        self.max_generations = max_generations
        self.selection = selection
        self.maximize = maximize
        self.random_seed = random_seed
        self.log = log
        # Unified directory name for logs and visualization
        self.dir_name = dir_name if dir_name else (log_dir_name if log_dir_name else type(self.problem).__name__)
        self.log_dir_name = self.dir_name
        self.show_progress = show_progress
        self.progress_interval = max(1, progress_interval)
        self.parallel_workers = parallel_workers if (parallel_workers or 0) > 1 else None

    def _evaluate(self, population: Sequence[Any]) -> List[float]:
        return [self.problem.fitness(ind) for ind in population]

    def _rank(self, population: Sequence[Any], fitness: Sequence[float]) -> List[Tuple[Any, float]]:
        return sorted(
            zip(population, fitness), key=lambda t: t[1], reverse=self.maximize
        )

    def _run_body(self, progress_label: Optional[str] = None) -> Tuple[Any, float, List[float], List[Tuple[Any, float]]]:
        population = self.problem.initialize_population(self.population_size)

        executor: Optional[ThreadPoolExecutor] = None
        if self.parallel_workers:
            executor = ThreadPoolExecutor(max_workers=self.parallel_workers)

        def _evaluate(pop: Sequence[Any]) -> List[float]:
            if executor is not None:
                return list(executor.map(self.problem.fitness, pop))
            return [self.problem.fitness(ind) for ind in pop]

        fitness = _evaluate(population)
        ranked = self._rank(population, fitness)
        best_ind, best_fit = ranked[0]
        best_history: List[float] = [best_fit]
        per_gen_best: List[Tuple[Any, float]] = [(best_ind, best_fit)]

        # Progress
        pbar = None
        start_time = time.perf_counter()
        if self.show_progress:
            pbar = _tqdm(total=self.max_generations, desc=progress_label or "GA", leave=False)

        for gen in range(self.max_generations):
            ranked = self._rank(population, fitness)
            elites = [ind for ind, _ in ranked[: self.elite_count]]

            needed = self.population_size - self.elite_count
            offspring: List[Any] = []
            while len(offspring) < needed:
                p1 = self.selection.select(ranked, self.maximize)
                p2 = self.selection.select(ranked, self.maximize)
                c1, c2 = self.problem.crossover(p1, p2)

                c1 = self.problem.mutate(c1, self.mutation_prob)
                if len(offspring) + 1 < needed:
                    c2 = self.problem.mutate(c2, self.mutation_prob)
                    offspring.extend([c1, c2])
                else:
                    offspring.append(c1)

            population = elites + offspring
            fitness = _evaluate(population)
            ranked = self._rank(population, fitness)
            if self.maximize:
                if ranked[0][1] > best_fit:
                    best_ind, best_fit = ranked[0]
            else:
                if ranked[0][1] < best_fit:
                    best_ind, best_fit = ranked[0]
            best_history.append(best_fit)
            per_gen_best.append((ranked[0][0], ranked[0][1]))

            # Update progress
            if self.show_progress:
                if pbar is not None:
                    pbar.set_postfix({"best": fmt_best(best_fit)})
                    pbar.update(1)
                else:
                    if (gen + 1) % self.progress_interval == 0 or (gen + 1) == self.max_generations:
                        elapsed = time.perf_counter() - start_time
                        rate = elapsed / (gen + 1)
                        remaining = rate * (self.max_generations - (gen + 1))
                        label = f"[{progress_label}] " if progress_label else ""
                        print(f"{label}gen {gen+1}/{self.max_generations} | best={fmt_best(best_fit)} | eta ~ {remaining:.1f}s")

        if pbar is not None:
            pbar.close()

        if executor is not None:
            executor.shutdown(wait=True)

        return best_ind, best_fit, best_history, per_gen_best

    def _write_log(self, filepath: str | Path, per_gen_best: List[Tuple[Any, float]], seed: Optional[int]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            if seed is not None:
                f.write(f"seed={seed}\n")
            for gen_idx, (ind, fit) in enumerate(per_gen_best):
                chrom_str, decoded_str = self.problem.stringify(ind)
                f.write(
                    f"gen={gen_idx}\tfitness={fit:.12f}\tchromosome={chrom_str}\tdecoded={decoded_str}\n"
                )

    def _resolve_log_dir(self) -> str:
        return str(Path('ejercicio_1') / 'log' / self.dir_name)

    def _resolve_vis_dir(self) -> Path:
        return Path('ejercicio_1') / 'visualization' / self.dir_name

    def _prepare_log_dir(self, dir_path: str | Path) -> None:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    def run(self) -> Tuple[Any, float, str, List[float], float]:
        _t0 = time.perf_counter()
        best_ind, best_fit, best_hist, per_gen = self._run_body(progress_label="run")
        run_time = time.perf_counter() - _t0
        if self.log:
            log_dir = self._resolve_log_dir()
            self._prepare_log_dir(log_dir)
            log_path = Path(log_dir) / 'run.txt'
            self._write_log(log_path, per_gen, seed=None)
        # Save convergence plot
        vis_dir = self._resolve_vis_dir()
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_convergence(best_hist, vis_dir / 'convergence_run.png', title=f"GA - {self.dir_name}")
        return best_ind, best_fit, fmt_best(best_fit), best_hist, run_time

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
                run_seed = self.random_seed + i
                random.seed(run_seed)
                label = f"run {i+1}/{runs}"
                _t0 = time.perf_counter()
                best, fit, hist, per_gen = self._run_body(progress_label=label)
                run_time = time.perf_counter() - _t0
                results.append((best, fit, fmt_best(fit), hist, run_seed, run_time))
                if self.log:
                    log_dir = self._resolve_log_dir()
                    log_path = Path(log_dir) / f'run_{i+1}.txt'
                    self._write_log(log_path, per_gen, seed=run_seed)
                if best_overall is None:
                    best_overall = (best, fit, fmt_best(fit), hist, run_seed, run_time)
                else:
                    if self.maximize:
                        if fit > best_overall[1]:
                            best_overall = (best, fit, fmt_best(fit), hist, run_seed, run_time)
                    else:
                        if fit < best_overall[1]:
                            best_overall = (best, fit, fmt_best(fit), hist, run_seed, run_time)
        finally:
            random.setstate(prev_state)

        # Save best convergence plot from multiple runs
        if best_overall is not None:
            _best_hist = best_overall[3]
            vis_dir = self._resolve_vis_dir()
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_convergence(_best_hist, vis_dir / 'convergence_best.png', title=f"GA - {self.dir_name} (best)")
        return best_overall, results