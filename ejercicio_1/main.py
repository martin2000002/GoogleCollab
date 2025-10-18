from __future__ import annotations
from pathlib import Path
import sys
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
    from typing import List, Tuple
from ga.core import GeneticAlgorithm
from ga.selections.tournament import TournamentSelection
from problems.continuous import BinaryContinuousProblem, beale_function, easom_function
from problems.tsp import TSPProblem
from shared.tsp.functions import make_grid_positions, make_random_positions
from shared.tsp.generator import export_tsp_graph
from shared.results import append_results, compute_summary, format_summary_block
from shared.utils import fmt_vec

def run_beale():
    bits = 16
    problem = BinaryContinuousProblem(
        func=beale_function,
        num_vars=2,
        bits_per_var=bits,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
    )

    ga = GeneticAlgorithm(
        problem=problem,
        population_size=80,
        mutation_prob=0.2,
        elite_ratio=0.20,
        max_generations=175,
        selection=TournamentSelection(k=3),
        maximize=False,
        random_seed=1,
        log=True,
        show_progress=True,
        dir_name="beale",
    )
    best_overall, results = ga.run_multiple(runs=10)
    best_ind, best_fit, best_fit_str, _, _seed, run_time = best_overall
    x = problem._decode(best_ind)
    line = f"Beale best f(x,y)= {best_fit_str} at {fmt_vec(x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_1', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    summ = compute_summary(vals, times)
    for l in format_summary_block(summ, title="Summary"):
        append_results('ejercicio_1', l)

def run_easom():
    bits = 16
    problem = BinaryContinuousProblem(
        func=easom_function,
        num_vars=2,
        bits_per_var=bits,
        lower_bounds=[-10.0, -10.0],
        upper_bounds=[10.0, 10.0],
    )

    ga = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        mutation_prob=0.2,
        elite_ratio=0.25,
        max_generations=40,
        selection=TournamentSelection(k=2),
        maximize=False,
        random_seed=2,
        log=True,
        show_progress=True,
        dir_name="easom",
    )
    best_overall, results = ga.run_multiple(runs=10)
    best_ind, best_fit, best_fit_str, _, _seed, run_time = best_overall
    x = problem._decode(best_ind)
    line = f"Easom best f(x,y)= {best_fit_str} at {fmt_vec(x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_1', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    summ = compute_summary(vals, times)
    for l in format_summary_block(summ, title="Summary"):
        append_results('ejercicio_1', l)

def run_tsp_for_positions(name: str, positions: List[Tuple[float, float]], population_size: int = 100, elite_ratio: float = 0.2, max_generations: int = 500, k: int = 2, seed: int = 3):
    tsp = TSPProblem(positions, seed=seed)
    ga = GeneticAlgorithm(
        problem=tsp,
        population_size=population_size,
        mutation_prob=0.3,
        elite_ratio=elite_ratio,
        max_generations=max_generations,
        selection=TournamentSelection(k=k),
        maximize=False,
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}",
        show_progress=True,
        parallel_workers=4
    )
    best_overall, results = ga.run_multiple(runs=5)
    best_tour, best_len, best_len_str, _, _seed, run_time = best_overall
    line = f"TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_1', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    summ = compute_summary(vals, times)
    for l in format_summary_block(summ, title="Summary"):
        append_results('ejercicio_1', l)
    export_tsp_graph(positions, best_tour, f"{name}.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_1', dir_name=f"tsp_{name}")

def run_tsp():
    n=25
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, 100, 0.3, 260, 4, n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, 200, 0.3, 170, 2, n)

    n=100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, 800, 0.2, 2000, 4, n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, 1000, 0.15, 2000, 4, n)

    n=225
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, 1600, 0.2, 3000, 4, n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, 1800, 0.15, 3000, 4, n)

if __name__ == "__main__":
    header = "== Continuous optimization =="
    print(header)
    append_results('ejercicio_1', header)
    run_beale()
    run_easom()

    # header = "\n== Traveling Salesman Problem =="
    # print(header)
    # append_results('ejercicio_1', header)
    # run_tsp()