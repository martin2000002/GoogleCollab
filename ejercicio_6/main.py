from __future__ import annotations

from pathlib import Path
import sys

# Ensure parent (contains shared, ejercicio_1, etc.) is importable
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from typing import List, Tuple

from abc_c.core import ArtificialBeeColony
from problems.continuous import ABCContinuousProblem, beale_function, easom_function
from problems.tsp import ABCTSPProblem
from shared.tsp.functions import make_grid_positions, make_random_positions
from shared.tsp.graphml_generator import export_tsp_graph
from shared.results import append_results, compute_summary, format_summary_block
from shared.utils import fmt_vec


def run_beale():
    problem = ABCContinuousProblem(
        func=beale_function,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
    )
    abc = ArtificialBeeColony(
        problem=problem,
        colony_size=120,
        max_epochs=1400,
        limit=100,
        maximize=False,
        random_seed=1,
        log=True,
        dir_name="beale",
        exe_root="ejercicio_6",
        show_progress=True,
    )
    best_overall, results = abc.run_multiple(runs=10)
    bx, bf, bf_str, _hist, _seed, run_time = best_overall
    line = f"Beale (ABC) best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_6', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    for l in format_summary_block(compute_summary(vals, times), title="Summary"):
        append_results('ejercicio_6', l)


def run_easom():
    problem = ABCContinuousProblem(
        func=easom_function,
        lower_bounds=[-10.0, -10.0],
        upper_bounds=[10.0, 10.0],
    )
    abc = ArtificialBeeColony(
        problem=problem,
        colony_size=30,
        max_epochs=100,
        limit=20,
        maximize=False,
        random_seed=1,
        log=True,
        dir_name="easom",
        exe_root="ejercicio_6",
        show_progress=True,
    )
    best_overall, results = abc.run_multiple(runs=1)
    bx, bf, bf_str, _hist, _seed, run_time = best_overall
    line = f"Easom (ABC) best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_6', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    for l in format_summary_block(compute_summary(vals, times), title="Summary"):
        append_results('ejercicio_6', l)


def _run_tsp_for_positions(name: str, positions: List[Tuple[float, float]], colony_size: int, max_epochs: int, limit: int, seed: int) -> None:
    tsp = ABCTSPProblem(positions, seed=seed)
    abc = ArtificialBeeColony(
        problem=tsp,
        colony_size=colony_size,
        max_epochs=max_epochs,
        limit=limit,
        maximize=False,
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}",
        exe_root="ejercicio_6",
        show_progress=True,
        parallel_workers=4,
    )
    best_overall, results = abc.run_multiple(runs=1)
    best_tour, best_len, best_len_str, _hist, _seed, run_time = best_overall
    line = f"TSP [{name}] (ABC) best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_6', line)
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    for l in format_summary_block(compute_summary(vals, times), title="Summary"):
        append_results('ejercicio_6', l)
    export_tsp_graph(positions, best_tour, f"{name}.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_6', dir_name=f"tsp_{name}")


def run_tsp():
    # n = 25
    # positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    # _run_tsp_for_positions(f"random_{n}", positions_rand, colony_size=200, max_epochs=1000, limit=150, seed=n)

    # positions_grid = make_grid_positions(n, spacing=10)
    # _run_tsp_for_positions(f"grid_{n}", positions_grid, colony_size=200, max_epochs=100, limit=100, seed=n)

    n = 100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    _run_tsp_for_positions(f"random_{n}", positions_rand, colony_size=8000, max_epochs=1000, limit=1000, seed=n)

    # positions_grid = make_grid_positions(n, spacing=10)
    # _run_tsp_for_positions(f"grid_{n}", positions_grid, colony_size=800, max_epochs=1200, limit=120, seed=n)

    # n = 225
    # positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    # _run_tsp_for_positions(f"random_{n}", positions_rand, colony_size=1200, max_epochs=2000, limit=200, seed=n)

    # positions_grid = make_grid_positions(n, spacing=10)
    # _run_tsp_for_positions(f"grid_{n}", positions_grid, colony_size=1500, max_epochs=2200, limit=220, seed=n)


if __name__ == "__main__":
    header = "== Continuous optimization (ABC) =="
    print(header)
    append_results('ejercicio_6', header)
    # run_beale()
    # run_easom()

    # header = "\n== Traveling Salesman Problem (ABC) =="
    # print(header)
    # append_results('ejercicio_6', header)
    run_tsp()
