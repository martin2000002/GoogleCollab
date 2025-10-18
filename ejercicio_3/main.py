from __future__ import annotations

from pathlib import Path
import sys
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
from typing import List, Tuple
from ACO import AntSystem
from shared.tsp.functions import make_random_positions, make_grid_positions
from shared.tsp.generator import export_tsp_graph
from shared.results import append_results, compute_summary, format_summary_block


def run_tsp_for_positions(name: str, positions: List[Tuple[float, float]], num_ants: int, max_epochs: int, alpha: float = 1.0, beta: float = 5.0, rho: float = 0.5, q: float = 1.0, seed: int = 3) -> None:
    aco = AntSystem(
        positions=positions,
        num_ants=num_ants,
        max_epochs=max_epochs,
        alpha=alpha,
        beta=beta,
        rho=rho,
        q=q,
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}",
        show_progress=True,
    )
    best_overall, results = aco.run_multiple(runs=1)
    best_tour, best_len, best_len_str, _hist, _seed, run_time = best_overall
    line = f"ACO TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_3', line)
    export_tsp_graph(positions, best_tour, f"{name}.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_3', dir_name=f"tsp_{name}")
    vals = [r[1] for r in results]
    times = [r[-1] for r in results]
    summ = compute_summary(vals, times)
    for l in format_summary_block(summ, title="Summary"):
        append_results('ejercicio_3', l)

def run_tsp():
    n = 25
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, num_ants=50, max_epochs=45, alpha=1, beta=4, rho=0.5, q=1.0, seed=n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, num_ants=50, max_epochs=10, alpha=1, beta=4, rho=0.5, q=1.0, seed=n)

    n=100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, num_ants=100, max_epochs=70, alpha=2, beta=5, rho=0.4, q=1.0, seed=n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, num_ants=110, max_epochs=40, alpha=2, beta=5, rho=0.3, q=1.0, seed=n)

    n=225
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    run_tsp_for_positions(f"random_{n}", positions_rand, num_ants=400, max_epochs=140, alpha=2, beta=5, rho=0.3, q=1.0, seed=n)

    positions_grid = make_grid_positions(n, spacing=10)
    run_tsp_for_positions(f"grid_{n}", positions_grid, num_ants=500, max_epochs=100, alpha=2, beta=5, rho=0.2, q=1.0, seed=n)

if __name__ == "__main__":
    header = "== ACO-AS on TSP =="
    print(header)
    append_results('ejercicio_3', header)
    run_tsp()
