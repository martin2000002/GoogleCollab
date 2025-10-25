from __future__ import annotations

from typing import List, Tuple

# External pieces used by the ablations
from ejercicio_2.pso.functions import beale as pso_beale, rastrigin
from pure_adapted.acor import ACOR  # only for types; not used directly here
from hybrid import HybridGAACO_TSP, HybridGAACOR

from shared.tsp.functions import make_random_positions, make_grid_positions
from shared.tsp.graphml_generator import export_tsp_graph
from shared.results import append_results, compute_summary, format_summary_block
from shared.utils import fmt_vec


RUNS = 10


def run_beale_ablation_all() -> None:
    """Ablation study for Hybrid(GA→ACOR) on Beale function.

    Mirrors the structure of executions/no_ablation.py: prints a sub-header,
    runs variants, logs best and summary to shared results.
    """
    sub = "-- Beale function (GA, PSO, ACOR) --"
    print(sub)
    append_results('ejercicio_5', sub)

    beale_base = dict(
        func=pso_beale,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
        ga_population_size=100,
        ga_max_generations=100,
        ga_runs=1,
        archive_size=40,
        samples_per_iter=40,
        max_epochs=50,
        q=0.1,
        xi=0.3,
        random_seed=14,
        log=True,
        exe_root="ejercicio_5",
        show_progress=True,
    )

    for label, mods in ((
        'no_elite', {'ga_elite_ratio': 0.0}
    ), (
        'no_pheromone', {'q': 0.0}
    )):
        params = dict(beale_base)
        params.update(mods)
        params['dir_name'] = f"beale_hybrid_ablation_{label}"
        hyb = HybridGAACOR(**params)
        best, runs = hyb.run_multiple(runs=RUNS)
        bx, bf, bf_str, _hist, _seed, run_time = best
        line = f"Hybrid(GA→ACOR) Beale [{label}] best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
        print(line)
        append_results('ejercicio_5', line)
        vals = [r[1] for r in runs]
        times = [r[-1] for r in runs]
        for l in format_summary_block(compute_summary(vals, times), title=f"Summary ({label})"):
            append_results('ejercicio_5', l)


def run_rastrigin_n10_ablation_all() -> None:
    """Ablation study for Hybrid(GA→ACOR) on Rastrigin (n=10)."""
    sub = "-- Rastrigin n=10 (GA, PSO, ACOR) --"
    print(sub)
    append_results('ejercicio_5', sub)

    n = 10
    lb = [-5.12] * n
    ub = [5.12] * n
    ras_base = dict(
        func=lambda x: rastrigin(x, A=10),
        lower_bounds=lb,
        upper_bounds=ub,
        ga_population_size=1000,
        ga_mutation_prob=0.01,
        ga_elite_ratio=0.1,
        ga_k=2,
        ga_max_generations=55,
        ga_runs=1,
        archive_size=120,
        samples_per_iter=180,
        max_epochs=70,
        q=0.1,
        xi=0.5,
        random_seed=24,
        log=True,
        exe_root="ejercicio_5",
        show_progress=True,
        parallel_workers=4,
    )

    for label, mods in ((
        'no_elite', {'ga_elite_ratio': 0.0}
    ), (
        'no_pheromone', {'q': 0.0}
    )):
        params = dict(ras_base)
        params.update(mods)
        params['dir_name'] = f"rastrigin_n10_hybrid_ablation_{label}"
        hyb = HybridGAACOR(**params)
        best, runs = hyb.run_multiple(runs=RUNS)
        bx, bf, bf_str, _hist, _seed, run_time = best
        line = f"Hybrid(GA→ACOR) Rastrigin(n=10) [{label}] best f(x)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
        print(line)
        append_results('ejercicio_5', line)
        vals = [r[1] for r in runs]
        times = [r[-1] for r in runs]
        for l in format_summary_block(compute_summary(vals, times), title=f"Summary ({label})"):
            append_results('ejercicio_5', l)


def run_tsp_ablation_all() -> None:
    """Ablation study for Hybrid(GA→ACO) on TSP for sizes 25/100/225.

    Follows the structure used by executions/no_ablation.py: for each dataset
    prints a sub-header and logs best and summary.
    """

    def _tsp_ablation(name: str, positions: List[Tuple[float, float]], seed: int, base_hp: dict):
        for label, mods in ((
            'no_elite', {'ga_elite_ratio': 0.0}
        ), (
            'no_pheromone', {'q': 0.0}
        )):
            hp = dict(base_hp)
            hp.update(mods)
            hp['dir_name'] = f"tsp_{name}_hybrid_ablation_{label}"
            hyb = HybridGAACO_TSP(
                positions=positions,
                ga_population_size=hp.get("ga_population_size", max(100, len(positions) * 4)),
                ga_elite_ratio=hp.get("ga_elite_ratio", 0.2),
                ga_mutation_prob=hp.get("ga_mutation_prob", 0.3),
                ga_max_generations=hp.get("ga_max_generations", max(200, len(positions) * 4)),
                ga_k=hp.get("ga_k", 3),
                ga_runs=hp.get("ga_runs", 2),
                num_ants=hp.get("num_ants", max(30, len(positions) // 2)),
                max_epochs=hp.get("max_epochs", max(40, len(positions) // 2)),
                alpha=hp.get("alpha", 1),
                beta=hp.get("beta", 5),
                rho=hp.get("rho", 0.5),
                q=hp.get("q", 1),
                random_seed=seed,
                log=True,
                dir_name=hp['dir_name'],
                exe_root="ejercicio_5",
                show_progress=True,
                parallel_workers=4,
            )
            best, runs = hyb.run_multiple(runs=RUNS)
            best_tour, best_len, best_len_str, _hist, _seed, run_time = best
            line = f"Hybrid(GA→ACO) TSP [{name}] [{label}] best length= {best_len_str} | time={run_time:.2f}s"
            print(line)
            append_results('ejercicio_5', line)
            export_tsp_graph(
                positions,
                best_tour,
                f"{name}_hybrid_ablation_{label}.graphml",
                annotate_indices=True,
                min_visual_distance=100 if name.startswith("grid_") else 30,
                export_root='ejercicio_5',
                dir_name=hp['dir_name'],
            )
            vals = [r[1] for r in runs]
            times = [r[-1] for r in runs]
            for l in format_summary_block(compute_summary(vals, times), title=f"Summary ({label})"):
                append_results('ejercicio_5', l)

    # TSP 25
    n = 25
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    base_25_rand = dict(
        ga_population_size=100, ga_elite_ratio=0.3, ga_mutation_prob=0.3, ga_max_generations=10, ga_k=4,
        ga_runs=2, num_ants=50, max_epochs=20, alpha=1, beta=4, rho=0.6, q=1,
    )
    sub = f"-- TSP random_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_25_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_25_grid = dict(
        ga_population_size=200, ga_elite_ratio=0.3, ga_mutation_prob=0.3, ga_max_generations=2, ga_k=2,
        ga_runs=2, num_ants=50, max_epochs=10, alpha=1, beta=4, rho=0.5, q=1,
    )
    sub = f"-- TSP grid_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_25_grid)

    # TSP 100
    n = 100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    base_100_rand = dict(
        ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.1, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=100, max_epochs=70, alpha=2, beta=5, rho=0.4, q=1,
    )
    sub = f"-- TSP random_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_100_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_100_grid = dict(
        ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=1, ga_k=4,
        ga_runs=2, num_ants=110, max_epochs=32, alpha=2, beta=5, rho=0.3, q=1,
    )
    sub = f"-- TSP grid_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_100_grid)

    # TSP 225
    n = 225
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    base_225_rand = dict(
        ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=230, max_epochs=100, alpha=2, beta=5, rho=0.3, q=1,
    )
    sub = f"-- TSP random_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_225_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_225_grid = dict(
        ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=500, max_epochs=90, alpha=2, beta=5, rho=0.2, q=1,
    )
    sub = f"-- TSP grid_{n} (GA, ACO, PSO-swap) --"
    print(sub)
    append_results('ejercicio_5', sub)
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_225_grid)
