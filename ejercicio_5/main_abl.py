from __future__ import annotations

from pathlib import Path
import sys
_ROOT_DIR = Path(__file__).resolve().parent.parent
# Ensure parent folder (contains ejercicio_1/2/3 and shared/) is importable
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
    
# Also add each exercise folder so their internal absolute imports (e.g., 'ga.interfaces') resolve
for sub in ("ejercicio_1", "ejercicio_2", "ejercicio_3"):
    p = _ROOT_DIR / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from typing import List, Tuple, Sequence

# GA (Ejercicio 1)
from ejercicio_1.ga.core import GeneticAlgorithm
from ejercicio_1.ga.selections.tournament import TournamentSelection
from ejercicio_1.problems.continuous import BinaryContinuousProblem, beale_function
from ejercicio_1.problems.tsp import TSPProblem

# PSO (Ejercicio 2)
from ejercicio_2.pso.core import ParticleSwarm
from ejercicio_2.pso.functions import beale as pso_beale, rastrigin

# ACO-AS (Ejercicio 3)
from ejercicio_3.aco_c.core import AntSystem

# Adapted versions (this exercise)
from pure_adapted.pso_tsp import SwapSequencePSO
from pure_adapted.acor import ACOR
from hybrid import HybridGAACO_TSP, HybridGAACOR

from shared.tsp.functions import make_random_positions, make_grid_positions
from shared.tsp.graphml_generator import export_tsp_graph
from shared.results import append_results, compute_summary, format_summary_block
from shared.utils import fmt_best, fmt_vec

RUNS = 10

def run_beale_all():
    header = "-- Beale function (GA, PSO, ACOR) --"
    print(header)
    append_results('ejercicio_5', header)

    ga_problem = BinaryContinuousProblem(
        func=beale_function,
        num_vars=2,
        bits_per_var=16,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
    )
    ga = GeneticAlgorithm(
        problem=ga_problem,
        population_size=80,
        mutation_prob=0.2,
        elite_ratio=0.20,
        max_generations=175,
        selection=TournamentSelection(k=3),
        maximize=False,
        random_seed=1,
        log=True,
        show_progress=True,
        exe_root="ejercicio_5",
        dir_name="beale",
    )
    ga_best, ga_runs = ga.run_multiple(runs=RUNS)
    best_ind, best_fit, best_fit_str, _hist, _seed, run_time = ga_best
    x = ga_problem._decode(best_ind)
    line = f"GA Beale best f(x,y)= {best_fit_str} at {fmt_vec(x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in ga_runs]
    times = [r[-1] for r in ga_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (GA)"):
        append_results('ejercicio_5', l)

    pso = ParticleSwarm(
        func=pso_beale,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
        swarm_size=100,
        max_epochs=50,
        alpha1=1,
        alpha2=0.5,
        inertia=0.5,
        maximize=False,
        random_seed=12,
        log=True,
        dir_name="beale_pso",
        exe_root="ejercicio_5",
        show_progress=True,
    )
    pso_best, pso_runs = pso.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = pso_best
    line = f"PSO Beale best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in pso_runs]
    times = [r[-1] for r in pso_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (PSO)"):
        append_results('ejercicio_5', l)

    acor = ACOR(
        func=pso_beale,
        lower_bounds=[-4.5, -4.5],
        upper_bounds=[4.5, 4.5],
        archive_size=40,
        samples_per_iter=40,
        max_epochs=50,
        q=0.1,
        xi=0.85,
        maximize=False,
        random_seed=13,
        log=True,
        dir_name="beale_acor",
        show_progress=True,
    )
    acor_best, acor_runs = acor.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = acor_best
    line = f"ACOR Beale best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in acor_runs]
    times = [r[-1] for r in acor_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (ACOR)"):
        append_results('ejercicio_5', l)

    hyb = HybridGAACOR(
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
        dir_name="beale_hybrid",
        exe_root="ejercicio_5",
        show_progress=True,
    )
    hyb_best, hyb_runs = hyb.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = hyb_best
    line = f"Hybrid(GA→ACOR) Beale best f(x,y)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in hyb_runs]
    times = [r[-1] for r in hyb_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (Hybrid)"):
        append_results('ejercicio_5', l)

def run_rastrigin_n10_all():
    header = "-- Rastrigin n=10 (GA, PSO, ACOR) --"
    print(header)
    append_results('ejercicio_5', header)

    n = 10
    lb = [-5.12] * n
    ub = [5.12] * n

    ga_problem = BinaryContinuousProblem(
        func=lambda x: rastrigin(x, A=10),
        num_vars=n,
        bits_per_var=16,
        lower_bounds=lb,
        upper_bounds=ub,
    )
    ga = GeneticAlgorithm(
        problem=ga_problem,
        population_size=1000,
        mutation_prob=0.0625,
        elite_ratio=0.1,
        max_generations=2000,
        selection=TournamentSelection(k=4),
        maximize=False,
        random_seed=21,
        log=True,
        dir_name="rastrigin_n10_ga",
        exe_root="ejercicio_5",
        show_progress=True,
        parallel_workers=4,
    )
    ga_best, ga_runs = ga.run_multiple(runs=RUNS)
    best_ind, best_fit, best_fit_str, _hist, _seed, run_time = ga_best
    x = ga_problem._decode(best_ind)
    line = f"GA Rastrigin(n=10) best f(x)={best_fit_str} at {fmt_vec(x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in ga_runs]
    times = [r[-1] for r in ga_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (GA)"):
        append_results('ejercicio_5', l)

    pso = ParticleSwarm(
        func=lambda x: rastrigin(x, A=10),
        lower_bounds=lb,
        upper_bounds=ub,
        swarm_size=1300,
        max_epochs=4000,
        alpha1=0.55,
        alpha2=0.4,
        inertia=1,
        maximize=False,
        random_seed=3,
        log=True,
        dir_name="rastrigin_n10_pso",
        exe_root="ejercicio_5",
        show_progress=True,
    )
    pso_best, pso_runs = pso.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = pso_best
    line = f"PSO Rastrigin(n=10) best f(x)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in pso_runs]
    times = [r[-1] for r in pso_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (PSO)"):
        append_results('ejercicio_5', l)

    acor = ACOR(
        func=lambda x: rastrigin(x, A=10),
        lower_bounds=lb,
        upper_bounds=ub,
        archive_size=120,
        samples_per_iter=180,
        max_epochs=2000,
        q=0.5,
        xi=0.85,
        maximize=False,
        random_seed=23,
        log=True,
        dir_name="rastrigin_n10_acor",
        show_progress=True,
        parallel_workers=4,
    )
    acor_best, acor_runs = acor.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = acor_best
    line = f"ACOR Rastrigin(n=10) best f(x)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in acor_runs]
    times = [r[-1] for r in acor_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (ACOR)"):
        append_results('ejercicio_5', l)

    hyb = HybridGAACOR(
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
        dir_name="rastrigin_n10_hybrid",
        exe_root="ejercicio_5",
        show_progress=True,
        parallel_workers=4,
    )
    hyb_best, hyb_runs = hyb.run_multiple(runs=RUNS)
    bx, bf, bf_str, _hist, _seed, run_time = hyb_best
    line = f"Hybrid(GA→ACOR) Rastrigin(n=10) best f(x)={bf_str} at {fmt_vec(bx)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    vals = [r[1] for r in hyb_runs]
    times = [r[-1] for r in hyb_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (Hybrid)"):
        append_results('ejercicio_5', l)

def _run_tsp_with_all(
    name: str,
    positions: List[Tuple[float, float]],
    seed: int,
    *,
    ga_params: dict | None = None,
    aco_params: dict | None = None,
    pso_params: dict | None = None,
    hybrid_params: dict | None = None,
) -> None:
    header = f"-- TSP {name} (GA, ACO, PSO-swap) --"
    print(header)
    append_results('ejercicio_5', header)

    ga_prob = TSPProblem(positions, seed=seed)
    if ga_params is None:
        ga_params = {}
    ga = GeneticAlgorithm(
        problem=ga_prob,
        population_size=ga_params.get("population_size", max(100, len(positions) * 4)),
        mutation_prob=ga_params.get("mutation_prob", 0.3),
        elite_ratio=ga_params.get("elite_ratio", 0.2),
        max_generations=ga_params.get("max_generations", max(200, len(positions) * 8)),
        selection=TournamentSelection(k=ga_params.get("k", 3)),
        maximize=False,
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}_ga",
        exe_root="ejercicio_5",
        show_progress=True,
        parallel_workers=4,
    )
    ga_best, ga_runs = ga.run_multiple(runs=RUNS)
    best_tour, best_len, best_len_str, _hist, _seed, run_time = ga_best
    line = f"GA TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    export_tsp_graph(positions, best_tour, f"{name}_ga.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_5', dir_name=f"tsp_{name}_ga")
    vals = [r[1] for r in ga_runs]
    times = [r[-1] for r in ga_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (GA)"):
        append_results('ejercicio_5', l)

    n = len(positions)
    if aco_params is None:
        aco_params = {}
    aco = AntSystem(
        positions=positions,
        num_ants=aco_params.get("num_ants", max(30, n // 2)),
        max_epochs=aco_params.get("max_epochs", max(50, n)),
        alpha=aco_params.get("alpha", 1),
        beta=aco_params.get("beta", 5),
        rho=aco_params.get("rho", 0.5),
        q=aco_params.get("q", 1),
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}_aco",
        exe_root="ejercicio_5",
        show_progress=True,
    )
    aco_best, aco_runs = aco.run_multiple(runs=RUNS)
    best_tour, best_len, best_len_str, _hist, _seed, run_time = aco_best
    line = f"ACO TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    export_tsp_graph(positions, best_tour, f"{name}_aco.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_5', dir_name=f"tsp_{name}_aco")
    vals = [r[1] for r in aco_runs]
    times = [r[-1] for r in aco_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (ACO)"):
        append_results('ejercicio_5', l)

    if pso_params is None:
        pso_params = {}
    pso = SwapSequencePSO(
        positions=positions,
        swarm_size=pso_params.get("swarm_size", max(60, n // 2)),
        max_epochs=pso_params.get("max_epochs", max(80, n)),
        alpha1=pso_params.get("alpha1", 0.9),
        alpha2=pso_params.get("alpha2", 0.9),
        inertia=pso_params.get("inertia", 0.8),
        vmax_frac=pso_params.get("vmax_frac", 0.7),
        random_seed=seed,
        log=True,
        dir_name=f"tsp_{name}_pso",
        show_progress=True,
        parallel_workers=4,
    )
    pso_best, pso_runs = pso.run_multiple(runs=RUNS)
    best_tour, best_len, best_len_str, _hist, _seed, run_time = pso_best
    line = f"PSO-swap TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_5', line)
    export_tsp_graph(positions, best_tour, f"{name}_pso.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_5', dir_name=f"tsp_{name}_pso")
    vals = [r[1] for r in pso_runs]
    times = [r[-1] for r in pso_runs]
    for l in format_summary_block(compute_summary(vals, times), title="Summary (PSO-swap)"):
        append_results('ejercicio_5', l)

    if hybrid_params is not None:
        hp = hybrid_params
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
            dir_name=f"tsp_{name}_hybrid",
            exe_root="ejercicio_5",
            show_progress=True,
            parallel_workers=4,
        )
        hyb_best, hyb_runs = hyb.run_multiple(runs=RUNS)
        best_tour, best_len, best_len_str, _hist, _seed, run_time = hyb_best
        line = f"Hybrid(GA→ACO) TSP [{name}] best length= {best_len_str} | time={run_time:.2f}s"
        print(line)
        append_results('ejercicio_5', line)
        export_tsp_graph(positions, best_tour, f"{name}_hybrid.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_5', dir_name=f"tsp_{name}_hybrid")
        vals = [r[1] for r in hyb_runs]
        times = [r[-1] for r in hyb_runs]
        for l in format_summary_block(compute_summary(vals, times), title="Summary (Hybrid)"):
            append_results('ejercicio_5', l)

def run_tsp_all():
    n = 25
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    _run_tsp_with_all(
        f"random_{n}", positions_rand, seed=n,
        ga_params=dict(population_size=100, elite_ratio=0.3, max_generations=260, k=4, mutation_prob=0.3),
        aco_params=dict(num_ants=50, max_epochs=45, alpha=1, beta=4, rho=0.5, q=1),
        pso_params=dict(swarm_size=1000, max_epochs=1100, alpha1=1, alpha2=0.5, inertia=0.5, vmax_frac=1),
        hybrid_params=dict(
            ga_population_size=100, ga_elite_ratio=0.3, ga_mutation_prob=0.3, ga_max_generations=10, ga_k=4,
            num_ants=50, max_epochs=20, alpha=1, beta=4, rho=0.6, q=1
        ),
    )

    positions_grid = make_grid_positions(n, spacing=10)
    _run_tsp_with_all(
        f"grid_{n}", positions_grid, seed=n,
        ga_params=dict(population_size=200, elite_ratio=0.3, max_generations=170, k=2, mutation_prob=0.3),
        aco_params=dict(num_ants=50, max_epochs=10, alpha=1, beta=4, rho=0.5, q=1),
        pso_params=dict(swarm_size=2000, max_epochs=300, alpha1=2.3, alpha2=0.5, inertia=0.4, vmax_frac=1),
        hybrid_params=dict(
            ga_population_size=200, ga_elite_ratio=0.3, ga_mutation_prob=0.3, ga_max_generations=2, ga_k=2,
            num_ants=50, max_epochs=10, alpha=1, beta=4, rho=0.5, q=1,
        ),
    )

    n = 100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    _run_tsp_with_all(
        f"random_{n}", positions_rand, seed=n,
        ga_params=dict(population_size=800, elite_ratio=0.2, max_generations=2000, k=4, mutation_prob=0.3),
        aco_params=dict(num_ants=100, max_epochs=70, alpha=2, beta=5, rho=0.4, q=1),
        pso_params=dict(swarm_size=4000, max_epochs=2000, alpha1=1, alpha2=0.9, inertia=0.88, vmax_frac=0.7),
        hybrid_params=dict(
            ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.1, ga_max_generations=100, ga_k=4,
            num_ants=100, max_epochs=70, alpha=2, beta=5, rho=0.4, q=1,
        ),
    )

    positions_grid = make_grid_positions(n, spacing=10)
    _run_tsp_with_all(
        f"grid_{n}", positions_grid, seed=n,
        ga_params=dict(population_size=1000, elite_ratio=0.15, max_generations=2000, k=4, mutation_prob=0.3),
        aco_params=dict(num_ants=110, max_epochs=40, alpha=2, beta=5, rho=0.3, q=1),
        pso_params=dict(swarm_size=8000, max_epochs=1800, alpha1=1, alpha2=0.9, inertia=0.88, vmax_frac=0.65),
        hybrid_params=dict(
            ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=1, ga_k=4,
            num_ants=110, max_epochs=32, alpha=2, beta=5, rho=0.3, q=1
        ),
    )

    n = 225
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    _run_tsp_with_all(
        f"random_{n}", positions_rand, seed=n,
        ga_params=dict(population_size=1600, elite_ratio=0.2, max_generations=3000, k=4, mutation_prob=0.3),
        aco_params=dict(num_ants=400, max_epochs=140, alpha=2, beta=5, rho=0.3, q=1),
        pso_params=dict(swarm_size=9000, max_epochs=3000, alpha1=1, alpha2=0.9, inertia=0.90, vmax_frac=0.7),
        hybrid_params=dict(
            ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
            num_ants=230, max_epochs=100, alpha=2, beta=5, rho=0.3, q=1,
        ),
    )

    positions_grid = make_grid_positions(n, spacing=10)
    _run_tsp_with_all(
        f"grid_{n}", positions_grid, seed=n,
        ga_params=dict(population_size=1800, elite_ratio=0.15, max_generations=3000, k=4, mutation_prob=0.3),
        aco_params=dict(num_ants=500, max_epochs=100, alpha=2, beta=5, rho=0.2, q=1),
        pso_params=dict(swarm_size=18000, max_epochs=2800, alpha1=1, alpha2=0.9, inertia=0.90, vmax_frac=0.65),
        hybrid_params=dict(
            ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
            num_ants=500, max_epochs=90, alpha=2, beta=5, rho=0.2, q=1,
        ),
    )


def run_ablation_all():
    header = "-- Ablation: sin elitismo (GA) y sin feromona (ACO/ACOR) --"
    print(header)
    append_results('ejercicio_5', header)

    # ===== Beale (Hybrid GA→ACOR) =====
    print("Ablation on Beale (Hybrid GA→ACOR)")
    append_results('ejercicio_5', "Ablation on Beale (Hybrid GA→ACOR)")
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
    for label, mods in (('no_elite', {'ga_elite_ratio': 0.0}), ('no_pheromone', {'q': 0.0})):  # only requested ablations
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

    # ===== Rastrigin n=10 (Hybrid GA→ACOR) =====
    print("Ablation on Rastrigin n=10 (Hybrid GA→ACOR)")
    append_results('ejercicio_5', "Ablation on Rastrigin n=10 (Hybrid GA→ACOR)")
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
    for label, mods in (('no_elite', {'ga_elite_ratio': 0.0}), ('no_pheromone', {'q': 0.0})):
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

    # ===== TSP (Hybrid GA→ACO) for all sizes used in run_tsp_all =====
    print("Ablation on TSP (Hybrid GA→ACO)")
    append_results('ejercicio_5', "Ablation on TSP (Hybrid GA→ACO)")

    def _tsp_ablation(name: str, positions: List[Tuple[float, float]], seed: int, base_hp: dict):
        for label, mods in (('no_elite', {'ga_elite_ratio': 0.0}), ('no_pheromone', {'q': 0.0})):
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
            export_tsp_graph(positions, best_tour, f"{name}_hybrid_ablation_{label}.graphml", annotate_indices=True, min_visual_distance=100 if name.startswith("grid_") else 30, export_root='ejercicio_5', dir_name=hp['dir_name'])
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
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_25_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_25_grid = dict(
        ga_population_size=200, ga_elite_ratio=0.3, ga_mutation_prob=0.3, ga_max_generations=2, ga_k=2,
        ga_runs=2, num_ants=50, max_epochs=10, alpha=1, beta=4, rho=0.5, q=1,
    )
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_25_grid)

    # TSP 100
    n = 100
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    base_100_rand = dict(
        ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.1, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=100, max_epochs=70, alpha=2, beta=5, rho=0.4, q=1,
    )
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_100_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_100_grid = dict(
        ga_population_size=200, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=1, ga_k=4,
        ga_runs=2, num_ants=110, max_epochs=32, alpha=2, beta=5, rho=0.3, q=1,
    )
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_100_grid)

    # TSP 225
    n = 225
    positions_rand = make_random_positions(n, width=100, height=100, seed=n)
    base_225_rand = dict(
        ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=230, max_epochs=100, alpha=2, beta=5, rho=0.3, q=1,
    )
    _tsp_ablation(f"random_{n}", positions_rand, seed=n, base_hp=base_225_rand)

    positions_grid = make_grid_positions(n, spacing=10)
    base_225_grid = dict(
        ga_population_size=100, ga_elite_ratio=0.2, ga_mutation_prob=0.3, ga_max_generations=100, ga_k=4,
        ga_runs=2, num_ants=500, max_epochs=90, alpha=2, beta=5, rho=0.2, q=1,
    )
    _tsp_ablation(f"grid_{n}", positions_grid, seed=n, base_hp=base_225_grid)


if __name__ == "__main__":
    # header = "== Continuous: Beale & Rastrigin (GA, PSO, ACOR) =="
    # print(header)
    # append_results('ejercicio_5', header)
    # run_beale_all()
    # run_rastrigin_n10_all()

    # header = "\n== TSP 25/100/225 (GA, ACO, PSO-swap) =="
    # print(header)
    # append_results('ejercicio_5', header)
    # run_tsp_all()

    # Ejecutar ablation requerida (solo dos variantes: sin elitismo y sin feromona)
    header = "\n== Ablation (elitismo y feromona) en Beale, Rastrigin y TSP =="
    print(header)
    append_results('ejercicio_5', header)
    run_ablation_all()
