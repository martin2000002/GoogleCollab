from __future__ import annotations
from pathlib import Path
import sys
_PARENT_DIR = Path(__file__).resolve().parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
from typing import Sequence
from pso.functions import beale, easom, rastrigin
from pso import ParticleSwarm
from shared.results import append_results

def _fmt_vec(x: Sequence[float]) -> str:
    return '[' + ', '.join(f"{float(v):.6f}" for v in x) + ']'

def run_beale():
    lb = [-4.5, -4.5]
    ub = [4.5, 4.5]
    pso = ParticleSwarm(
        func=beale,
        lower_bounds=lb,
        upper_bounds=ub,
        swarm_size=60,
        max_epochs=150,
        alpha1=1.0,
        alpha2=1.0,
        inertia=1.0,
        maximize=False,
        random_seed=1,
        log=True,
        dir_name="beale",
        show_progress=True,
    )
    best_overall, _ = pso.run_multiple(runs=10)
    best_x, best_f, best_f_str, _hist, _seed, run_time = best_overall
    line = f"Beale best f(x,y)={best_f_str} at {_fmt_vec(best_x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_2', line)


def run_easom():
    lb = [-10.0, -10.0]
    ub = [10.0, 10.0]
    pso = ParticleSwarm(
        func=easom,
        lower_bounds=lb,
        upper_bounds=ub,
        swarm_size=50,
        max_epochs=150,
        alpha1=1.0,
        alpha2=1.0,
        inertia=1.0,
        maximize=False,
        random_seed=2,
        log=True,
        dir_name="easom",
        show_progress=True,
    )
    best_overall, _ = pso.run_multiple(runs=10)
    best_x, best_f, best_f_str, _hist, _seed, run_time = best_overall
    line = f"Easom best f(x,y)={best_f_str} at {_fmt_vec(best_x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_2', line)


def run_rastrigin_n10():
    n = 10
    lb = [-5.12] * n
    ub = [5.12] * n
    pso = ParticleSwarm(
        func=lambda x: rastrigin(x, A=10.0),
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
        dir_name="rastrigin_n10",
        show_progress=True
    )
    best_overall, _ = pso.run_multiple(runs=10)
    best_x, best_f, best_f_str, _hist, _seed, run_time = best_overall
    line = f"Rastrigin(n=10) best f(x)={best_f_str} at {_fmt_vec(best_x)} | time={run_time:.2f}s"
    print(line)
    append_results('ejercicio_2', line)


if __name__ == "__main__":
    header = "== PSO on Continuous Functions =="
    print(header)
    append_results('ejercicio_2', header)
    run_beale()
    run_easom()
    # run_rastrigin_n10()
