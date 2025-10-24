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

from shared.results import append_results

from executions.no_ablation import run_beale_all, run_rastrigin_n10_all, run_tsp_all
from executions.ablation import run_beale_ablation_all, run_rastrigin_n10_ablation_all, run_tsp_ablation_all


if __name__ == "__main__":
    # ======= ORIGINAL (sin ablación) =======
    sep = "====================================="
    # print(sep)
    # append_results('ejercicio_5', sep)
    # title = "WITHOUT ABLATION"
    # print(title)
    # append_results('ejercicio_5', title)
    # print(sep)
    # append_results('ejercicio_5', sep)

    # header = "== Continuous: Beale & Rastrigin (GA, PSO, ACOR) =="
    # print(header)
    # append_results('ejercicio_5', header)
    # run_beale_all()
    # run_rastrigin_n10_all()

    # header = "\n== TSP 25/100/225 (GA, ACO, PSO-swap) =="
    # print(header)
    # append_results('ejercicio_5', header)
    # run_tsp_all()

    # ======= CON ABLACIÓN =======
    print(sep)
    append_results('ejercicio_5', sep)
    title = "With Ablation"
    print(title)
    append_results('ejercicio_5', title)
    print(sep)
    append_results('ejercicio_5', sep)

    header = "== Continuous: Beale & Rastrigin (GA, PSO, ACOR) =="
    print(header)
    append_results('ejercicio_5', header)
    run_beale_ablation_all()
    run_rastrigin_n10_ablation_all()

    header = "\n== TSP 25/100/225 (GA, ACO, PSO-swap) =="
    print(header)
    append_results('ejercicio_5', header)
    run_tsp_ablation_all()