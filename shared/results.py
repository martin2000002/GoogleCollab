from __future__ import annotations
from pathlib import Path
from typing import Union
from statistics import mean, stdev
from shared.utils import fmt_best


def append_results(exercise_root: Union[str, Path], text: str) -> str:
	"""Append a line of text to ejercicio_n/visualization/results.txt.

	Creates the directory if needed. Returns the absolute path to the results file.
	"""
	root = Path(exercise_root)
	out_dir = root / 'visualization'
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / 'results.txt'
	with out_path.open('a', encoding='utf-8') as f:
		f.write(text.rstrip('\n') + '\n')
	return str(out_path.resolve())

def compute_summary(values: list[float], times: list[float]) -> dict:
	"""Compute summary metrics for a list of run values and times."""
	if not values or not times or len(values) != len(times):
		raise ValueError("values and times must be non-empty and of same length")
	n = len(values)
	avg_val = mean(values)
	std_val = stdev(values) if n > 1 else 0.0
	avg_time = mean(times)
	best_val = min(values)  # callers should pass minimization metrics
	return {
		"runs": n,
		"best": best_val,
		"mean": avg_val,
		"std": std_val,
		"mean_time": avg_time,
	}

def format_summary_block(summary: dict, title: str | None = None) -> list[str]:
	"""Return pretty formatted lines for a summary block.
	The 'best', 'mean', and 'std' are formatted with fmt_best, time with 2 decimals.
	"""
	lines: list[str] = []
	if title:
		lines.append(f"    {title}")
	lines.append(f"    runs: {summary['runs']}")
	lines.append(f"    best: {fmt_best(summary['best'])}")
	lines.append(f"    mean: {fmt_best(summary['mean'])}")
	lines.append(f"    std:  {fmt_best(summary['std'])}")
	lines.append(f"    mean time: {summary['mean_time']:.2f}s")
	return lines

