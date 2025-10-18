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
	"""Compute summary metrics for a list of run values and times.
	Returns keys: runs, mean_best, mean_time, std
	"""
	if not values or not times or len(values) != len(times):
		raise ValueError("values and times must be non-empty and of same length")
	n = len(values)
	avg_val = mean(values)
	std_val = stdev(values) if n > 1 else 0.0
	avg_time = mean(times)
	return {
		"runs": n,
		"mean_best": avg_val,
		"mean_time": avg_time,
		"std": std_val,
	}

def format_summary_block(summary: dict, title: str | None = None) -> list[str]:
	"""Return pretty formatted lines for a summary block.
	Formats mean_best and std with fmt_best, mean time with 2 decimals.
	"""
	lines: list[str] = []
	if title:
		lines.append(f"    {title}")
	lines.append(f"    runs: {summary['runs']}")
	lines.append(f"    mean best: {fmt_best(summary['mean_best'])}")
	lines.append(f"    mean time: {summary['mean_time']:.2f}s")
	lines.append(f"    std:  {fmt_best(summary['std'])}")
	return lines

