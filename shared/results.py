from __future__ import annotations
from pathlib import Path
from typing import Union


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

