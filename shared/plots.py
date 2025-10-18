from __future__ import annotations
from typing import Sequence, Union
from pathlib import Path
import matplotlib.pyplot as plt


def save_convergence(history: Sequence[float], output_path: Union[str, Path], *, title: str | None = None, ylabel: str = "Best Fitness", xlabel: str = "Generation") -> str:
    """Save a convergence plot (y vs generations) to output_path. Returns absolute path as string."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(history)), history, color='tab:red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path.resolve())
