from __future__ import annotations
from typing import Sequence

def fmt_best(f: float) -> str:
    """Format best fitness: scientific notation when |f| < 1e-3 and not exactly 0, else 12 decimals."""
    return f"{f:.6e}" if (abs(f) < 1e-3 and f != 0.0) else f"{f:.12f}"


def fmt_vec(x: Sequence[float]) -> str:
    """Format a vector as [x1, x2, ...] with 6 decimals."""
    return '[' + ', '.join(f"{float(v):.6f}" for v in x) + ']'
