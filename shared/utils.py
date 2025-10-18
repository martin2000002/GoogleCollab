from __future__ import annotations

def fmt_best(f: float) -> str:
    """Format best fitness: scientific notation when |f| < 1e-3 and not exactly 0, else 12 decimals."""
    return f"{f:.6e}" if (abs(f) < 1e-3 and f != 0.0) else f"{f:.12f}"
