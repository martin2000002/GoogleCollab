from typing import List, Sequence, Tuple
from pathlib import Path
import math
import networkx as nx

COLOR = "#8B0000"

NODE_WIDTH = 35
NODE_GAP = 20.0
_SPACING = NODE_WIDTH + NODE_GAP


def export_tsp_graph(
    positions: List[Tuple[float, float]],
    tour: Sequence[int],
    filename: str,
    annotate_indices: bool = True,
    min_visual_distance: float = 0,
    export_root: str = 'ejercicio_1',
    dir_name: str | None = None,
) -> str:
    n = len(positions)
    if n == 0:
        raise ValueError("positions must be non-empty")
    if not tour or len(tour) != n:
        raise ValueError("tour must contain exactly n nodes in order")

    G = nx.Graph()

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    def _is_grid(xs, ys) -> bool:
        rx = sorted({round(v, 6) for v in xs})
        ry = sorted({round(v, 6) for v in ys})
        return len(rx) * len(ry) == n

    is_grid = _is_grid(xs, ys)
    if is_grid:
        rx = sorted({round(v, 6) for v in xs})
        ry = sorted({round(v, 6) for v in ys})
        x_index = {val: idx for idx, val in enumerate(rx)}
        y_index = {val: idx for idx, val in enumerate(ry)}
        layout = []
        for (x, y) in positions:
            ix = x_index[round(x, 6)]
            iy = y_index[round(y, 6)]
            layout.append((ix * _SPACING, iy * _SPACING))
    else:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        scale = _SPACING / 10.0
        layout = [((x - min_x) * scale, (y - min_y) * scale) for (x, y) in positions]

    if min_visual_distance and min_visual_distance > 0:
        if is_grid:
            if _SPACING < min_visual_distance:
                scale_up = min_visual_distance / _SPACING
                layout = [(x * scale_up, y * scale_up) for (x, y) in layout]
        else:
            def _min_pairwise_distance(points: List[Tuple[float, float]]) -> float:
                m = float('inf')
                n_points = len(points)
                for i in range(n_points):
                    x1, y1 = points[i]
                    for j in range(i + 1, n_points):
                        x2, y2 = points[j]
                        d = math.hypot(x1 - x2, y1 - y2)
                        if d < m:
                            m = d
                return 0.0 if m == float('inf') else m

            current_min = _min_pairwise_distance(layout)
            if current_min > 0 and current_min < min_visual_distance:
                scale_up = min_visual_distance / current_min
                layout = [(x * scale_up, y * scale_up) for (x, y) in layout]

    for i, (x, y) in enumerate(layout):
        label = str(i) if annotate_indices else ""
        G.add_node(str(i), x=float(x), y=float(y), color=COLOR, label=label, width=NODE_WIDTH)

    edges_in_tour = []
    for idx in range(n - 1):
        u = str(tour[idx])
        v = str(tour[idx + 1])
        edges_in_tour.append((u, v))
    edges_in_tour.append((str(tour[-1]), str(tour[0])))

    for u, v in edges_in_tour:
        G.add_edge(u, v, color=COLOR)

    subdir = dir_name if dir_name else 'export'
    export_dir = Path(export_root) / 'visualization' / subdir
    export_dir.mkdir(parents=True, exist_ok=True)
    if not filename.lower().endswith('.graphml'):
        filename = f"{filename}.graphml"
    export_path = export_dir / filename
    nx.write_graphml(G, export_path)
    return str(export_path.resolve())
