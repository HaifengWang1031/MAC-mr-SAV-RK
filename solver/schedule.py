"""Pure fixed/prescribed schedule arithmetic shared by stepping and saved-run validation."""
import numpy as np

from .mac.grid import Array


def schedule_nodes(initial_time: float, sequence: Array) -> Array:
    """Accumulate long prescribed schedules without losing their endpoint."""
    nodes = np.empty(sequence.size+1, dtype=float)
    nodes[0] = initial_time
    total = float(initial_time)
    correction = 0.
    for index, step in enumerate(sequence, start=1):
        increment = float(step)-correction
        updated = total+increment
        correction = (updated-total)-increment
        total = updated
        nodes[index] = total
    return nodes


def nearest_node_indices(nodes: Array, requests: list[float]) -> list[int]:
    """Map requests to nearest scheduled nodes, choosing the earlier node in a tie."""
    indices = []
    for t in requests:
        distance = np.abs(nodes-t)
        tie_tolerance = 8*np.finfo(float).eps*max(abs(t), float(np.max(np.abs(nodes))), np.finfo(float).tiny)
        indices.append(int(np.flatnonzero(distance <= distance.min()+tie_tolerance)[0]))
    return indices
