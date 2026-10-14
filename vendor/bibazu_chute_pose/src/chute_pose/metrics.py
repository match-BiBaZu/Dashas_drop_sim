"""Shared, colour-coded metric labels; ranks never replace pose identity."""
from __future__ import annotations

METHOD_LABELS = {"rocking": "Rocking", "cwsa": "CWSA", "standard_csa": "CSA (chute)", "crsa": "CRSA (chute)"}
METHOD_COLORS = {"rocking": "#0072B2", "cwsa": "#D55E00", "standard_csa": "#AA4499", "crsa": "#009E73"}


def value_for(node, method):
    if method == "rocking":
        return node.rocking_barrier_mm
    if method == "cwsa":
        return node.csa_stability_index
    return node.classical_metrics.get(method, {}).get("raw_score")


def metric_lines(roadmap, methods):
    output = {node.node_id: [] for node in roadmap.nodes}
    for method in methods:
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown display method: {method}")
        values = {node.node_id: value_for(node, method) for node in roadmap.nodes}
        # Rank the displayed values: visually equal scores have equal ranks.
        precision = 3 if method in {"rocking", "cwsa"} else 6
        rounded = {key: round(value, precision) for key, value in values.items() if value is not None}
        ranks = {value: rank for rank, value in enumerate(sorted(set(rounded.values()), reverse=True))}
        units = " mm" if method == "rocking" else " sr/mm" if method in {"standard_csa", "crsa"} else ""
        for node in roadmap.nodes:
            score = values[node.node_id]
            text = (f"{METHOD_LABELS[method]}: rank {ranks[rounded[node.node_id]]} | {score:.{precision}f}{units}"
                    if score is not None else f"{METHOD_LABELS[method]}: N/A")
            output[node.node_id].append((text, METHOD_COLORS[method]))
    return output
