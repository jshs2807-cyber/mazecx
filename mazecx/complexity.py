"""흡수 마르코프 연쇄로 C_geo와 C_arith를 계산한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mazecx.graph import SpatialGraph

AXIS_WEIGHTS = ("standard", "none", "a", "inv")


@dataclass(frozen=True)
class ComplexityResult:
    """길이 가중·차수 가중 흡수시간과 두 복잡도."""

    tau_length: float
    tau_degree: float
    c_geo: float
    c_arith: float


def vertex_weight(axis_count: int, mode: str) -> float:
    """축 개수 a(v)와 옵션에 따라 정점 가중치 w(v)를 반환한다."""
    if axis_count < 1:
        raise ValueError(f"축 개수는 1 이상이어야 합니다. 받은 값: {axis_count}")
    if mode == "standard":
        return float(max(1, axis_count - 1))
    if mode == "none":
        return 1.0
    if mode == "a":
        return float(axis_count)
    if mode == "inv":
        return 1.0 / float(axis_count)
    raise ValueError(
        f"알 수 없는 축 가중치 '{mode}'. 사용할 수 있는 값: {', '.join(AXIS_WEIGHTS)}"
    )


def compute_complexity(graph: SpatialGraph, axis_weight: str = "standard") -> ComplexityResult:
    """길이 가중치와 차수 가중치로 입구의 흡수시간을 구하고 복잡도를 만든다."""
    transient = [node_id for node_id in graph.ids if node_id != graph.exit]
    if graph.entry not in transient:
        raise RuntimeError("entry가 전이 상태 집합에 없습니다.")
    index = {node_id: position for position, node_id in enumerate(transient)}
    tau_length = _entry_absorption(
        graph,
        _transition_matrix(graph, transient, index, "length"),
        transient,
        axis_weight,
    )
    tau_degree = _entry_absorption(
        graph,
        _transition_matrix(graph, transient, index, "degree"),
        transient,
        axis_weight,
    )
    if tau_length < 0.0 or tau_degree < 0.0:
        raise RuntimeError("흡수시간 τ에 음수가 있습니다.")
    c_geo = float(np.sqrt(tau_length * tau_degree))
    c_arith = (tau_length + tau_degree) / 2.0
    if not np.isfinite(c_geo) or not np.isfinite(c_arith):
        raise RuntimeError("복잡도에 NaN 또는 inf가 있습니다.")
    return ComplexityResult(tau_length, tau_degree, c_geo, c_arith)


def _transition_matrix(
    graph: SpatialGraph,
    transient: list[str],
    index: dict[str, int],
    mode: str,
) -> np.ndarray:
    """전이 정점 사이의 이동 확률 행렬 Θ를 만든다.

    길이 가중치는 W_ij = 1/L_ij 이다.
    차수 가중치는 도착 정점 j의 k_eff(j) = max(2, deg(j))로 W_ij = 1/k_eff(j) 이다.
    s_i에는 출구로 가는 가중치도 포함하고, 출구 열은 Θ에 넣지 않는다.
    """
    if mode not in ("length", "degree"):
        raise ValueError(f"알 수 없는 간선 가중치 '{mode}'")
    size = len(transient)
    theta = np.zeros((size, size), dtype=np.float64)
    degrees = {node_id: graph.degree(node_id) for node_id in graph.ids}
    for row, source in enumerate(transient):
        weighted: list[tuple[int, float]] = []
        total = 0.0
        for neighbor, length, _edge_index in graph.adj[source]:
            if mode == "length":
                weight = 1.0 / float(length)
            else:
                weight = 1.0 / float(max(2, degrees[neighbor]))
            total += weight
            column = index.get(neighbor)
            if column is not None:
                weighted.append((column, weight))
        if total == 0.0:
            raise RuntimeError(f"정점 {source}에서 나가는 가중치 합이 0입니다.")
        for column, weight in weighted:
            theta[row, column] = weight / total
    return theta


def _entry_absorption(
    graph: SpatialGraph,
    theta: np.ndarray,
    transient: list[str],
    axis_weight: str,
) -> float:
    """(I − Θ) τ = w_T 를 풀어 입구의 흡수시간을 반환한다."""
    size = len(transient)
    system = np.eye(size, dtype=np.float64) - theta
    weights = np.array(
        [vertex_weight(graph.axis_count(node_id), axis_weight) for node_id in transient],
        dtype=np.float64,
    )
    try:
        tau = np.linalg.solve(system, weights)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "행렬 (I − Θ)가 특이(singular)하여 흡수시간을 구할 수 없습니다."
        ) from exc
    if not np.all(np.isfinite(tau)):
        raise RuntimeError("흡수시간 τ에 NaN 또는 inf가 있습니다.")
    if np.any(tau < -1e-9):
        raise RuntimeError("흡수시간 τ에 음수가 있습니다.")
    return float(tau[transient.index(graph.entry)])
