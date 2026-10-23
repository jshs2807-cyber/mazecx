"""출구 좌표를 아는 탐욕 라우팅의 성공률과 GR_score를 계산한다."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from mazecx.graph import SpatialGraph, taxi_distance


@dataclass(frozen=True)
class GreedyResult:
    """탐욕 라우팅 성공률, 복잡도, 입구 성공 여부, GR_score."""

    success_rate: float
    difficulty: float
    entry_success: bool
    gr_score: float


def greedy_route(graph: SpatialGraph, start: str) -> tuple[bool, int]:
    """start에서 출구로 가는 탐욕 경로의 성공 여부와 경로 길이를 반환한다."""
    succeeded, path_length, _visits = trace_greedy(graph, start)
    return succeeded, path_length


def trace_greedy(graph: SpatialGraph, start: str) -> tuple[bool, int, tuple[str, ...]]:
    """탐욕 경로의 성공 여부, 길이, 방문 정점 순서를 반환한다.

    후보는 출구까지 택시거리가 지금보다 엄격히 작은 이웃뿐이다.
    동점이면 간선 길이가 짧은 쪽, 그래도 같으면 정점 id 사전순으로 고른다.
    """
    if start == graph.exit:
        return True, 0, (start,)
    exit_coord = graph.nodes[graph.exit]
    node = start
    visits = [start]
    path_length = 0
    steps = 0
    while node != graph.exit:
        steps += 1
        if steps > graph.n_nodes:
            raise RuntimeError("탐욕 라우팅 반복 횟수가 정점 수를 넘었습니다.")
        here = taxi_distance(graph.nodes[node], exit_coord)
        candidates: list[tuple[int, int, str]] = []
        for neighbor, length, _edge_index in graph.adj[node]:
            distance = taxi_distance(graph.nodes[neighbor], exit_coord)
            if distance < here:
                candidates.append((distance, length, neighbor))
        if not candidates:
            return False, path_length, tuple(visits)
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        _distance, length, neighbor = candidates[0]
        path_length += length
        node = neighbor
        visits.append(node)
    return True, path_length, tuple(visits)


def greedy_statistics(graph: SpatialGraph) -> GreedyResult:
    """출구를 뺀 모든 정점에서 탐욕 라우팅 통계를 계산한다."""
    distances = graph.shortest_distances_from(graph.exit)
    starts = [node_id for node_id in graph.ids if node_id != graph.exit]
    if not starts:
        raise RuntimeError("탐욕 라우팅 출발점이 없습니다.")
    scores: list[float] = []
    successes = 0
    entry_success = False
    for start in starts:
        if start not in distances:
            raise RuntimeError(f"출구에서 정점 '{start}'까지의 최단거리가 없습니다.")
        succeeded, path_length = greedy_route(graph, start)
        if start == graph.entry:
            entry_success = succeeded
        if succeeded:
            if path_length <= 0:
                raise RuntimeError("성공한 탐욕 경로의 길이가 0입니다.")
            successes += 1
            scores.append(distances[start] / path_length)
        else:
            scores.append(0.0)
    success_rate = successes / len(starts)
    gr_score = sum(scores) / len(starts)
    if abs(gr_score - success_rate) >= 1e-9:
        warnings.warn(
            f"GR_score({gr_score})와 success_rate({success_rate})가 1e-9 이상 다릅니다. "
            "축에 평행한 간선이면 성공한 탐욕 경로는 최단 경로여야 합니다.",
            UserWarning,
            stacklevel=2,
        )
    return GreedyResult(
        success_rate=success_rate,
        difficulty=1.0 - success_rate,
        entry_success=entry_success,
        gr_score=gr_score,
    )
