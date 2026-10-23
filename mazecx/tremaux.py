"""트레모 알고리즘의 걸음 수를 시뮬레이션한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mazecx.graph import SpatialGraph


@dataclass(frozen=True)
class TremauxStats:
    """걸음 수의 평균, 표본 표준편차, 최소, 최대, L_total로 나눈 값."""

    mean: float
    std: float
    minimum: int
    maximum: int
    norm: float
    samples: tuple[int, ...]

    def __repr__(self) -> str:
        """표본 전체를 출력하지 않고 요약만 보여 준다."""
        return (
            f"TremauxStats(mean={self.mean}, std={self.std}, "
            f"minimum={self.minimum}, maximum={self.maximum}, norm={self.norm}, "
            f"runs={len(self.samples)})"
        )


def tremaux_statistics(graph: SpatialGraph, runs: int = 100, seed: int = 0) -> TremauxStats:
    """같은 난수 생성기로 runs번 실행해 걸음 수 통계를 반환한다."""
    samples = tuple(tremaux_samples(graph, runs, seed))
    values = np.asarray(samples, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(samples) > 1 else 0.0
    total = graph.total_length()
    if total <= 0:
        raise RuntimeError("L_total이 0이라 걸음 수를 정규화할 수 없습니다.")
    return TremauxStats(
        mean=mean,
        std=std,
        minimum=min(samples),
        maximum=max(samples),
        norm=mean / float(total),
        samples=samples,
    )


def tremaux_samples(graph: SpatialGraph, runs: int, seed: int) -> list[int]:
    """한 난수 생성기를 이어서 쓰며 실행마다의 걸음 수를 반환한다."""
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs는 1 이상의 정수여야 합니다.")
    rng = np.random.default_rng(seed)
    incident = _incident_edges(graph)
    return [run_tremaux(graph, rng, incident) for _ in range(runs)]


def run_tremaux(
    graph: SpatialGraph,
    rng: np.random.Generator,
    incident: dict[str, tuple[int, ...]] | None = None,
) -> int:
    """트레모를 한 번 실행하고 걸음 수(지난 간선 길이의 합)를 반환한다.

    들어온 간선을 따라 되돌아갈지는 방문 기록을 갱신하기 전에 판단한다.
    """
    if incident is None:
        incident = _incident_edges(graph)
    marks = [0] * graph.n_edges
    visited: set[str] = set()
    node = graph.entry
    incoming: int | None = None
    steps = 0
    traversals = 0
    limit = 2 * graph.n_edges
    while True:
        if node == graph.exit:
            return steps
        chosen = _choose_edge(marks, incident[node], incoming, node in visited, rng)
        visited.add(node)
        marks[chosen] += 1
        assert marks[chosen] <= 2, "간선 표시가 2를 넘었습니다."
        left, right, length = graph.edges[chosen]
        steps += length
        traversals += 1
        if traversals > limit:
            raise RuntimeError(
                f"트레모 탐색이 간선을 {limit}번(2×|E|) 넘게 지났습니다. 무한루프 버그입니다."
            )
        if node == left:
            node = right
        elif node == right:
            node = left
        else:
            raise RuntimeError(f"간선 {chosen}이 정점 {node}에 연결되어 있지 않습니다.")
        incoming = chosen


def _incident_edges(graph: SpatialGraph) -> dict[str, tuple[int, ...]]:
    """정점별 간선 번호를 이웃 id 순으로 모아 둔다."""
    return {
        node_id: tuple(edge_index for _neighbor, _length, edge_index in graph.adj[node_id])
        for node_id in graph.ids
    }


def _choose_edge(
    marks: list[int],
    incident: tuple[int, ...],
    incoming: int | None,
    already_visited: bool,
    rng: np.random.Generator,
) -> int:
    """되돌아가기, 새 길, 돌아 나가기 순서로 다음 간선을 고른다."""
    if incoming is not None and already_visited and marks[incoming] == 1:
        return incoming
    unmarked = [edge_index for edge_index in incident if marks[edge_index] == 0]
    if unmarked:
        return unmarked[int(rng.integers(0, len(unmarked)))]
    once = [edge_index for edge_index in incident if marks[edge_index] == 1]
    if not once:
        raise RuntimeError("선택할 간선이 없습니다. 모든 간선의 표시가 2입니다.")
    return once[int(rng.integers(0, len(once)))]
