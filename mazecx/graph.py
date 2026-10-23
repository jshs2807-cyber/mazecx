"""공간 네트워크 그래프와 최단거리."""

from __future__ import annotations

import heapq
from typing import Iterable

Coord = tuple[int, int, int]
Edge = tuple[str, str, int]


def coordinate_span(nodes: dict[str, Coord]) -> int:
    """각 축의 (최댓값 − 최솟값 + 1) 가운데 가장 큰 값을 반환한다."""
    if not nodes:
        return 0
    coords = list(nodes.values())
    spans = []
    for axis in range(3):
        values = [coord[axis] for coord in coords]
        spans.append(max(values) - min(values) + 1)
    return max(spans)


def taxi_distance(p: Coord, q: Coord) -> int:
    """두 좌표의 택시거리 |Δx|+|Δy|+|Δz|를 반환한다."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) + abs(p[2] - q[2])


def differing_axes(p: Coord, q: Coord) -> list[int]:
    """두 좌표가 서로 다른 축의 인덱스 목록을 반환한다. 0=x, 1=y, 2=z."""
    return [axis for axis in range(3) if p[axis] != q[axis]]


class SpatialGraph:
    """좌표가 있는 무방향 공간 그래프.

    간선은 입력 순서를 유지하고, 각 정점의 인접 목록은 이웃 id 순으로 정렬한다.
    """

    def __init__(
        self,
        nodes: dict[str, Coord],
        edges: Iterable[Edge],
        entry: str,
        exit: str,
    ) -> None:
        """정점, 간선, 입구, 출구로 그래프를 만든다."""
        self.nodes: dict[str, Coord] = {
            node_id: (int(coord[0]), int(coord[1]), int(coord[2]))
            for node_id, coord in nodes.items()
        }
        self.ids: tuple[str, ...] = tuple(self.nodes.keys())
        self.edges: tuple[Edge, ...] = tuple(
            (u, v, int(length)) for u, v, length in edges
        )
        self.entry = entry
        self.exit = exit
        self.meta: dict[str, str | int | float] = {}
        self.source_dim: int | None = None
        if self.entry not in self.nodes or self.exit not in self.nodes:
            raise ValueError("entry와 exit는 nodes에 있어야 합니다.")
        for u, v, length in self.edges:
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"간선 {u}-{v}의 끝점이 nodes에 없습니다.")
            if length <= 0:
                raise ValueError(f"간선 {u}-{v}의 길이는 양수여야 합니다.")
        self.adj: dict[str, list[tuple[str, int, int]]] = {
            node_id: [] for node_id in self.ids
        }
        for index, (u, v, length) in enumerate(self.edges):
            self.adj[u].append((v, length, index))
            self.adj[v].append((u, length, index))
        for node_id in self.ids:
            self.adj[node_id].sort(key=lambda item: (item[0], item[2]))

    def __repr__(self) -> str:
        """그래프 크기를 짧게 보여 준다."""
        return (
            f"SpatialGraph(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"entry={self.entry!r}, exit={self.exit!r})"
        )

    @property
    def n_nodes(self) -> int:
        """정점 수를 반환한다."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """간선 수를 반환한다."""
        return len(self.edges)

    def degree(self, node_id: str) -> int:
        """정점 node_id에 연결된 간선 수 deg(v)를 반환한다."""
        return len(self.adj[node_id])

    def axis_count(self, node_id: str) -> int:
        """정점 node_id에 닿은 간선이 쓰는 서로 다른 축의 개수 a(v)를 반환한다."""
        origin = self.nodes[node_id]
        axes: set[int] = set()
        for neighbor, _length, _index in self.adj[node_id]:
            diffs = differing_axes(origin, self.nodes[neighbor])
            if len(diffs) != 1:
                raise ValueError(
                    f"간선 {node_id}-{neighbor}가 한 축에만 평행하지 않습니다."
                )
            axes.add(diffs[0])
        return len(axes)

    def total_length(self) -> int:
        """모든 간선 길이의 합 L_total을 반환한다."""
        return sum(length for _u, _v, length in self.edges)

    def n_loops(self) -> int:
        """이 연결 성분의 고리 수 |E| − |V| + 1 을 반환한다."""
        return self.n_edges - self.n_nodes + 1

    def is_3d(self) -> bool:
        """정점 z 좌표가 두 가지 이상이면 True를 반환한다."""
        return len({coord[2] for coord in self.nodes.values()}) >= 2

    def grid_size(self) -> int:
        """미로 형식이면 입력 dim, 그래프 형식이면 축별 좌표 폭의 최댓값을 반환한다."""
        if self.source_dim is not None:
            return self.source_dim
        return coordinate_span(self.nodes)

    def shortest_path(self) -> int:
        """entry에서 exit까지 간선 길이 기준 최단거리를 반환한다."""
        distances = self.shortest_distances_from(self.entry)
        if self.exit not in distances:
            raise RuntimeError(
                f"entry '{self.entry}'에서 exit '{self.exit}'까지의 최단거리가 없습니다."
            )
        return distances[self.exit]

    def component_from(self, start: str) -> set[str]:
        """start와 같은 연결 성분에 속한 정점 id 집합을 반환한다."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            for neighbor, _length, _index in self.adj[node_id]:
                if neighbor not in seen:
                    stack.append(neighbor)
        return seen

    def induced_subgraph(self, keep: set[str]) -> SpatialGraph:
        """keep에 속한 정점과 그 사이의 간선만 남긴 그래프를 반환한다."""
        nodes = {node_id: self.nodes[node_id] for node_id in self.ids if node_id in keep}
        edges = [
            (u, v, length)
            for u, v, length in self.edges
            if u in keep and v in keep
        ]
        child = SpatialGraph(nodes, edges, self.entry, self.exit)
        child.meta = dict(self.meta)
        child.source_dim = self.source_dim
        return child

    def shortest_distances_from(self, source: str) -> dict[str, int]:
        """간선 길이를 가중치로 source에서 각 정점까지의 최단거리를 구한다."""
        if source not in self.nodes:
            raise KeyError(source)
        dist: dict[str, int] = {source: 0}
        heap: list[tuple[int, str]] = [(0, source)]
        while heap:
            current, node_id = heapq.heappop(heap)
            if current != dist[node_id]:
                continue
            for neighbor, length, _index in self.adj[node_id]:
                candidate = current + length
                if neighbor not in dist or candidate < dist[neighbor]:
                    dist[neighbor] = candidate
                    heapq.heappush(heap, (candidate, neighbor))
        return dist
