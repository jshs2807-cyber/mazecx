"""미로·그래프 입력을 읽고 검증해 SpatialGraph로 만든다."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from mazecx.graph import Coord, SpatialGraph, differing_axes, taxi_distance

RawEdge = tuple[str, str, int | None]


def load_path(path: str | Path) -> SpatialGraph:
    """JSON 파일을 읽어 SpatialGraph로 만든다."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"파일을 읽을 수 없습니다: {file_path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 형식이 잘못되었습니다: {file_path}: {exc}") from exc
    return load_document(data)


def load_document(data: dict) -> SpatialGraph:
    """그래프 또는 미로 JSON 객체를 검증해 SpatialGraph로 만든다."""
    if not isinstance(data, dict):
        raise ValueError("입력 JSON의 최상위는 객체여야 합니다.")
    meta = parse_meta(data)
    if "dim" in data:
        nodes, edges, entry, exit_id = _maze_components(data)
        source_dim = int(data["dim"])
    else:
        nodes, edges, entry, exit_id = _graph_components(data)
        source_dim = None
    return _finalize(nodes, edges, entry, exit_id, meta, source_dim)


def parse_meta(data: dict) -> dict[str, str | int | float]:
    """선택 항목 meta를 읽는다. 값은 문자열과 숫자만 허용한다."""
    if "meta" not in data:
        return {}
    raw = data["meta"]
    if not isinstance(raw, dict):
        raise ValueError("meta는 문자열·숫자 값만 있는 객체여야 합니다.")
    meta: dict[str, str | int | float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("meta의 키는 문자열이어야 합니다.")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(
                f"meta '{key}' 값은 문자열이나 숫자여야 합니다. 받은 값: {value!r}"
            )
        meta[key] = value
    return meta


def _graph_components(data: dict) -> tuple[dict[str, Coord], list[RawEdge], str, str]:
    """그래프 형식 JSON에서 정점, 간선, 입구, 출구를 꺼낸다."""
    for key in ("nodes", "edges", "entry", "exit"):
        if key not in data:
            raise ValueError(f"그래프 형식에는 '{key}' 키가 필요합니다.")
    if not isinstance(data["nodes"], dict):
        raise ValueError("nodes는 정점 id를 좌표로 대응시킨 객체여야 합니다.")
    if not isinstance(data["edges"], list):
        raise ValueError("edges는 간선 목록이어야 합니다.")
    entry = _node_id(data["entry"], "entry")
    exit_id = _node_id(data["exit"], "exit")
    nodes = {
        _node_id(node_id, "정점 id"): _normalize_coord(coord, f"정점 '{node_id}'")
        for node_id, coord in data["nodes"].items()
    }
    edges: list[RawEdge] = []
    for raw in data["edges"]:
        edges.append(_parse_raw_edge(raw))
    return nodes, edges, entry, exit_id


def _maze_components(data: dict) -> tuple[dict[str, Coord], list[RawEdge], str, str]:
    """미로 형식을 곧은 통로를 압축한 그래프 요소로 변환한다."""
    for key in ("dim", "passages", "entry", "exit"):
        if key not in data:
            raise ValueError(f"미로 형식에는 '{key}' 키가 필요합니다.")
    dim = data["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim은 1 이상의 정수여야 합니다. 받은 값: {dim!r}")
    if not isinstance(data["passages"], list):
        raise ValueError("passages는 통로 목록이어야 합니다.")
    entry = _normalize_coord(data["entry"], "entry")
    exit_cell = _normalize_coord(data["exit"], "exit")
    _require_in_grid(entry, dim, "entry")
    _require_in_grid(exit_cell, dim, "exit")

    passages = _unique_passages(data["passages"], dim)
    cells: set[Coord] = set()
    adjacency: dict[Coord, set[Coord]] = {}
    for left, right in passages:
        cells.add(left)
        cells.add(right)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    if entry not in cells:
        raise ValueError(f"entry 칸 {list(entry)}가 통로에 등장하지 않습니다.")
    if exit_cell not in cells:
        raise ValueError(f"exit 칸 {list(exit_cell)}가 통로에 등장하지 않습니다.")

    vertices = {
        cell
        for cell in cells
        if _is_vertex(cell, adjacency[cell], entry, exit_cell)
    }
    edges = _compress_corridors(adjacency, vertices)
    nodes = {_cell_id(cell): cell for cell in sorted(vertices)}
    entry_id = _cell_id(entry)
    exit_id = _cell_id(exit_cell)
    assert entry_id in nodes and exit_id in nodes, "변환 버그: entry 또는 exit가 정점이 아닙니다."
    return nodes, [(u, v, length) for u, v, length in edges], entry_id, exit_id


def _finalize(
    nodes: dict[str, Coord],
    raw_edges: list[RawEdge],
    entry: str,
    exit_id: str,
    meta: dict[str, str | int | float],
    source_dim: int | None,
) -> SpatialGraph:
    """3-1 검증을 적용하고, 입구와 끊긴 정점은 경고 후 제거한다."""
    if entry not in nodes:
        raise ValueError(f"entry '{entry}'가 nodes에 없습니다.")
    if exit_id not in nodes:
        raise ValueError(f"exit '{exit_id}'가 nodes에 없습니다.")
    if entry == exit_id:
        raise ValueError(f"entry와 exit는 서로 달라야 합니다: '{entry}'.")

    seen_at: dict[Coord, str] = {}
    for node_id, coord in nodes.items():
        if coord in seen_at:
            raise ValueError(
                f"정점 '{seen_at[coord]}'와 '{node_id}'의 좌표가 같습니다: {list(coord)}."
            )
        seen_at[coord] = node_id

    edges: list[tuple[str, str, int]] = []
    seen_pairs: set[frozenset[str]] = set()
    for u, v, given_length in raw_edges:
        if u not in nodes or v not in nodes:
            missing = u if u not in nodes else v
            raise ValueError(f"간선 [{u}, {v}]의 끝점 '{missing}'가 nodes에 없습니다.")
        if u == v:
            raise ValueError(f"자기 자신으로 가는 간선은 허용되지 않습니다: '{u}'.")
        pair = frozenset((u, v))
        if pair in seen_pairs:
            raise ValueError(f"정점 '{u}'와 '{v}' 사이 간선이 중복되었습니다.")
        seen_pairs.add(pair)
        diffs = differing_axes(nodes[u], nodes[v])
        if len(diffs) != 1:
            raise ValueError(
                f"간선 {u}-{v}는 한 축에만 평행해야 합니다. "
                f"좌표 {list(nodes[u])}와 {list(nodes[v])}."
            )
        distance = taxi_distance(nodes[u], nodes[v])
        if given_length is not None and given_length != distance:
            raise ValueError(
                f"간선 {u}-{v}의 length {given_length}이 택시거리 {distance}와 다릅니다."
            )
        edges.append((u, v, distance))

    graph = SpatialGraph(nodes, edges, entry, exit_id)
    graph.meta = dict(meta)
    graph.source_dim = source_dim
    return _keep_entry_component(graph)


def _keep_entry_component(graph: SpatialGraph) -> SpatialGraph:
    """입구 연결 성분만 남긴다. 출구가 없으면 오류, 다른 정점은 경고 후 제거한다."""
    component = graph.component_from(graph.entry)
    if graph.exit not in component:
        raise ValueError(
            f"entry '{graph.entry}'에서 exit '{graph.exit}'로 가는 경로가 없습니다."
        )
    removed = [node_id for node_id in graph.ids if node_id not in component]
    if not removed:
        return graph
    warnings.warn(
        "entry와 연결되지 않은 정점을 제거하고 계속합니다: " + ", ".join(removed),
        UserWarning,
        stacklevel=3,
    )
    return graph.induced_subgraph(component)


def _normalize_coord(raw: object, context: str) -> Coord:
    """[x, y] 또는 [x, y, z]를 정수 좌표 (x, y, z)로 맞춘다. 2차원은 z=0."""
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError(f"{context}: 좌표는 [x, y] 또는 [x, y, z]여야 합니다. 받은 값: {raw!r}")
    numbers: list[int] = []
    for component in raw:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{context}: 좌표 성분은 정수여야 합니다. 받은 값: {raw!r}")
        if isinstance(component, float) and not component.is_integer():
            raise ValueError(f"{context}: 좌표 성분은 정수여야 합니다. 받은 값: {raw!r}")
        numbers.append(int(component))
    if len(numbers) == 2:
        numbers.append(0)
    return (numbers[0], numbers[1], numbers[2])


def _node_id(raw: object, context: str) -> str:
    """정점 id가 문자열인지 확인하고 그대로 반환한다."""
    if not isinstance(raw, str):
        raise ValueError(f"{context}는 문자열이어야 합니다. 받은 값: {raw!r}")
    return raw


def _parse_raw_edge(raw: object) -> RawEdge:
    """간선 [u, v] 또는 [u, v, length]를 해석한다. 길이가 없으면 None."""
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError(f"간선은 [u, v] 또는 [u, v, length]여야 합니다. 받은 값: {raw!r}")
    u = _node_id(raw[0], "간선 끝점")
    v = _node_id(raw[1], "간선 끝점")
    if len(raw) == 2:
        return (u, v, None)
    length = raw[2]
    if isinstance(length, bool) or not isinstance(length, (int, float)):
        raise ValueError(f"간선 [{u}, {v}]의 length는 정수여야 합니다. 받은 값: {length!r}")
    if isinstance(length, float) and not length.is_integer():
        raise ValueError(f"간선 [{u}, {v}]의 length는 정수여야 합니다. 받은 값: {length!r}")
    return (u, v, int(length))


def _require_in_grid(coord: Coord, dim: int, context: str) -> None:
    """칸 좌표의 모든 성분이 0 이상 dim 미만인지 확인한다."""
    if any(component < 0 or component >= dim for component in coord):
        raise ValueError(
            f"{context} 칸 좌표 {list(coord)}가 0~{dim - 1} 범위를 벗어났습니다."
        )


def _unique_passages(raw_passages: list, dim: int) -> list[tuple[Coord, Coord]]:
    """통로를 검사하고, 순서만 다른 중복은 경고한 뒤 한 번만 남긴다."""
    unique: list[tuple[Coord, Coord]] = []
    seen: set[tuple[Coord, Coord]] = set()
    for index, raw in enumerate(raw_passages):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"통로는 칸 두 개의 쌍이어야 합니다. 받은 값: {raw!r}")
        left = _normalize_coord(raw[0], f"통로 {index}의 첫 칸")
        right = _normalize_coord(raw[1], f"통로 {index}의 둘째 칸")
        _require_in_grid(left, dim, "통로")
        _require_in_grid(right, dim, "통로")
        distance = taxi_distance(left, right)
        if distance != 1:
            raise ValueError(
                f"통로 {list(left)}-{list(right)}의 택시거리가 1이 아닙니다 (거리 {distance}). "
                "이웃한 칸만 통로로 이을 수 있습니다."
            )
        key = (left, right) if left <= right else (right, left)
        if key in seen:
            warnings.warn(
                f"중복된 통로를 한 번만 사용합니다: {list(left)}-{list(right)}.",
                UserWarning,
                stacklevel=3,
            )
            continue
        seen.add(key)
        unique.append((left, right))
    return unique


def _is_straight(cell: Coord, left: Coord, right: Coord) -> bool:
    """cell을 중심으로 두 이웃이 같은 축의 반대 방향인지 확인한다."""
    delta_left = tuple(left[axis] - cell[axis] for axis in range(3))
    delta_right = tuple(right[axis] - cell[axis] for axis in range(3))
    return all(a + b == 0 for a, b in zip(delta_left, delta_right)) and any(
        component != 0 for component in delta_left
    )


def _is_vertex(cell: Coord, neighbors: set[Coord], entry: Coord, exit_cell: Coord) -> bool:
    """입구, 출구, 막다른 길, 갈림길, 꺾이는 점을 정점으로 둔다."""
    if cell == entry or cell == exit_cell:
        return True
    if len(neighbors) != 2:
        return True
    left, right = tuple(neighbors)
    return not _is_straight(cell, left, right)


def _cell_id(cell: Coord) -> str:
    """칸 좌표를 정점 id 문자열 'x,y,z'로 만든다."""
    return f"{cell[0]},{cell[1]},{cell[2]}"


def _compress_corridors(
    adjacency: dict[Coord, set[Coord]],
    vertices: set[Coord],
) -> list[tuple[str, str, int]]:
    """곧은 통로 칸을 없애고 양옆 정점을 잇는 간선으로 합친다."""
    used: set[tuple[Coord, Coord]] = set()
    edges: list[tuple[str, str, int]] = []
    for start in sorted(vertices):
        for neighbor in sorted(adjacency[start]):
            first_key = _passage_key(start, neighbor)
            if first_key in used:
                continue
            prev = start
            curr = neighbor
            length = 1
            used.add(first_key)
            while curr not in vertices:
                nxts = [cell for cell in adjacency[curr] if cell != prev]
                if len(nxts) != 1:
                    raise RuntimeError(f"곧은 통로 칸 {list(curr)}에서 다음 칸을 정할 수 없습니다.")
                nxt = nxts[0]
                prev, curr = curr, nxt
                length += 1
                key = _passage_key(prev, curr)
                if key in used:
                    raise RuntimeError("통로 압축 중 같은 통로를 다시 지났습니다.")
                used.add(key)
                if length > len(adjacency):
                    raise RuntimeError("통로 압축 중 무한루프가 의심됩니다.")
            if curr == start:
                raise RuntimeError(
                    f"변환 버그: 정점 {list(start)}에서 시작한 통로가 자기 자신으로 돌아옵니다."
                )
            diffs = differing_axes(start, curr)
            assert len(diffs) == 1, (
                f"변환 버그: 간선 {list(start)}-{list(curr)}가 한 축에 평행하지 않습니다."
            )
            assert length == taxi_distance(start, curr), (
                f"변환 버그: 간선 길이 {length}가 택시거리와 다릅니다."
            )
            edges.append((_cell_id(start), _cell_id(curr), length))
    return edges


def _passage_key(left: Coord, right: Coord) -> tuple[Coord, Coord]:
    """무방향 통로를 순서에 무관한 키로 만든다."""
    return (left, right) if left <= right else (right, left)
