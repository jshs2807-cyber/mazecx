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


def split_documents(text: str, label: str) -> list[tuple[str, str]]:
    """JSON 문자열을 미로 하나당 (이름, JSON 문자열)로 나눈다.

    최상위가 객체면 파일 하나다. 최상위가 배열이면 원소마다 미로 하나이고,
    이름은 id가 있으면 그것을 쓴다.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 형식이 잘못되었습니다: {exc}") from exc
    if isinstance(data, dict):
        return [(label, text)]
    if not isinstance(data, list):
        raise ValueError("입력 JSON의 최상위는 객체이거나 미로 배열이어야 합니다.")
    if not data:
        raise ValueError("묶음 JSON에 미로가 없습니다.")
    stem = Path(label).stem or label
    used: set[str] = set()
    named: list[tuple[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"묶음 JSON의 {index}번째 항목은 객체여야 합니다.")
        raw_name = item.get("id")
        base = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else f"{stem}_{index}"
        name = base
        number = 2
        while name in used:
            name = f"{base}_{number}"
            number += 1
        used.add(name)
        named.append((name, json.dumps(item, ensure_ascii=False)))
    return named


def load_document(data: dict) -> SpatialGraph:
    """그래프 또는 미로 JSON 객체를 검증해 SpatialGraph로 만든다."""
    if not isinstance(data, dict):
        raise ValueError("입력 JSON의 최상위는 객체여야 합니다.")
    if "dim" not in data and "nodes" not in data and _has_coordinate_edges(data):
        data = _coordinate_edges_document(data)
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


def _has_coordinate_edges(data: dict) -> bool:
    """edges가 정점 이름이 아니라 [x, y, z] 칸 좌표로 이어져 있는지 본다."""
    edges = data.get("edges")
    if not isinstance(edges, list) or not edges:
        return False
    raw = edges[0]
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        return False
    return _is_coord_like(raw[0]) and _is_coord_like(raw[1])


def _is_coord_like(raw: object) -> bool:
    """값이 [x, y] 또는 [x, y, z] 숫자 좌표인지 확인한다."""
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw)


def _coordinate_edges_document(data: dict) -> dict:
    """칸 좌표 간선을 미로 형식 또는 그래프 형식으로 바꾼다."""
    parsed: list[tuple[Coord, Coord, int | None]] = []
    for index, raw in enumerate(data["edges"]):
        if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
            raise ValueError(f"간선은 칸 둘과 선택적 길이여야 합니다. 받은 값: {raw!r}")
        if not _is_coord_like(raw[0]) or not _is_coord_like(raw[1]):
            raise ValueError(f"간선 {index}의 끝은 좌표여야 합니다. 받은 값: {raw!r}")
        left = _normalize_coord(raw[0], f"간선 {index}의 첫 칸")
        right = _normalize_coord(raw[1], f"간선 {index}의 둘째 칸")
        length = None
        if len(raw) == 3:
            length = _whole_number(raw[2], f"간선 {index}의 길이")
        parsed.append((left, right, length))
    entry = _normalize_coord(data.get("entry"), "entry")
    exit_cell = _normalize_coord(data.get("exit"), "exit")
    dim = _size_dim(data, [entry, exit_cell, *(cell for pair in parsed for cell in pair[:2])])
    meta = _scalar_meta(data)
    unit = all(
        taxi_distance(left, right) == 1 and (length is None or length == 1)
        for left, right, length in parsed
    )
    if unit:
        return {
            "dim": dim,
            "passages": [[list(left), list(right)] for left, right, _length in parsed],
            "entry": list(entry),
            "exit": list(exit_cell),
            "meta": meta,
        }
    nodes: dict[str, list[int]] = {}
    for cell in (entry, exit_cell, *(cell for pair in parsed for cell in pair[:2])):
        nodes[_cell_id(cell)] = list(cell)
    edges: list[list[object]] = []
    for left, right, length in parsed:
        edge: list[object] = [_cell_id(left), _cell_id(right)]
        if length is not None:
            edge.append(length)
        edges.append(edge)
    return {
        "nodes": nodes,
        "edges": edges,
        "entry": _cell_id(entry),
        "exit": _cell_id(exit_cell),
        "meta": meta,
    }


def _size_dim(data: dict, coords: list[Coord]) -> int:
    """size가 있으면 그 격자 한 변을, 없으면 좌표 범위로 dim을 정한다."""
    size = data.get("size")
    if isinstance(size, list) and size:
        limits: list[int] = []
        for item in size:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ValueError(f"size의 각 값은 1 이상의 정수여야 합니다. 받은 값: {size!r}")
            limits.append(item)
        while len(limits) < 3:
            limits.append(1)
        for coord in coords:
            for axis, limit in enumerate(limits[:3]):
                if coord[axis] < 0 or coord[axis] >= limit:
                    raise ValueError(
                        f"좌표 {list(coord)}가 size {limits[:3]} 밖입니다."
                    )
        return max(limits[:3])
    if not coords:
        raise ValueError("size 또는 칸 좌표가 있어야 격자 크기를 정할 수 있습니다.")
    return max(max(coord) for coord in coords) + 1


def _scalar_meta(data: dict) -> dict[str, str | int | float]:
    """좌표 간선 파일의 문자열·숫자 항목을 meta로 옮긴다."""
    meta: dict[str, str | int | float] = {}
    existing = data.get("meta")
    if isinstance(existing, dict):
        meta.update(parse_meta({"meta": existing}))
    for key in ("id", "algorithm", "loop_ratio", "seed", "dimension"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        meta[key] = value
    return meta


def _whole_number(raw: object, context: str) -> int:
    """정수 또는 정수로 떨어지는 실수를 정수로 바꾼다."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{context}는 정수여야 합니다. 받은 값: {raw!r}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{context}는 정수여야 합니다. 받은 값: {raw!r}")
    return int(raw)


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
