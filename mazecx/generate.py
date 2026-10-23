"""무작위 깊이우선탐색으로 시험용 미로를 만든다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mazecx.graph import Coord

_DELTAS_3D = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
_DELTAS_2D = _DELTAS_3D[:4]
_PLUS_3D = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_PLUS_2D = ((1, 0, 0), (0, 1, 0))


def generate_dataset(
    out: str | Path,
    sizes: list[int],
    loops: list[float],
    count: int,
    seed: int,
    dim2d: bool = False,
) -> list[Path]:
    """(크기, 고리 확률) 조합마다 count개의 미로 JSON을 저장한다."""
    _check_generation(sizes, loops, count, seed)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    n_width = max(2, len(str(max(sizes))))
    index_width = max(3, len(str(count - 1)))
    written: list[Path] = []
    running = 0
    for size in sizes:
        for loop_p in loops:
            for index in range(count):
                file_seed = seed + running
                document = build_maze(size, float(loop_p), file_seed, dim2d)
                name = maze_filename(dim2d, size, float(loop_p), index, n_width, index_width)
                path = destination / name
                path.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written.append(path)
                running += 1
    return written


def build_maze(size: int, loop_p: float, seed: int, dim2d: bool) -> dict:
    """한 개의 미로 형식 문서를 만든다. meta.seed는 이 미로에 쓴 시드다."""
    rng = np.random.default_rng(seed)
    passages = _carve_maze(size, loop_p, rng, dim2d)
    exit_cell = [size - 1, size - 1, 0 if dim2d else size - 1]
    return {
        "dim": size,
        "passages": passages,
        "entry": [0, 0, 0],
        "exit": exit_cell,
        "meta": {
            "generator": "dfs",
            "n": size,
            "loop_p": loop_p,
            "seed": seed,
        },
    }


def maze_filename(
    dim2d: bool,
    size: int,
    loop_p: float,
    index: int,
    n_width: int,
    index_width: int,
) -> str:
    """정렬해도 크기·확률·번호 순서가 유지되는 파일 이름을 만든다."""
    kind = "2d" if dim2d else "3d"
    return (
        f"maze_{kind}_n{size:0{n_width}d}_p{format_probability(loop_p)}_"
        f"{index:0{index_width}d}.json"
    )


def format_probability(loop_p: float) -> str:
    """확률을 파일 이름에 넣을 고정 자릿수 문자열로 만든다.

    자릿수를 고정해야 0, 0.05, 0.15를 이름순으로 정렬해도 크기순이 유지된다.
    """
    return f"{float(loop_p):.6f}"


def _carve_maze(
    size: int,
    loop_p: float,
    rng: np.random.Generator,
    dim2d: bool,
) -> list[list[list[int]]]:
    """완전 미로를 만든 뒤, 남아 있는 벽은 확률 loop_p로 뚫는다."""
    deltas = _DELTAS_2D if dim2d else _DELTAS_3D
    start = (0, 0, 0)
    visited = {start}
    stack = [start]
    raw: list[tuple[Coord, Coord]] = []
    while stack:
        current = stack[-1]
        options = [
            nxt
            for nxt in _neighbors(current, deltas, size, dim2d)
            if nxt not in visited
        ]
        if not options:
            stack.pop()
            continue
        chosen = options[int(rng.integers(0, len(options)))]
        raw.append((current, chosen))
        visited.add(chosen)
        stack.append(chosen)
    expected = size * size if dim2d else size * size * size
    if len(visited) != expected:
        raise RuntimeError("미로 생성 중 모든 칸을 방문하지 못했습니다.")

    existing = {_undirected(left, right) for left, right in raw}
    plus = _PLUS_2D if dim2d else _PLUS_3D
    zs = (0,) if dim2d else range(size)
    for z in zs:
        for y in range(size):
            for x in range(size):
                cell = (x, y, z)
                for dx, dy, dz in plus:
                    nxt = (x + dx, y + dy, z + dz)
                    if not _inside(nxt, size, dim2d):
                        continue
                    key = _undirected(cell, nxt)
                    if key in existing:
                        continue
                    if float(rng.random()) < loop_p:
                        raw.append((cell, nxt))
                        existing.add(key)
    return [
        [list(left), list(right)]
        for left, right in sorted(existing)
    ]


def _neighbors(
    cell: Coord,
    deltas: tuple[tuple[int, int, int], ...],
    size: int,
    dim2d: bool,
) -> list[Coord]:
    """격자 안의 이웃 칸을 고정된 방향 순서로 반환한다."""
    found = []
    for dx, dy, dz in deltas:
        nxt = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
        if _inside(nxt, size, dim2d):
            found.append(nxt)
    return found


def _inside(cell: Coord, size: int, dim2d: bool) -> bool:
    """칸이 격자 안에 있는지 확인한다. 2차원은 z=0만 허용한다."""
    x, y, z = cell
    if dim2d:
        return 0 <= x < size and 0 <= y < size and z == 0
    return 0 <= x < size and 0 <= y < size and 0 <= z < size


def _undirected(left: Coord, right: Coord) -> tuple[Coord, Coord]:
    """통로 양 끝을 좌표 순으로 정렬해 중복을 없앤다."""
    if left <= right:
        return left, right
    return right, left


def _check_generation(sizes: list[int], loops: list[float], count: int, seed: int) -> None:
    """생성 인자가 격자 크기·확률·반복·시드 조건에 맞는지 확인한다."""
    if not sizes:
        raise ValueError("sizes에는 격자 크기가 하나 이상 있어야 합니다.")
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError(f"격자 크기는 2 이상의 정수여야 합니다. 받은 값: {size!r}")
    if not loops:
        raise ValueError("loops에는 확률이 하나 이상 있어야 합니다.")
    for loop_p in loops:
        if isinstance(loop_p, bool) or not isinstance(loop_p, (int, float)):
            raise ValueError(f"고리 확률은 0 이상 1 이하의 숫자여야 합니다. 받은 값: {loop_p!r}")
        if not 0.0 <= float(loop_p) <= 1.0:
            raise ValueError(f"고리 확률은 0 이상 1 이하여야 합니다. 받은 값: {loop_p!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count는 1 이상의 정수여야 합니다.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed는 정수여야 합니다.")
