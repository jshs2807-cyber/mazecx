"""시험용 미로 생성기 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mazecx.generate import generate_dataset
from mazecx.io import load_path


def _cells_and_passages(path: Path) -> tuple[set[tuple[int, ...]], int]:
    """저장된 미로에서 칸 집합과 통로 수를 센다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cells: set[tuple[int, ...]] = set()
    for left, right in data["passages"]:
        cells.add(tuple(left))
        cells.add(tuple(right))
    return cells, len(data["passages"])


def test_perfect_maze_is_a_tree(tmp_path: Path) -> None:
    """고리 확률 0이면 모든 칸이 이어지고 변환 그래프의 고리 수는 0이다."""
    written = generate_dataset(tmp_path, sizes=[3], loops=[0], count=4, seed=0)
    assert len(written) == 4
    for path in written:
        cells, passage_count = _cells_and_passages(path)
        assert len(cells) == 27
        assert passage_count == len(cells) - 1
        graph = load_path(path)
        assert graph.n_loops() == 0
        assert graph.grid_size() == 3
        assert graph.meta["generator"] == "dfs"
        assert graph.meta["n"] == 3
        assert graph.meta["loop_p"] == 0


def test_same_seed_writes_identical_files(tmp_path: Path) -> None:
    """같은 시드로 두 번 만들면 파일 내용이 완전히 같다."""
    first = generate_dataset(tmp_path / "a", sizes=[3, 4], loops=[0, 0.15], count=2, seed=11)
    second = generate_dataset(tmp_path / "b", sizes=[3, 4], loops=[0, 0.15], count=2, seed=11)
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.name for path in first] == sorted(path.name for path in first)
    for left, right in zip(first, second):
        assert left.read_bytes() == right.read_bytes()
    other = generate_dataset(tmp_path / "c", sizes=[3, 4], loops=[0, 0.15], count=2, seed=12)
    assert first[0].read_bytes() != other[0].read_bytes()


def test_filename_keeps_sort_order(tmp_path: Path) -> None:
    """파일 이름을 정렬하면 크기, 확률, 번호 순서가 유지된다."""
    written = generate_dataset(
        tmp_path,
        sizes=[4, 5],
        loops=[0, 0.05, 0.15],
        count=3,
        seed=0,
    )
    names = [path.name for path in written]
    assert names == sorted(names)
    assert names[0] == "maze_3d_n04_p0.000000_000.json"
    assert "maze_3d_n05_p0.050000_002.json" in names


def test_dim2d_uses_plane(tmp_path: Path) -> None:
    """2차원 미로는 z=0 평면이고 출구도 z=0이다."""
    written = generate_dataset(tmp_path, sizes=[3], loops=[0], count=1, seed=2, dim2d=True)
    data = json.loads(written[0].read_text(encoding="utf-8"))
    assert written[0].name.startswith("maze_2d_")
    assert data["exit"] == [2, 2, 0]
    cells, passage_count = _cells_and_passages(written[0])
    assert len(cells) == 9
    assert passage_count == 8
    assert all(cell[2] == 0 for cell in cells)
    graph = load_path(written[0])
    assert graph.is_3d() is False
    assert graph.n_loops() == 0


def test_rejects_bad_probability() -> None:
    """확률 범위 밖은 ValueError."""
    with pytest.raises(ValueError):
        generate_dataset("unused", sizes=[3], loops=[1.5], count=1, seed=0)
