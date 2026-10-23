"""입력 검증과 미로 변환 테스트."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mazecx.io import load_document, load_path, parse_meta

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def load_clean(name: str):
    """예제 파일을 읽는다. 예상 밖 경고가 나면 실패한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return load_path(EX / name)


def test_t1_graph_shape() -> None:
    """곧은 통로 하나는 정점 2개, 길이 5인 간선 하나다."""
    graph = load_clean("t1.json")
    assert graph.entry == "A"
    assert graph.exit == "E"
    assert graph.n_nodes == 2
    assert graph.n_edges == 1
    assert graph.edges[0][2] == 5
    assert graph.total_length() == 5
    assert graph.nodes["A"] == (0, 0, 0)
    assert graph.nodes["E"] == (5, 0, 0)


def test_t2_axis_counts() -> None:
    """갈림길 B의 축 개수는 2, 나머지는 1이다."""
    graph = load_clean("t2.json")
    assert graph.axis_count("A") == 1
    assert graph.axis_count("B") == 2
    assert graph.axis_count("C") == 1
    assert graph.axis_count("D") == 1
    assert graph.degree("B") == 3
    assert graph.degree("D") == 1
    assert sorted(length for _u, _v, length in graph.edges) == [1, 2, 3]


def test_t4_axis_count_three() -> None:
    """3축이 만나는 B의 축 개수는 3이다."""
    graph = load_clean("t4.json")
    assert graph.axis_count("A") == 1
    assert graph.axis_count("B") == 3
    assert graph.axis_count("U") == 1
    assert graph.axis_count("C") == 1


def test_t5_maze_compression() -> None:
    """곧은 칸은 빠지고 꺾이는 점만 정점이 되며 간선 길이는 2다."""
    graph = load_clean("t5.json")
    assert set(graph.nodes) == {"0,0,0", "2,0,0", "2,2,0"}
    assert "1,0,0" not in graph.nodes
    assert "2,1,0" not in graph.nodes
    assert graph.entry == "0,0,0"
    assert graph.exit == "2,2,0"
    assert graph.n_edges == 2
    assert sorted(length for _u, _v, length in graph.edges) == [2, 2]
    endpoints = {frozenset((u, v)) for u, v, _length in graph.edges}
    assert endpoints == {
        frozenset(("0,0,0", "2,0,0")),
        frozenset(("2,0,0", "2,2,0")),
    }
    assert graph.nodes["2,0,0"] == (2, 0, 0)


def test_t6_diagonal() -> None:
    """대각선 간선은 ValueError."""
    with pytest.raises(ValueError, match="평행"):
        load_path(EX / "t6_diagonal.json")


def test_t6_same_coord() -> None:
    """같은 좌표를 가진 두 정점은 ValueError."""
    with pytest.raises(ValueError, match="좌표"):
        load_path(EX / "t6_same_coord.json")


def test_t6_entry_equals_exit() -> None:
    """entry와 exit가 같으면 ValueError."""
    with pytest.raises(ValueError, match="서로 달라"):
        load_path(EX / "t6_same_entry_exit.json")


def test_t6_no_path() -> None:
    """entry에서 exit로 갈 수 없으면 ValueError."""
    with pytest.raises(ValueError, match="경로"):
        load_path(EX / "t6_no_path.json")


def test_t6_duplicate_edge() -> None:
    """같은 두 정점 사이 간선이 두 번이면 ValueError."""
    with pytest.raises(ValueError, match="중복"):
        load_path(EX / "t6_duplicate_edge.json")


def test_t6_maze_not_neighbor() -> None:
    """이웃이 아닌 두 칸을 잇는 통로는 ValueError."""
    with pytest.raises(ValueError, match="이웃"):
        load_path(EX / "t6_maze_not_neighbor.json")


def test_2d_coordinates_pad_z() -> None:
    """2차원 좌표 [x, y]는 z=0으로 채운다."""
    graph = load_document(
        {
            "nodes": {"A": [0, 0], "E": [5, 0]},
            "edges": [["A", "E"]],
            "entry": "A",
            "exit": "E",
        }
    )
    assert graph.nodes["A"] == (0, 0, 0)
    assert graph.nodes["E"] == (5, 0, 0)
    assert graph.total_length() == 5


def test_matching_length_is_accepted() -> None:
    """택시거리와 같은 length는 그대로 둔다."""
    graph = load_document(
        {
            "nodes": {"A": [0, 0, 0], "E": [5, 0, 0]},
            "edges": [["A", "E", 5]],
            "entry": "A",
            "exit": "E",
        }
    )
    assert graph.edges[0][2] == 5


def test_length_mismatch() -> None:
    """주어진 length가 택시거리와 다르면 ValueError."""
    with pytest.raises(ValueError, match="택시거리"):
        load_document(
            {
                "nodes": {"A": [0, 0, 0], "E": [5, 0, 0]},
                "edges": [["A", "E", 4]],
                "entry": "A",
                "exit": "E",
            }
        )


def test_self_loop() -> None:
    """자기 자신으로 가는 간선은 ValueError."""
    with pytest.raises(ValueError, match="자기 자신"):
        load_document(
            {
                "nodes": {"A": [0, 0, 0], "E": [1, 0, 0]},
                "edges": [["A", "A"]],
                "entry": "A",
                "exit": "E",
            }
        )


def test_disconnected_node_is_removed_with_warning() -> None:
    """입구와 끊긴 정점과 그 간선은 경고 후 제거한다."""
    with pytest.warns(UserWarning, match="연결되지 않은"):
        graph = load_document(
            {
                "nodes": {
                    "A": [0, 0, 0],
                    "E": [1, 0, 0],
                    "Y": [5, 5, 0],
                    "Z": [6, 5, 0],
                },
                "edges": [["A", "E"], ["Y", "Z"]],
                "entry": "A",
                "exit": "E",
            }
        )
    assert set(graph.nodes) == {"A", "E"}
    assert graph.n_edges == 1
    assert graph.total_length() == 1


def test_maze_straight_only_compression() -> None:
    """일직선 통로의 중간 칸은 정점이 아니다."""
    graph = load_document(
        {
            "dim": 3,
            "passages": [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
            ],
            "entry": [0, 0, 0],
            "exit": [2, 0, 0],
        }
    )
    assert set(graph.nodes) == {"0,0,0", "2,0,0"}
    assert graph.n_edges == 1
    assert graph.edges[0][2] == 2


def test_entry_splits_straight_corridor() -> None:
    """곧은 통로 위의 entry는 정점으로 남아 통로를 둘로 나눈다."""
    graph = load_document(
        {
            "dim": 5,
            "passages": [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
                [[2, 0, 0], [3, 0, 0]],
            ],
            "entry": [1, 0, 0],
            "exit": [3, 0, 0],
        }
    )
    assert set(graph.nodes) == {"0,0,0", "1,0,0", "3,0,0"}
    endpoints = {frozenset((u, v)): length for u, v, length in graph.edges}
    assert endpoints[frozenset(("0,0,0", "1,0,0"))] == 1
    assert endpoints[frozenset(("1,0,0", "3,0,0"))] == 2


def test_duplicate_passage_warns_once() -> None:
    """순서만 바뀐 중복 통로는 경고하고 한 번만 쓴다."""
    with pytest.warns(UserWarning, match="중복된 통로"):
        graph = load_document(
            {
                "dim": 3,
                "passages": [
                    [[0, 0, 0], [1, 0, 0]],
                    [[1, 0, 0], [0, 0, 0]],
                ],
                "entry": [0, 0, 0],
                "exit": [1, 0, 0],
            }
        )
    assert graph.n_edges == 1
    assert graph.edges[0][2] == 1


def test_maze_entry_must_appear() -> None:
    """통로에 없는 entry는 ValueError."""
    with pytest.raises(ValueError, match="entry"):
        load_document(
            {
                "dim": 3,
                "passages": [[[0, 0, 0], [1, 0, 0]]],
                "entry": [2, 2, 0],
                "exit": [0, 0, 0],
            }
        )


def test_maze_junction_compression() -> None:
    """갈림길은 정점으로 남고, 곧은 중간 칸만 합쳐진다."""
    graph = load_document(
        {
            "dim": 4,
            "passages": [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
                [[2, 0, 0], [2, 1, 0]],
                [[2, 0, 0], [3, 0, 0]],
            ],
            "entry": [0, 0, 0],
            "exit": [2, 1, 0],
        }
    )
    assert set(graph.nodes) == {"0,0,0", "2,0,0", "2,1,0", "3,0,0"}
    lengths = {frozenset((u, v)): length for u, v, length in graph.edges}
    assert lengths[frozenset(("0,0,0", "2,0,0"))] == 2
    assert lengths[frozenset(("2,0,0", "2,1,0"))] == 1
    assert lengths[frozenset(("2,0,0", "3,0,0"))] == 1


def test_meta_survives_maze_conversion_and_sets_dim() -> None:
    """meta는 변환 뒤에도 남고, 격자 크기는 좌표 폭이 아니라 dim이다."""
    graph = load_document(
        {
            "dim": 5,
            "passages": [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [2, 0, 0]],
            ],
            "entry": [0, 0, 0],
            "exit": [2, 0, 0],
            "meta": {"name": "short", "trial": 3},
        }
    )
    assert graph.meta == {"name": "short", "trial": 3}
    assert graph.grid_size() == 5
    assert graph.is_3d() is False


def test_meta_rejects_non_scalar() -> None:
    """meta 값이 목록이면 ValueError."""
    with pytest.raises(ValueError, match="meta"):
        parse_meta({"meta": {"tags": ["a"]}})


def test_graph_grid_size_uses_coordinate_span() -> None:
    """그래프 형식의 격자 크기는 축별 좌표 폭의 최댓값이다."""
    graph = load_clean("t2.json")
    assert graph.grid_size() == 5
    assert graph.n_loops() == 0
    assert graph.meta == {}


def test_coordinate_edges_load_as_a_maze() -> None:
    """칸 좌표로 적힌 edges도 미로로 읽어 곧은 통로를 압축한다."""
    import json

    raw = json.loads((EX / "external_dfs.json").read_text(encoding="utf-8"))
    raw["stats"] = {"cells": 27, "loops": 0}
    raw["simplified"] = {"vertices": [], "edges": []}
    graph = load_document(raw)
    assert graph.n_nodes == 21
    assert graph.n_edges == 20
    assert graph.n_loops() == 0
    assert graph.grid_size() == 3
    assert graph.is_3d() is True
    assert graph.shortest_path() == 6
    assert graph.total_length() == 26
    assert graph.entry == "0,0,0"
    assert graph.exit == "2,2,2"
    assert graph.meta["id"] == "3d_n03_dfs_loop00_000"
    assert graph.meta["seed"] == 30300000
    assert graph.meta["loop_ratio"] == 0.0


def test_shortest_distance_t2() -> None:
    """간선 길이 기준 최단거리는 출구에서 한 번에 구한다."""
    graph = load_clean("t2.json")
    dist = graph.shortest_distances_from(graph.exit)
    assert dist["C"] == 0
    assert dist["B"] == 3
    assert dist["A"] == 5
    assert dist["D"] == 4
