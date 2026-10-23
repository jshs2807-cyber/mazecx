"""흡수시간과 C_geo, C_arith 테스트."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from mazecx.complexity import compute_complexity, vertex_weight
from mazecx.io import load_document, load_path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def near(expected: float):
    """절대오차 1e-9로 비교한다."""
    return pytest.approx(expected, abs=1e-9, rel=0)


def load_clean(name: str):
    """예제 파일을 읽는다. 예상 밖 경고가 나면 실패한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return load_path(EX / name)


def test_vertex_weight_modes() -> None:
    """축 가중치 네 가지의 정의."""
    assert vertex_weight(1, "standard") == 1
    assert vertex_weight(2, "standard") == 1
    assert vertex_weight(3, "standard") == 2
    assert vertex_weight(3, "none") == 1.0
    assert vertex_weight(3, "a") == 3.0
    assert vertex_weight(2, "inv") == near(0.5)
    with pytest.raises(ValueError):
        vertex_weight(1, "unknown")


def test_t1_straight_corridor() -> None:
    """곧은 통로는 축 가중치와 상관없이 복잡도 1이다."""
    graph = load_clean("t1.json")
    for mode in ("standard", "none"):
        result = compute_complexity(graph, mode)
        assert result.tau_length == near(1)
        assert result.tau_degree == near(1)
        assert result.c_geo == near(1)
        assert result.c_arith == near(1)


def test_t2_junction_and_dead_end() -> None:
    """갈림길 하나와 막다른 길 하나의 검산값."""
    graph = load_clean("t2.json")
    for mode in ("standard", "none"):
        result = compute_complexity(graph, mode)
        assert result.tau_length == near(11)
        assert result.tau_degree == near(6)
        assert result.c_geo == near(8.124038404635961)
        assert result.c_arith == near(8.5)


def test_t2_does_not_depend_on_node_order() -> None:
    """정점 순서가 바뀌어도 입구 성분을 T에서 다시 찾는다."""
    data = json.loads((EX / "t2.json").read_text(encoding="utf-8"))
    data["nodes"] = dict(reversed(list(data["nodes"].items())))
    graph = load_document(data)
    result = compute_complexity(graph, "standard")
    assert result.tau_length == near(11)
    assert result.tau_degree == near(6)


def test_t3_greedy_trap() -> None:
    """탐욕 라우팅이 막히는 미로의 흡수시간."""
    graph = load_clean("t3.json")
    for mode in ("standard", "none"):
        result = compute_complexity(graph, mode)
        assert result.tau_length == near(22)
        assert result.tau_degree == near(18)
        assert result.c_geo == near(19.8997487421324)
        assert result.c_arith == near(20)


def test_t4_three_axis_weight() -> None:
    """3축 갈림길은 standard와 none의 흡수시간이 다르다."""
    graph = load_clean("t4.json")
    assert graph.axis_count("B") == 3
    standard = compute_complexity(graph, "standard")
    assert standard.tau_length == near(12)
    assert standard.tau_degree == near(9)
    assert standard.c_geo == near(10.392304845413264)
    assert standard.c_arith == near(10.5)
    none = compute_complexity(graph, "none")
    assert none.tau_length == near(8)
    assert none.tau_degree == near(6)
    assert none.c_geo == near(6.928203230275509)
    assert none.c_arith == near(7)


def test_t4_axis_weight_a_and_inv() -> None:
    """T4의 a·inv 가중치 검산값. B의 축 개수는 3이다."""
    graph = load_clean("t4.json")
    assert graph.is_3d() is True
    weighted = compute_complexity(graph, "a")
    assert weighted.tau_length == near(16)
    assert weighted.tau_degree == near(12)
    assert weighted.c_geo == near(13.856406460551016)
    assert weighted.c_arith == near(14)
    inverse = compute_complexity(graph, "inv")
    assert inverse.tau_length == near(5.333333333333333)
    assert inverse.tau_degree == near(4)
    assert inverse.c_geo == near(4.618802153517006)
    assert inverse.c_arith == near(4.666666666666667)


def test_t3_shortest_path_and_loops() -> None:
    """T3는 나무가 되어 고리가 없고 최단거리는 10이다."""
    graph = load_clean("t3.json")
    assert graph.n_loops() == 0
    assert graph.shortest_path() == 10
    assert graph.is_3d() is False


def test_t8_square_loop() -> None:
    """정사각 고리는 고리 1개다. 꼭짓점의 축이 모두 2개라 standard·none은 4다.

    a는 w=2라 8, inv는 w=1/2라 2이다. 길이 가중과 차수 가중은 같다.
    """
    graph = load_clean("t8.json")
    assert graph.n_loops() == 1
    assert graph.is_3d() is False
    for mode in ("standard", "none"):
        result = compute_complexity(graph, mode)
        assert result.tau_length == near(4)
        assert result.tau_degree == near(4)
        assert result.c_geo == near(4)
        assert result.c_arith == near(4)
    doubled = compute_complexity(graph, "a")
    assert doubled.tau_length == near(8)
    assert doubled.tau_degree == near(8)
    assert doubled.c_geo == near(8)
    assert doubled.c_arith == near(8)
    halved = compute_complexity(graph, "inv")
    assert halved.tau_length == near(2)
    assert halved.tau_degree == near(2)
    assert halved.c_geo == near(2)
    assert halved.c_arith == near(2)


def test_2d_coordinates_match_t1() -> None:
    """z를 0으로 채운 2차원 좌표도 같은 복잡도를 준다."""
    graph = load_document(
        {
            "nodes": {"A": [0, 0], "E": [5, 0]},
            "edges": [["A", "E", 5]],
            "entry": "A",
            "exit": "E",
        }
    )
    result = compute_complexity(graph, "standard")
    assert result.tau_length == near(1)
    assert result.c_geo == near(1)


def test_disconnected_node_is_ignored() -> None:
    """입구와 끊긴 정점은 흡수시간 계산에 들어가지 않는다."""
    with pytest.warns(UserWarning):
        graph = load_document(
            {
                "nodes": {
                    "A": [0, 0, 0],
                    "E": [5, 0, 0],
                    "Z": [9, 9, 9],
                },
                "edges": [["A", "E"]],
                "entry": "A",
                "exit": "E",
            }
        )
    result = compute_complexity(graph, "standard")
    assert set(graph.nodes) == {"A", "E"}
    assert result.tau_length == near(1)
    assert result.tau_degree == near(1)


def test_solver_uses_solve_not_inverse() -> None:
    """역행렬 대신 연립방정식을 풀고, scipy·networkx를 쓰지 않는다."""
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "mazecx").glob("*.py")
    )
    assert "linalg.inv" not in source
    assert "linalg.solve" in source
    assert "import networkx" not in source
    assert "import scipy" not in source
    assert "import pandas" not in source
    assert "import jinja2" not in source
    assert "import random" not in source
