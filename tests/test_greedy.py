"""탐욕 라우팅 테스트."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mazecx.greedy import greedy_route, greedy_statistics, trace_greedy
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


def stats(name: str):
    """예제 그래프의 탐욕 통계. GR_score 불일치는 오류로 본다."""
    graph = load_clean(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return greedy_statistics(graph)


def test_t1_success() -> None:
    """곧은 통로는 모든 출발점에서 성공한다."""
    result = stats("t1.json")
    assert result.success_rate == near(1.0)
    assert result.difficulty == near(0.0)
    assert result.entry_success is True
    assert result.gr_score == near(1.0)


def test_t2_success_and_gr_score() -> None:
    """갈림길이 있어도 탐욕 경로는 모두 최단 경로다."""
    result = stats("t2.json")
    assert result.success_rate == near(1.0)
    assert result.gr_score == near(1.0)
    assert result.entry_success is True


def test_t3_greedy_gets_stuck() -> None:
    """출발점 다섯 개 중 F, G만 성공한다."""
    graph = load_clean("t3.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = greedy_statistics(graph)
    assert result.success_rate == near(0.4)
    assert result.difficulty == near(0.6)
    assert result.entry_success is False
    assert result.gr_score == near(0.4)
    succeeded = [
        node_id
        for node_id in graph.ids
        if node_id != graph.exit and greedy_route(graph, node_id)[0]
    ]
    assert succeeded == ["F", "G"]
    assert greedy_route(graph, "F")[1] == 5
    assert greedy_route(graph, "G")[1] == 2


def test_t4_success() -> None:
    """3축 갈림길도 출구 쪽으로만 가면 모두 성공한다."""
    result = stats("t4.json")
    assert result.success_rate == near(1.0)
    assert result.gr_score == near(1.0)
    assert result.entry_success is True


def test_t8_square_chooses_b_on_tie() -> None:
    """정사각 고리의 입구에서는 B와 D가 동점이라 사전순으로 B를 고른다."""
    graph = load_clean("t8.json")
    result = stats("t8.json")
    assert result.success_rate == near(1.0)
    assert result.gr_score == near(1.0)
    assert result.entry_success is True
    succeeded, length, visits = trace_greedy(graph, "A")
    assert succeeded is True
    assert length == 4
    assert visits == ("A", "B", "C")


def test_tie_break_uses_node_id() -> None:
    """거리와 간선 길이가 같으면 정점 id 사전순으로 고르고, 막다른 쪽이면 실패한다."""
    graph = load_document(
        {
            "nodes": {
                "S": [2, 2, 0],
                "B": [0, 2, 0],
                "M": [2, 0, 0],
                "E": [0, 0, 0],
            },
            "edges": [["S", "B"], ["S", "M"], ["M", "E"]],
            "entry": "S",
            "exit": "E",
        }
    )
    assert greedy_route(graph, "S") == (False, 2)
    assert greedy_route(graph, "M")[0] is True
