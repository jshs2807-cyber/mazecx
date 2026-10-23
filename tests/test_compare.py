"""여러 미로 비교 표와 순위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from mazecx.cli import analyze_graph
from mazecx.compare import (
    COMPARE_METRICS,
    METRIC_CGEO,
    METRIC_GREEDY,
    METRIC_TREMAUX,
    build_table,
    normalize_by_max,
    rank_agreement,
    rank_desc,
)
from mazecx.io import load_path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
COMPARE_SOURCE = ROOT / "mazecx" / "compare.py"


def test_rank_desc_shares_the_best_tied_rank() -> None:
    """큰 값이 1위이고, 동점은 공동 순위이며 다음 순위는 건너뛴다."""
    assert rank_desc([5, 3, 3, 1]) == [1, 2, 2, 4]


def test_normalize_all_zero_does_not_divide() -> None:
    """최댓값이 0이면 나누지 않고 모두 0을 돌려준다."""
    assert normalize_by_max([0, 0, 0]) == [0, 0, 0]


def test_normalize_divides_by_the_peak() -> None:
    """최댓값으로 나눈 비율이 1을 넘지 않는다."""
    assert normalize_by_max([2, 4, 1]) == [
        pytest.approx(0.5, abs=1e-9),
        pytest.approx(1.0, abs=1e-9),
        pytest.approx(0.25, abs=1e-9),
    ]


def test_compare_module_does_not_use_pandas() -> None:
    """비교 계산은 pandas에 기대지 않는다."""
    source = COMPARE_SOURCE.read_text(encoding="utf-8")
    assert "import pandas" not in source
    assert "import pd" not in source


@pytest.fixture(scope="module")
def standard_trio() -> tuple[list[dict], list[dict]]:
    """T2, T3, T4를 같은 반복과 시드로 계산한 비교 표다."""
    results = []
    names = []
    for filename in ("t2.json", "t3.json", "t4.json"):
        graph = load_path(EX / filename)
        names.append(Path(filename).stem)
        results.append(analyze_graph(graph, filename, 20000, 0))
    groups = ["3D" if result["is_3d"] else "2D" for result in results]
    return build_table(results, names, groups, "standard"), results


def test_steps_all_matches_runs_and_mean(standard_trio: tuple[list[dict], list[dict]]) -> None:
    """걸음 수 목록의 길이는 반복 횟수이고, 그 평균이 트레모 평균이다."""
    _table, results = standard_trio
    for result in results:
        steps = result["steps_all"]
        assert len(steps) == result["runs"] == 20000
        assert sum(steps) / len(steps) == pytest.approx(result["tremaux_mean"], abs=1e-9)


def test_example_metrics_ranks_and_agreement(standard_trio: tuple[list[dict], list[dict]]) -> None:
    """T2, T3, T4의 기하평균, 트레모, 탐욕 순위와 비율, 순위 일치도가 검산값과 같다."""
    table, _results = standard_trio
    by_name = {row["이름"]: row for row in table}
    assert by_name["t2"][METRIC_CGEO] == pytest.approx(8.124038404635961, abs=1e-9)
    assert by_name["t3"][METRIC_CGEO] == pytest.approx(19.8997487421324, abs=1e-9)
    assert by_name["t4"][METRIC_CGEO] == pytest.approx(10.392304845413264, abs=1e-9)
    assert by_name["t3"][f"{METRIC_CGEO} 순위"] == 1
    assert by_name["t4"][f"{METRIC_CGEO} 순위"] == 2
    assert by_name["t2"][f"{METRIC_CGEO} 순위"] == 3
    assert by_name["t2"][METRIC_TREMAUX] == pytest.approx(6, abs=0.1)
    assert by_name["t4"][METRIC_TREMAUX] == pytest.approx(7, abs=0.1)
    assert by_name["t3"][METRIC_TREMAUX] == pytest.approx(12, abs=0.1)
    assert by_name["t3"][f"{METRIC_TREMAUX} 순위"] == 1
    assert by_name["t4"][f"{METRIC_TREMAUX} 순위"] == 2
    assert by_name["t2"][f"{METRIC_TREMAUX} 순위"] == 3
    assert by_name["t2"][METRIC_GREEDY] == pytest.approx(0, abs=1e-9)
    assert by_name["t3"][METRIC_GREEDY] == pytest.approx(0.6, abs=1e-9)
    assert by_name["t4"][METRIC_GREEDY] == pytest.approx(0, abs=1e-9)
    assert by_name["t3"][f"{METRIC_GREEDY} 순위"] == 1
    assert by_name["t2"][f"{METRIC_GREEDY} 순위"] == 2
    assert by_name["t4"][f"{METRIC_GREEDY} 순위"] == 2
    ratios = normalize_by_max([row[METRIC_CGEO] for row in table])
    assert ratios == [
        pytest.approx(0.408248290463863, abs=1e-9),
        pytest.approx(1.0, abs=1e-9),
        pytest.approx(0.5222329678670935, abs=1e-9),
    ]
    matrix = rank_agreement(table, list(COMPARE_METRICS))
    geo = COMPARE_METRICS.index(METRIC_CGEO)
    assert matrix[geo][COMPARE_METRICS.index(METRIC_TREMAUX)] == pytest.approx(1.0, abs=1e-9)
    assert matrix[geo][COMPARE_METRICS.index(METRIC_GREEDY)] == pytest.approx(0.8660254037844387, abs=1e-9)
    assert matrix[geo][geo] == 1.0
