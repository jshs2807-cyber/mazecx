"""여러 미로를 나란히 비교하는 표와 순위."""

from __future__ import annotations

from mazecx.complexity import AXIS_WEIGHTS
from mazecx.stats import spearman

METRIC_TREMAUX = "트레모 평균 걸음 수"
METRIC_GREEDY = "탐욕 라우팅 복잡도"
METRIC_CGEO = "우리 복잡도 기하평균"
METRIC_CARITH = "우리 복잡도 산술평균"
COMPARE_METRICS = (METRIC_TREMAUX, METRIC_GREEDY, METRIC_CGEO, METRIC_CARITH)


def rank_desc(values: list[float]) -> list[int]:
    """큰 값일수록 1위다. 동점은 그중 가장 높은 순위를 함께 받는다."""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        shared = cursor + 1
        for index in order[cursor:end]:
            ranks[index] = shared
        cursor = end
    return ranks


def normalize_by_max(values: list[float]) -> list[float]:
    """최댓값으로 나눈 비율을 돌려준다. 최댓값이 0이면 모두 0이다."""
    if not values:
        return []
    peak = max(values)
    if peak == 0:
        return [0.0] * len(values)
    return [float(value) / float(peak) for value in values]


def build_table(
    results: list[dict],
    names: list[str],
    groups: list[str],
    axis_variant: str,
) -> list[dict]:
    """비교 표의 행을 만든다. 순위는 값이 큰 미로가 앞선다."""
    if axis_variant not in AXIS_WEIGHTS:
        raise ValueError(f"축 변형은 {', '.join(AXIS_WEIGHTS)} 중 하나여야 합니다.")
    if not (len(results) == len(names) == len(groups)):
        raise ValueError("결과, 이름, 그룹의 개수가 같아야 합니다.")
    columns = _metric_columns(results, axis_variant)
    rank_columns = {name: rank_desc(values) for name, values in columns.items()}
    rows: list[dict] = []
    for index, result in enumerate(results):
        row: dict = {
            "이름": names[index],
            "그룹": groups[index],
            "정점 수": result["n_nodes"],
            "간선 수": result["n_edges"],
            "고리 수": result["n_loops"],
            "전체 통로 길이": result["L_total"],
            "최단거리": result["shortest_path"],
        }
        for name, values in columns.items():
            row[name] = values[index]
        for name, ranks in rank_columns.items():
            row[f"{name} 순위"] = ranks[index]
        rows.append(row)
    return rows


def rank_agreement(rows: list[dict], metrics: list[str]) -> list[list[float]]:
    """지표 열끼리의 스피어만 상관 행렬을 만든다. 대각선은 1이고 NaN을 허용한다."""
    columns = [[float(row[metric]) for row in rows] for metric in metrics]
    size = len(metrics)
    matrix = [[0.0] * size for _ in range(size)]
    for left in range(size):
        for right in range(size):
            if left == right:
                matrix[left][right] = 1.0
            else:
                matrix[left][right] = float(spearman(columns[left], columns[right]))
    return matrix


def _metric_columns(results: list[dict], axis_variant: str) -> dict[str, list[float]]:
    """비교 지표 4개의 값 목록을 고정된 순서로 꺼낸다."""
    return {
        METRIC_TREMAUX: [float(result["tremaux_mean"]) for result in results],
        METRIC_GREEDY: [float(result["greedy_difficulty"]) for result in results],
        METRIC_CGEO: [float(result[f"C_geo_{axis_variant}"]) for result in results],
        METRIC_CARITH: [float(result[f"C_arith_{axis_variant}"]) for result in results],
    }
