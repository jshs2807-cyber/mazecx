"""비교 그래프가 plotly Figure를 돌려주는지 확인한다."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from mazecx.charts import (
    agreement_heatmap,
    comparison_bars,
    group_color,
    maze_figure,
    metric_scatter,
    ratio_bars,
    steps_boxes,
)
from mazecx.cli import analyze_graph
from mazecx.compare import COMPARE_METRICS, METRIC_CGEO, METRIC_TREMAUX, build_table, normalize_by_max, rank_agreement
from mazecx.io import load_path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def test_group_color_is_stable() -> None:
    """같은 그룹은 항상 같은 색이고, 2D와 3D는 서로 다르다."""
    assert group_color("3D") == group_color("3D")
    assert group_color("2D") == group_color("2D")
    assert group_color("3D") != group_color("2D")
    assert group_color("실험") == group_color("실험")


def test_chart_functions_return_figures_for_examples() -> None:
    """T2, T3, T4 결과로 비교 그래프 함수가 예외 없이 Figure를 만든다."""
    names = ["t2", "t3", "t4"]
    results = [analyze_graph(load_path(EX / f"{name}.json"), name, 10, 0) for name in names]
    groups = ["3D" if result["is_3d"] else "2D" for result in results]
    table = build_table(results, names, groups, "standard")
    metrics = [(metric, [float(row[metric]) for row in table]) for metric in COMPARE_METRICS]
    figures = [
        comparison_bars(names, groups, metrics, [float(result["tremaux_std"]) for result in results]),
        ratio_bars(names, groups, [(label, normalize_by_max(values)) for label, values in metrics]),
        agreement_heatmap(rank_agreement(table, list(COMPARE_METRICS)), list(COMPARE_METRICS)),
        agreement_heatmap([[1.0, float("nan")], [float("nan"), 1.0]], ["가", "나"]),
        metric_scatter(
            names,
            groups,
            [float(row[METRIC_CGEO]) for row in table],
            [float(row[METRIC_TREMAUX]) for row in table],
            METRIC_CGEO,
            METRIC_TREMAUX,
        ),
        steps_boxes(names, groups, [list(result["steps_all"]) for result in results]),
        maze_figure(load_path(EX / "t2.json")),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
