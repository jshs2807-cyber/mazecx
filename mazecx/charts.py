"""비교 화면용 plotly 그림. 제목과 축 이름은 한국어다."""

from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mazecx.compare import METRIC_TREMAUX
from mazecx.graph import SpatialGraph

_GROUP_COLORS = {
    "3D": "#1f77b4",
    "2D": "#ff7f0e",
}
_EXTRA_COLORS = (
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


def group_color(group: str) -> str:
    """그룹 이름마다 항상 같은 색을 돌려준다."""
    if group in _GROUP_COLORS:
        return _GROUP_COLORS[group]
    digest = 0
    for char in group:
        digest = (digest * 33 + ord(char)) % len(_EXTRA_COLORS)
    return _EXTRA_COLORS[digest]


def comparison_bars(
    names: list[str],
    groups: list[str],
    metrics: list[tuple[str, list[float]]],
    tremaux_std: list[float] | None = None,
) -> go.Figure:
    """지표 4개를 2행 2열 막대로 그린다. 각 칸은 자기 세로축을 쓴다."""
    figure = make_subplots(rows=2, cols=2, subplot_titles=[label for label, _values in metrics])
    legend_shown: set[str] = set()
    for slot, (label, values) in enumerate(metrics):
        row = slot // 2 + 1
        column = slot % 2 + 1
        for group in _group_order(groups):
            picked = [index for index, name in enumerate(groups) if name == group]
            errors = None
            if label == METRIC_TREMAUX and tremaux_std is not None:
                errors = {"type": "data", "array": [tremaux_std[index] for index in picked], "visible": True}
            show = group not in legend_shown
            legend_shown.add(group)
            figure.add_trace(
                go.Bar(
                    x=[names[index] for index in picked],
                    y=[values[index] for index in picked],
                    name=group,
                    legendgroup=group,
                    showlegend=show,
                    marker_color=group_color(group),
                    text=[_bar_text(values[index]) for index in picked],
                    textposition="outside",
                    error_y=errors,
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(title_text="미로", categoryorder="array", categoryarray=names, row=row, col=column)
        figure.update_yaxes(title_text=label, row=row, col=column)
    _polish(figure, "지표별 비교", 780)
    return figure


def ratio_bars(
    names: list[str],
    groups: list[str],
    metrics: list[tuple[str, list[float]]],
) -> go.Figure:
    """지표별 최댓값 비율을 묶음 막대로 그린다. 색은 지표다."""
    del groups
    figure = go.Figure()
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for index, (label, values) in enumerate(metrics):
        figure.add_trace(
            go.Bar(
                x=names,
                y=values,
                name=label,
                marker_color=palette[index % len(palette)],
                hovertemplate="%{x}<br>" + label + " 비율 %{y:.3f}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group")
    figure.update_xaxes(title_text="미로", categoryorder="array", categoryarray=names)
    figure.update_yaxes(title_text="최댓값 대비 비율", range=[0, 1.05])
    _polish(figure, "한눈에 비교", 520, "지표")
    return figure


def agreement_heatmap(matrix: list[list[float]], labels: list[str]) -> go.Figure:
    """스피어만 순위 상관을 히트맵으로 그린다. NaN 칸은 계산 불가다."""
    size = len(labels)
    text = [[_heat_text(matrix[row][column]) for column in range(size)] for row in range(size)]
    numeric = [
        [None if isinstance(matrix[row][column], float) and math.isnan(matrix[row][column]) else matrix[row][column] for column in range(size)]
        for row in range(size)
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=[[0] * size for _ in range(size)],
            x=labels,
            y=labels,
            colorscale=[[0, "#bdbdbd"], [1, "#bdbdbd"]],
            showscale=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=numeric,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar={"title": "스피어만 ρ"},
            hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="지표")
    figure.update_yaxes(title_text="지표", autorange="reversed")
    _polish(figure, "순위 일치도", 560)
    return figure


def metric_scatter(
    names: list[str],
    groups: list[str],
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    """두 지표의 산점도를 그린다. 점 색은 그룹이고 옆에 이름을 붙인다."""
    figure = go.Figure()
    for group in _group_order(groups):
        picked = [index for index, name in enumerate(groups) if name == group]
        figure.add_trace(
            go.Scatter(
                x=[x_values[index] for index in picked],
                y=[y_values[index] for index in picked],
                mode="markers+text",
                name=group,
                text=[names[index] for index in picked],
                textposition="top right",
                marker={"color": group_color(group), "size": 12},
                hovertemplate="%{text}<br>" + x_label + " %{x}<br>" + y_label + " %{y}<extra></extra>",
            )
        )
    figure.update_xaxes(title_text=x_label, type="log" if log_x else "linear")
    figure.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    _polish(figure, "두 지표 산점도", 520)
    return figure


def steps_boxes(names: list[str], groups: list[str], steps: list[list[int]]) -> go.Figure:
    """미로별 트레모 걸음 수 분포를 상자그림으로 그린다."""
    figure = go.Figure()
    for group in _group_order(groups):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=group,
                marker={"color": group_color(group), "size": 10},
                showlegend=True,
            )
        )
    for index, name in enumerate(names):
        group = groups[index]
        figure.add_trace(
            go.Box(
                y=steps[index],
                name=name,
                marker_color=group_color(group),
                showlegend=False,
                hovertemplate=name + "<br>걸음 수 %{y}<extra>" + group + "</extra>",
            )
        )
    figure.update_layout(boxmode="group")
    figure.update_xaxes(title_text="미로", categoryorder="array", categoryarray=names)
    figure.update_yaxes(title_text="걸음 수")
    _polish(figure, "트레모 걸음 수 분포", 520)
    return figure


def maze_figure(graph: SpatialGraph) -> go.Figure:
    """정점과 간선을 3차원으로 그린다. 입구는 초록, 출구는 빨강, 3축 갈림길은 주황이다."""
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for left, right, _length in graph.edges:
        start = graph.nodes[left]
        end = graph.nodes[right]
        edge_x.extend((start[0], end[0], None))
        edge_y.extend((start[1], end[1], None))
        edge_z.extend((start[2], end[2], None))
    normal: list[str] = []
    junctions: list[str] = []
    for node_id in graph.ids:
        if node_id in (graph.entry, graph.exit):
            continue
        if graph.axis_count(node_id) >= 3:
            junctions.append(node_id)
        else:
            normal.append(node_id)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            name="간선",
            line={"color": "#7f7f7f", "width": 4},
            hoverinfo="skip",
        )
    )
    _add_nodes(figure, graph, normal, "정점", "#1f77b4")
    _add_nodes(figure, graph, junctions, "3축 갈림길", "#ff7f0e")
    _add_nodes(figure, graph, [graph.entry], "입구", "#2ca02c")
    _add_nodes(figure, graph, [graph.exit], "출구", "#d62728")
    figure.update_layout(
        title="미로 그래프",
        paper_bgcolor="white",
        legend_title="구분",
        scene={
            "xaxis_title": "x 좌표",
            "yaxis_title": "y 좌표",
            "zaxis_title": "z 좌표",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        height=640,
        font={"size": 12},
    )
    return figure


def _add_nodes(figure: go.Figure, graph: SpatialGraph, node_ids: list[str], name: str, color: str) -> None:
    """같은 색으로 묶을 정점들을 그림에 추가한다."""
    if not node_ids:
        return
    figure.add_trace(
        go.Scatter3d(
            x=[graph.nodes[node_id][0] for node_id in node_ids],
            y=[graph.nodes[node_id][1] for node_id in node_ids],
            z=[graph.nodes[node_id][2] for node_id in node_ids],
            mode="markers",
            name=name,
            text=node_ids,
            marker={"color": color, "size": 7},
            hovertemplate="%{text}<br>(%{x}, %{y}, %{z})<extra>" + name + "</extra>",
        )
    )


def _group_order(groups: list[str]) -> list[str]:
    """처음 나온 순서를 유지한 그룹 목록을 만든다."""
    seen: list[str] = []
    for group in groups:
        if group not in seen:
            seen.append(group)
    return seen


def _bar_text(value: float) -> str:
    """10 이상이면 소수 첫째 자리, 그 미만이면 셋째 자리로 적는다."""
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _heat_text(value: float) -> str:
    """상관 칸의 글자다. NaN이면 계산 불가다."""
    if isinstance(value, float) and math.isnan(value):
        return "계산 불가"
    return f"{value:.2f}"


def _polish(figure: go.Figure, title: str, height: int, legend_title: str = "구분") -> None:
    """흰 배경과 한국어 제목, 12pt 글자를 맞춘다."""
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 12},
        height=height,
        legend_title=legend_title,
    )
