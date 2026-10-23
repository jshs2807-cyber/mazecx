"""결과 CSV를 비교하고 HTML 보고서를 만든다."""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mazecx.stats import bootstrap_ci, paired_diff_ci, pearson, spearman

CANDIDATES = (
    "C_geo_standard",
    "C_arith_standard",
    "C_geo_none",
    "C_arith_none",
    "C_geo_a",
    "C_arith_a",
    "C_geo_inv",
    "C_arith_inv",
)
BASELINES = ("n_nodes", "L_total", "shortest_path")
TARGETS = ("tremaux_mean", "greedy_difficulty")
PREDICTORS = CANDIDATES + BASELINES
AXIS_MODES = ("standard", "none", "a", "inv")
MIN_ROWS = 8
KST = timezone(timedelta(hours=9))
FIGURES = (
    "fig1_tremaux.svg",
    "fig2_greedy.svg",
    "fig3_correlations.svg",
    "fig4_differences.svg",
)
_HANGUL = re.compile(r"[가-힣]")


@dataclass
class CorrRow:
    """한 예측 변수와 검증 기준의 상관."""

    predictor: str
    target: str
    spearman: float
    ci_low: float
    ci_high: float
    pearson: float


@dataclass
class DiffRow:
    """두 스피어만 상관 차이의 구간과 판정."""

    kind: str
    front: str
    back: str
    target: str
    ci_low: float
    ci_high: float
    judgment: str


@dataclass
class GroupReport:
    """그룹 하나의 상관, 차이 판정, 경고."""

    name: str
    count: int
    enough: bool
    correlations: list[CorrRow] = field(default_factory=list)
    differences: list[DiffRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_report(csv_path: str | Path, out_dir: str | Path, banner: str | None = None) -> None:
    """results.csv로 그림, summary.csv, index.html을 만든다."""
    source = Path(csv_path)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    valid, errors = load_results(source)
    reports = analyze_results(valid)
    save_figures(valid, _find(reports, "전체"), destination)
    write_summary(reports, destination / "summary.csv")
    shutil.copyfile(source, destination / "results.csv")
    html = render_html(
        reports,
        errors,
        analyzed=len(valid),
        banner=banner,
        generated_at=datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
        commit=short_commit(),
    )
    (destination / "index.html").write_text(html, encoding="utf-8")


def load_results(path: Path) -> tuple[list[dict], list[dict[str, str]]]:
    """error가 비어 있는 행만 숫자로 읽고, 오류 행은 따로 돌려준다."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
    valid: list[dict] = []
    errors: list[dict[str, str]] = []
    for raw in raw_rows:
        message = (raw.get("error") or "").strip()
        file_name = raw.get("file", "")
        if message:
            errors.append({"file": file_name, "error": message})
            continue
        valid.append(_parse_valid_row(raw))
    return valid, errors


def analyze_results(rows: list[dict]) -> list[GroupReport]:
    """명세의 그룹마다 상관과 판정을 계산한다. 8행 미만은 표본 부족이다."""
    reports = [analyze_group(name, subset) for name, subset in build_groups(rows)]
    _add_loop_sign_warnings(rows, reports)
    return reports


def build_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """전체, 차원, 격자 크기, 고리 유무와 그 조합 그룹을 만든다."""
    groups: list[tuple[str, list[dict]]] = [("전체", list(rows))]
    for flag, name in ((True, "3D"), (False, "2D")):
        subset = [row for row in rows if row["is_3d"] is flag]
        if subset:
            groups.append((name, subset))
    for size in sorted({int(row["grid_size"]) for row in rows}):
        groups.append((f"격자 크기 {size}", [row for row in rows if int(row["grid_size"]) == size]))
    absent = [row for row in rows if int(row["n_loops"]) == 0]
    present = [row for row in rows if int(row["n_loops"]) > 0]
    if absent:
        groups.append(("고리 없음", absent))
    if present:
        groups.append(("고리 있음", present))
    for size in sorted({int(row["grid_size"]) for row in rows}):
        for name, subset in (
            (f"격자 크기 {size} · 고리 없음", [row for row in absent if int(row["grid_size"]) == size]),
            (f"격자 크기 {size} · 고리 있음", [row for row in present if int(row["grid_size"]) == size]),
        ):
            if subset:
                groups.append((name, subset))
    return groups


def analyze_group(name: str, rows: list[dict]) -> GroupReport:
    """그룹 하나의 상관, 결합·축 차이, 단순 기준 경고를 계산한다."""
    if len(rows) < MIN_ROWS:
        return GroupReport(name, len(rows), False)
    correlations: list[CorrRow] = []
    for predictor in PREDICTORS:
        x_values = _column(rows, predictor)
        for target in TARGETS:
            y_values = _column(rows, target)
            low, high = bootstrap_ci(x_values, y_values, spearman)
            correlations.append(
                CorrRow(
                    predictor=predictor,
                    target=target,
                    spearman=spearman(x_values, y_values),
                    ci_low=low,
                    ci_high=high,
                    pearson=pearson(x_values, y_values),
                )
            )
    differences: list[DiffRow] = []
    for mode in AXIS_MODES:
        front = f"C_geo_{mode}"
        back = f"C_arith_{mode}"
        for target in TARGETS:
            low, high = paired_diff_ci(_column(rows, front), _column(rows, back), _column(rows, target), spearman)
            differences.append(
                DiffRow("combine", front, back, target, low, high, judge_interval(low, high))
            )
    for mode in ("standard", "a", "inv"):
        front = f"C_geo_{mode}"
        for target in TARGETS:
            low, high = paired_diff_ci(
                _column(rows, front),
                _column(rows, "C_geo_none"),
                _column(rows, target),
                spearman,
            )
            differences.append(
                DiffRow("axis", front, "C_geo_none", target, low, high, judge_interval(low, high))
            )
    return GroupReport(
        name,
        len(rows),
        True,
        correlations,
        differences,
        baseline_warnings(correlations),
    )


def judge_interval(low: float, high: float) -> str:
    """차이 구간이 0의 어느 쪽에 있는지 판정 문장으로 반환한다."""
    if not np.isfinite(low) or not np.isfinite(high):
        return "차이를 구분할 수 없음"
    if low > 0:
        return "앞쪽이 더 잘 맞음"
    if high < 0:
        return "뒤쪽이 더 잘 맞음"
    return "차이를 구분할 수 없음"


def baseline_warnings(correlations: list[CorrRow]) -> list[str]:
    """C_geo_standard의 ρ가 단순 기준의 최대 ρ보다 작으면 경고한다."""
    warnings: list[str] = []
    for target in TARGETS:
        ours = _rho(correlations, "C_geo_standard", target)
        baselines = [_rho(correlations, name, target) for name in BASELINES]
        finite = [value for value in baselines if np.isfinite(value)]
        if finite and np.isfinite(ours) and ours < max(finite):
            warnings.append(f"{target}: 단순 기준보다 낫지 않음")
    return warnings


def build_figures(rows: list[dict], overall: GroupReport | None) -> list[plt.Figure]:
    """비교 그림 네 장을 메모리에서 만든다. 그림 글자는 영어만 쓴다."""
    return [
        _scatter(rows, "tremaux_mean", "tremaux_mean (mean number of Tremaux steps)", None),
        _scatter(rows, "greedy_difficulty", "greedy_difficulty (1 minus greedy success rate)", None),
        _correlation_figure(overall, None),
        _difference_figure(overall, None),
    ]


def save_figures(rows: list[dict], overall: GroupReport | None, directory: Path) -> None:
    """산점도 두 장과 구간 그림 두 장을 SVG로 저장한다. 그림 글자는 영어만 쓴다."""
    for figure, name in zip(build_figures(rows, overall), FIGURES):
        _save(figure, directory / name)


def summary_csv_bytes(reports: list[GroupReport]) -> bytes:
    """그룹별 ρ, 구간, 판정을 UTF-8 BOM CSV 바이트로 만든다."""
    return _summary_text(reports).encode("utf-8-sig")


def write_summary(reports: list[GroupReport], path: Path) -> None:
    """그룹별 ρ, 구간, 판정을 CSV로 저장한다."""
    path.write_bytes(summary_csv_bytes(reports))


def _summary_text(reports: list[GroupReport]) -> str:
    """그룹별 요약 CSV 본문을 만든다."""
    fieldnames = [
        "group",
        "n",
        "status",
        "kind",
        "left",
        "right",
        "target",
        "spearman",
        "ci_low",
        "ci_high",
        "pearson",
        "judgment",
        "warnings",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for report in reports:
        status = "ok" if report.enough else "표본 부족"
        warnings = " | ".join(report.warnings)
        if not report.enough:
            writer.writerow(
                {
                    "group": report.name,
                    "n": report.count,
                    "status": status,
                    "warnings": warnings,
                }
            )
            continue
        for row in report.correlations:
            writer.writerow(
                {
                    "group": report.name,
                    "n": report.count,
                    "status": status,
                    "kind": "correlation",
                    "left": row.predictor,
                    "right": "",
                    "target": row.target,
                    "spearman": row.spearman,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "pearson": row.pearson,
                    "judgment": "",
                    "warnings": warnings,
                }
            )
        for row in report.differences:
            writer.writerow(
                {
                    "group": report.name,
                    "n": report.count,
                    "status": status,
                    "kind": row.kind,
                    "left": row.front,
                    "right": row.back,
                    "target": row.target,
                    "spearman": "",
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "pearson": "",
                    "judgment": row.judgment,
                    "warnings": warnings,
                }
            )
    return handle.getvalue()


def render_html(
    reports: list[GroupReport],
    errors: list[dict[str, str]],
    analyzed: int,
    banner: str | None,
    generated_at: str,
    commit: str,
) -> str:
    """외부 파일 없이 한 장의 한국어 HTML 보고서를 만든다."""
    parts = [
        "<!DOCTYPE html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>미로 복잡도 비교 보고서</title>",
        "<style>",
        _CSS,
        "</style>",
        "</head>",
        "<body>",
        "<h1>미로 복잡도 비교 보고서</h1>",
    ]
    if banner:
        parts.append(f'<p class="banner">{_esc(banner)}</p>')
    parts.append(
        "<p>"
        f"생성 시각: {_esc(generated_at)} · "
        f"커밋: {_esc(commit)} · "
        f"분석한 미로 수: {analyzed} · "
        f"오류 난 파일 수: {len(errors)}"
        "</p>"
    )
    parts.append("<h2>핵심 결론</h2>")
    parts.append("<ul>")
    for report in reports:
        if not report.enough:
            parts.append(f"<li>{_esc(report.name)}: 표본 부족 (n={report.count})</li>")
            continue
        for row in report.differences:
            parts.append(
                "<li>"
                f"{_esc(report.name)}: {_esc(row.front)} 대 {_esc(row.back)}, "
                f"{_esc(row.target)} — {_esc(row.judgment)}"
                "</li>"
            )
        for warning in report.warnings:
            parts.append(f'<li class="warn">{_esc(report.name)}: {_esc(warning)}</li>')
    parts.append("</ul>")
    parts.append("<h2>그룹별 상관</h2>")
    for report in reports:
        parts.append(f"<h3>{_esc(report.name)} (n={report.count})</h3>")
        if not report.enough:
            parts.append("<p>표본 부족</p>")
            continue
        parts.append(_correlation_table(report))
        parts.append(_difference_table(report))
    parts.append("<h2>그림</h2>")
    parts.extend(
        [
            "<h3>기하평균 복잡도와 트레모 걸음 수</h3>",
            f'<img src="{FIGURES[0]}" alt="C_geo_standard versus tremaux mean">',
            "<h3>기하평균 복잡도와 탐욕 라우팅 복잡도</h3>",
            f'<img src="{FIGURES[1]}" alt="C_geo_standard versus greedy difficulty">',
            "<h3>스피어만 상관과 95% 구간</h3>",
            f'<img src="{FIGURES[2]}" alt="Spearman correlations with intervals">',
            "<h3>결합 방식과 축 가중치의 차이</h3>",
            f'<img src="{FIGURES[3]}" alt="Paired differences of Spearman rho">',
        ]
    )
    parts.append("<h2>오류 파일</h2>")
    if errors:
        parts.append("<ul>")
        for item in errors:
            parts.append(f"<li>{_esc(item['file'])}: {_esc(item['error'])}</li>")
        parts.append("</ul>")
    else:
        parts.append("<p>오류 난 파일이 없습니다.</p>")
    parts.append("<h2>내려받기</h2>")
    parts.append('<p><a href="results.csv">results.csv</a> · <a href="summary.csv">summary.csv</a></p>')
    parts.append("<h2>용어</h2>")
    parts.append("<ul>")
    for line in _GLOSSARY:
        parts.append(f"<li>{line}</li>")
    parts.append("</ul>")
    parts.append("</body></html>")
    html = "\n".join(parts)
    if 'href="/' in html or 'src="/' in html:
        raise RuntimeError("HTML 링크는 상대경로여야 합니다.")
    return html


def short_commit() -> str:
    """GITHUB_SHA, git 커밋, 둘 다 실패하면 unknown 가운데 앞 7자리를 반환한다."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if not sha:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    return sha[:7] if sha else "unknown"


def _parse_valid_row(raw: dict[str, str]) -> dict:
    """성공한 CSV 행의 수치 열을 숫자로 바꾼다."""
    parsed: dict = {"file": raw.get("file", ""), "error": ""}
    parsed["is_3d"] = (raw.get("is_3d") or "").strip().lower() == "true"
    for key in ("grid_size", "n_nodes", "n_edges", "n_loops", "L_total", "shortest_path"):
        parsed[key] = int(float(raw[key]))
    for key in PREDICTORS + TARGETS + (
        "tremaux_std",
        "tremaux_min",
        "tremaux_max",
        "tremaux_norm",
        "greedy_success_rate",
        "greedy_GR_score",
    ):
        parsed[key] = float(raw[key])
    for mode in AXIS_MODES:
        for prefix in ("tau_length", "tau_degree", "C_geo", "C_arith"):
            key = f"{prefix}_{mode}"
            if key not in parsed:
                parsed[key] = float(raw[key])
    return parsed


def _column(rows: list[dict], key: str) -> np.ndarray:
    """행들에서 한 수치 열을 꺼낸다."""
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def _rho(correlations: list[CorrRow], predictor: str, target: str) -> float:
    """저장된 상관에서 해당 쌍의 스피어만 ρ를 찾는다."""
    for row in correlations:
        if row.predictor == predictor and row.target == target:
            return row.spearman
    return float("nan")


def _add_loop_sign_warnings(rows: list[dict], reports: list[GroupReport]) -> None:
    """전체 그룹에 고리 유무가 섞이고 ρ 부호가 반대면 경고를 붙인다."""
    overall = _find(reports, "전체")
    absent = _find(reports, "고리 없음")
    present = _find(reports, "고리 있음")
    if overall is None or absent is None or present is None:
        return
    mixed = any(int(row["n_loops"]) == 0 for row in rows) and any(int(row["n_loops"]) > 0 for row in rows)
    if not mixed or not absent.enough or not present.enough:
        return
    for target in TARGETS:
        left = _rho(absent.correlations, "C_geo_standard", target)
        right = _rho(present.correlations, "C_geo_standard", target)
        if np.isfinite(left) and np.isfinite(right) and left * right < 0:
            overall.warnings.append(f"{target}: 고리 유무에 따라 경향이 반대")


def _find(reports: list[GroupReport], name: str) -> GroupReport | None:
    """이름이 같은 그룹 결과를 찾는다."""
    for report in reports:
        if report.name == name:
            return report
    return None


def _scatter(rows: list[dict], y_key: str, y_label: str, path: Path | None) -> plt.Figure:
    """C_geo_standard를 로그 눈금 가로축으로 한 산점도를 저장한다."""
    figure, axis = plt.subplots(figsize=(8, 5.4))
    _paint(figure, axis)
    series = (
        ("no loop", [row for row in rows if int(row["n_loops"]) == 0], "#1f77b4", "o"),
        ("has loop", [row for row in rows if int(row["n_loops"]) > 0], "#d62728", "s"),
    )
    plotted = False
    for label, subset, color, marker in series:
        xs = []
        ys = []
        for row in subset:
            x_value = float(row["C_geo_standard"])
            y_value = float(row[y_key])
            if np.isfinite(x_value) and x_value > 0 and np.isfinite(y_value):
                xs.append(x_value)
                ys.append(y_value)
        if xs:
            axis.scatter(xs, ys, c=color, marker=marker, label=label, s=36)
            plotted = True
    if plotted:
        axis.set_xscale("log")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No positive C_geo_standard", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("C_geo_standard (geometric-mean complexity, log scale)")
    axis.set_ylabel(y_label)
    axis.set_title("Complexity against difficulty")
    return _save(figure, path)


def _correlation_figure(overall: GroupReport | None, path: Path | None) -> plt.Figure:
    """후보와 단순 기준의 ρ와 95% 구간을 검증 기준별로 그린다."""
    if overall is None or not overall.enough:
        return _insufficient(path, "Spearman rho", "measure")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 7.2))
    titles = {
        "tremaux_mean": "Tremaux mean",
        "greedy_difficulty": "Greedy difficulty",
    }
    for axis, target in zip(axes, TARGETS):
        _paint(figure, axis)
        rows = [row for row in overall.correlations if row.target == target]
        positions = list(range(len(rows) - 1, -1, -1))
        for row, position in zip(rows, positions):
            if not np.isfinite(row.spearman):
                continue
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                left = max(0.0, row.spearman - row.ci_low)
                right = max(0.0, row.ci_high - row.spearman)
                axis.errorbar(row.spearman, position, xerr=[[left], [right]], fmt="o", color="#1f77b4", capsize=3)
            else:
                axis.plot(row.spearman, position, "o", color="#1f77b4")
        axis.set_yticks(positions)
        axis.set_yticklabels([row.predictor for row in rows])
        axis.axvline(0, linestyle=":", color="black")
        axis.set_xlabel("Spearman rho (95% bootstrap interval)")
        axis.set_ylabel("measure")
        axis.set_title(titles[target])
    return _save(figure, path)


def _difference_figure(overall: GroupReport | None, path: Path | None) -> plt.Figure:
    """전체 그룹의 결합 방식 차이와 축 가중치 차이를 구간으로 그린다."""
    if overall is None or not overall.enough:
        return _insufficient(path, "Difference of Spearman rho", "comparison")
    figure, axes = plt.subplots(2, 1, figsize=(11, 9))
    panels = (
        (axes[0], "combine", "C_geo minus C_arith"),
        (axes[1], "axis", "Axis-weight C_geo minus C_geo_none"),
    )
    for axis, kind, title in panels:
        _paint(figure, axis)
        rows = [row for row in overall.differences if row.kind == kind]
        positions = list(range(len(rows) - 1, -1, -1))
        labels = []
        for row, position in zip(rows, positions):
            target_name = "Tremaux" if row.target == "tremaux_mean" else "greedy difficulty"
            labels.append(f"{row.front} - {row.back}, {target_name}")
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                midpoint = 0.5 * (row.ci_low + row.ci_high)
                left = max(0.0, midpoint - row.ci_low)
                right = max(0.0, row.ci_high - midpoint)
                axis.errorbar(midpoint, position, xerr=[[left], [right]], fmt="o", color="#2ca02c", capsize=3)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.axvline(0, linestyle=":", color="black")
        axis.set_xlabel("Difference of Spearman rho (95% interval)")
        axis.set_ylabel("comparison")
        axis.set_title(title)
    return _save(figure, path)


def _insufficient(path: Path | None, xlabel: str, ylabel: str) -> plt.Figure:
    """표본이 부족할 때 영어 안내만 있는 그림을 저장한다."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    _paint(figure, axis)
    axis.text(0.5, 0.5, "Insufficient sample", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return _save(figure, path)


def _paint(figure: plt.Figure, axis: plt.Axes) -> None:
    """흰 배경과 10 이상의 글자 크기를 맞춘다."""
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.tick_params(labelsize=10)
    axis.xaxis.label.set_size(11)
    axis.yaxis.label.set_size(11)
    axis.title.set_size(12)


def _save(figure: plt.Figure, path: Path | None) -> plt.Figure:
    """여백을 맞춘다. 경로가 있으면 SVG로 저장하고 한글이 없는지 확인한다."""
    figure.tight_layout()
    if path is None:
        return figure
    figure.savefig(path, format="svg", facecolor="white")
    plt.close(figure)
    text = path.read_text(encoding="utf-8")
    if _HANGUL.search(text):
        raise RuntimeError(f"그림에 한글이 들어 있습니다: {path.name}")
    return figure


def _correlation_table(report: GroupReport) -> str:
    """ρ, 구간, 피어슨 r 표를 가로 스크롤 영역에 넣는다."""
    body = [
        '<div class="scroll"><table>',
        "<thead><tr><th>변수</th><th>검증 기준</th><th>스피어만 ρ</th><th>95% 구간</th><th>피어슨 r</th></tr></thead>",
        "<tbody>",
    ]
    for row in report.correlations:
        body.append(
            "<tr>"
            f"<td>{_esc(row.predictor)}</td>"
            f"<td>{_esc(row.target)}</td>"
            f"<td>{_num(row.spearman)}</td>"
            f"<td>{_num(row.ci_low)} ~ {_num(row.ci_high)}</td>"
            f"<td>{_num(row.pearson)}</td>"
            "</tr>"
        )
    body.append("</tbody></table></div>")
    return "\n".join(body)


def _difference_table(report: GroupReport) -> str:
    """결합·축 차이의 구간과 판정 표를 만든다."""
    body = [
        '<div class="scroll"><table>',
        "<thead><tr><th>비교</th><th>앞</th><th>뒤</th><th>검증 기준</th><th>95% 구간</th><th>판정</th></tr></thead>",
        "<tbody>",
    ]
    for row in report.differences:
        label = "결합 방식" if row.kind == "combine" else "축 가중치"
        body.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{_esc(row.front)}</td>"
            f"<td>{_esc(row.back)}</td>"
            f"<td>{_esc(row.target)}</td>"
            f"<td>{_num(row.ci_low)} ~ {_num(row.ci_high)}</td>"
            f"<td>{_esc(row.judgment)}</td>"
            "</tr>"
        )
    body.append("</tbody></table></div>")
    return "\n".join(body)


def _num(value: float) -> str:
    """유한한 수는 소수점 넷째 자리까지, 그 외에는 NaN으로 적는다."""
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.4f}"


def _esc(value: object) -> str:
    """HTML 본문에 넣을 문자열을 이스케이프한다."""
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_CSS = """
body { font-family: sans-serif; margin: 1rem; line-height: 1.5; background: #fff; color: #222; }
h1, h2, h3 { line-height: 1.3; }
.banner { background: #fff4d6; padding: 0.6rem 0.8rem; }
.warn { color: #8a3b00; }
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; min-width: 720px; margin-bottom: 1rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.5rem; text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
img { max-width: 100%; height: auto; background: #fff; }
"""

_GLOSSARY = (
    "C_geo는 길이 가중 흡수시간과 차수 가중 흡수시간의 기하평균이다.",
    "C_arith는 두 흡수시간의 산술평균이다.",
    "tau_length는 간선 길이의 역수를 가중치로 둔 입구의 흡수시간이다.",
    "tau_degree는 도착 정점의 유효 차수 max(2, 차수)를 가중치로 둔 흡수시간이다.",
    "standard는 축 가중치 max(1, 축 개수−1), none은 1, a는 축 개수, inv는 1/축 개수이다.",
    "tremaux_mean은 트레모 알고리즘 걸음 수의 평균이며 클수록 헤매기 쉽다.",
    "greedy_difficulty는 1−탐욕 라우팅 성공률이며 클수록 출구로 곧장 가기 어렵다.",
    "n_nodes, L_total, shortest_path는 비교용 단순 기준이다.",
    "n_loops는 간선 수−정점 수+1이고, 0이면 고리가 없는 미로다.",
    "스피어만 ρ는 순위의 상관이고, 피어슨 r은 값 자체의 상관이다.",
    "95% 구간은 행을 복원추출한 부트스트랩 2.5%와 97.5% 분위수다.",
)
