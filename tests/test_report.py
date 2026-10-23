"""비교 보고서 테스트."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from mazecx.cli import RESULT_KEYS, batch_folder, main
from mazecx.generate import generate_dataset
from mazecx.report import (
    FIGURES,
    CorrRow,
    GroupReport,
    TARGETS,
    _add_loop_sign_warnings,
    baseline_warnings,
    build_report,
    judge_interval,
)

_HANGUL = re.compile(r"[가-힣]")


def test_judgment_sentences() -> None:
    """차이 구간이 0보다 큰지, 작은지, 포함하는지에 따라 판정 문장이 정해진다."""
    assert judge_interval(0.1, 0.2) == "앞쪽이 더 잘 맞음"
    assert judge_interval(-0.3, -0.1) == "뒤쪽이 더 잘 맞음"
    assert judge_interval(-0.1, 0.2) == "차이를 구분할 수 없음"
    assert judge_interval(0.0, 0.2) == "차이를 구분할 수 없음"
    assert judge_interval(float("nan"), 1.0) == "차이를 구분할 수 없음"


def test_baseline_warning_when_simple_rho_is_larger() -> None:
    """C_geo_standard의 ρ가 단순 기준보다 작으면 경고를 붙인다."""
    rows = []
    for name, rho in (
        ("C_geo_standard", 0.1),
        ("n_nodes", 0.9),
        ("L_total", 0.2),
        ("shortest_path", 0.3),
    ):
        for target in TARGETS:
            rows.append(CorrRow(name, target, rho, 0.0, 1.0, rho))
    warnings = baseline_warnings(rows)
    assert warnings == [
        "tremaux_mean: 단순 기준보다 낫지 않음",
        "greedy_difficulty: 단순 기준보다 낫지 않음",
    ]


def test_loop_sign_warning_on_overall_group() -> None:
    """고리 있는 미로와 없는 미로의 ρ 부호가 반대면 전체 그룹에 경고를 붙인다."""
    overall = GroupReport("전체", 16, True)
    absent = GroupReport("고리 없음", 8, True, [_corr("C_geo_standard", "tremaux_mean", 0.4)])
    present = GroupReport("고리 있음", 8, True, [_corr("C_geo_standard", "tremaux_mean", -0.3)])
    rows = [{"n_loops": 0}] * 8 + [{"n_loops": 1}] * 8
    _add_loop_sign_warnings(rows, [overall, absent, present])
    assert overall.warnings == ["tremaux_mean: 고리 유무에 따라 경향이 반대"]


def test_generated_mazes_produce_report(tmp_path: Path) -> None:
    """3D 미로 12개를 계산하면 HTML, 그림 4개, summary.csv가 생기고 기하평균이 적힌다."""
    maze_dir = tmp_path / "mazes"
    written = generate_dataset(maze_dir, sizes=[3], loops=[0, 0.15], count=6, seed=0)
    assert len(written) == 12
    csv_path = tmp_path / "results.csv"
    batch_folder(maze_dir, runs=20, seed=0, out=csv_path)
    site = tmp_path / "site"
    build_report(csv_path, site)
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "기하평균" in html
    assert 'href="/' not in html
    assert 'src="/' not in html
    assert (site / "summary.csv").is_file()
    assert (site / "results.csv").is_file()
    for name in FIGURES:
        figure = site / name
        assert figure.is_file()
        assert _HANGUL.search(figure.read_text(encoding="utf-8")) is None


def test_five_rows_are_marked_insufficient(tmp_path: Path) -> None:
    """유효한 행이 5개뿐이면 오류 없이 표본 부족이라고 표시한다."""
    csv_path = tmp_path / "small.csv"
    _write_rows(csv_path, 5)
    site = tmp_path / "site"
    main(["report", str(csv_path), "--out", str(site)])
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "표본 부족" in html
    for name in FIGURES:
        assert (site / name).is_file()


def _corr(predictor: str, target: str, rho: float) -> CorrRow:
    """판정 테스트용 상관 한 줄을 만든다."""
    return CorrRow(predictor, target, rho, rho, rho, rho)


def _write_rows(path: Path, count: int) -> None:
    """보고서가 읽는 열을 모두 채운 작은 CSV를 만든다."""
    rows = []
    for index in range(count):
        row = {key: "1" for key in RESULT_KEYS}
        row["file"] = f"m{index}.json"
        row["error"] = ""
        row["is_3d"] = "true"
        row["grid_size"] = "3"
        row["n_nodes"] = str(8 + index)
        row["n_edges"] = str(7 + index)
        row["n_loops"] = "0"
        row["L_total"] = str(10 + index)
        row["shortest_path"] = str(3 + index)
        row["greedy_entry_success"] = "true"
        row["runs"] = "20"
        row["seed"] = str(index)
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_KEYS))
        writer.writeheader()
        writer.writerows(rows)
