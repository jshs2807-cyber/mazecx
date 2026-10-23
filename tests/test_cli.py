"""명령행 분석과 일괄 처리 테스트."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from mazecx.cli import RESULT_KEYS, analyze_file, format_report, main

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def near(expected: float):
    """절대오차 1e-9로 비교한다."""
    return pytest.approx(expected, abs=1e-9, rel=0)


def test_analyze_t1_and_t3_values() -> None:
    """단일 분석 결과에 명세의 키가 있고 T1·T3 검산값과 맞다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        straight = analyze_file(EX / "t1.json", runs=20, seed=0)
        trapped = analyze_file(EX / "t3.json", runs=10, seed=0)
    assert [key for key in straight if key != "steps_all"] == list(RESULT_KEYS)
    assert len(straight["steps_all"]) == 20
    assert sum(straight["steps_all"]) / len(straight["steps_all"]) == near(straight["tremaux_mean"])
    assert "steps_all" not in RESULT_KEYS
    assert straight["error"] == ""
    assert straight["is_3d"] is False
    assert straight["grid_size"] == 6
    assert straight["n_nodes"] == 2
    assert straight["n_loops"] == 0
    assert straight["L_total"] == 5
    assert straight["shortest_path"] == 5
    assert straight["tau_length_standard"] == near(1)
    assert straight["tau_degree_standard"] == near(1)
    assert straight["C_geo_standard"] == near(1)
    assert straight["C_arith_standard"] == near(1)
    assert straight["tau_length_none"] == near(1)
    assert straight["tremaux_mean"] == near(5)
    assert straight["tremaux_norm"] == near(1)
    assert straight["greedy_success_rate"] == near(1)
    assert straight["greedy_entry_success"] is True
    assert trapped["is_3d"] is False
    assert trapped["n_loops"] == 0
    assert trapped["shortest_path"] == 10
    assert trapped["tau_length_standard"] == near(22)
    assert trapped["tau_degree_standard"] == near(18)
    assert trapped["C_geo_standard"] == near(19.8997487421324)
    assert trapped["C_arith_standard"] == near(20)
    assert trapped["greedy_success_rate"] == near(0.4)
    assert trapped["greedy_difficulty"] == near(0.6)
    assert trapped["greedy_entry_success"] is False
    assert trapped["greedy_GR_score"] == near(0.4)


def test_analyze_t4_all_variants() -> None:
    """T4는 3차원이고 네 축 가중치를 모두 함께 돌려준다."""
    result = analyze_file(EX / "t4.json", runs=8, seed=0)
    assert result["is_3d"] is True
    assert result["tau_length_standard"] == near(12)
    assert result["tau_degree_standard"] == near(9)
    assert result["C_geo_standard"] == near(10.392304845413264)
    assert result["C_arith_standard"] == near(10.5)
    assert result["tau_length_none"] == near(8)
    assert result["tau_degree_none"] == near(6)
    assert result["C_geo_none"] == near(6.928203230275509)
    assert result["C_arith_none"] == near(7)
    assert result["tau_length_a"] == near(16)
    assert result["tau_degree_a"] == near(12)
    assert result["C_geo_a"] == near(13.856406460551016)
    assert result["C_arith_a"] == near(14)
    assert result["tau_length_inv"] == near(5.333333333333333)
    assert result["tau_degree_inv"] == near(4)
    assert result["C_geo_inv"] == near(4.618802153517006)
    assert result["C_arith_inv"] == near(4.666666666666667)
    report = format_report(result)
    assert "C_geo" in report
    assert "none" in report
    assert "탐욕 라우팅" in report


def test_analyze_t5_maze_file() -> None:
    """미로 형식 파일도 변환 뒤 계산된다."""
    result = analyze_file(EX / "t5.json", runs=5, seed=0)
    assert result["n_nodes"] == 3
    assert result["n_edges"] == 2
    assert result["L_total"] == 4
    assert result["grid_size"] == 3
    assert result["C_geo_standard"] > 0


def test_analyze_tremaux_reproducible() -> None:
    """같은 파일과 같은 seed면 트레모 요약 통계가 같다."""
    first = analyze_file(EX / "t2.json", runs=40, seed=3)
    second = analyze_file(EX / "t2.json", runs=40, seed=3)
    assert first["tremaux_mean"] == second["tremaux_mean"]
    assert first["tremaux_std"] == second["tremaux_std"]
    assert first["tremaux_min"] == second["tremaux_min"]
    assert first["tremaux_max"] == second["tremaux_max"]


def test_invalid_file_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """검증 오류는 상태 코드 1과 이유 메시지로 끝난다."""
    with pytest.raises(SystemExit) as caught:
        main(["analyze", str(EX / "t6_diagonal.json")])
    assert caught.value.code == 1
    assert "평행" in capsys.readouterr().err


def test_batch_continues_after_invalid_file(tmp_path: Path) -> None:
    """파일 하나가 실패해도 다음 파일을 처리하고 시드는 0번부터 더한다."""
    folder = tmp_path / "data"
    folder.mkdir()
    shutil.copy(EX / "t1.json", folder / "a_ok.json")
    shutil.copy(EX / "t6_diagonal.json", folder / "b_bad.json")
    shutil.copy(EX / "t4.json", folder / "c_axis.json")
    out = tmp_path / "results.csv"
    main(
        [
            "batch",
            str(folder),
            "--runs",
            "8",
            "--seed",
            "5",
            "--out",
            str(out),
        ]
    )
    with out.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(RESULT_KEYS)
        rows = list(reader)
    assert [row["file"] for row in rows] == ["a_ok.json", "b_bad.json", "c_axis.json"]
    assert rows[0]["error"] == ""
    assert float(rows[0]["C_geo_standard"]) == near(1)
    assert rows[0]["seed"] == "5"
    assert rows[0]["greedy_entry_success"] == "true"
    assert float(rows[0]["tremaux_mean"]) == near(5)
    assert "평행" in rows[1]["error"]
    assert rows[1]["tau_length_standard"] == ""
    assert rows[1]["seed"] == "6"
    assert float(rows[2]["tau_length_standard"]) == near(12)
    assert float(rows[2]["tau_length_none"]) == near(8)
    assert rows[2]["is_3d"] == "true"
    assert rows[2]["seed"] == "7"


def test_batch_meta_columns_are_sorted_union(tmp_path: Path) -> None:
    """meta 키의 합집합이 사전순 meta_ 열로 붙고, 없는 칸은 비운다."""
    folder = tmp_path / "mazes"
    folder.mkdir()
    (folder / "a.json").write_text(
        json.dumps(
            {
                "nodes": {"A": [0, 0, 0], "E": [1, 0, 0]},
                "edges": [["A", "E"]],
                "entry": "A",
                "exit": "E",
                "meta": {"name": "first", "kind": "line"},
            }
        ),
        encoding="utf-8",
    )
    (folder / "b.json").write_text(
        json.dumps(
            {
                "nodes": {"A": [0, 0, 0], "B": [1, 1, 0]},
                "edges": [["A", "B"]],
                "entry": "A",
                "exit": "B",
                "meta": {"kind": "bad", "seed": 7},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "meta.csv"
    main(["batch", str(folder), "--runs", "2", "--seed", "0", "--out", str(out)])
    with out.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[-3:] == ["meta_kind", "meta_name", "meta_seed"]
        rows = list(reader)
    assert rows[0]["error"] == ""
    assert rows[0]["meta_name"] == "first"
    assert rows[0]["meta_kind"] == "line"
    assert rows[0]["meta_seed"] == ""
    assert rows[1]["error"] != ""
    assert rows[1]["meta_kind"] == "bad"
    assert rows[1]["meta_seed"] == "7"
    assert rows[1]["meta_name"] == ""


def test_module_analyze_writes_json(tmp_path: Path) -> None:
    """python -m mazecx.cli analyze 가 표를 출력하고 JSON을 저장한다."""
    out = tmp_path / "result.json"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mazecx.cli",
            "analyze",
            str(EX / "t1.json"),
            "--runs",
            "12",
            "--seed",
            "0",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "C_geo" in completed.stdout
    assert "tremaux_mean" in completed.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data.keys()) == list(RESULT_KEYS)
    assert data["tau_length_standard"] == near(1)
    assert data["C_arith_standard"] == near(1)
    assert data["tremaux_mean"] == near(5)
    assert data["tremaux_min"] == 5
    assert data["greedy_success_rate"] == near(1)
    assert data["greedy_entry_success"] is True
    assert data["runs"] == 12
    assert data["seed"] == 0
