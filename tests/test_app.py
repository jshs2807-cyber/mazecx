"""Streamlit 화면 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import _apply_roster_edits
from mazecx.generate import generate_documents

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
C_GEO_T3 = 19.8997487421324
C_GEO_T4_A = 13.856406460551016


def test_app_opens_without_exception() -> None:
    """앱이 예외 없이 뜬다."""
    at = _app()
    at.run()
    assert not at.exception


def test_example_t3_shows_standard_c_geo() -> None:
    """예시 T3를 계산하면 C_geo(standard)가 소수 넷째 자리까지 맞다."""
    at = _app()
    at.run()
    at.selectbox(key="example_choice").set_value("T3").run()
    at.button(key="calc_example").click().run()
    assert not at.exception
    assert _has_metric(at, C_GEO_T3)


def test_bad_json_shows_error_without_crashing() -> None:
    """잘못된 JSON은 오류 문장으로 보여 주고 앱은 멈추지 않는다."""
    at = _app()
    at.run()
    at.text_area(key="json_paste").set_value("{").run()
    at.button(key="calc_paste").click().run()
    assert at.error
    assert not at.exception


def test_changing_runs_keeps_previous_result() -> None:
    """반복 횟수를 바꿔도 이미 계산한 결과는 남아 있다."""
    at = _app()
    at.run()
    at.selectbox(key="example_choice").set_value("T3").run()
    at.button(key="calc_example").click().run()
    at.number_input(key="runs_one").set_value(20).run()
    assert not at.exception
    assert _has_metric(at, C_GEO_T3)


def test_three_screens_open() -> None:
    """세 화면을 옮겨도 예외가 나지 않는다."""
    at = _app()
    at.run()
    at.radio(key="page").set_value("여러 미로 비교").run()
    assert not at.exception
    at.radio(key="page").set_value("미로 만들기").run()
    assert not at.exception
    at.number_input(key="count_make").set_value(100).run()
    assert at.button(key="make_mazes").disabled
    assert any("500" in warning.value for warning in at.warning)


def test_three_mazes_make_a_comparison_table() -> None:
    """화면 2에 T2, T3, T4를 넣고 모두 계산하면 비교 표가 3행이다."""
    at = _compare(("t2.json", "t3.json", "t4.json"), runs=10)
    assert not at.exception
    assert len(_comparison_frame(at)) == 3


def test_two_mazes_need_three_for_rank_agreement() -> None:
    """미로가 2개면 순위 일치도 자리에 3개 이상 필요하다는 안내가 나온다."""
    at = _compare(("t2.json", "t3.json"), runs=10)
    assert not at.exception
    assert any("3개 이상 필요" in info.value for info in at.info)


def test_name_and_group_edits_do_not_recalculate() -> None:
    """이름과 그룹을 고쳐도 이미 계산한 걸음 수 평균은 바뀌지 않는다."""
    item = {
        "name": "t2",
        "group": "2D",
        "content": "{}",
        "result": {"file": "t2", "tremaux_mean": 6.0, "runs": 10, "seed": 0},
    }
    _apply_roster_edits([item], [{"이름": "다른이름", "그룹": "실험"}])
    assert item["name"] == "다른이름"
    assert item["group"] == "실험"
    assert item["result"]["tremaux_mean"] == 6.0
    assert item["result"]["runs"] == 10
    assert item["result"]["seed"] == 0


def test_runs_edit_keeps_results_and_axis_changes_the_table() -> None:
    """반복 횟수만 바꾸면 다시 계산하지 않고, 축 변형을 바꾸면 표의 복잡도가 바뀐다."""
    at = _compare(("t2.json", "t3.json", "t4.json"), runs=10)
    before = [item["result"]["tremaux_mean"] for item in at.session_state["roster"]]
    at.number_input(key="runs_many").set_value(40).run()
    assert not at.exception
    assert [item["result"]["tremaux_mean"] for item in at.session_state["roster"]] == before
    at.selectbox(key="axis_variant").set_value("a").run()
    assert not at.exception
    frame = _comparison_frame(at)
    row = frame.loc[frame["이름"] == "t4"].iloc[0]
    assert row["우리 복잡도 기하평균"] == pytest.approx(C_GEO_T4_A, abs=1e-9)


def test_ten_generated_mazes_compare_without_error() -> None:
    """만든 미로 10개를 목록에 넣으면 비교 표가 10행이고 예외가 없다."""
    files = [
        (name, json.dumps(document).encode("utf-8"), "application/json")
        for name, document in generate_documents([3], [0.0], 10, 0)
    ]
    at = _open_compare(timeout=180)
    at.number_input(key="runs_many").set_value(10).run()
    at.file_uploader(key="many_json").set_value(files).run()
    at.button(key="calc_many").click().run()
    assert not at.exception
    assert len(_comparison_frame(at)) == 10


def test_app_does_not_reimplement_formulas() -> None:
    """화면 코드는 계산식을 다시 만들지 않고 mazecx를 호출한다."""
    source = APP.read_text(encoding="utf-8")
    assert "linalg" not in source
    assert "def spearman" not in source
    assert "def vertex_weight" not in source
    assert "tempfile" not in source


def _app() -> AppTest:
    """앱 테스트 러너를 만든다."""
    return AppTest.from_file(str(APP), default_timeout=90)


def _open_compare(timeout: int = 120) -> AppTest:
    """여러 미로 비교 화면까지 연다."""
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    at.radio(key="page").set_value("여러 미로 비교").run()
    return at


def _compare(filenames: tuple[str, ...], runs: int) -> AppTest:
    """예제 JSON을 비교 목록에 넣고 모두 계산한다."""
    files = [(name, (ROOT / "examples" / name).read_bytes(), "application/json") for name in filenames]
    at = _open_compare()
    at.number_input(key="runs_many").set_value(runs).run()
    at.file_uploader(key="many_json").set_value(files).run()
    at.button(key="calc_many").click().run()
    return at


def _comparison_frame(at: AppTest) -> pd.DataFrame:
    """미로별 비교 표를 찾는다."""
    for element in at.dataframe:
        if "우리 복잡도 기하평균" in list(element.value.columns):
            return element.value
    raise AssertionError("비교 표가 없습니다")


def _has_metric(at: AppTest, expected: float) -> bool:
    """큰 숫자 가운데 기대값과 소수 넷째 자리까지 같은 것이 있는지 본다."""
    for metric in at.metric:
        if float(metric.value) == pytest.approx(expected, abs=5e-5):
            return True
    return False
