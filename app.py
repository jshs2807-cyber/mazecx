"""미로 복잡도를 브라우저에서 계산하는 화면. 계산식은 mazecx에 둔다."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mazecx.bundle import read_jsons_from_zip
from mazecx.charts import (
    agreement_heatmap,
    comparison_bars,
    maze_figure,
    metric_scatter,
    ratio_bars,
    steps_boxes,
)
from mazecx.cli import RESULT_KEYS, analyze_graph, ordered_result
from mazecx.compare import (
    COMPARE_METRICS,
    METRIC_CGEO,
    METRIC_TREMAUX,
    build_table,
    normalize_by_max,
    rank_agreement,
)
from mazecx.complexity import AXIS_WEIGHTS
from mazecx.generate import generate_documents
from mazecx.io import load_document
from mazecx.report import GroupReport, analyze_results, build_figures, summary_csv_bytes

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 5 * 1024 * 1024
MAX_FILES = 1000
MAX_GENERATED = 500
PAGES = ("미로 하나 계산", "여러 미로 비교", "미로 만들기")


def main() -> None:
    """사이드바에서 고른 화면을 그린다."""
    st.set_page_config(page_title="미로 복잡도", layout="wide")
    st.sidebar.header("메뉴")
    page = st.sidebar.radio("화면", PAGES, key="page")
    if page == PAGES[0]:
        page_single()
    elif page == PAGES[1]:
        page_compare()
    else:
        page_generate()


def page_single() -> None:
    """미로 하나를 읽어 복잡도와 3차원 그림을 보여 준다."""
    st.header("미로 하나 계산")
    runs = st.number_input("트레모 반복 횟수", min_value=10, max_value=5000, value=100, step=10, key="runs_one")
    seed = st.number_input("seed", value=0, step=1, key="seed_one")
    upload_tab, paste_tab, example_tab = st.tabs(("JSON 파일 올리기", "JSON 직접 붙여 넣기", "예시 고르기"))
    with upload_tab:
        uploaded = st.file_uploader("미로 JSON", type=["json"], key="one_file")
        if st.button("계산하기", key="calc_upload"):
            _calculate_upload(uploaded, int(runs), int(seed))
    with paste_tab:
        pasted = st.text_area("JSON", height=220, key="json_paste")
        if st.button("계산하기", key="calc_paste"):
            _calculate_text(pasted, "붙여 넣은 JSON", int(runs), int(seed))
    with example_tab:
        choices = _example_map()
        if not choices:
            st.error("examples 폴더에 예시 JSON이 없습니다.")
        else:
            label = st.selectbox("예시", list(choices), key="example_choice")
            if st.button("계산하기", key="calc_example"):
                text = choices[label].read_text(encoding="utf-8")
                _calculate_text(text, choices[label].name, int(runs), int(seed))
    _show_single()


def page_compare() -> None:
    """비교 목록을 계산하고, 미로별 비교와 전체 경향을 탭으로 보여 준다."""
    st.header("여러 미로 비교")
    st.caption("계산은 모두 계산을 눌렀을 때만 합니다. 같은 파일 내용, 반복 횟수, 시드는 다시 계산하지 않습니다.")
    runs = st.number_input("트레모 반복 횟수", min_value=10, max_value=5000, value=100, step=10, key="runs_many")
    seed = st.number_input("seed", value=0, step=1, key="seed_many")
    mode = st.radio("입력", ("JSON 여러 개", "ZIP 하나"), horizontal=True, key="compare_mode")
    if mode == "JSON 여러 개":
        uploaded = st.file_uploader("미로 JSON", type=["json"], accept_multiple_files=True, key="many_json")
        files = list(uploaded or [])
    else:
        zip_file = st.file_uploader("미로 ZIP", type=["zip"], key="many_zip")
        files = [zip_file] if zip_file is not None else []
    if st.button("모두 계산", key="calc_many"):
        _calculate_roster(files, mode, int(runs), int(seed))
    _delete_roster_controls()
    _edit_roster()
    ready = [item for item in _roster() if item.get("result")]
    per_maze, trends = st.tabs(("미로별 비교", "전체 경향 분석"))
    with per_maze:
        if len(ready) < 2:
            st.info("미로별 비교는 계산된 미로가 2개 이상일 때 사용할 수 있습니다.")
        else:
            _show_per_maze(ready)
    with trends:
        bundle = st.session_state.get("trend")
        if bundle is None:
            st.info("전체 경향 분석은 모두 계산을 누른 뒤, 성공한 미로가 8개 이상일 때 나옵니다.")
        else:
            _refresh_trend_names(bundle)
            _show_comparison(bundle, "compare")


def page_generate() -> None:
    """시험용 미로를 메모리에서 만들고, 바로 비교할 수 있게 한다."""
    st.header("미로 만들기")
    sizes_text = st.text_input("크기 목록", value="4, 5, 6", key="sizes_text")
    loops_text = st.text_input("고리 확률 목록", value="0, 0.05, 0.15", key="loops_text")
    count = st.number_input("크기·확률 조합당 개수", min_value=1, value=20, step=1, key="count_make")
    seed = st.number_input("seed", value=0, step=1, key="seed_make")
    dimension = st.radio("차원", ("3D", "2D"), horizontal=True, key="dim_make")
    sizes = _parse_sizes(sizes_text)
    loops = _parse_loops(loops_text)
    ready = sizes is not None and loops is not None
    total = len(sizes) * len(loops) * int(count) if ready else 0
    if not ready:
        st.error("크기 목록은 2 이상의 정수, 고리 확률 목록은 0 이상 1 이하의 숫자로 적어 주세요. 쉼표로 구분합니다.")
    else:
        st.write(f"만들 미로 총수: {total}개")
    blocked = ready and total > MAX_GENERATED
    if blocked:
        st.warning(f"만들 미로가 {total}개입니다. 500개 이하로 줄여 주세요.")
    if st.button("만들기", key="make_mazes", disabled=not ready or blocked):
        _make_mazes(sizes or [], loops or [], int(count), int(seed), dimension == "2D")
    made = st.session_state.get("made")
    if made is not None:
        st.download_button(
            "ZIP 내려받기",
            data=made["zip"],
            file_name="mazes.zip",
            mime="application/zip",
            key="download_made_zip",
        )
        compare_runs = st.number_input(
            "비교에 쓸 트레모 반복 횟수",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            key="runs_made",
        )
        if st.button("만든 미로로 바로 비교하기", key="compare_made"):
            _compare_made(made["docs"], int(compare_runs), int(seed))
    bundle = st.session_state.get("made_compare")
    if bundle is not None:
        _show_comparison(bundle, "made")


def _calculate_upload(uploaded: object, runs: int, seed: int) -> None:
    """올린 JSON 파일 하나를 계산한다."""
    if uploaded is None:
        st.error("JSON 파일을 올려 주세요.")
        return
    if uploaded.size > MAX_BYTES:
        st.error(f"{uploaded.name} 파일이 5MB를 넘습니다. 계산을 중단합니다.")
        return
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeError:
        st.error("JSON 파일을 UTF-8로 읽을 수 없습니다.")
        return
    _calculate_text(text, uploaded.name, runs, seed)


def _calculate_text(text: str, label: str, runs: int, seed: int) -> None:
    """JSON 문자열 하나를 계산해 세션에 남긴다."""
    if not text.strip():
        st.error("계산할 JSON이 비어 있습니다.")
        return
    try:
        result = dict(cached_analysis(text, runs, seed))
        result["file"] = label
        st.session_state["single"] = {"result": result, "content": text, "label": label}
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)


def _show_single() -> None:
    """마지막으로 성공한 단일 계산 결과를 그린다."""
    stored = st.session_state.get("single")
    if stored is None:
        return
    result = stored["result"]
    st.subheader(str(result["file"]))
    left, mid_left, mid_right, right = st.columns(4)
    left.metric("트레모 평균 걸음 수", _four(result["tremaux_mean"]))
    mid_left.metric("탐욕 라우팅 복잡도", _four(result["greedy_difficulty"]))
    mid_right.metric("C_geo (standard)", _four(result["C_geo_standard"]))
    right.metric("C_arith (standard)", _four(result["C_arith_standard"]))
    st.dataframe(_axis_rows(result), use_container_width=True)
    st.dataframe(_detail_rows(result), use_container_width=True)
    try:
        graph = load_document(json.loads(stored["content"]))
    except ValueError as exc:
        st.error(str(exc))
        graph = None
    if graph is not None:
        st.plotly_chart(maze_figure(graph), use_container_width=True)
    if st.button("비교 목록에 추가", key="add_to_roster"):
        _remember_maze(str(result["file"]), stored["content"])
    payload = json.dumps(ordered_result(result), ensure_ascii=False, indent=2) + "\n"
    st.download_button(
        "결과 JSON 내려받기",
        data=payload.encode("utf-8"),
        file_name="result.json",
        mime="application/json",
        key="download_one_json",
    )


def _named_texts(files: list, mode: str) -> list[tuple[str, str]] | None:
    """업로드를 (파일 이름, JSON 문자열)로 바꾼다. 한도를 넘으면 안내하고 None을 반환한다."""
    if mode == "ZIP 하나":
        archive = files[0]
        if archive.size > MAX_BYTES:
            st.error(f"{archive.name} 파일이 5MB를 넘습니다. 계산을 중단합니다.")
            return None
        extracted = read_jsons_from_zip(archive.getvalue())
        pairs = [(name, payload) for name, payload in extracted]
    else:
        if len(files) > MAX_FILES:
            st.error("파일은 최대 1000개까지 올릴 수 있습니다. 계산을 중단합니다.")
            return None
        pairs = []
        for item in files:
            if item.size > MAX_BYTES:
                st.error(f"{item.name} 파일이 5MB를 넘습니다. 계산을 중단합니다.")
                return None
            pairs.append((item.name, item.getvalue()))
    if len(pairs) > MAX_FILES:
        st.error("파일은 최대 1000개까지 올릴 수 있습니다. 계산을 중단합니다.")
        return None
    named: list[tuple[str, str]] = []
    for name, payload in pairs:
        if len(payload) > MAX_BYTES:
            st.error(f"{name} 파일이 5MB를 넘습니다. 계산을 중단합니다.")
            return None
        named.append((name, payload.decode("utf-8-sig")))
    if not named:
        st.error("읽을 JSON 파일이 없습니다.")
        return None
    return named


def _run_batch(
    named: list[tuple[str, str]],
    runs: int,
    seed: int,
    progress: object,
) -> tuple[list[dict], list[dict[str, str]]]:
    """파일 이름 순으로 계산하고, 실패한 파일은 건너뛴다."""
    ordered = sorted(named, key=lambda item: item[0])
    rows: list[dict] = []
    errors: list[dict[str, str]] = []
    total = len(ordered)
    for index, (name, text) in enumerate(ordered):
        progress.progress(index / total, text=f"계산 중: {name}")
        try:
            result = dict(cached_analysis(text, runs, seed + index))
            result["file"] = name
            rows.append(result)
        except Exception as exc:
            errors.append({"file": name, "error": str(exc)})
    progress.progress(1.0, text="완료")
    return rows, errors


def _show_comparison(bundle: dict, key_prefix: str) -> None:
    """결과 표, 실패 목록, 그리고 8개 이상일 때의 상관 분석을 그린다."""
    rows: list[dict] = bundle["rows"]
    errors: list[dict[str, str]] = bundle["errors"]
    st.subheader("전체 결과")
    visible = [{key: value for key, value in row.items() if key != "steps_all"} for row in rows]
    st.dataframe(visible, use_container_width=True)
    st.download_button(
        "CSV 내려받기",
        data=_rows_csv_bytes(rows),
        file_name="results.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )
    if errors:
        st.subheader("실패한 파일")
        for item in errors:
            st.write(f"{item['file']}: {item['error']}")
    if len(rows) < 8:
        st.info("비교하려면 미로가 8개 이상 필요합니다")
        return
    reports: list[GroupReport] = bundle["reports"]
    st.subheader("판정")
    for line in _judgment_lines(reports):
        st.write(line)
    st.subheader("그룹별 상관")
    for report in reports:
        st.markdown(f"**{report.name}** (n={report.count})")
        if not report.enough:
            st.write("표본 부족")
            continue
        st.dataframe(_correlation_rows(report), use_container_width=True)
        st.dataframe(_difference_rows(report), use_container_width=True)
    overall = next((report for report in reports if report.name == "전체"), None)
    captions = (
        "기하평균 복잡도와 트레모 걸음 수",
        "기하평균 복잡도와 탐욕 라우팅 복잡도",
        "스피어만 상관과 95% 구간",
        "결합 방식과 축 가중치의 차이",
    )
    for caption, figure in zip(captions, build_figures(rows, overall)):
        st.caption(caption)
        st.pyplot(figure, clear_figure=True)
    st.download_button(
        "summary.csv 내려받기",
        data=summary_csv_bytes(reports),
        file_name="summary.csv",
        mime="text/csv",
        key=f"{key_prefix}_summary",
    )


def _make_mazes(sizes: list[int], loops: list[float], count: int, seed: int, dim2d: bool) -> None:
    """생성기로 미로를 만들어 세션에 담는다."""
    try:
        documents = generate_documents(sizes, loops, count, seed, dim2d)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return
    st.session_state["made"] = {"docs": documents, "zip": _documents_zip(documents)}
    st.session_state.pop("made_compare", None)
    st.success(f"{len(documents)}개를 만들었습니다.")


def _compare_made(documents: list[tuple[str, dict]], runs: int, seed: int) -> None:
    """만든 미로를 화면 2와 같은 방식으로 계산한다."""
    named = [(name, json.dumps(document, ensure_ascii=False)) for name, document in documents]
    progress = st.progress(0.0, text="준비")
    rows, errors = _run_batch(named, runs, seed, progress)
    reports = analyze_results(rows) if len(rows) >= 8 else []
    st.session_state["made_compare"] = {"rows": rows, "errors": errors, "reports": reports}


def _roster() -> list[dict]:
    """화면 2 비교 목록을 세션에서 꺼낸다."""
    return st.session_state.setdefault("roster", [])


def _remember_maze(filename: str, text: str) -> None:
    """계산해 둔 미로 하나를 비교 목록에 넣는다. 내용은 중복해서 넣지 않는다."""
    if any(item["content"] == text for item in _roster()):
        st.info("이 미로는 이미 비교 목록에 있습니다.")
        return
    _append_maze(filename, text)
    st.success(f"비교 목록에 {_roster()[-1]['name']}을 넣었습니다.")


def _append_maze(filename: str, text: str) -> None:
    """목록에 미로 하나를 추가하고 편집 표의 판을 새로 연다."""
    used = {item["name"] for item in _roster()}
    _roster().append(
        {
            "name": _unique_name(Path(filename).stem or filename, used),
            "group": _default_group(text),
            "content": text,
            "result": None,
        }
    )
    st.session_state["roster_version"] = st.session_state.get("roster_version", 0) + 1


def _calculate_roster(files: list, mode: str, runs: int, seed: int) -> None:
    """올린 파일을 목록에 더한 뒤, 목록 전체를 같은 runs와 seed로 계산한다."""
    if files:
        try:
            named = _named_texts(files, mode)
        except ValueError as exc:
            st.error(str(exc))
            return
        except zipfile.BadZipFile:
            st.error("ZIP 파일을 읽을 수 없습니다.")
            return
        except UnicodeError:
            st.error("JSON 파일을 UTF-8로 읽을 수 없습니다.")
            return
        except Exception as exc:
            st.exception(exc)
            return
        if named is None:
            return
        known = {item["content"] for item in _roster()}
        for filename, text in named:
            if text not in known:
                _append_maze(filename, text)
                known.add(text)
    roster = _roster()
    if not roster:
        st.error("계산할 미로가 없습니다.")
        return
    progress = st.progress(0.0, text="준비")
    errors: list[dict[str, str]] = []
    total = len(roster)
    for index, item in enumerate(roster):
        progress.progress(index / total, text=f"계산 중: {item['name']}")
        try:
            result = dict(cached_analysis(item["content"], runs, seed))
            result["file"] = item["name"]
            item["result"] = result
        except Exception as exc:
            item["result"] = None
            errors.append({"file": item["name"], "error": str(exc)})
    progress.progress(1.0, text="완료")
    rows = []
    contents = []
    for item in roster:
        if item.get("result"):
            rows.append(_public_row(item))
            contents.append(item["content"])
    st.session_state["trend"] = {
        "rows": rows,
        "errors": errors,
        "reports": analyze_results(rows) if len(rows) >= 8 else [],
        "contents": contents,
    }


def _public_row(item: dict) -> dict:
    """CSV와 경향 분석에 넘길 결과에서 걸음 수 목록을 뺀다."""
    result = item["result"]
    row = {key: value for key, value in result.items() if key != "steps_all"}
    row["file"] = item["name"]
    return row


def _refresh_trend_names(bundle: dict) -> None:
    """이름만 바뀐 목록을 경향 분석 표의 파일 열에 반영한다."""
    by_content = {item["content"]: item["name"] for item in _roster()}
    for row, content in zip(bundle["rows"], bundle.get("contents", [])):
        name = by_content.get(content)
        if name is not None:
            row["file"] = name


def _delete_roster_controls() -> None:
    """목록에서 하나를 지우거나 전체를 비운다."""
    roster = _roster()
    if not roster:
        return
    labels = [f"{index + 1}. {item['name']}" for index, item in enumerate(roster)]
    choice = st.selectbox("지울 미로", labels, key="delete_choice")
    left, right = st.columns(2)
    if left.button("목록에서 지우기", key="delete_one"):
        del roster[labels.index(choice)]
        st.session_state["roster_version"] = st.session_state.get("roster_version", 0) + 1
        st.session_state.pop("trend", None)
    if right.button("전체 비우기", key="delete_all"):
        st.session_state["roster"] = []
        st.session_state["roster_version"] = st.session_state.get("roster_version", 0) + 1
        st.session_state.pop("trend", None)


def _edit_roster() -> None:
    """이름과 그룹만 고친다. 계산 결과는 그대로 둔다."""
    roster = _roster()
    if not roster:
        st.info("비교 목록이 비어 있습니다. JSON을 올리거나 화면 1에서 추가하세요.")
        return
    version = st.session_state.get("roster_version", 0)
    edited = st.data_editor(
        [{"이름": item["name"], "그룹": item["group"]} for item in roster],
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key=f"roster_editor_{version}",
    )
    records = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    _apply_roster_edits(roster, records)


def _apply_roster_edits(roster: list[dict], records: list[dict]) -> None:
    """이름과 그룹만 반영한다. 복잡도 계산은 호출하지 않는다."""
    for item, record in zip(roster, records):
        item["name"] = str(record.get("이름", item["name"]))
        item["group"] = str(record.get("그룹", item["group"]))
        if item.get("result"):
            item["result"]["file"] = item["name"]


def _show_per_maze(ready: list[dict]) -> None:
    """계산된 미로를 표와 그래프로 나란히 비교한다."""
    variant = st.selectbox("축 변형", AXIS_WEIGHTS, key="axis_variant")
    names = [item["name"] for item in ready]
    groups = [item["group"] for item in ready]
    results = [item["result"] for item in ready]
    table = build_table(results, names, groups, variant)
    sort_by = st.selectbox("정렬 기준", ("목록 순서", *COMPARE_METRICS), key="sort_metrics")
    if sort_by != "목록 순서":
        order = sorted(range(len(table)), key=lambda index: table[index][sort_by], reverse=True)
        table = [table[index] for index in order]
        ready = [ready[index] for index in order]
        names = [item["name"] for item in ready]
        groups = [item["group"] for item in ready]
        results = [item["result"] for item in ready]
    st.subheader("비교 표")
    frame = pd.DataFrame(table)
    st.dataframe(_highlight_hardest(frame), use_container_width=True, key="compare_table")
    st.download_button(
        "비교 표 CSV 내려받기",
        data=_table_csv(table),
        file_name="compare.csv",
        mime="text/csv",
        key="per_maze_csv",
    )
    st.plotly_chart(
        comparison_bars(
            names,
            groups,
            [(metric, [float(row[metric]) for row in table]) for metric in COMPARE_METRICS],
            [float(result["tremaux_std"]) for result in results],
        ),
        use_container_width=True,
    )
    st.caption("막대가 클수록 그 지표에서 더 어려운 미로입니다. 트레모 칸의 오차막대는 걸음 수의 표준편차입니다.")
    ratios = [(metric, normalize_by_max([float(row[metric]) for row in table])) for metric in COMPARE_METRICS]
    st.plotly_chart(ratio_bars(names, groups, ratios), use_container_width=True)
    blocked = [metric for metric, _values in ratios if max(float(row[metric]) for row in table) == 0]
    if blocked:
        st.caption("모든 미로에서 값이 0이라 비교할 수 없는 지표: " + ", ".join(blocked))
    st.caption("막대 높이가 1이면 그 지표에서 가장 어려운 미로입니다.")
    st.subheader("순위 일치도")
    if len(table) < 3:
        st.info("순위를 비교하려면 미로가 3개 이상 필요합니다.")
    else:
        if len(table) < 8:
            st.caption("미로 수가 적어 참고용입니다.")
        matrix = rank_agreement(table, list(COMPARE_METRICS))
        st.plotly_chart(agreement_heatmap(matrix, list(COMPARE_METRICS)), use_container_width=True)
        st.caption("1에 가까울수록 두 지표가 미로의 어려운 순서를 같게 매긴다는 뜻입니다.")
    st.subheader("두 지표 산점도")
    scatter_fields = [*COMPARE_METRICS, "정점 수", "전체 통로 길이", "최단거리", "고리 수"]
    x_name = st.selectbox("가로축", scatter_fields, index=scatter_fields.index(METRIC_CGEO), key="scatter_x")
    y_name = st.selectbox("세로축", scatter_fields, index=scatter_fields.index(METRIC_TREMAUX), key="scatter_y")
    x_values = [float(row[x_name]) for row in table]
    y_values = [float(row[y_name]) for row in table]
    log_x_blocked = any(value <= 0 for value in x_values)
    log_y_blocked = any(value <= 0 for value in y_values)
    log_x = st.checkbox("가로축 로그 눈금", key="log_x", disabled=log_x_blocked)
    log_y = st.checkbox("세로축 로그 눈금", key="log_y", disabled=log_y_blocked)
    if log_x_blocked:
        st.caption("가로축에 0 이하인 값이 있어 로그 눈금을 쓸 수 없습니다.")
    if log_y_blocked:
        st.caption("세로축에 0 이하인 값이 있어 로그 눈금을 쓸 수 없습니다.")
    st.plotly_chart(
        metric_scatter(names, groups, x_values, y_values, x_name, y_name, log_x and not log_x_blocked, log_y and not log_y_blocked),
        use_container_width=True,
    )
    st.caption("점의 위치는 고른 두 지표의 값이고, 색은 그룹입니다. 점 옆에 미로 이름이 붙습니다.")
    st.subheader("트레모 걸음 수 분포")
    st.plotly_chart(
        steps_boxes(names, groups, [list(result["steps_all"]) for result in results]),
        use_container_width=True,
    )
    st.caption("상자가 길수록 갈림길에서 어느 길을 고르느냐에 따라 걸음 수가 크게 달라지는 미로입니다.")
    st.subheader("미로 그림 나란히 보기")
    picked = st.multiselect("미로 (최대 4개)", names, max_selections=4, key="side_by_side")
    ordered = [name for name in names if name in picked]
    by_name = {item["name"]: item for item in ready}
    for start in range(0, len(ordered), 2):
        columns = st.columns(2)
        for offset, name in enumerate(ordered[start : start + 2]):
            with columns[offset]:
                st.markdown(f"**{name}**")
                try:
                    graph = load_document(json.loads(by_name[name]["content"]))
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.plotly_chart(maze_figure(graph), use_container_width=True)
    st.caption("입구는 초록, 출구는 빨강, 3축 갈림길은 주황입니다. 마우스로 돌려 볼 수 있습니다.")


def _highlight_hardest(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """지표 열에서 가장 어려운 값의 칸을 칠한다."""

    def paint(column: pd.Series) -> list[str]:
        if column.name not in COMPARE_METRICS:
            return [""] * len(column)
        hardest = column.max()
        return ["background-color: #ffe08a" if value == hardest else "" for value in column]

    return frame.style.apply(paint)


def _table_csv(rows: list[dict]) -> bytes:
    """비교 표를 UTF-8 BOM CSV로 만든다."""
    fieldnames = list(rows[0]) if rows else []
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8-sig")


def _unique_name(base: str, used: set[str]) -> str:
    """겹치는 이름 뒤에 _2, _3을 붙여 새 이름을 만든다."""
    if base not in used:
        return base
    number = 2
    while f"{base}_{number}" in used:
        number += 1
    return f"{base}_{number}"


def _default_group(text: str) -> str:
    """meta.group이 있으면 그것을, 없으면 3D 또는 2D를 쓴다."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "2D"
    if isinstance(data, dict):
        meta = data.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("group"), str) and meta["group"]:
            return str(meta["group"])
        try:
            return "3D" if load_document(data).is_3d() else "2D"
        except ValueError:
            return "2D"
    return "2D"


@st.cache_data(show_spinner=False)
def cached_analysis(content: str, runs: int, seed: int) -> dict:
    """같은 JSON, 반복 횟수, 시드면 다시 계산하지 않는다."""
    graph = load_document(json.loads(content))
    result = analyze_graph(graph, "memory.json", runs, seed)
    return {key: _plain(value) for key, value in result.items()}


def _plain(value: object) -> object:
    """캐시에 넣기 쉽게 numpy 스칼라를 파이썬 값으로 바꾼다."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _example_map() -> dict[str, Path]:
    """examples의 T1부터 T8 파일을 화면 라벨로 묶는다."""
    labeled: dict[str, Path] = {}
    for path in sorted((ROOT / "examples").glob("t*.json")):
        stem = path.stem
        if "_" in stem:
            head, tail = stem.split("_", 1)
            label = f"{head.upper()} {tail}"
        else:
            label = stem.upper()
        labeled[label] = path
    return labeled


def _axis_rows(result: dict) -> list[dict[str, object]]:
    """축 가중치 네 종의 흡수시간과 복잡도를 표로 만든다."""
    rows = []
    for mode in AXIS_WEIGHTS:
        rows.append(
            {
                "축 가중치": mode,
                "tau_length": result[f"tau_length_{mode}"],
                "tau_degree": result[f"tau_degree_{mode}"],
                "C_geo": result[f"C_geo_{mode}"],
                "C_arith": result[f"C_arith_{mode}"],
            }
        )
    return rows


def _detail_rows(result: dict) -> list[dict[str, object]]:
    """크기, 고리, 트레모, 탐욕 라우팅 요약을 표로 만든다."""
    pairs = (
        ("정점 수", result["n_nodes"]),
        ("간선 수", result["n_edges"]),
        ("고리 수", result["n_loops"]),
        ("전체 통로 길이", result["L_total"]),
        ("최단거리", result["shortest_path"]),
        ("3D 여부", "예" if result["is_3d"] else "아니오"),
        ("트레모 표준편차", result["tremaux_std"]),
        ("트레모 최소", result["tremaux_min"]),
        ("트레모 최대", result["tremaux_max"]),
        ("탐욕 성공률", result["greedy_success_rate"]),
        ("입구 출발 성공 여부", "예" if result["greedy_entry_success"] else "아니오"),
    )
    return [{"항목": name, "값": value} for name, value in pairs]


def _judgment_lines(reports: list[GroupReport]) -> list[str]:
    """그룹별 판정 문장과 경고를 한 줄씩 만든다."""
    lines: list[str] = []
    for report in reports:
        if not report.enough:
            lines.append(f"{report.name}: 표본 부족 (n={report.count})")
            continue
        for row in report.differences:
            lines.append(f"{report.name}: {row.front} 대 {row.back}, {row.target} — {row.judgment}")
        for warning in report.warnings:
            lines.append(f"{report.name}: {warning}")
    return lines


def _correlation_rows(report: GroupReport) -> list[dict[str, object]]:
    """한 그룹의 스피어만 ρ, 구간, 피어슨 r을 표로 만든다."""
    return [
        {
            "변수": row.predictor,
            "검증 기준": row.target,
            "스피어만 ρ": row.spearman,
            "95% 하한": row.ci_low,
            "95% 상한": row.ci_high,
            "피어슨 r": row.pearson,
        }
        for row in report.correlations
    ]


def _difference_rows(report: GroupReport) -> list[dict[str, object]]:
    """결합 방식과 축 가중치 차이의 구간과 판정을 표로 만든다."""
    return [
        {
            "비교": "결합 방식" if row.kind == "combine" else "축 가중치",
            "앞": row.front,
            "뒤": row.back,
            "검증 기준": row.target,
            "95% 하한": row.ci_low,
            "95% 상한": row.ci_high,
            "판정": row.judgment,
        }
        for row in report.differences
    ]


def _rows_csv_bytes(rows: list[dict]) -> bytes:
    """일괄 결과 표를 UTF-8 BOM CSV로 만든다."""
    meta = sorted({key for row in rows for key in row if key.startswith("meta_") and key not in RESULT_KEYS})
    fieldnames = list(RESULT_KEYS) + meta
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key, "")) for key in fieldnames})
    return handle.getvalue().encode("utf-8-sig")


def _documents_zip(documents: list[tuple[str, dict]]) -> bytes:
    """미로 문서들을 메모리 안의 zip으로 묶는다."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, document in documents:
            payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
            archive.writestr(name, payload.encode("utf-8"))
    return buffer.getvalue()


def _parse_sizes(text: str) -> list[int] | None:
    """쉼표나 공백으로 적힌 격자 크기 목록을 읽는다."""
    parts = text.replace(",", " ").split()
    if not parts:
        return None
    values: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        values.append(int(part))
    return values


def _parse_loops(text: str) -> list[float] | None:
    """쉼표나 공백으로 적힌 고리 확률 목록을 읽는다."""
    parts = text.replace(",", " ").split()
    if not parts:
        return None
    values: list[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            return None
        if not 0.0 <= value <= 1.0:
            return None
        values.append(value)
    return values


def _four(value: object) -> str:
    """화면의 큰 숫자를 소수 넷째 자리까지 적는다."""
    return f"{float(value):.4f}"


def _csv_cell(value: object) -> object:
    """CSV 칸의 참·거짓을 true/false로 적는다."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


if __name__ == "__main__":
    main()
