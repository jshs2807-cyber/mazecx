"""미로 복잡도 계산의 명령행 진입점."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from mazecx.complexity import AXIS_WEIGHTS, compute_complexity
from mazecx.generate import generate_dataset
from mazecx.graph import SpatialGraph
from mazecx.greedy import greedy_statistics
from mazecx.io import load_path, parse_meta
from mazecx.tremaux import tremaux_statistics

RESULT_KEYS = (
    "file",
    "error",
    "is_3d",
    "grid_size",
    "n_nodes",
    "n_edges",
    "n_loops",
    "L_total",
    "shortest_path",
    "tau_length_standard",
    "tau_degree_standard",
    "C_geo_standard",
    "C_arith_standard",
    "tau_length_none",
    "tau_degree_none",
    "C_geo_none",
    "C_arith_none",
    "tau_length_a",
    "tau_degree_a",
    "C_geo_a",
    "C_arith_a",
    "tau_length_inv",
    "tau_degree_inv",
    "C_geo_inv",
    "C_arith_inv",
    "tremaux_mean",
    "tremaux_std",
    "tremaux_min",
    "tremaux_max",
    "tremaux_norm",
    "greedy_success_rate",
    "greedy_difficulty",
    "greedy_entry_success",
    "greedy_GR_score",
    "runs",
    "seed",
)


def analyze_graph(graph: SpatialGraph, file_label: str, runs: int, seed: int) -> dict:
    """그래프 하나의 복잡도 네 변형, 트레모, 탐욕 라우팅 결과를 반환한다."""
    _check_options(runs, seed)
    result: dict = {
        "file": file_label,
        "error": "",
        "is_3d": graph.is_3d(),
        "grid_size": graph.grid_size(),
        "n_nodes": graph.n_nodes,
        "n_edges": graph.n_edges,
        "n_loops": graph.n_loops(),
        "L_total": graph.total_length(),
        "shortest_path": graph.shortest_path(),
    }
    for mode in AXIS_WEIGHTS:
        complexity = compute_complexity(graph, mode)
        result[f"tau_length_{mode}"] = complexity.tau_length
        result[f"tau_degree_{mode}"] = complexity.tau_degree
        result[f"C_geo_{mode}"] = complexity.c_geo
        result[f"C_arith_{mode}"] = complexity.c_arith
    tremaux = tremaux_statistics(graph, runs, seed)
    greedy = greedy_statistics(graph)
    result.update(
        {
            "tremaux_mean": tremaux.mean,
            "tremaux_std": tremaux.std,
            "tremaux_min": tremaux.minimum,
            "tremaux_max": tremaux.maximum,
            "tremaux_norm": tremaux.norm,
            "greedy_success_rate": greedy.success_rate,
            "greedy_difficulty": greedy.difficulty,
            "greedy_entry_success": greedy.entry_success,
            "greedy_GR_score": greedy.gr_score,
            "runs": runs,
            "seed": seed,
        }
    )
    for key in sorted(graph.meta):
        result[f"meta_{key}"] = graph.meta[key]
    return result


def analyze_file(
    path: str | Path,
    runs: int = 100,
    seed: int = 0,
    file_label: str | None = None,
) -> dict:
    """JSON 파일 하나를 읽어 분석 결과 딕셔너리를 반환한다."""
    label = str(path) if file_label is None else file_label
    graph = load_path(path)
    return analyze_graph(graph, label, runs, seed)


def format_report(result: dict) -> str:
    """분석 결과를 사람이 읽기 좋은 표 문자열로 만든다."""
    lines = [
        "기본",
        _row("file", result["file"]),
        _row("is_3d", result["is_3d"]),
        _row("grid_size", result["grid_size"]),
        _row("n_nodes", result["n_nodes"]),
        _row("n_edges", result["n_edges"]),
        _row("n_loops", result["n_loops"]),
        _row("L_total", result["L_total"]),
        _row("shortest_path", result["shortest_path"]),
        _row("runs", result["runs"]),
        _row("seed", result["seed"]),
        "",
        "복잡도",
        f"{'variant':<16}{'tau_length':>14}{'tau_degree':>14}{'C_geo':>14}{'C_arith':>14}",
    ]
    for mode in AXIS_WEIGHTS:
        lines.append(
            f"{mode:<16}"
            f"{_fmt(result[f'tau_length_{mode}']):>14}"
            f"{_fmt(result[f'tau_degree_{mode}']):>14}"
            f"{_fmt(result[f'C_geo_{mode}']):>14}"
            f"{_fmt(result[f'C_arith_{mode}']):>14}"
        )
    lines.extend(
        [
            "",
            "트레모",
            _row("tremaux_mean", result["tremaux_mean"]),
            _row("tremaux_std", result["tremaux_std"]),
            _row("tremaux_min", result["tremaux_min"]),
            _row("tremaux_max", result["tremaux_max"]),
            _row("tremaux_norm", result["tremaux_norm"]),
            "",
            "탐욕 라우팅",
            _row("greedy_success_rate", result["greedy_success_rate"]),
            _row("greedy_difficulty", result["greedy_difficulty"]),
            _row("greedy_entry_success", result["greedy_entry_success"]),
            _row("greedy_GR_score", result["greedy_GR_score"]),
        ]
    )
    meta_lines = [_row(key, result[key]) for key in result if key.startswith("meta_")]
    if meta_lines:
        lines.extend(["", "meta", *meta_lines])
    return "\n".join(lines)


def write_result_json(result: dict, path: str | Path) -> None:
    """분석 결과를 UTF-8 JSON 파일로 저장한다."""
    Path(path).write_text(
        json.dumps(ordered_result(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ordered_result(result: dict) -> dict:
    """명세 열 순서 뒤에 meta 키를 사전순으로 붙인 딕셔너리를 반환한다."""
    payload = {key: result.get(key, "") for key in RESULT_KEYS}
    for key in sorted(name for name in result if name.startswith("meta_")):
        payload[key] = result[key]
    return payload


def batch_folder(folder: str | Path, runs: int, seed: int, out: str | Path) -> None:
    """폴더 바로 아래의 JSON을 파일명 순으로 계산해 CSV 한 개로 저장한다.

    파일 순번은 0부터다. i번째 파일의 시드는 seed + i 이고,
    검증에 실패한 파일은 error 열에 이유를 적고 다음 파일로 넘어간다.
    meta 키는 모든 파일의 합집합을 사전순 meta_<키> 열로 붙인다.
    """
    _check_options(runs, seed)
    directory = Path(folder)
    if not directory.is_dir():
        raise ValueError(f"폴더가 아닙니다: {directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix == ".json"),
        key=lambda path: path.name,
    )
    packed: list[tuple[dict, dict[str, str | int | float]]] = []
    for index, path in enumerate(files):
        file_seed = seed + index
        meta: dict[str, str | int | float] = {}
        try:
            meta = _peek_meta(path)
            row = analyze_file(path, runs=runs, seed=file_seed, file_label=path.name)
            print(f"{path.name}: 완료")
        except (ValueError, RuntimeError) as exc:
            row = {key: "" for key in RESULT_KEYS}
            row["file"] = path.name
            row["error"] = str(exc)
            row["runs"] = runs
            row["seed"] = file_seed
            print(f"{path.name}: 오류: {exc}")
        packed.append((row, meta))
    meta_names = sorted({key for _row, meta in packed for key in meta})
    rows: list[dict] = []
    for row, meta in packed:
        for name in meta_names:
            row[f"meta_{name}"] = meta.get(name, "")
        rows.append(row)
    _write_csv(rows, meta_names, out)
    failed = sum(1 for row, _meta in packed if row["error"])
    print(f"저장: {out} (성공 {len(rows) - failed}, 실패 {failed})")


def build_parser() -> argparse.ArgumentParser:
    """analyze, batch, generate, report 하위 명령을 갖춘 인자 해석기를 만든다."""
    parser = argparse.ArgumentParser(
        prog="mazecx.cli",
        description="공간 네트워크 미로의 복잡도를 계산한다.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="JSON 파일 하나를 분석한다.")
    analyze.add_argument("file", help="그래프 또는 미로 JSON 파일")
    analyze.add_argument("--runs", type=int, default=100, help="트레모 반복 횟수 (기본 100)")
    analyze.add_argument("--seed", type=int, default=0, help="난수 시드 (기본 0)")
    analyze.add_argument("--out", default=None, help="결과를 저장할 JSON 경로")

    batch = subparsers.add_parser("batch", help="폴더의 JSON을 CSV로 일괄 계산한다.")
    batch.add_argument("folder", help="JSON 파일이 들어 있는 폴더")
    batch.add_argument("--runs", type=int, default=100, help="트레모 반복 횟수 (기본 100)")
    batch.add_argument("--seed", type=int, default=0, help="첫 파일의 난수 시드 (기본 0)")
    batch.add_argument("--out", required=True, help="결과를 저장할 CSV 경로")

    generate = subparsers.add_parser("generate", help="시험용 미로 JSON을 만든다.")
    generate.add_argument("--out", required=True, help="JSON을 저장할 폴더")
    generate.add_argument("--sizes", type=int, nargs="+", required=True, help="한 변의 칸 수")
    generate.add_argument("--loops", type=float, nargs="+", required=True, help="추가로 벽을 뚫을 확률")
    generate.add_argument("--count", type=int, default=20, help="조합마다 만들 개수 (기본 20)")
    generate.add_argument("--seed", type=int, default=0, help="난수 시드 (기본 0)")
    generate.add_argument("--dim2d", action="store_true", help="n×n 평면 미로를 만든다")

    report = subparsers.add_parser("report", help="결과 CSV로 HTML 보고서를 만든다.")
    report.add_argument("csv", help="batch가 만든 results.csv")
    report.add_argument("--out", required=True, help="보고서를 저장할 폴더")
    report.add_argument("--banner", default=None, help="보고서 맨 위에 넣을 안내 문장")
    return parser


def main(argv: list[str] | None = None) -> None:
    """명령행 인자를 읽어 분석, 생성, 보고서 작성을 실행한다."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze_file(args.file, runs=args.runs, seed=args.seed)
            print(format_report(result))
            if args.out:
                write_result_json(result, args.out)
                print(f"저장: {args.out}")
        elif args.command == "batch":
            batch_folder(args.folder, runs=args.runs, seed=args.seed, out=args.out)
        elif args.command == "generate":
            written = generate_dataset(
                args.out,
                sizes=list(args.sizes),
                loops=list(args.loops),
                count=args.count,
                seed=args.seed,
                dim2d=args.dim2d,
            )
            print(f"생성: {len(written)}개 → {args.out}")
        elif args.command == "report":
            from mazecx.report import build_report

            build_report(args.csv, args.out, banner=args.banner)
            print(f"보고서: {args.out}")
        else:
            raise ValueError(f"알 수 없는 명령입니다: {args.command}")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)


def _check_options(runs: int, seed: int) -> None:
    """반복 횟수와 시드가 정수 조건에 맞는지 확인한다."""
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs는 1 이상의 정수여야 합니다.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed는 정수여야 합니다.")


def _peek_meta(path: Path) -> dict[str, str | int | float]:
    """파일의 meta만 읽는다. meta가 없으면 빈 딕셔너리를 반환한다."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return parse_meta(data)


def _fmt(value: object) -> str:
    """표에 넣을 값을 짧은 문자열로 만든다."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _row(key: str, value: object) -> str:
    """키와 값 한 줄을 만든다."""
    return f"{key:<24}{_fmt(value)}"


def _csv_cell(value: object) -> object:
    """CSV 칸에 쓸 값으로 바꾼다. 참·거짓은 true/false로 적는다."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(rows: list[dict], meta_names: list[str], path: str | Path) -> None:
    """결과 행을 UTF-8 BOM CSV로 저장한다."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RESULT_KEYS) + [f"meta_{name}" for name in meta_names]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in fieldnames})


if __name__ == "__main__":
    main()
