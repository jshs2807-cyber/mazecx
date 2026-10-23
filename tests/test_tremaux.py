"""트레모 걸음 수 테스트."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from mazecx.io import load_path
from mazecx.tremaux import tremaux_statistics

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples"


def near(expected: float, tolerance: float):
    """주어진 절대오차로 비교한다."""
    return pytest.approx(expected, abs=tolerance, rel=0)


def load_clean(name: str):
    """예제 파일을 읽는다. 예상 밖 경고가 나면 실패한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return load_path(EX / name)


def test_t1_always_five_steps() -> None:
    """곧은 통로는 모든 실행의 걸음 수가 5다."""
    stats = tremaux_statistics(load_clean("t1.json"), runs=50, seed=0)
    assert set(stats.samples) == {5}
    assert stats.mean == near(5, 1e-9)
    assert stats.std == near(0, 1e-9)
    assert stats.minimum == 5
    assert stats.maximum == 5
    assert stats.norm == near(1, 1e-9)


def test_t2_mean_is_six() -> None:
    """가능한 걸음 수는 5 또는 7이고, 20000회 평균은 6 ± 0.05다."""
    stats = tremaux_statistics(load_clean("t2.json"), runs=20000, seed=0)
    assert set(stats.samples) <= {5, 7}
    assert stats.mean == near(6, 0.05)
    assert stats.minimum == 5
    assert stats.maximum == 7


def test_t3_mean_is_twelve() -> None:
    """가능한 걸음 수는 10 또는 14이고, 20000회 평균은 12 ± 0.1이다."""
    stats = tremaux_statistics(load_clean("t3.json"), runs=20000, seed=0)
    assert set(stats.samples) <= {10, 14}
    assert stats.mean == near(12, 0.1)
    assert stats.minimum == 10
    assert stats.maximum == 14


def test_t4_mean_is_seven() -> None:
    """가능한 걸음 수는 5 또는 9이고, 20000회 평균은 7 ± 0.1이다."""
    stats = tremaux_statistics(load_clean("t4.json"), runs=20000, seed=0)
    assert set(stats.samples) <= {5, 9}
    assert stats.mean == near(7, 0.1)
    assert stats.minimum == 5
    assert stats.maximum == 9


def test_t8_always_four_steps() -> None:
    """정사각 고리는 어느 쪽으로 먼저 가도 걸음 수가 4다."""
    stats = tremaux_statistics(load_clean("t8.json"), runs=30, seed=0)
    assert set(stats.samples) == {4}
    assert stats.mean == near(4, 1e-9)


def test_t7_same_seed_is_identical() -> None:
    """같은 입력과 같은 seed면 평균, 표준편차, 최소, 최대가 같다."""
    graph = load_clean("t3.json")
    first = tremaux_statistics(graph, runs=100, seed=0)
    second = tremaux_statistics(graph, runs=100, seed=0)
    assert first.mean == second.mean
    assert first.std == second.std
    assert first.minimum == second.minimum
    assert first.maximum == second.maximum
    assert first.samples == second.samples


def test_different_seeds_change_the_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """실행마다 생성기를 새로 만들지 않고, seed가 다르면 표본도 달라진다."""
    created: list[np.random.Generator] = []
    real_rng = np.random.default_rng

    def spy(seed: int) -> np.random.Generator:
        generator = real_rng(seed)
        created.append(generator)
        return generator

    monkeypatch.setattr(np.random, "default_rng", spy)
    graph = load_clean("t2.json")
    stats = tremaux_statistics(graph, runs=30, seed=0)
    assert len(created) == 1
    assert set(stats.samples) <= {5, 7}
    other = tremaux_statistics(graph, runs=30, seed=1)
    assert stats.samples != other.samples


def test_numpy_generator_is_the_only_rng() -> None:
    """난수는 numpy Generator만 사용한다."""
    source = (ROOT / "mazecx" / "tremaux.py").read_text(encoding="utf-8")
    assert "default_rng" in source
    assert "import random" not in source
