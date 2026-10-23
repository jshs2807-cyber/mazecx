"""상관계수와 부트스트랩 구간 테스트."""

from __future__ import annotations

import math

import pytest

from mazecx.stats import bootstrap_ci, paired_diff_ci, pearson, spearman


def near(expected: float):
    """절대오차 1e-9로 비교한다."""
    return pytest.approx(expected, abs=1e-9, rel=0)


def test_spearman_without_ties() -> None:
    """동점이 없는 순위의 스피어만 상관."""
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == near(0.8)


def test_spearman_with_ties() -> None:
    """동점은 평균 순위로 처리한다."""
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == near(0.9486832980505139)
    assert spearman([10, 20, 20, 30, 40], [3, 1, 2, 5, 4]) == near(0.5642880936468347)


def test_pearson_known_value() -> None:
    """피어슨 상관의 검산값."""
    assert pearson([1, 2, 3, 4], [2, 4, 5, 9]) == near(0.9647638212377322)


def test_constant_input_is_nan() -> None:
    """한쪽 값이 모두 같으면 예외 없이 NaN을 반환한다."""
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(pearson([4, 4, 4, 4], [1, 2, 3, 4]))


def test_bootstrap_same_seed_matches() -> None:
    """같은 시드로 부트스트랩 구간을 두 번 구하면 결과가 같다."""
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [2, 1, 4, 3, 8, 7, 5, 6]
    first = bootstrap_ci(x, y, spearman, n_boot=40, seed=3)
    second = bootstrap_ci(x, y, spearman, n_boot=40, seed=3)
    assert first == second
    assert math.isfinite(first[0]) and math.isfinite(first[1])
    paired = paired_diff_ci(x, y, y, spearman, n_boot=40, seed=3)
    again = paired_diff_ci(x, y, y, spearman, n_boot=40, seed=3)
    assert paired == again
