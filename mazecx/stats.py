"""스피어만·피어슨 상관과 부트스트랩 구간. numpy만 사용한다."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Correlation = Callable[[np.ndarray, np.ndarray], float]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """동점은 평균 순위로 바꾼 뒤 순위끼리의 피어슨 상관을 반환한다."""
    left, right = _as_pair(x, y)
    if left.size < 2 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("nan")
    return pearson(_average_ranks(left), _average_ranks(right))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """피어슨 상관계수를 반환한다. 한쪽이 모두 같으면 NaN이다."""
    left, right = _as_pair(x, y)
    if left.size < 2 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    func: Correlation,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """행을 복원추출해 상관계수의 2.5%와 97.5% 분위수를 반환한다."""
    left, right = _as_pair(x, y)
    values = _resample(left, right, None, None, func, n_boot, seed, paired=False)
    return _quantile_interval(values, n_boot)


def paired_diff_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    func: Correlation,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """같은 복원추출로 func(x1, y) − func(x2, y)의 95% 구간을 반환한다."""
    first, target = _as_pair(x1, y)
    second, target_again = _as_pair(x2, y)
    if target.shape != target_again.shape:
        raise ValueError("x1, x2, y의 길이가 같아야 합니다.")
    values = _resample(first, target, second, target, func, n_boot, seed, paired=True)
    return _quantile_interval(values, n_boot)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """작은 값부터 1로 매기고, 동점은 그 자리들의 평균 순위를 준다."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    size = values.size
    while start < size:
        end = start + 1
        while end < size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _as_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """두 입력을 float 1차원 배열로 맞추고 길이가 같은지 확인한다."""
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("x와 y는 같은 길이여야 합니다.")
    return left, right


def _resample(
    x1: np.ndarray,
    y: np.ndarray,
    x2: np.ndarray | None,
    y2: np.ndarray | None,
    func: Correlation,
    n_boot: int,
    seed: int,
    paired: bool,
) -> list[float]:
    """복원추출 반복에서 유한한 통계량만 모은다."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot는 1 이상의 정수여야 합니다.")
    rng = np.random.default_rng(seed)
    size = y.size
    kept: list[float] = []
    for _ in range(n_boot):
        index = rng.integers(0, size, size=size)
        if paired:
            assert x2 is not None and y2 is not None
            first = func(x1[index], y[index])
            second = func(x2[index], y2[index])
            if np.isfinite(first) and np.isfinite(second):
                kept.append(float(first - second))
        else:
            value = func(x1[index], y[index])
            if np.isfinite(value):
                kept.append(float(value))
    return kept


def _quantile_interval(values: list[float], n_boot: int) -> tuple[float, float]:
    """유효 반복이 90% 미만이면 (NaN, NaN), 아니면 2.5%·97.5% 분위수를 반환한다."""
    if len(values) < 0.9 * n_boot:
        return (float("nan"), float("nan"))
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975], method="linear")
    return float(low), float(high)
