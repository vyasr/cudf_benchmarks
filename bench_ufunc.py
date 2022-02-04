import cudf
import numpy as np
import cupy as cp
import pytest


@pytest.mark.parametrize("ufunc", [np.add, np.logical_and, np.bitwise_and])
@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("has_nulls", [False, True])
def test_ufunc_series_binary(benchmark, ufunc, N, has_nulls):
    sers = [cudf.Series(cp.random.randint(low=1, high=100, size=N))
            for _ in range(2)]
    if has_nulls:
        for ser in sers:
            ser[::2] = None

    benchmark(ufunc, *sers)
