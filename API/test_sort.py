import cudf
import cupy as cp
import pytest


@pytest.mark.parametrize("cls", [cudf.Series, cudf.Index])
@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_series_index_sort_values(benchmark, cls, N):
    obj = cls(cp.random.rand(N))
    benchmark(obj.sort_values)


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
@pytest.mark.parametrize("n", [10])
def test_series_nsmallest(benchmark, N, n):
    ser = cudf.Series(cp.random.rand(N))
    benchmark(ser.nsmallest, n)
