import cudf
import pytest


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_rangeindex_column(benchmark, N):
    obj = cudf.RangeIndex(range(N))
    benchmark(obj._column)


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_rangeindex_columns(benchmark, N):
    obj = cudf.RangeIndex(range(N))
    benchmark(obj._columns)