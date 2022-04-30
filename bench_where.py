import cudf
import pytest


@pytest.mark.parametrize("N", [1_000, 100_000, 10_000_000])
def test_rangeindex_where(benchmark, N):
    obj = cudf.RangeIndex(range(N))
    benchmark(obj.where)