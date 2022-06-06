"""Benchmarks of Index methods."""

import pytest
from config import cudf, cupy


@pytest.mark.parametrize("N", [100, 1_000_000])
def test_construction(benchmark, N):
    benchmark(cudf.Index, cupy.random.rand(N))


def test_sort_values(benchmark, index_dtype_int_nulls_false):
    benchmark(index_dtype_int_nulls_false.sort_values)
