"""Benchmarks of Series methods."""

import pytest
from config import cudf, cupy


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.Series, cupy.random.rand(N))


def bench_sort_values(benchmark, series_dtype_int):
    benchmark(series_dtype_int.sort_values)


@pytest.mark.parametrize("n", [10])
def bench_series_nsmallest(benchmark, series_dtype_int, n):
    benchmark(series_dtype_int.nsmallest, n)