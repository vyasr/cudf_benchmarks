"""Benchmarks of Index methods."""

import pytest
from config import cudf, cupy
from utils import cudf_benchmark


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.Index, cupy.random.rand(N))


@cudf_benchmark(cls="index", dtype="int", nulls=False)
def bench_sort_values(benchmark, index):
    benchmark(index.sort_values)
