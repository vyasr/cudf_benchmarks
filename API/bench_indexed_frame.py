"""Benchmarks of IndexedFrame methods."""

import pytest


@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def bench_scans(benchmark, op, indexedframe_dtype_int):
    benchmark(getattr(indexedframe_dtype_int, op))


@pytest.mark.parametrize("op", ["sum", "product", "mean"])
def bench_reductions(benchmark, op, indexedframe_dtype_int):
    benchmark(getattr(indexedframe_dtype_int, op))


def bench_drop_duplicates(benchmark, indexedframe_dtype_int):
    benchmark(indexedframe_dtype_int.drop_duplicates)


def bench_rangeindex_replace(benchmark, indexedframe_dtype_int):
    # TODO: Consider adding more DataFrame-specific benchmarks for different
    # types of valid inputs (dicts, etc).
    benchmark(indexedframe_dtype_int.replace, 0, 2)
