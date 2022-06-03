"""Benchmarks of IndexedFrame methods."""

import pytest


@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def test_scans(benchmark, op, indexedframe_dtype_int):
    benchmark(getattr(indexedframe_dtype_int, op))


@pytest.mark.parametrize("op", ["sum", "product", "mean"])
def test_reductions(benchmark, op, indexedframe_dtype_int):
    benchmark(getattr(indexedframe_dtype_int, op))


def test_drop_duplicates(benchmark, indexedframe_dtype_int):
    benchmark(indexedframe_dtype_int.drop_duplicates)


def test_rangeindex_replace(benchmark, indexedframe_dtype_int):
    # TODO: Consider adding more DataFrame-specific benchmarks for different
    # types of valid inputs (dicts, etc).
    benchmark(indexedframe_dtype_int.replace, 0, 2)
