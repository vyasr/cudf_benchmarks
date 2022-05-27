"""Benchmarks of IndexedFrame methods."""

import pytest


@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def test_scans(benchmark, op, indexed_frame):
    benchmark(getattr(indexed_frame, op))


@pytest.mark.parametrize("op", ["sum", "product", "mean"])
def test_reductions(benchmark, op, indexed_frame):
    benchmark(getattr(indexed_frame, op))


def test_drop_duplicates(benchmark, indexed_frame):
    benchmark(indexed_frame.drop_duplicates)


def test_rangeindex_replace(benchmark, indexed_frame):
    # TODO: Consider adding more DataFrame-specific benchmarks for different
    # types of valid inputs (dicts, etc).
    benchmark(indexed_frame.replace, 0, 2)
