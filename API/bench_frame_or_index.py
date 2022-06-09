"""Benchmarks of methods that exist for both Frame and BaseIndex."""

import operator

import numpy as np
import pytest
from utils import cudf_benchmark, make_gather_map


@cudf_benchmark(cls="frame_or_index", dtype="int")
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
@pytest.mark.parametrize("fraction", [0.4])
def bench_take(benchmark, gather_how, fraction, frame_or_index):
    nr = len(frame_or_index)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index.take, gather_map)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_argsort(benchmark, frame_or_index):
    benchmark(frame_or_index.argsort)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_min(benchmark, frame_or_index):
    benchmark(frame_or_index.min)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_where(benchmark, frame_or_index):
    cond = frame_or_index % 2 == 0
    benchmark(frame_or_index.where, cond, 0)


@cudf_benchmark(cls="frame_or_index", dtype="int", nulls=False)
def bench_values_host(benchmark, frame_or_index):
    benchmark(lambda: frame_or_index.values_host)


@cudf_benchmark(cls="frame_or_index", dtype="int", nulls=False)
def bench_values(benchmark, frame_or_index):
    benchmark(lambda: frame_or_index.values)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_nunique(benchmark, frame_or_index):
    benchmark(frame_or_index.nunique)


@cudf_benchmark(cls="frame_or_index", dtype="int", nulls=False)
def bench_to_numpy(benchmark, frame_or_index):
    benchmark(frame_or_index.to_numpy)


@cudf_benchmark(cls="frame_or_index", dtype="int", nulls=False)
def bench_to_cupy(benchmark, frame_or_index):
    benchmark(frame_or_index.to_cupy)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_to_arrow(benchmark, frame_or_index):
    benchmark(frame_or_index.to_arrow)


@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_astype(benchmark, frame_or_index):
    benchmark(frame_or_index.astype, float)


@pytest.mark.parametrize("ufunc", [np.add, np.logical_and, np.bitwise_and])
@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_ufunc_series_binary(benchmark, frame_or_index, ufunc):
    benchmark(ufunc, frame_or_index, frame_or_index)


@pytest.mark.parametrize(
    "op",
    [operator.add, operator.mul, operator.__and__, operator.eq],
)
@cudf_benchmark(cls="frame_or_index", dtype="int")
def bench_binops(benchmark, op, frame_or_index):
    # Use integer data so that __and__ is well-defined.
    benchmark(lambda: op(frame_or_index, frame_or_index))
