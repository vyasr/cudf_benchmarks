"""Benchmarks of methods that exist for both Frame and BaseIndex."""

import operator

import numpy as np
import pytest
from utils import make_gather_map


@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
@pytest.mark.parametrize("fraction", [0.4])
def bench_take(benchmark, gather_how, fraction, frame_or_index_dtype_int):
    nr = len(frame_or_index_dtype_int)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index_dtype_int.take, gather_map)


def bench_argsort(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.argsort)


def bench_min(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.min)


def bench_where(benchmark, frame_or_index_dtype_int):
    cond = frame_or_index_dtype_int % 2 == 0
    benchmark(frame_or_index_dtype_int.where, cond, 0)


def bench_values_host(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(lambda: frame_or_index_dtype_int_nulls_false.values_host)


def bench_values(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(lambda: frame_or_index_dtype_int_nulls_false.values)


def bench_nunique(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.nunique)


def bench_to_numpy(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(frame_or_index_dtype_int_nulls_false.to_numpy)


def bench_to_cupy(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(frame_or_index_dtype_int_nulls_false.to_cupy)


def bench_to_arrow(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.to_arrow)


def bench_astype(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.astype, float)


@pytest.mark.parametrize("ufunc", [np.add, np.logical_and, np.bitwise_and])
def bench_ufunc_series_binary(benchmark, frame_or_index_dtype_int, ufunc):
    benchmark(ufunc, frame_or_index_dtype_int, frame_or_index_dtype_int)


@pytest.mark.parametrize(
    "op",
    [operator.add, operator.mul, operator.__and__, operator.eq],
)
def bench_binops(benchmark, op, frame_or_index_dtype_int):
    # Use integer data so that __and__ is well-defined.
    benchmark(lambda: op(frame_or_index_dtype_int, frame_or_index_dtype_int))
