"""Benchmarks of methods that exist for both Frame and BaseIndex."""

import numpy as np
import pytest
from utils import make_gather_map


@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
@pytest.mark.parametrize("fraction", [0.4])
def test_take(benchmark, gather_how, fraction, frame_or_index_dtype_int):
    nr = len(frame_or_index_dtype_int)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index_dtype_int.take, gather_map)


def test_argsort(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.argsort)


def test_min(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.min)


def test_where(benchmark, frame_or_index_dtype_int):
    cond = frame_or_index_dtype_int % 2 == 0
    benchmark(frame_or_index_dtype_int.where, cond, 0)


def test_values_host(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(lambda: frame_or_index_dtype_int_nulls_false.values_host)


def test_values(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(lambda: frame_or_index_dtype_int_nulls_false.values)


def test_nunique(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.nunique)


def test_to_numpy(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(frame_or_index_dtype_int_nulls_false.to_numpy)


def test_to_cupy(benchmark, frame_or_index_dtype_int_nulls_false):
    benchmark(frame_or_index_dtype_int_nulls_false.to_cupy)


def test_to_arrow(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.to_arrow)


def test_astype(benchmark, frame_or_index_dtype_int):
    benchmark(frame_or_index_dtype_int.astype, float)


@pytest.mark.parametrize("ufunc", [np.add, np.logical_and, np.bitwise_and])
def test_ufunc_series_binary(benchmark, frame_or_index_dtype_int, ufunc):
    benchmark(ufunc, frame_or_index_dtype_int, frame_or_index_dtype_int)
