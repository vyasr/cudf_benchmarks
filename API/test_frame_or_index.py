"""Benchmarks of methods that exist for both Frame and BaseIndex."""

import pytest
from utils import make_gather_map


@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
@pytest.mark.parametrize("fraction", [0.4])
def test_take(benchmark, gather_how, fraction, frame_or_index):
    nr = len(frame_or_index)
    gather_map = make_gather_map(nr * fraction, nr, gather_how)
    benchmark(frame_or_index.take, gather_map)


def test_argsort(benchmark, frame_or_index):
    benchmark(frame_or_index.argsort)


def test_min(benchmark, frame_or_index):
    benchmark(frame_or_index.min)


def test_where(benchmark, frame_or_index):
    cond = frame_or_index % 2 == 0
    benchmark(frame_or_index.where, cond, 0)


def test_values_host(benchmark, frame_or_index_nulls_false):
    benchmark(lambda: frame_or_index_nulls_false.values_host)


def test_values(benchmark, frame_or_index_nulls_false):
    benchmark(lambda: frame_or_index_nulls_false.values)


def test_nunique(benchmark, frame_or_index):
    benchmark(frame_or_index.nunique)


def test_to_numpy(benchmark, frame_or_index_nulls_false):
    benchmark(frame_or_index_nulls_false.to_numpy)


def test_to_cupy(benchmark, frame_or_index_nulls_false):
    benchmark(frame_or_index_nulls_false.to_cupy)


def test_to_arrow(benchmark, frame_or_index):
    benchmark(frame_or_index.to_arrow)


def test_astype(benchmark, frame_or_index):
    benchmark(frame_or_index.astype, float)
