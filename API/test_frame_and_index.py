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
