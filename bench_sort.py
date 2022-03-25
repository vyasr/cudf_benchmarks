import pytest

import utils


@pytest.mark.parametrize(
    "ncols",
    [4, 7, 10],
)
@pytest.mark.parametrize(
    "nrows",
    [100, 10_000, 1_000_000],
)
def test_sort(benchmark, ncols, nrows):
    frame = utils.make_frame(ncols, ncols, nrows)
    benchmark(frame.sort_values, by=[f"key{i}" for i in range(ncols)])
