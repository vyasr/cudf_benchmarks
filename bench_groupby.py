import pytest

import utils


@pytest.mark.parametrize(
    "ncols",
    [4, 7, 10],
)
@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize(
    "nrows",
    [100, 10_000, 1_000_000],
)
def test_groupby(benchmark, ncols, nkey_cols, nrows):
    frame = utils.make_frame(ncols, nkey_cols, nrows)
    benchmark(frame.groupby, by=[f"key{i}" for i in range(nkey_cols)])
