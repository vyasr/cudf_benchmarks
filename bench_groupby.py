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


@pytest.mark.parametrize(
    "nrows",
    [100, 10_000, 1_000_000],
)
@pytest.mark.parametrize(
    "agg",
    ["sum", ["sum", "mean"], {f"val{i}": ["sum", "mean", "count"] for i in range(6)}],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_groupby_agg(benchmark, nrows, agg, as_index, sort):
    ncols = 10
    nkey_cols = 4
    frame = utils.make_frame(ncols, nkey_cols, nrows)
    benchmark(
        frame.groupby(
            by=[f"key{i}" for i in range(nkey_cols)], as_index=as_index, sort=sort
        ).agg,
        agg,
    )
