import cudf
import cupy as cp
import pytest

# low = 0
# high = 100
# nkey_cols = 2
# nrows =


def make_frame(ncols, nkey_cols, nrows, low=0, high=100):
    nval_cols = ncols - nkey_cols
    key_columns = {
        f"key{i}": cp.random.randint(low, high, nrows)
        for i in range(nkey_cols)
    }
    val_columns = {
        f"val{i}": cp.random.rand(nrows)
        for i in range(nval_cols)
    }
    return cudf.DataFrame({**key_columns, **val_columns})


@pytest.mark.parametrize(
    'ncols', [4, 7, 10],
)
@pytest.mark.parametrize(
    'nkey_cols', [2, 3, 4],
)
@pytest.mark.parametrize(
    'nrows', [10, 100, 1000, 10_000, 100_000, 1_000_000],
)
def test_merge(benchmark, ncols, nkey_cols, nrows):
    lhs = make_frame(ncols, nkey_cols, nrows)
    rhs = make_frame(ncols, nkey_cols, nrows // 2)
    benchmark(lhs.merge, rhs, on=[f"key{i}" for i in range(nkey_cols)])
