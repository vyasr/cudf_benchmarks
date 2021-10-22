import cupy as cp

import cudf
import pytest

from utils import make_frame

n = [100, 10000]

@pytest.fixture(params=n)
def col(request):
    return make_frame(ncols=1, nkey_cols=0, nrows=request.param)['val0']._column

@pytest.fixture(params=n)
def df(request):
    return make_frame(ncols=5, nkey_cols=0, nrows=request.param)

def make_gather_map(len_gather_map, len_column, how):
    if how == "sequence":
        return cudf.Series(cp.arange(0, len_gather_map))._column
    elif how == "reverse":
        return cudf.Series(cp.arange(len_column - 1, len_column - len_gather_map - 1, -1))._column
    elif how == "random":
        return cudf.Series(cp.random.randint(0, len_column, len_gather_map))._column

@pytest.mark.parametrize(
    "nullify", [True, False]
)
@pytest.mark.parametrize(
    "gather_how", ["sequence", "reverse", "random"]
)
def test_gather_single_column(benchmark, col, gather_how, nullify):
    gather_map = make_gather_map(int(col.size * 0.4), col.size, gather_how)
    benchmark(lambda: col.take(gather_map, nullify))

@pytest.mark.parametrize(
    "keep_index", [True, False]
)
@pytest.mark.parametrize(
    "gather_how", ["sequence", "reverse", "random"]
)
def test_take_multiple_column(benchmark, df, gather_how, keep_index):
    gather_map = make_gather_map(int(len(df) * 0.4), len(df), gather_how)
    benchmark(lambda: df.take(gather_map, keep_index=keep_index))