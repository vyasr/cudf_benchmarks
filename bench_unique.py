import cupy as cp

import cudf
import pytest

from utils import make_frame

n = [100, 10000]

@pytest.fixture(params=n)
def col(request):
    return make_frame(ncols=1, nkey_cols=0, nrows=request.param, low=0, high=10)['val0']._column

@pytest.fixture(params=n)
def df(request):
    return make_frame(ncols=5, nkey_cols=0, nrows=request.param)

def test_unique_single_column(benchmark, col):
    benchmark(lambda: col.unique())

def test_drop_duplicate_multiple_column(benchmark, df):
    benchmark(lambda: df.drop_duplicates())