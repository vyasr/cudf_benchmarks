import cupy as cp

import cudf
import pytest

from utils import make_frame, make_col

n = [100, 10000]

@pytest.fixture(params=n)
def col(request):
    return make_col(request.param)

@pytest.fixture(params=n)
def df(request):
    return make_frame(ncols=5, nkey_cols=0, nrows=request.param)

def test_unique_single_column(benchmark, col):
    benchmark(col.unique)

def test_drop_duplicate_df(benchmark, df):
    benchmark(df.drop_duplicates)