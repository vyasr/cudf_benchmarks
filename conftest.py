from itertools import product

import pytest

from utils import make_col, make_frame

n = [100, 10000]
has_nulls = [True, False]


@pytest.fixture(params=product(n, has_nulls))
def col(request):
    return make_col(*request.param)


@pytest.fixture(params=n)
def df(request):
    return make_frame(ncols=5, nkey_cols=0, nrows=request.param)
