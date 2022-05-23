import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.getcwd(), "common"))

import pytest

from common.config import cudf
from common.utils import make_col, make_frame


@pytest.fixture(params=product([100, 10000], [True, False]))
def col(request):
    """Create a cudf column.

    The two parameters are `nrows` and `has_nulls`
    """
    return make_col(*request.param)


@pytest.fixture(params=[100, 10000])
def df(request):
    """Create a cudf DataFrame.

    The two parameters are `nrows`
    """
    return make_frame(ncols=5, nkey_cols=0, nrows=request.param)


@pytest.fixture(params=[0, 1], ids=["AxisIndex", "AxisColumn"])
def axis(request):
    return request.param


@pytest.fixture(params=[1_000, 100_000, 10_000_000])
def rangeindex(request):
    """Create a cudf RangeIndex of different size `nrows`"""
    return cudf.RangeIndex(range(request.param))


def pytest_sessionstart(session):
    sys.path.insert(0, os.path.join(os.getcwd(), "common"))


def pytest_sessionfinish(session, exitstatus):
    if "common" in sys.path[0]:
        del sys.path[0]
