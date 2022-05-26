import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.getcwd(), "common"))

import pytest
import pytest_cases
from pytest_cases import fixture_union

from common.config import cudf, cupy
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


"""
New fixture logic.
"""

num_rows = [100, 10_000]
num_cols = [1, 5]


# Core Series fixtures
@pytest_cases.fixture(params=num_rows)
def series_nulls_false(request):
    return cudf.Series(cupy.arange(request.param))


@pytest_cases.fixture(params=num_rows)
def series_nulls_true(request):
    s = cudf.Series(cupy.arange(request.param))
    s.iloc[::2] = None
    return s


fixture_union(
    name="series",
    fixtures=(["series_nulls_false", "series_nulls_true"]),
)


# Core DataFrame fixtures
def make_dataframe(nr, nc):
    return cudf.DataFrame({f"{i}": cupy.arange(nr) for i in range(nc)})


def make_nullable_dataframe(nr, nc):
    df = cudf.DataFrame({f"{i}": cupy.arange(nr) for i in range(nc)})
    df.iloc[::2, :] = None
    return df


for nr in num_rows:
    for nc in num_cols:
        name = f"dataframe_nulls_false_rows_{nr}_cols_{nc}"
        globals()[name] = pytest.fixture(name=name)(
            lambda nr=nr, nc=nc: make_dataframe(nr, nc)
        )

        name = f"dataframe_nulls_true_rows_{nr}_cols_{nc}"
        globals()[name] = pytest.fixture(name=name)(
            lambda nr=nr, nc=nc: make_nullable_dataframe(nr, nc)
        )

fixture_union(
    name="dataframe_nulls_false_one_col",
    fixtures=[f"dataframe_nulls_false_rows_{nr}_cols_1" for nr in num_rows],
)

fixture_union(
    name="dataframe_nulls_true_one_col",
    fixtures=[f"dataframe_nulls_true_rows_{nr}_cols_1" for nr in num_rows],
)

fixture_union(
    name="dataframe_nulls_false",
    fixtures=[
        f"dataframe_nulls_false_rows_{nr}_cols_{nc}"
        for nr in num_rows
        for nc in num_cols
    ],
)

fixture_union(
    name="dataframe_nulls_true",
    fixtures=[
        f"dataframe_nulls_true_rows_{nr}_cols_{nc}"
        for nr in num_rows
        for nc in num_cols
    ],
)

fixture_union(
    name="dataframe",
    fixtures=("dataframe_nulls_true", "dataframe_nulls_false"),
)

fixture_union(
    name="indexed_frame_nulls_false",
    fixtures=("series_nulls_false", "dataframe_nulls_false"),
)

fixture_union(
    name="indexed_frame_nulls_true",
    fixtures=("series_nulls_true", "dataframe_nulls_true"),
)

fixture_union(name="indexed_frame", fixtures=("series", "dataframe"))
