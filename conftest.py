import os
import string
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
# Dynamic fixture creation as discussed in
# https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206

num_rows = [10]
num_cols = [1]

column_generators = {
    "int": (lambda nr: cupy.arange(nr)),
    "float": (lambda nr: cupy.arange(nr).astype(float)),
}


# Core fixtures generated for each common type of object.
def series_nulls_false(request):
    return cudf.Series(cupy.arange(request.param))


name = "series_nulls_false"
globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(series_nulls_false)


def series_nulls_true(request):
    s = cudf.Series(cupy.arange(request.param))
    s.iloc[::2] = None
    return s


name = "series_nulls_true"
globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(series_nulls_true)


# Since we may in some cases want to benchmark just single-columned DataFrame
# objects, we generate separate fixtures for each num_rows/num_cols pair so
# that we can recombine all the num_cols==1 fixtures into one union rather than
# using a parametrized fixture as we do for the series case above.
def make_dataframe(nr, nc):
    if nc > len(string.ascii_lowercase):
        raise ValueError(
            "make_dataframe does not support more than "
            f"{len(string.ascii_lowercase)} columns, but {nc} were requested."
        )
    return cudf.DataFrame(
        {f"{string.ascii_lowercase[i]}": cupy.arange(nr) for i in range(nc)}
    )


def make_nullable_dataframe(nr, nc):
    df = make_dataframe(nr, nc)
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


def range_index(request):
    return cudf.RangeIndex(request.param)


name = "range_index"
globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(range_index)


def int64_index(request):
    return cudf.Index(cupy.arange(request.param), dtype="int64")


name = "int64_index"
globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(int64_index)


# Various common important fixture unions
fixture_union(
    name="series",
    fixtures=(["series_nulls_false", "series_nulls_true"]),
)


fixture_union(
    name="dataframe_nulls_false_cols_5",
    fixtures=[f"dataframe_nulls_false_rows_{nr}_cols_5" for nr in num_rows],
)

fixture_union(
    name="dataframe_nulls_true_cols_5",
    fixtures=[f"dataframe_nulls_true_rows_{nr}_cols_5" for nr in num_rows],
)


fixture_union(
    name="dataframe_cols_5",
    fixtures=("dataframe_nulls_true_cols_5", "dataframe_nulls_false_cols_5"),
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

fixture_union(name="generic_index", fixtures=("int64_index",))

# TODO: Add MultiIndex
fixture_union(name="index", fixtures=("generic_index", "range_index"))

fixture_union(name="frame", fixtures=("indexed_frame", "generic_index"))

# Note: pytest_cases isn't smart enough to recognize that the same fixture
# (generic_index) gets included twice if we directly union "index" and "frame".
fixture_union(
    name="frame_or_index_nulls_false",
    fixtures=("indexed_frame_nulls_false", "generic_index", "range_index"),
)


fixture_union(
    name="frame_or_index", fixtures=("indexed_frame", "generic_index", "range_index")
)
