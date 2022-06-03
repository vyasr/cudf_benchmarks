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

We dynamically generate fixtures to cover the most common matrix of objects required
throughout benchmarking of cuDF. Specifically, we need to account for the following
different parameters:
    - Class of object (DataFrame, Series, Index)
    - Dtype
    - Nullability
    - Size (rows for Series/Index, rows/columns for DataFrame)

We make one assumption, which is that all benchmarks should be tested for data
with all different possible numbers of rows, so that level of granularity is
not exposed in the fixtures but is instead parametrized. For DataFrames, we do
provide specific fixtures for different numbers of columns since not all
benchmarks need to be run for data with different numbers of columns but could
instead use a single fixed number and only vary the rows.

Each element of this matrix is constructed as a separate fixture. We employ
the standard naming scheme
`classname_dtype_{dtype}_nulls_{true|false}[_cols_{num_cols}]`
where classname is a lowercased version of the classname. Note that in the case of
indexes, to match Series/DataFrame we simply set `classname=index` and rely on
the `dtype_{dtype}` component to delineate which index class is actually in use.
The `num_cols` component of the name is only used for dataframes.

In addition to the above fixtures, we also provide the following more specialized
fixtures:
    - rangeindex: Since RangeIndex always holds int64 data we cannot conflate
      it with index_dtype_int64 (a true Int64Index), and it cannot hold nulls.
      As a result, it is provided as a separate fixture.
    - multiindex: MultiIndex does not support nulls

We then provide a large collection of fixture unions that combine the above fixtures.
In general, these unions collapse along the three possible dimensions above:
    - classname: Unions by class result in fixtures that will iterate over
      multiple types. These unions typically map to actual classes within cuDF,
      for instance unioning `series*` and `dataframe*` results in `indexedframe*` fixtures.
    - dtype: Collapsing along dtype is only done for specific combinations that
      we expect to be useful. Since we don't generally have a use case for a fixture that
      encompasses _all_ dtypes, these collapses take the form e.g. `dtype_{int_or_float}`
    - nulls: Collapsing along nulls results in fixtures with this component of
      the name dropped, e.g. union(`series_nulls_true`, `series_nulls_false`) ->
      `series`.
"""
# Dynamic fixture creation as discussed in
# https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206

num_rows = [10]
num_cols = [1]

# A dictionary of callables that create a column of a specified length
column_generators = {
    "int": cupy.arange,
    # "float": (lambda nr: cupy.arange(nr, dtype=float)),
}

# def create_fixture_from_function(name, func, **kwargs):
#     globals()[name] = pytest_cases.fixture(name=name, **kwargs)(
#         series_nulls_false
#     )


# First generate all the base fixtures.
for dtype, column_generator in column_generators.items():

    def series_nulls_false(request):
        return cudf.Series(column_generator(request.param))

    name = f"series_dtype_{dtype}_nulls_false"
    globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(
        series_nulls_false
    )

    def series_nulls_true(request):
        s = cudf.Series(column_generator(request.param))
        s.iloc[::2] = None
        return s

    name = f"series_dtype_{dtype}_nulls_true"
    globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(
        series_nulls_true
    )

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
            # TODO: pytest_cases seems to have a bug where the first argument
            # being a kwarg (nr=nr, nc=nc) raises errors. I'll need to track
            # that upstream.
            def dataframe(request, nr=nr, nc=nc):
                return make_dataframe(nr, nc)

            name = f"dataframe_dtype_{dtype}_nulls_false_rows_{nr}_cols_{nc}"
            globals()[name] = pytest_cases.fixture(name=name)(dataframe)

            def dataframe(request, nr=nr, nc=nc):
                return make_nullable_dataframe(nr, nc)

            name = f"dataframe_dtype_{dtype}_nulls_true_rows_{nr}_cols_{nc}"
            globals()[name] = pytest_cases.fixture(name=name)(dataframe)

    # Index fixture. Note that we choose not to create a nullable index fixture
    # since that's such an unnecessary and esoteric use-case.
    def int64_index(request):
        return cudf.Index(column_generator(request.param))

    name = f"index_dtype_{dtype}"
    globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(int64_index)


# Inside the main loop, perform the collapses over classes and over
# nullability. Collapsing over dtype can happen later.


for dtype, column_generator in column_generators.items():
    for nulls in ["false", "true"]:
        fixture_union(
            name=f"dataframe_dtype_{dtype}_nulls_{nulls}",
            fixtures=[
                f"dataframe_dtype_{dtype}_nulls_{nulls}_rows_{nr}_cols_{nc}"
                for nr in num_rows
                for nc in num_cols
            ],
        )

        fixture_union(
            name=f"dataframe_dtype_{dtype}_nulls_{nulls}_cols_5",
            fixtures=[
                f"dataframe_dtype_{dtype}_nulls_{nulls}_rows_{nr}_cols_5"
                for nr in num_rows
            ],
        )

    fixture_union(
        name=f"dataframe_dtype_{dtype}_cols_5",
        fixtures=(
            f"dataframe_dtype_{dtype}_nulls_true_cols_5",
            f"dataframe_dtype_{dtype}_nulls_false_cols_5",
        ),
    )

    # Collapse over nulls:
    for classname in ["series", "dataframe"]:
        # Various common important fixture unions
        fixture_union(
            name=f"{classname}_dtype_{dtype}",
            fixtures=(
                f"{classname}_dtype_{dtype}_nulls_false",
                f"{classname}_dtype_{dtype}_nulls_true",
            ),
        )

    for nulls in ["_nulls_false", "_nulls_true", ""]:
        fixture_union(
            name=f"indexedframe_dtype_{dtype}{nulls}",
            fixtures=(
                f"series_dtype_{dtype}{nulls}",
                f"dataframe_dtype_{dtype}{nulls}",
            ),
        )

    # TODO: Add MultiIndex
    fixture_union(
        name=f"frame_dtype_{dtype}",
        fixtures=(f"indexedframe_dtype_{dtype}", f"index_dtype_{dtype}"),
    )

    # Note: pytest_cases isn't smart enough to recognize that the same fixture
    # (generic_index) gets included twice if we directly union "index" and
    # "frame".
    fixture_union(
        name=f"frame_or_index_dtype_{dtype}_nulls_false",
        fixtures=(f"indexedframe_dtype_{dtype}_nulls_false", f"index_dtype_{dtype}"),
    )

    fixture_union(
        name=f"frame_or_index_dtype_{dtype}",
        fixtures=(f"indexedframe_dtype_{dtype}", f"index_dtype_{dtype}"),
    )


# TODO: Figure out how to incorporate RangeIndex and MultiIndex fixtures.
def range_index(request):
    return cudf.RangeIndex(request.param)


name = "range_index"
globals()[name] = pytest_cases.fixture(name=name, params=num_rows)(range_index)


# fixture_union(name="generic_index", fixtures=("int64_index",))
#
# # TODO: Add MultiIndex, also of different dtypes...
# fixture_union(name="index", fixtures=("generic_index", "range_index"))
