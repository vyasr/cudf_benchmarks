import os
import re
import string
import sys
from itertools import groupby, product

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

fixtures = {}
fixtures[0] = set()


def make_fixture(name, func, **kwargs):
    globals()[name] = pytest_cases.fixture(name=name, **kwargs)(func)
    global fixtures
    fixtures[0].add(name)


# First generate all the base fixtures.
for dtype, column_generator in column_generators.items():

    def series_nulls_false(request, column_generator=column_generator):
        return cudf.Series(column_generator(request.param))

    make_fixture(
        f"series_dtype_{dtype}_nulls_false", series_nulls_false, params=num_rows
    )

    def series_nulls_true(request, column_generator=column_generator):
        s = cudf.Series(column_generator(request.param))
        s.iloc[::2] = None
        return s

    make_fixture(f"series_dtype_{dtype}_nulls_true", series_nulls_true, params=num_rows)

    # Since we may in some cases want to benchmark just DataFrames with a
    # specific number of columns rather than iterating over all numbers of
    # columns, we don't include the number of columns in the fixture
    # parametrization and instead create separate fixtures to be recombined
    # later.
    def make_dataframe(nr, nc, column_generator=column_generator):
        assert nc <= len(string.ascii_lowercase)
        return cudf.DataFrame(
            {f"{string.ascii_lowercase[i]}": column_generator(nr) for i in range(nc)}
        )

    for nc in num_cols:
        # TODO: pytest_cases seems to have a bug where the first argument
        # being a kwarg (nr=nr, nc=nc) raises errors. I'll need to track
        # that upstream, but for now that's no longer an issue since I'm
        # passing request as a positional parameter.
        def dataframe_nulls_false(request, nc=nc):
            return make_dataframe(request.param, nc)

        make_fixture(
            f"dataframe_dtype_{dtype}_nulls_false_cols_{nc}",
            dataframe_nulls_false,
            params=num_rows,
        )

        def dataframe_nulls_true(request, nc=nc):
            df = make_dataframe(request.param, nc)
            df.iloc[::2, :] = None
            return df

        make_fixture(
            f"dataframe_dtype_{dtype}_nulls_true_cols_{nc}",
            dataframe_nulls_true,
            params=num_rows,
        )

    # Create the combined dataframe fixtures for different numbers of columns.
    for nulls in ["false", "true"]:
        name = f"dataframe_dtype_{dtype}_nulls_{nulls}"
        fixture_union(
            name=name,
            fixtures=[
                f"dataframe_dtype_{dtype}_nulls_{nulls}_cols_{nc}" for nc in num_cols
            ],
        )
        fixtures[0].add(name)

    # Index fixture. Note that we choose not to create a nullable index fixture
    # since that's such an unnecessary and esoteric use-case.
    def index_nulls_false(request, column_generator=column_generator):
        return cudf.Index(column_generator(request.param))

    make_fixture(f"index_dtype_{dtype}_nulls_false", index_nulls_false, params=num_rows)


def collapse_fixtures(fixtures, collapser, new_fixture_set, never_added):
    """Create unions of fixtures based on specific name mappings.

    A collapser is a callable that maps fixture names into unions. The
    assumption is that this will be a many to one mapping. We need to keep
    track of fixtures that were not added to any union to know that they were
    not collapsed and should be considered in future iterations.
    """

    for name, group_fixtures in groupby(sorted(fixtures, key=collapser), key=collapser):
        # If the groupby doesn't actually collapse anything, just toss the fixture
        # back into the pool for the next round of collapse.
        group_fixtures = list(group_fixtures)
        if len(group_fixtures) > 1:
            # After one level of collapsing we can arrive at the same fixtures
            # more than one way (e.g. we could collapse nulls and
            # series|dataframe->index in either order). As long as either of
            # those happened, we don't count the fixture as one that was never
            # added to anything. Therefore, any non-singleton group is
            # sufficient to indicate that the constitutent fixtures were added
            # to a new union (and hence the for loop below is not within the
            # conditional below), but that union may already have been created
            # so the conditional below is necessary to avoid duplicates.
            if name not in new_fixture_set:
                fixture_union(name=name, fixtures=group_fixtures)
                new_fixture_set.add(name)

            for f in group_fixtures:
                # It's OK if it's already been removed.
                try:
                    never_added.remove(f)
                except KeyError:
                    pass


# TODO: Rather than using sets, we need to use dicts with empty values because
# we need the results to be ordered. Without that, when we use pytest-xdist it
# is possible for different threads to collapse fixtures in different orders,
# and then it will fail because they look like different fixtures.
cur_level = 0
cur_level_fixtures = fixtures[cur_level]
never_added = set()

# Loop until no new fixtures are added.
while cur_level_fixtures:
    # Anything that wasn't added to any of the unions at the previous level is
    # effectively already one level higher because none of the previous
    # collapsers had any effect.
    prev_level_fixtures = cur_level_fixtures | never_added
    never_added = prev_level_fixtures.copy()

    cur_level += 1
    cur_level_fixtures = fixtures[cur_level] = set()

    # TODO: May need to have different collapsers at different levels?
    for collapser in [
        lambda n: re.sub("_nulls_(true|false)", "", n),
        lambda n: re.sub("series|dataframe", "indexedframe", n),
        lambda n: re.sub("indexedframe|index", "frame_or_index", n),
    ]:
        collapse_fixtures(
            prev_level_fixtures,
            collapser,
            cur_level_fixtures,
            never_added,
        )

for dtype, column_generator in column_generators.items():
    # Have to manually add this one because we aren't including nullable
    # indexes but we want to be able to run some benchmarks on Series/DataFrame
    # that may or may not be nullable as well as Index objects.
    fixture_union(
        name=f"frame_or_index_dtype_{dtype}",
        fixtures=(f"indexedframe_dtype_{dtype}", f"index_dtype_{dtype}_nulls_false"),
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
