import os
import re
import string
import sys
from collections.abc import MutableSet
from functools import partial
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


def make_fixture(name, func, new_fixtures, **kwargs):
    """Create a named fixture and inject it into the global namespace.

    https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
    explains why this hack is necessary.
    """
    globals()[name] = pytest_cases.fixture(name=name, **kwargs)(func)
    new_fixtures.add(name)


def collapse_fixtures(fixtures, pattern, repl, new_fixtures, used):
    """Create unions of fixtures based on specific name mappings.

    `fixtures` are grouped into unions according the regex replacement
    `re.sub(pattern, repl)` and placed into `new_fixtures`. `used` is
    updated with any element of `fixtures` that participated in a union.
    """

    def collapser(n):
        return re.sub(pattern, repl, n)

    for name, group in groupby(sorted(fixtures, key=collapser), key=collapser):
        group = list(group)
        if len(group) > 1:
            # The presence of a fixture in any non-singleton group indicates
            # that it is included in some resulting union. There may be
            # multiple paths to that union (and those paths could have been
            # traversed in previous calls to collapse fixtures with the same
            # new_fixtures), so we must update this set even if the fixture
            # has already been created (i.e. ahead of the below conditional).
            used |= OrderedSet(group)

            if name not in new_fixtures:
                fixture_union(name=name, fixtures=group)
                new_fixtures.add(name)


class OrderedSet(MutableSet):
    """A minimal OrderedSet implementation built on a dict.

    This implementation exploits the fact that dicts are ordered as of Python
    3.7. It is not intended to be performant, so only the minimal set of
    methods are implemented. We need this class to ensure that fixture names
    are constructed deterministically, otherwise pytest-xdist will complain if
    different threads have seemingly different tests.
    """

    def __init__(self, args=None):
        args = args or []
        self._data = {value: None for value in args}

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        # Helpful for debugging.
        data = ", ".join(str(i) for i in self._data)
        return f"{self.__class__.__name__}({data})"

    def add(self, value):
        self._data[value] = None

    def discard(self, value):
        self._data.pop(value, None)

    def copy(self):
        return OrderedSet(self._data)


# A dictionary of callables that create a column of a specified length
column_generators = {
    "int": cupy.arange,
    "float": (lambda nr: cupy.arange(nr, dtype=float)),
}

num_rows = [10]
num_cols = [1, 6]
fixtures = {0: OrderedSet()}
make_fixture_level_0 = partial(make_fixture, new_fixtures=fixtures[0], params=num_rows)

# First generate all the base fixtures.
for dtype, column_generator in column_generators.items():

    def series_nulls_false(request, column_generator=column_generator):
        return cudf.Series(column_generator(request.param))

    make_fixture_level_0(f"series_dtype_{dtype}_nulls_false", series_nulls_false)

    def series_nulls_true(request, column_generator=column_generator):
        s = cudf.Series(column_generator(request.param))
        s.iloc[::2] = None
        return s

    make_fixture_level_0(f"series_dtype_{dtype}_nulls_true", series_nulls_true)

    # Since we may in some cases want to benchmark just DataFrames with a
    # specific number of columns, we create separate fixtures for different
    # numbers of columns to be combined later.
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

        make_fixture_level_0(
            f"dataframe_dtype_{dtype}_nulls_false_cols_{nc}",
            dataframe_nulls_false,
        )

        def dataframe_nulls_true(request, nc=nc):
            df = make_dataframe(request.param, nc)
            df.iloc[::2, :] = None
            return df

        make_fixture_level_0(
            f"dataframe_dtype_{dtype}_nulls_true_cols_{nc}",
            dataframe_nulls_true,
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

    # For now, not bothering to include a nullable index fixture.
    def index_nulls_false(request, column_generator=column_generator):
        return cudf.Index(column_generator(request.param))

    make_fixture_level_0(f"index_dtype_{dtype}_nulls_false", index_nulls_false)


cur_level = 0
prev_fixtures = fixtures[cur_level]

# Loop through "levels" of merging fixtures until no new fixtures are added.
while fixtures[cur_level]:
    cur_level += 1
    fixtures[cur_level] = OrderedSet()

    used = OrderedSet()
    for pat, repl in [
        ("_nulls_(true|false)", ""),
        ("series|dataframe", "indexedframe"),
        ("indexedframe|index", "frame_or_index"),
    ]:
        collapse_fixtures(
            prev_fixtures,
            pat,
            repl,
            fixtures[cur_level],
            used,
        )
    # Anything that wasn't added to any of the unions is effectively already
    # collapsed, so we need to reconsider those in the next stage.
    prev_fixtures = fixtures[cur_level] | (fixtures[cur_level - 1] - used)

for dtype, column_generator in column_generators.items():
    # We have to manually add this one because we aren't including nullable
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
