"""Standard location for shared fixtures and code across tests.

Most fixtures defined in this file represent one of the primary classes in the
cuDF ecosystem such as DataFrame, Series, or Index. These fixtures may in turn
be broken up into two categories: base fixtures and fixture unions. Each base
fixture represents a specific type of object as well as certain of its
properties crucial for benchmarking. Specifically, fixtures must account for
the following different parameters:
    - Class of object (DataFrame, Series, Index)
    - Dtype
    - Nullability
    - Size (rows for all, rows/columns for DataFrame)

One such fixture is a series of nullable integer data. Given that we generally
want data across different sizes, we parametrize all fixtures across different
numbers of rows rather than generating separate fixtures for each different
possible number of rows. The number of columns is only relevant for DataFrame.

While this core set of fixtures means that any benchmark can be run for any
combination of these parameters, it also means that we would effectively have
to parametrize our benchmarks with many fixtures. Not only is parametrizing
tests with fixtures in this manner unsupported by pytest, it is also an
inelegant solution leading to cumbersome parameter lists that must be
maintained across all tests. Instead we make use of the
`pytest_cases <https://smarie.github.io/python-pytest-cases/>_` pytest plugin,
which supports the creation of fixture unions: fixtures that result from
combining other fixtures together. The result is a set of well-defined fixtures
that allow us to write benchmarks that naturally express the set of objects for
which they are valid, e.g. `def bench_sort_values(frame_or_index)`.

The generated fixtures are named according to the following convention:
`classname_dtype_{dtype}[_nulls_{true|false}][_cols_{num_cols}]`
where classname is one of the following: index, series, dataframe, indexedframe,
frame, frame_or_index. Note that in the case of
indexes, to match Series/DataFrame we simply set `classname=index` and rely on
the `dtype_{dtype}` component to delineate which index class is actually in use.

In addition to the above fixtures, we also provide the following more specialized
fixtures:
    - rangeindex: Since RangeIndex always holds int64 data we cannot conflate
      it with index_dtype_int64 (a true Int64Index), and it cannot hold nulls.
      As a result, it is provided as a separate fixture.
"""


import os
import re
import string
import sys
from collections.abc import MutableSet
from functools import partial
from itertools import groupby

import pytest_cases

sys.path.insert(0, os.path.join(os.getcwd(), "common"))

from common.config import NUM_COLS, NUM_ROWS, cudf, cupy  # noqa: E402


def pytest_sessionstart(session):
    """Add the common files to the path for all tests to import."""
    sys.path.insert(0, os.path.join(os.getcwd(), "common"))


def pytest_sessionfinish(session, exitstatus):
    """Clean up sys.path after exit."""
    if "common" in sys.path[0]:
        del sys.path[0]


@pytest_cases.fixture(params=[0, 1], ids=["AxisIndex", "AxisColumn"])
def axis(request):
    return request.param


def make_fixture(name, func, new_fixtures, **kwargs):
    """Create a named fixture and inject it into the global namespace.

    https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
    explains why this hack is necessary. Essentially, dynamically generated
    fixtures must exist in globals() to be found by pytest.
    """
    globals()[name] = pytest_cases.fixture(name=name, **kwargs)(func)
    new_fixtures.add(name)


def l1_id(val):
    return val.alternative_name


def default_id(val):
    """A default index used to disambiguate tests.

    Although we explicitly construct fixtures in such a way as to guarantee
    that duplicates will not be present, pytest does not know this. At each
    level of fixture unions, pytest requires that a unique name be defined for
    each member, otherwise it assigns an id by default to avoid collisions.
    Rather than leaving a raw id in place, we prefix it by 'alt' for easier
    identification across all tests.
    """
    return f"alt{val.get_alternative_idx()}"


def collapse_fixtures(fixtures, pattern, repl, new_fixtures, used, idfunc):
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
                pytest_cases.fixture_union(name=name, fixtures=group, ids=idfunc)
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
random_state = cupy.random.RandomState(42)
column_generators = {
    "int": (lambda nr: random_state.randint(low=0, high=100, size=nr)),
    "float": (lambda nr: random_state.rand(nr)),
}
fixtures = {0: OrderedSet()}

# TODO: We need to decide whether making the number of rows part of the fixture
# parametrization makes sense, or if we need those separated as well. It's
# possible that some benchmarks will need to prevent using too large a frame
# (or will need a larger one), but the downside then is that we'll need to
# update fixture names in all tests if we change the number of rows.
make_fixture_level_0 = partial(make_fixture, new_fixtures=fixtures[0], params=NUM_ROWS)

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

    for nc in NUM_COLS:
        # TODO: pytest_cases seems to have a bug where the first argument being
        # a kwarg (nr=nr, nc=nc) raises errors. I'll need to track that
        # upstream, but for now that's no longer an issue since I'm passing
        # request as a positional parameter.
        def dataframe_nulls_false(request, nc=nc, make_dataframe=make_dataframe):
            return make_dataframe(request.param, nc)

        make_fixture_level_0(
            f"dataframe_dtype_{dtype}_nulls_false_cols_{nc}",
            dataframe_nulls_false,
        )

        def dataframe_nulls_true(request, nc=nc, make_dataframe=make_dataframe):
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
        pytest_cases.fixture_union(
            name=name,
            fixtures=[
                f"dataframe_dtype_{dtype}_nulls_{nulls}_cols_{nc}" for nc in NUM_COLS
            ],
            ids=[f"cols_{nc}" for nc in NUM_COLS],
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

        idfunc = default_id if cur_level > 1 else l1_id
        collapse_fixtures(prev_fixtures, pat, repl, fixtures[cur_level], used, idfunc)

    # Anything that wasn't added to any of the unions is effectively already
    # collapsed, so we need to reconsider those in the next stage.
    prev_fixtures = fixtures[cur_level] | (fixtures[cur_level - 1] - used)


for dtype, column_generator in column_generators.items():
    # We have to manually add this one because we aren't including nullable
    # indexes but we want to be able to run some benchmarks on Series/DataFrame
    # that may or may not be nullable as well as Index objects.
    pytest_cases.fixture_union(
        name=f"frame_or_index_dtype_{dtype}",
        fixtures=(f"indexedframe_dtype_{dtype}", f"index_dtype_{dtype}_nulls_false"),
        ids=["", f"index_dtype_{dtype}_nulls_false"],
    )


# TODO: Decide where to incorporate RangeIndex and MultiIndex fixtures.
@pytest_cases.fixture(params=NUM_ROWS)
def rangeindex(request):
    return cudf.RangeIndex(request.param)
