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
`classname_dtype_{dtype}[_nulls_{true|false}][[_cols_{num_cols}]_rows_{num_rows}]`
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
from itertools import groupby

import pytest_cases

# TODO: Rather than doing this path hacking (including the sessionstart and
# sessionfinish hooks), we could just make the benchmarks a (sub)package to
# enable relative imports. A minor change to consider when these are ported
# into the main repo.
sys.path.insert(0, os.path.join(os.getcwd(), "common"))

from config import NUM_ROWS  # noqa: W0611, E402, F401
from config import column_generators  # noqa: F401
from config import cudf  # noqa: E402
from config import (
    NUM_COLS,
    collect_ignore,
    pytest_collection_modifyitems,
    pytest_sessionfinish,
    pytest_sessionstart,
)


@pytest_cases.fixture(params=[0, 1], ids=["AxisIndex", "AxisColumn"])
def axis(request):
    return request.param


def collapse_fixtures(fixtures, pattern, repl, new_fixtures, idfunc):
    """Create unions of fixtures based on specific name mappings.

    `fixtures` are grouped into unions according the regex replacement
    `re.sub(pattern, repl)` and placed into `new_fixtures`.
    """

    def collapser(n):
        return re.sub(pattern, repl, n)

    for name, group in groupby(sorted(fixtures, key=collapser), key=collapser):
        group = list(group)
        if len(group) > 1:
            if name not in fixtures | new_fixtures:
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


def make_fixture(name, func):
    """Create a named fixture and inject it into the global namespace.

    https://github.com/pytest-dev/pytest/issues/2424#issuecomment-333387206
    explains why this hack is necessary. Essentially, dynamically generated
    fixtures must exist in globals() to be found by pytest.
    """
    globals()[name] = pytest_cases.fixture(name=name)(func)
    global fixtures
    fixtures.add(name)


# First generate all the base fixtures.
fixtures = OrderedSet()
for dtype, column_generator in column_generators.items():

    def make_dataframe(nr, nc, column_generator=column_generator):
        assert nc <= len(string.ascii_lowercase)
        return cudf.DataFrame(
            {f"{string.ascii_lowercase[i]}": column_generator(nr) for i in range(nc)}
        )

    for nr in NUM_ROWS:
        # TODO: pytest_cases.fixture doesn't appear to support lambdas where
        # pytest does.
        # TODO: pytest_cases seems to have a bug where the first argument being
        # a kwarg (nr=nr, nc=nc) raises errors. I'll need to track that
        # upstream, for now I'm just passing the request fixture and not using
        # it as a way to bypass the issue.
        def series_nulls_false(request, nr=nr, column_generator=column_generator):
            return cudf.Series(column_generator(nr))

        make_fixture(f"series_dtype_{dtype}_nulls_false_rows_{nr}", series_nulls_false)

        def series_nulls_true(request, nr=nr, column_generator=column_generator):
            s = cudf.Series(column_generator(nr))
            s.iloc[::2] = None
            return s

        make_fixture(f"series_dtype_{dtype}_nulls_true_rows_{nr}", series_nulls_true)

        # For now, not bothering to include a nullable index fixture.
        def index_nulls_false(request, nr=nr, column_generator=column_generator):
            return cudf.Index(column_generator(nr))

        make_fixture(f"index_dtype_{dtype}_nulls_false_rows_{nr}", index_nulls_false)

        for nc in NUM_COLS:

            def dataframe_nulls_false(
                request, nr=nr, nc=nc, make_dataframe=make_dataframe
            ):
                return make_dataframe(nr, nc)

            make_fixture(
                f"dataframe_dtype_{dtype}_nulls_false_cols_{nc}_rows_{nr}",
                dataframe_nulls_false,
            )

            def dataframe_nulls_true(
                request, nr=nr, nc=nc, make_dataframe=make_dataframe
            ):
                df = make_dataframe(nr, nc)
                df.iloc[::2, :] = None
                return df

            make_fixture(
                f"dataframe_dtype_{dtype}_nulls_true_cols_{nc}_rows_{nr}",
                dataframe_nulls_true,
            )


# We define some custom naming functions for use in the creation of fixture
# unions to create more readable test function names that don't contain the
# entire union, which quickly becomes intractably long.
def l1_id(val):
    return val.alternative_name


def default_id(val):
    return f"alt{val.get_alternative_idx()}"


# Label the first level differently from others since there's no redundancy.
idfunc = l1_id
# This is a temporary assignment to effect a do-while loop below. new_fixtures
# really starts off empty.
new_fixtures = fixtures

# Keep trying to merge existing fixtures until no new fixtures are added.
while new_fixtures:
    new_fixtures = OrderedSet()

    # Note: If we start also introducing unions across dtypes, most likely
    # those will take the form `*int_and_float*` or similar since we won't want
    # to union _all_ dtypes. In that case, the regexes will need to use
    # suitable lookaheads etc to avoid infinite loops here.
    for pat, repl in [
        ("_nulls_(true|false)", ""),
        ("series|dataframe", "indexedframe"),
        ("indexedframe|index", "frame_or_index"),
        (r"_rows_\d+", ""),
        (r"_cols_\d+", ""),
    ]:

        collapse_fixtures(fixtures, pat, repl, new_fixtures, idfunc)

    fixtures |= new_fixtures
    idfunc = default_id


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
