"""Module used for global configuration of benchmarks.

This file contains global definitions that are important for configuring all
benchmarks such as fixture sizes. In addition, this file supports the following
features:
    - Defining the CUDF_BENCHMARKS_USE_PANDAS environment variable will change
      all benchmarks to run with pandas instead of cudf (and numpy instead of
      cupy). This feature enables easy comparisons of benchmarks between cudf
      and pandas. All common modules (cudf, cupy) should be imported from here
      by benchmark modules to allow configuration if needed.
    - Defining CUDF_BENCHMARKS_TEST_ONLY will set global configuration
      variables to avoid running large benchmarks, instead using minimal values
      to simply ensure that benchmarks are functional.
"""
import os
import sys

# Environment variable-based configuration of benchmarking pandas or cudf.
collect_ignore = []
if "CUDF_BENCHMARKS_USE_PANDAS" in os.environ:
    import numpy as cupy
    import pandas as cudf

    # cudf internals offer no pandas compatibility guarantees, and we also
    # never need to compare those benchmarks to pandas.
    collect_ignore.append("internal/")

    # Also filter out benchmarks of APIs that are not compatible with pandas.
    def is_pandas_compatible(item):
        return all(m.name != "pandas_incompatible" for m in item.own_markers)

    def pytest_collection_modifyitems(session, config, items):
        items[:] = list(filter(is_pandas_compatible, items))

else:
    import cudf  # noqa: W0611, F401
    import cupy

    def pytest_collection_modifyitems(session, config, items):
        pass


def pytest_sessionstart(session):
    """Add the common files to the path for all tests to import."""
    sys.path.insert(0, os.path.join(os.getcwd(), "common"))


def pytest_sessionfinish(session, exitstatus):
    """Clean up sys.path after exit."""
    if "common" in sys.path[0]:
        del sys.path[0]


# Constants used to define benchmarking standards.
if "CUDF_BENCHMARKS_TEST_ONLY" in os.environ:
    NUM_ROWS = [10, 20]
    NUM_COLS = [1, 6]

    # Some benchmarks aren't solely reliant on fixtures for size and become too
    # unwieldy. In the long run we may want a custom mark like
    # `pandas_incompatible` for more granular identification of such functions.
    collect_ignore.append("API/bench_functions.py")
else:
    NUM_ROWS = [100, 10_000, 1_000_000]
    NUM_COLS = [1, 6]

# A dictionary of callables that create a column of a specified length
random_state = cupy.random.RandomState(42)
column_generators = {
    "int": (lambda nr: random_state.randint(low=0, high=100, size=nr)),
    "float": (lambda nr: random_state.rand(nr)),
}
