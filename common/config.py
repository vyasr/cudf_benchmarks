"""Module used for global configuration of benchmarks.

All common modules (cudf, cupy) should be imported from here by benchmark
modules to allow configuration if needed.
"""
import os

# Environment variable-based configuration of benchmarking pandas or cudf.
if "CUDF_BENCHMARKS_USE_PANDAS" in os.environ:
    import numpy as cupy
    import pandas as cudf
else:
    import cudf  # noqa: W0611, F401
    import cupy


# TODO: Choose real values here before merging.
# Constants used to define benchmarking standards.
NUM_ROWS = [10]  # The column lengths to use for benchmarked objects
NUM_COLS = [1, 6]  # The numbers of columns to use for benchmarked DataFrames

# A dictionary of callables that create a column of a specified length
random_state = cupy.random.RandomState(42)
column_generators = {
    "int": (lambda nr: random_state.randint(low=0, high=100, size=nr)),
    "float": (lambda nr: random_state.rand(nr)),
}
