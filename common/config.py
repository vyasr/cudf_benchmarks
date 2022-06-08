"""Module used for global configuration of benchmarks.

All common modules (cudf, cupy) should be imported from here by benchmark
modules to allow configuration if needed.
"""
import inspect
import os

# Environment variable-based configuration of benchmarking pandas or cudf.
if "CUDF_BENCHMARKS_USE_PANDAS" in os.environ:
    import numpy as cupy
    import pandas as cudf
else:
    import cudf  # noqa: W0611, F401
    import cupy  # noqa: W0611, F401


# TODO: Choose real values here before merging.
# Constants used to define benchmarking standards.
NUM_ROWS = [10]  # The column lengths to use for benchmarked objects
NUM_COLS = [1, 6]  # The numbers of columns to use for benchmarked DataFrames


def cudf_benchmark(cls, dtype="int", nulls=None, cols=None, name="obj"):
    if inspect.isclass(cls):
        cls = cls.__name__
    cls = cls.lower()

    if not isinstance(dtype, list):
        dtype = [dtype]
    dtype_str = "_dtype_" + "_or_".join(dtype)

    null_str = ""
    if nulls is not None:
        null_str = f"_nulls_{nulls}".lower()

    col_str = ""
    if cols is not None:
        col_str = f"_cols_{cols}"

    fixture_name = f"{cls}{dtype_str}{null_str}{col_str}"

    def deco(func):
        src = f"""
def wrapped(benchmark, {fixture_name}):
    func(benchmark, {fixture_name})
"""
        globals_ = {"func": func}
        exec(src, globals_)
        return globals_["wrapped"]

    return deco
