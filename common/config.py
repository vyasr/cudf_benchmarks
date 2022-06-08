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


def flatten(xs):
    for x in xs:
        if not isinstance(x, str):
            yield from x
        else:
            yield x


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
        # Note: Marks must be applied _before_ the cudf_benchmark decorator.
        # Extract all marks and apply them directly to the wrapped function
        # except for parametrize. For parametrize, we also need to augment the
        # signature and forward the parameters.
        marks = func.pytestmark
        mark_parameters = [m for m in marks if m.name == "parametrize"]

        # Parameters may be specified as tuples, so we need to flatten.
        parameters = list(flatten([m.args[0] for m in mark_parameters]))
        param_string = ", ".join(parameters)
        src = f"""
def wrapped(benchmark, {fixture_name}, {param_string}):
    func(benchmark, {fixture_name}, {param_string})
"""
        globals_ = {"func": func}
        exec(src, globals_)
        wrapped = globals_["wrapped"]
        wrapped.pytestmark = marks
        return wrapped

    return deco
