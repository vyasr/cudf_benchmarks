import string

import numpy
import pytest
from config import cudf, cupy

exprs = ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tan(d*e*f)"]


@pytest.mark.parametrize("expr", exprs)
def test_eval_func(benchmark, expr, dataframe_dtype_float_cols_6):
    benchmark(dataframe_dtype_float_cols_6.eval, expr)


@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def test_merge(benchmark, dataframe_dtype_int_cols_6, nkey_cols):
    benchmark(
        dataframe_dtype_int_cols_6.merge,
        dataframe_dtype_int_cols_6,
        on=[f"{string.ascii_lowercase[i]}" for i in range(nkey_cols)],
    )


# TODO: Some of these cases could be generalized to an IndexedFrame benchmark
# instead of a DataFrame benchmark.
@pytest.mark.parametrize(
    "values",
    [
        range(1000),
        {f"key{i}": range(1000) for i in range(10)},
        cudf.DataFrame({f"key{i}": range(1000) for i in range(10)}),
        cudf.Series(range(1000)),
    ],
)
def test_isin(benchmark, dataframe_dtype_int, values):
    benchmark(dataframe_dtype_int.isin, values)


@pytest.fixture(
    params=[0, numpy.random.RandomState, cupy.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    return rs if isinstance(rs, int) else rs(seed=42)


@pytest.mark.parametrize("frac", [0.5])
def test_sample(benchmark, dataframe_dtype_int, axis, frac, random_state):
    if axis == 1 and isinstance(random_state, cupy.random.RandomState):
        pytest.skip("Unsupported params.")
    benchmark(
        dataframe_dtype_int.sample, frac=frac, axis=axis, random_state=random_state
    )
