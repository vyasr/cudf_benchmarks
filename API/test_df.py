import string

import pytest
from config import cudf

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
