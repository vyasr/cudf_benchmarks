import string

import pytest

exprs = ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tan(d*e*f)"]


@pytest.mark.parametrize("expr", exprs)
def test_eval_func(benchmark, expr, dataframe_dtype_float_cols_6):
    benchmark(dataframe_dtype_float_cols_6.eval, expr)


@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def test_merge(benchmark, dataframe_dtype_int_nulls_false_cols_6, nkey_cols):
    benchmark(
        dataframe_dtype_int_nulls_false_cols_6.merge,
        dataframe_dtype_int_nulls_false_cols_6,
        on=[f"{string.ascii_lowercase[i]}" for i in range(nkey_cols)],
    )
