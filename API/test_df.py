import pytest

exprs = ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tan(d*e*f)"]


@pytest.mark.parametrize("expr", exprs)
def test_eval_func(benchmark, expr, dataframe_dtype_float_cols_6):
    benchmark(dataframe_dtype_float_cols_6.eval, expr)
