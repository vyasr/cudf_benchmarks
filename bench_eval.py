import string

import pytest

from config import cudf

exprs = [
    "a+b",
    "a+b+c+d+e",
    "a / (sin(a) + cos(b)) * tan(d*e*f)"
]
col_names = tuple(s for s in string.ascii_lowercase)


def make_frame(ncol, nrow):
    return cudf.DataFrame({col_names[i]: range(nrow) for i in range(ncol)})


df_1000000 = make_frame(6, 1_000_000).astype(float)
df_100 = make_frame(6, 100).astype(float)


@pytest.mark.parametrize("expr", exprs)
@pytest.mark.parametrize("df", [df_100, df_1000000])
def test_eval_func(benchmark, expr, df):
    benchmark(
        df.eval,
        expr,
    )
