import string

import numpy
import pytest
from config import cudf, cudf_benchmark, cupy


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.DataFrame, {None: cupy.random.rand(N)})


@pytest.mark.parametrize(
    "expr", ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tan(d*e*f)"]
)
def bench_eval_func(benchmark, expr, dataframe_dtype_float_cols_6):
    benchmark(dataframe_dtype_float_cols_6.eval, expr)


@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def bench_merge(benchmark, dataframe_dtype_int_nulls_false_cols_6, nkey_cols):
    on = list(dataframe_dtype_int_nulls_false_cols_6.columns[:nkey_cols])
    benchmark(
        dataframe_dtype_int_nulls_false_cols_6.merge,
        dataframe_dtype_int_nulls_false_cols_6,
        on=on,
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
def bench_isin(benchmark, dataframe_dtype_int, values):
    benchmark(dataframe_dtype_int.isin, values)


@pytest.fixture(
    params=[0, numpy.random.RandomState, cupy.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    return rs if isinstance(rs, int) else rs(seed=42)


@pytest.mark.parametrize("frac", [0.5])
def bench_sample(benchmark, dataframe_dtype_int, axis, frac, random_state):
    if axis == 1 and isinstance(random_state, cupy.random.RandomState):
        pytest.skip("Unsupported params.")
    benchmark(
        dataframe_dtype_int.sample, frac=frac, axis=axis, random_state=random_state
    )


@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def bench_groupby(benchmark, dataframe_dtype_int_nulls_false_cols_6, nkey_cols):
    by = list(dataframe_dtype_int_nulls_false_cols_6.columns[:nkey_cols])
    benchmark(dataframe_dtype_int_nulls_false_cols_6.groupby, by=by)


@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        ["sum", "mean"],
        {f"{string.ascii_lowercase[i]}": ["sum", "mean", "count"] for i in range(6)},
    ],
)
@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def bench_groupby_agg(
    benchmark, dataframe_dtype_int_nulls_false_cols_6, agg, nkey_cols, as_index, sort
):
    by = list(dataframe_dtype_int_nulls_false_cols_6.columns[:nkey_cols])
    benchmark(
        dataframe_dtype_int_nulls_false_cols_6.groupby(
            by=by, as_index=as_index, sort=sort
        ).agg,
        agg,
    )


@pytest.mark.parametrize("ncol_sort", [1])
def bench_sort_values(benchmark, dataframe_dtype_int, ncol_sort):
    by = list(dataframe_dtype_int.columns[:ncol_sort])
    benchmark(dataframe_dtype_int.sort_values, by)


@pytest.mark.parametrize("ncol_sort", [1])
@pytest.mark.parametrize("n", [10])
def bench_nsmallest(benchmark, dataframe_dtype_int, ncol_sort, n):
    by = list(dataframe_dtype_int.columns[:ncol_sort])
    benchmark(dataframe_dtype_int.nsmallest, n, by)


@cudf_benchmark(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "expr", ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tan(d*e*f)"]
)
def bench_eval_func2(benchmark, obj, expr):
    benchmark(obj.eval, "a+b")
