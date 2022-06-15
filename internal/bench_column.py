import pytest
from utils import cudf_benchmark, make_boolean_mask_column, make_gather_map


@cudf_benchmark(cls="column", dtype="float")
def bench_apply_boolean_mask(benchmark, column):
    mask = make_boolean_mask_column(column.size)
    benchmark(column.apply_boolean_mask, mask)


@cudf_benchmark(cls="column", dtype="float")
@pytest.mark.parametrize("dropnan", [True, False])
def bench_dropna(benchmark, column, dropnan):
    benchmark(column.dropna, drop_nan=dropnan)


@cudf_benchmark(cls="column", dtype="float")
def bench_unique_single_column(benchmark, column):
    benchmark(column.unique)


@cudf_benchmark(cls="column", dtype="float")
@pytest.mark.parametrize("nullify", [True, False])
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
def bench_take(benchmark, column, gather_how, nullify):
    gather_map = make_gather_map(column.size * 0.4, column.size, gather_how)._column
    benchmark(column.take, gather_map, nullify=nullify)
