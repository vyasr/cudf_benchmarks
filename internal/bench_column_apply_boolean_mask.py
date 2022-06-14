import pytest
from utils import make_boolean_mask_column


def bench_apply_boolean_mask(benchmark, col):
    mask = make_boolean_mask_column(col.size)
    benchmark(col.apply_boolean_mask, mask)


@pytest.mark.parametrize("dropnan", [True, False])
def bench_dropna(benchmark, col, dropnan):
    benchmark(col.dropna, drop_nan=dropnan)


def bench_unique_single_column(benchmark, col):
    benchmark(col.unique)
