import pytest


@pytest.mark.parametrize("dropnan", [True, False])
def bench_dropna_column(benchmark, col, dropnan):
    benchmark(col.dropna, drop_nan=dropnan)
