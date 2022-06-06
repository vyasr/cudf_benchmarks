import pytest


def test_sort_values(benchmark, series_dtype_int):
    benchmark(series_dtype_int.sort_values)


@pytest.mark.parametrize("n", [10])
def test_series_nsmallest(benchmark, series_dtype_int, n):
    benchmark(series_dtype_int.nsmallest, n)
