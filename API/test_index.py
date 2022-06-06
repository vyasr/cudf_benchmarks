"""Benchmarks of Index methods."""


def test_sort_values(benchmark, index_dtype_int_nulls_false):
    benchmark(index_dtype_int_nulls_false.sort_values)
