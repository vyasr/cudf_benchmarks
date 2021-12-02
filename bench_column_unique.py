def test_unique_single_column(benchmark, col):
    benchmark(col.unique)
