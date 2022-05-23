def test_rangeindex_values_host(benchmark, rangeindex):
    benchmark(rangeindex.values_host)
