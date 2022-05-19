def test_rangeindex_values_host(benchmark, rangeindex):
    benchmark(lambda: rangeindex.values_host)
