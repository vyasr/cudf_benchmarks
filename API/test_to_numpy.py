def test_rangeindex_to_numpy(benchmark, rangeindex):
    benchmark(rangeindex.to_numpy)
