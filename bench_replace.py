def test_rangeindex_replace(benchmark, rangeindex):
    benchmark(rangeindex.replace, 0, 2)
