def test_rangeindex_to_arrow(benchmark, rangeindex):
    benchmark(rangeindex.to_arrow)
