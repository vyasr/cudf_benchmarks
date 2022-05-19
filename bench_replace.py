def test_rangeindex_replace(benchmark, rangeindex):
    to_replace = rangeindex % 2 == 0
    benchmark(rangeindex.replace, to_replace, 2)
