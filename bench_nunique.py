def test_rangeindex_nunique(benchmark, rangeindex):
    benchmark(rangeindex.nunique)
