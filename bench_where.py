import cudf


def test_rangeindex_where(benchmark, rangeindex):
    cond = rangeindex % 2 == 0
    benchmark(rangeindex.where, cond, 0)
