def bench_rangeindex_column(benchmark, rangeindex):
    benchmark(lambda: rangeindex._column)


def bench_rangeindex_columns(benchmark, rangeindex):
    benchmark(lambda: rangeindex._columns)
