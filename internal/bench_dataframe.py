from utils import cudf_benchmark, make_boolean_mask_column


@cudf_benchmark(cls="dataframe", dtype="int")
def bench_apply_boolean_mask(benchmark, dataframe):
    mask = make_boolean_mask_column(len(dataframe))
    benchmark(dataframe._apply_boolean_mask, mask)
