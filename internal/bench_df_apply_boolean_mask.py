from utils import make_boolean_mask_column


def bench_df_apply_boolean_mask(benchmark, dataframe_dtype_int):
    mask = make_boolean_mask_column(len(dataframe_dtype_int))
    benchmark(dataframe_dtype_int._apply_boolean_mask, mask)
