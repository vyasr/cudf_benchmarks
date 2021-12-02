from utils import make_boolean_mask_column


def test_df_apply_boolean_mask(benchmark, df):
    mask = make_boolean_mask_column(len(df))
    benchmark(df._apply_boolean_mask, mask)
