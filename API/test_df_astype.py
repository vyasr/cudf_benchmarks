def test_df_apply_boolean_mask(benchmark, df):
    benchmark(df.astype, float)
