def test_drop_duplicate_df(benchmark, df):
    benchmark(df.drop_duplicates)
