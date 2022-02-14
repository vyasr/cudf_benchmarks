from utils import make_boolean_mask_column


def test_column_apply_boolean_mask(benchmark, col):
    mask = make_boolean_mask_column(col.size)
    benchmark(col.apply_boolean_mask, mask)
