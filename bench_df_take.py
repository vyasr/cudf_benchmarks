import pytest

from utils import make_gather_map


@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
def test_take_multiple_column(benchmark, df, gather_how):
    gather_map = make_gather_map(len(df) * 0.4, len(df), gather_how)
    benchmark(df.take, gather_map)
