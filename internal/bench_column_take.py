import pytest
from utils import make_gather_map


@pytest.mark.parametrize("nullify", [True, False])
@pytest.mark.parametrize("gather_how", ["sequence", "reverse", "random"])
def bench_gather_single_column(benchmark, col, gather_how, nullify):
    gather_map = make_gather_map(col.size * 0.4, col.size, gather_how)
    benchmark(col.take, gather_map, nullify=nullify)
