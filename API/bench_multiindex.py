import numpy as np
import pandas as pd
import pytest
from config import cudf


@pytest.fixture
def pidx():
    num_elements = int(1e3)
    a = np.random.randint(0, num_elements // 10, num_elements)
    b = np.random.randint(0, num_elements // 10, num_elements)
    return pd.MultiIndex.from_arrays([a, b], names=("a", "b"))


@pytest.fixture
def midx(pidx):
    return cudf.MultiIndex.from_pandas(pidx)


def bench_from_pandas(benchmark, pidx):
    benchmark(cudf.MultiIndex.from_pandas, pidx)


def bench_constructor(benchmark, pidx):
    benchmark(cudf.MultiIndex, codes=pidx.codes, levels=pidx.levels, names=pidx.names)


def bench_from_frame(benchmark, pidx):
    benchmark(cudf.MultiIndex.from_frame, pidx.to_frame(index=False))


def bench_copy(benchmark, midx):
    benchmark(midx.copy, deep=False)