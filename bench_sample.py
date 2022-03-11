import cupy as cp
import numpy as np
import pytest

from utils import make_frame


@pytest.fixture(params=[10000, 100000000], ids=["size10K", "size100M"])
def df(request):
    size = request.param
    return make_frame(ncols=10, nkey_cols=0, nrows=size)


@pytest.fixture(
    params=[0, np.random.RandomState, cp.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    if isinstance(rs, int):
        return rs
    return rs(seed=42)


def test_sample_df(benchmark, df, axis, random_state):
    if axis == 1 and isinstance(random_state, cp.random.RandomState):
        pytest.skip("Unsupported params.")
    frac = 0.5
    benchmark(df.sample, frac=frac, axis=axis, random_state=random_state)
