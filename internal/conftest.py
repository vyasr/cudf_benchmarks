import itertools

import pytest

from common.config import cudf, cupy  # noqa: E402


@pytest.fixture(params=itertools.product([100, 10000], [True, False]))
def col(request):
    """Create a cudf column.

    The two parameters are `nrows` and `has_nulls`
    """
    nrows, has_nulls = request.param
    rstate = cupy.random.RandomState(seed=0)
    c = cudf.core.column.as_column(rstate.randn(nrows))
    if has_nulls:
        # The choice of null placement is arbitrary.
        c[::2] = None
    return c
