import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.getcwd(), "common"))
from common.utils import make_col  # noqa: E402


@pytest.fixture(params=itertools.product([100, 10000], [True, False]))
def col(request):
    """Create a cudf column.

    The two parameters are `nrows` and `has_nulls`
    """
    return make_col(*request.param)
