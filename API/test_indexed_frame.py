import pytest


@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def test_scans(benchmark, op, indexed_frame):
    benchmark(getattr(indexed_frame, op))
