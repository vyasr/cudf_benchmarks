"""Module used for global configuration of benchmarks.

All common modules (cudf, cupy) should be imported from here by benchmark
modules to allow configuration if needed.
"""
import os

# Environment variable-based configuration of benchmarking pandas or cudf.
if "CUDF_BENCHMARKS_USE_PANDAS" in os.environ:
    import pandas as cudf
    import numpy as cupy
else:
    import cudf  # noqa: W0611
    import cupy  # noqa: W0611
