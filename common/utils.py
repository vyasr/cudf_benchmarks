import inspect
import textwrap
from numbers import Real

from config import NUM_COLS, column_generators, cudf
from config import cupy as cp


def make_gather_map(len_gather_map: Real, len_column: Real, how: str):
    """Create a gather map based on "how" you'd like to gather from input.
    - sequence: gather the first `len_gather_map` rows, the first thread
                collects the first element
    - reverse:  gather the last `len_gather_map` rows, the first thread
                collects the last element
    - random:   create a pseudorandom gather map

    `len_gather_map`, `len_column` gets rounded to integer.
    """
    len_gather_map = round(len_gather_map)
    len_column = round(len_column)

    rstate = cp.random.RandomState(seed=0)
    if how == "sequence":
        return cudf.Series(cp.arange(0, len_gather_map))._column
    elif how == "reverse":
        return cudf.Series(
            cp.arange(len_column - 1, len_column - len_gather_map - 1, -1)
        )._column
    elif how == "random":
        return cudf.Series(rstate.randint(0, len_column, len_gather_map))._column


def make_boolean_mask_column(size):
    rstate = cp.random.RandomState(seed=0)
    return cudf.core.column.as_column(rstate.randint(0, 2, size).astype(bool))


def flatten(xs):
    for x in xs:
        if not isinstance(x, str):
            yield from x
        else:
            yield x


def cudf_benchmark(cls, dtype="int", nulls=None, cols=None, name=None):
    """A convenience wrapper for using cudf's 'standard' fixtures.

    The standard fixture generation logic provides a plethora of useful
    fixtures to allow developers to easily select an appropriate cross-section
    of the space of objects to apply a particular benchmark to. However, the
    usage of this fixtures is cumbersome because creating them in a principled
    fashion results in long names and very specific naming schemes. This
    decorator abstracts that naming logic away from the developer, allowing
    them to instead focus on defining the fixture semantically by describing
    its properties.

    Parameters
    ----------
    cls : Union[str, Type]
        The class of object to test. May either be specified as the type
        itself, or using the name (as a string). If a string, the case is
        irrelevant as the string will be converted to all lowercase.
    dtype : Union[str, Iterable[str]], default 'int'
        The dtype or set of dtypes to use.
    nulls : Optional[bool], default None
        Whether to test nullable or non-nullable data. If None, both nullable
        and non-nullable data are included.
    cols : Optional[int], None
        The number of columns. Only valid if cls == 'dataframe'. If None, use
        all possible numbers of columns. Specifying multiple values is
        unsupported.
    fixture_name : str, default None
        The name of the fixture as used in the decorated test. If None,
        defaults to `cls.lower()` if cls is a string, otherwise
        `cls.__name__.lower()`. Use of this name allows the decorated function
        to masquerade as a pytest receiving a fixture while substituting the
        real fixture (with a much longer name).

    Raises
    ------
    ValueError
        If any of the parameters do not correspond to extant fixtures.
    ValueError
        If cols != None and cls != 'dataframe'

    Examples
    --------
    # Note: As an internal function, this example is not meant for doctesting.

    @cudf_benchmark("dataframe", dtype="int", nulls=False, name="df")
    def bench_columns(benchmark, df):
        benchmark(df.columns)
    """
    if inspect.isclass(cls):
        cls = cls.__name__
    cls = cls.lower()

    # TODO: See if there's a better way to centralize this definition.
    supported_classes = (
        "series",
        "index",
        "dataframe",
        "indexedframe",
        "frame_or_index",
    )
    assert (
        cls in supported_classes
    ), f"cls {cls} is invalid, choose from {', '.join(c for c in supported_classes)}"

    name = name or cls

    if not isinstance(dtype, list):
        dtype = [dtype]
    assert all(
        dt in column_generators for dt in dtype
    ), f"The only supported dtypes are {', '.join(dt for dt in column_generators)}"
    dtype_str = "_dtype_" + "_or_".join(dtype)

    null_str = ""
    if nulls is not None:
        null_str = f"_nulls_{nulls}".lower()

    col_str = ""
    if cols is not None:
        assert cols in NUM_COLS, (
            f"You have requested a DataFrame with {cols} columns but fixtures "
            f"only exist for the values {', '.join(c for c in NUM_COLS)}"
        )
        col_str = f"_cols_{cols}"

    fixture_name = f"{cls}{dtype_str}{null_str}{col_str}"

    def deco(func):
        # Construct the signature of the wrapper function to mimic the wrapped
        # function and use that signature to construct the list of arguments
        # forwarded to the wrapped function.
        parameters = inspect.signature(func).parameters

        # Note: This assumes that any benchmark using this fixture has at least
        # two parameters, but that must be valid because they must be using
        # both the pytest-benchmark `benchmark` fixture and the cudf object.
        params_str = ", ".join(f"{p}" for p in parameters if p != name)
        arg_str = ", ".join(f"{p}={p}" for p in parameters if p != name)

        params_str += f", {fixture_name}"
        arg_str += f", {name}={fixture_name}"

        src = textwrap.dedent(
            f"""
            def wrapped({params_str}):
                func({arg_str})
            """
        )
        globals_ = {"func": func}
        exec(src, globals_)
        wrapped = globals_["wrapped"]
        # In case marks were applied to the wrapped function, copy them over.
        if marks := getattr(func, "pytestmark", None):
            wrapped.pytestmark = marks
        return wrapped

    return deco
