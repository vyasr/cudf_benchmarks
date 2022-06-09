import inspect
from numbers import Real

from config import cudf
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
    name = name or cls

    if not isinstance(dtype, list):
        dtype = [dtype]
    dtype_str = "_dtype_" + "_or_".join(dtype)

    null_str = ""
    if nulls is not None:
        null_str = f"_nulls_{nulls}".lower()

    col_str = ""
    if cols is not None:
        col_str = f"_cols_{cols}"

    fixture_name = f"{cls}{dtype_str}{null_str}{col_str}"

    def deco(func):
        # Note: Marks must be applied _before_ the cudf_benchmark decorator.
        # Extract all marks and apply them directly to the wrapped function
        # except for parametrize. For parametrize, we also need to augment the
        # signature and forward the parameters.
        marks = func.pytestmark
        mark_parameters = [m for m in marks if m.name == "parametrize"]

        # Parameters may be specified as tuples, so we need to flatten.
        parameters = list(flatten([m.args[0] for m in mark_parameters]))
        params_string = ", ".join(f"{p}" for p in parameters)
        passed_params_string = ", ".join(f"{p}={p}" for p in parameters)
        src = f"""
def wrapped(benchmark, {fixture_name}, {params_string}):
    func(benchmark, {passed_params_string}, {name}={fixture_name})
"""
        globals_ = {"func": func}
        exec(src, globals_)
        wrapped = globals_["wrapped"]
        wrapped.pytestmark = marks
        return wrapped

    return deco
