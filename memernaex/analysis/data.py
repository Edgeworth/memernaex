import inspect
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from matplotlib import ticker


@dataclass(frozen=True, eq=True, order=True, kw_only=True)
class Var:
    id: str
    name: str
    dtype: type[pl.DataType]
    derived: bool = False
    formatter: ticker.FuncFormatter | None = None


def read_var_data(cls: type, path: Path) -> pl.DataFrame:
    df = pl.read_ndjson(path)
    for _, var_obj in inspect.getmembers(cls, lambda member: isinstance(member, Var)):
        if var_obj.derived:
            continue
        df = df.with_columns(pl.col(var_obj.id).cast(var_obj.dtype, strict=True).alias(var_obj.id))
    return df
