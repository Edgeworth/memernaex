import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
from matplotlib import ticker


@dataclass(frozen=True, eq=True, order=True, kw_only=True)
class Var:
    id: str
    name: str
    dtype: type[pl.DataType]
    derived: bool = False
    formatter: ticker.FuncFormatter | None = None


def _get_vars(source: type | ModuleType | Iterable[Any]) -> list[Var]:
    if isinstance(source, (type, ModuleType)):
        return [v for _, v in inspect.getmembers(source, lambda o: isinstance(o, Var))]
    return [v for v in source if isinstance(v, Var)]


def read_var_data(var_source: type | ModuleType | Iterable[Var], path: Path) -> pl.DataFrame:
    df = pl.read_ndjson(path)
    varz = _get_vars(var_source)
    for var in varz:
        if var.derived:
            continue
        df = df.with_columns(pl.col(var.id).cast(var.dtype, strict=True).alias(var.id))
    return df
