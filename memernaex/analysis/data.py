

def from pathlib import Path
import polars as pl
from matplotlib import ticker


from dataclasses import dataclass

@dataclass(frozen=True, eq=True, order=True, kw_only=True)
class Var:
    id: str
    name: str
    dtype: type[pl.DataType]
    formatter: ticker.FuncFormatter | None = None


def read_var_data(self, path: Path) -> pl.DataFrame:
    return pl.read_ndjson(path)
