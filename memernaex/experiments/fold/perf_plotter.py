# Copyright 2022 Eliot Courtney.
from pathlib import Path

import polars as pl
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.analysis.data import Var, read_var_data
from memernaex.plot.plots import plot_mean_log_quantity, plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class FoldPerfPlotter:
    VAR_NAME = Var(id="name", name="Name", dtype=pl.String)
    VAR_LENGTH = Var(id="length", name="Length (nuc)", dtype=pl.Int64)
    VAR_REAL_SEC = Var(id="real_sec", name="Wall time (s)", dtype=pl.Float64)
    VAR_USER_SEC = Var(id="user_sec", name="User time (s)", dtype=pl.Float64)
    VAR_SYS_SEC = Var(id="sys_sec", name="Sys time (s)", dtype=pl.Float64)
    VAR_MAXRSS_BYTES = Var(
        id="maxrss_bytes",
        name="Maximum RSS (B)",
        dtype=pl.Int64,
        formatter=ticker.FuncFormatter(lambda x, _: human_size(x, False)),
    )
    VAR_PROGRAM = Var(id="program", name="Program", dtype=pl.String)
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.df = read_var_data(self.__class__, input_path)
        self.output_dir = output_dir
        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, df: pl.DataFrame, name: str) -> None:
        y_vars = [self.VAR_REAL_SEC, self.VAR_MAXRSS_BYTES]
        for y_var in y_vars:
            f = plot_mean_quantity(df, self.VAR_PROGRAM, self.VAR_LENGTH, y_var)
            save_figure(f, self._path(name + y_var.id))

    def run(self) -> None:
        # Plot quantities
        for group, df in self.df.group_by("dataset"):
            dataset_name = str(group[0])
            self._plot_quantity(df, dataset_name)

            # Also plot random dataset without RNAstructure and ViennaRNA-d3
            if dataset_name == "random":
                subset_df = df.filter(
                    ~pl.col("program").is_in(["RNAstructure", "ViennaRNA-d3", "ViennaRNA-d3-noLP"])
                )
                self._plot_quantity(subset_df, f"{dataset_name}_subset_")

            y_vars = [self.VAR_REAL_SEC, self.VAR_MAXRSS_BYTES]
            for y_var in y_vars:
                f = plot_mean_log_quantity(df, self.VAR_PROGRAM, self.VAR_LENGTH, y_var)
                save_figure(f, self._path(f"{dataset_name}_{y_var.id}_log"))
