# Copyright 2022 Eliot Courtney.
from pathlib import Path
from typing import ClassVar

import polars as pl
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.plot.plots import Column, plot_mean_log_quantity, plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class FoldPerfPlotter:
    COLS: ClassVar[dict[str, Column]] = {
        "name": Column(idx="name", name="Name"),
        "length": Column(idx="length", name="Length (nuc)"),
        "real_sec": Column(idx="real_sec", name="Wall time (s)"),
        "user_sec": Column(idx="user_sec", name="User time (s)"),
        "sys_sec": Column(idx="sys_sec", name="Sys time (s)"),
        "maxrss_bytes": Column(
            idx="maxrss_bytes",
            name="Maximum RSS (B)",
            formatter=ticker.FuncFormatter(lambda x, _: human_size(x, False)),
        ),
    }
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.df = pl.read_ndjson(input_path)
        self.output_dir = output_dir
        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, df: pl.DataFrame, name: str) -> None:
        for y in ["real_sec", "maxrss_bytes"]:
            f = plot_mean_quantity(df, "program", self.COLS["length"], self.COLS[y])
            save_figure(f, self._path(name + y))

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

            for y in ["real_sec", "maxrss_bytes"]:
                f = plot_mean_log_quantity(df, "program", self.COLS["length"], self.COLS[y])
                save_figure(f, self._path(f"{dataset_name}_{y}_log"))
