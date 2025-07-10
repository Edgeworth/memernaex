# Copyright 2022 Eliot Courtney.
from pathlib import Path

import polars as pl

from memernaex.experiments.subopt.perf_common import (
    SUBOPT_PERF_COLS,
    SUBOPT_PERF_GROUP_VARS,
    load_subopt_perf_df,
)
from memernaex.plot.plots import plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class SuboptPerfPlotter:
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.df = load_subopt_perf_df(input_path)
        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, name: str) -> None:
        cols = SUBOPT_PERF_COLS
        for group, df in self.df.group_by(SUBOPT_PERF_GROUP_VARS):
            group_name = "_".join(str(x) for x in group)
            for y in ["strucs_per_sec", "bases_per_byte", "maxrss_bytes"]:
                f = plot_mean_quantity(df, "program", cols["rna_length"], cols[y])
                save_figure(f, self._path(f"{name}_{group_name}_{y}"))

    def run(self) -> None:
        self._plot_quantity("quantity")
