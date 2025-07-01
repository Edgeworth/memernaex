# Copyright 2022 Eliot Courtney.
from pathlib import Path
from typing import ClassVar

import polars as pl
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.plot.plots import Column, plot_mean_log_quantity, plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class SuboptPerfPlotter:
    COLS: ClassVar[dict[str, Column]] = {
        "package_name": Column(idx="package_name", name="Package Name"),
        "ctd": Column(idx="ctd", name="CTD"),
        "lonely_pairs": Column(idx="lonely_pairs", name="Lonely Pairs"),
        "energy_model": Column(idx="energy_model", name="Energy Model"),
        "backend": Column(idx="backend", name="Backend"),
        "sorted_strucs": Column(idx="sorted_strucs", name="Sorted Structures"),
        "delta": Column(idx="delta", name="Delta"),
        "strucs": Column(idx="strucs", name="Structures"),
        "time_secs": Column(idx="time_secs", name="Time (s)"),
        "count_only": Column(idx="count_only", name="Count Only"),
        "algorithm": Column(idx="algorithm", name="Algorithm"),
        "dataset": Column(idx="dataset", name="Dataset"),
        "rna_name": Column(idx="rna_name", name="RNA Name"),
        "rna_length": Column(idx="rna_length", name="Length (nuc)"),
        "run_idx": Column(idx="run_idx", name="Run Index"),
        "output_strucs": Column(idx="output_strucs", name="Output Structures"),
        "maxrss_bytes": Column(
            idx="maxrss_bytes",
            name="Maximum RSS (B)",
            formatter=ticker.FuncFormatter(lambda x, _: human_size(x, False)),
        ),
        "user_sec": Column(idx="user_sec", name="User time (s)"),
        "sys_sec": Column(idx="sys_sec", name="Sys time (s)"),
        "real_sec": Column(idx="real_sec", name="Wall time (s)"),
        "failed": Column(idx="failed", name="Failed"),
    }
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.df = pl.read_ndjson(input_path)
        self.output_dir = output_dir
        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, name: str = "") -> None:
        for y in ["real_sec", "maxrss_bytes"]:
            f = plot_mean_quantity(self.df, "package_name", self.COLS["rna_length"], self.COLS[y])
            save_figure(f, self._path(name + y))

    def run(self) -> None:
        self._plot_quantity()

        for y in ["real_sec", "maxrss_bytes"]:
            f = plot_mean_log_quantity(
                self.df, "package_name", self.COLS["rna_length"], self.COLS[y]
            )
            save_figure(f, self._path(f"{y}_log"))
