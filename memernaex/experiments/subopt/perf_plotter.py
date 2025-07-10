# Copyright 2022 Eliot Courtney.
from pathlib import Path
from typing import ClassVar

import polars as pl
from matplotlib import pyplot, ticker
from rnapy.util.format import human_size

from memernaex.analysis.complexity import ComplexityFitter
from memernaex.plot.plots import Column, plot_mean_quantity
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
        # Derived columns:
        "strucs_per_sec": Column(idx="strucs_per_sec", name="Structures per second"),
        "bases_per_byte": Column(idx="bases_per_byte", name="Bases per byte"),
    }
    GROUP_VARS: ClassVar[list[str]] = [
        "count_only",
        "ctd",
        "dataset",
        "delta",
        "lonely_pairs",
        "sorted_strucs",
        "strucs",
        "time_secs",
    ]
    PACKAGE_VARS: ClassVar[list[str]] = [
        "package_name",
        "ctd",
        "lonely_pairs",
        "sorted_strucs",
        "energy_model",
        "algorithm",
        "backend",
    ]

    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.df = pl.read_ndjson(input_path)

        # Add column for program identifier.
        self.df = self.df.with_columns(
            pl.format("{}-{}-{}-{}", "package_name", "ctd", "algorithm", "backend").alias(
                "program"
            ),
            (pl.col("output_strucs") / pl.col("real_sec")).alias("strucs_per_sec"),
            (pl.col("output_strucs") * pl.col("rna_length") / pl.col("maxrss_bytes")).alias(
                "bases_per_byte"
            ),
        )

        # Remove any rows with failed true.
        self.df = self.df.filter(~pl.col("failed"))

        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, name: str) -> None:
        for group, df in self.df.group_by(self.GROUP_VARS):
            group_name = "_".join(str(x) for x in group)
            for y in ["strucs_per_sec", "bases_per_byte", "maxrss_bytes"]:
                f = plot_mean_quantity(df, "program", self.COLS["rna_length"], self.COLS[y])
                save_figure(f, self._path(f"{name}_{group_name}_{y}"))

    def _analyze_complexity(self) -> None:
        for group, df in self.df.group_by(self.PACKAGE_VARS):
            # Split into two dfs - ones with "delta" non-empty and ones with "strucs" non-empty.
            df = df.filter(pl.col("delta").str.len_chars() > 0)
            print(df)
            group_name = "_".join(str(x) for x in group)
            print(group_name)
            fitter = ComplexityFitter(df)
            name, result = fitter.fit2d(xs=("delta", "rna_length"), y="real_sec")
            print(f"Best model: {name}")
            print(result.fit_report())
            result.plot()
            pyplot.show(block=True)
            break

    def run(self) -> None:
        # self._plot_quantity("quantity")
        self._analyze_complexity()
