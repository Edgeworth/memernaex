# Copyright 2022 Eliot Courtney.
from pathlib import Path
from typing import ClassVar

import polars as pl
from matplotlib import pyplot as plt
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.analysis.complexity import ComplexityFitter
from memernaex.plot.plots import plot_mean_quantity
from memernaex.plot.util import Var, save_figure, set_style


class SuboptPerfPlotter:
    VAR_PACKAGE_NAME = Var(id="package_name", name="Package Name", dtype=pl.String)
    VAR_CTD = Var(id="ctd", name="CTD", dtype=pl.String)
    VAR_LONELY_PAIRS = Var(id="lonely_pairs", name="Lonely Pairs", dtype=pl.String)
    VAR_ENERGY_MODEL = Var(id="energy_model", name="Energy Model", dtype=pl.String)
    VAR_BACKEND = Var(id="backend", name="Backend", dtype=pl.String)
    VAR_SORTED_STRUCS = Var(id="sorted_strucs", name="Sorted Structures", dtype=pl.Boolean)
    VAR_DELTA = Var(id="delta", name="Delta", dtype=pl.Float64)
    VAR_STRUCS = Var(id="strucs", name="Structures", dtype=pl.Int64)
    VAR_TIME_SECS = Var(id="time_secs", name="Time (s)", dtype=pl.Float64)
    VAR_COUNT_ONLY = Var(id="count_only", name="Count Only", dtype=pl.Boolean)
    VAR_ALGORITHM = Var(id="algorithm", name="Algorithm", dtype=pl.String)
    VAR_DATASET = Var(id="dataset", name="Dataset", dtype=pl.String)
    VAR_RNA_NAME = Var(id="rna_name", name="RNA Name", dtype=pl.String)
    VAR_RNA_LENGTH = Var(id="rna_length", name="Length (nuc)", dtype=pl.Int64)
    VAR_RUN_ID = Var(id="run_id", name="Run Index", dtype=pl.Int64)
    VAR_OUTPUT_STRUCS = Var(id="output_strucs", name="Output Structures", dtype=pl.Int64)
    VAR_MAXRSS_BYTES = Var(
        id="maxrss_bytes",
        name="Maximum RSS (B)",
        dtype=pl.Int64,
        formatter=ticker.FuncFormatter(lambda x, _: human_size(x, False)),
    )
    VAR_USER_SEC = Var(id="user_sec", name="User time (s)", dtype=pl.Float64)
    VAR_SYS_SEC = Var(id="sys_sec", name="Sys time (s)", dtype=pl.Float64)
    VAR_REAL_SEC = Var(id="real_sec", name="Wall time (s)", dtype=pl.Float64)
    VAR_FAILED = Var(id="failed", name="Failed", dtype=pl.Boolean)
    # Derived vars:
    VAR_STRUCS_PER_SEC = Var(id="strucs_per_sec", name="Structures per second", dtype=pl.Float64)
    VAR_BASES_PER_BYTE = Var(id="bases_per_byte", name="Bases per byte", dtype=pl.Float64)
    VAR_PROGRAM = Var(id="program", name="Program", dtype=pl.String)

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
            y_vars = [self.VAR_STRUCS_PER_SEC, self.VAR_BASES_PER_BYTE, self.VAR_MAXRSS_BYTES]
            for y_var in y_vars:
                f = plot_mean_quantity(df, self.VAR_PROGRAM, self.VAR_RNA_LENGTH, y_var)
                save_figure(f, self._path(f"{name}_{group_name}_{y_var.id}"))

    def _analyze_complexity(self) -> None:
        for group, group_df in self.df.group_by(self.PACKAGE_VARS):
            # Split into two dfs - ones with "delta" non-empty and ones with "strucs" non-empty.
            df = group_df.filter(pl.col("delta").str.len_chars() > 0)
            group_name = "_".join(str(x) for x in group)
            print(group_name)
            fitter = ComplexityFitter(
                df=df, xs=(self.VAR_DELTA, self.VAR_RNA_LENGTH), y=self.VAR_REAL_SEC
            )
            name, result = fitter.fit()
            print(f"Best model: {name}")
            print(result.fit_report())
            f = fitter.plot(name)
            f.show()
            plt.show(block=True)
            break

    def run(self) -> None:
        # self._plot_quantity("quantity")
        self._analyze_complexity()
