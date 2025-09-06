# Copyright 2022 Eliot Courtney.
import sys
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.analysis.complexity import ComplexityFitter
from memernaex.analysis.data import Var, read_var_data
from memernaex.plot.plots import plot_mean_quantity
from memernaex.plot.util import save_figure, set_style

# Package variables
VAR_ALGORITHM = Var(id="algorithm", name="Algorithm", dtype=pl.String)
VAR_BACKEND = Var(id="backend", name="Backend", dtype=pl.String)
VAR_CTD = Var(id="ctd", name="CTD", dtype=pl.String)
VAR_PACKAGE_NAME = Var(id="package_name", name="Package Name", dtype=pl.String)
VAR_PACKAGE = Var(id="package", name="Program", dtype=pl.String, derived=True)

# Group variables
VAR_COUNT_ONLY = Var(id="count_only", name="Count Only", dtype=pl.String)
VAR_DATASET = Var(id="dataset", name="Dataset", dtype=pl.String)
VAR_DELTA = Var(id="delta", name="Delta", dtype=pl.String)
VAR_ENERGY_MODEL = Var(id="energy_model", name="Energy Model", dtype=pl.String)
VAR_LONELY_PAIRS = Var(id="lonely_pairs", name="Lonely Pairs", dtype=pl.String)
VAR_SORTED_STRUCS = Var(id="sorted_strucs", name="Sorted Structures", dtype=pl.String)
VAR_STRUCS = Var(id="strucs", name="Structures", dtype=pl.String)
VAR_TIME_SECS = Var(id="time_secs", name="Time (s)", dtype=pl.String)

# Independent variables
VAR_RNA_NAME = Var(id="rna_name", name="RNA Name", dtype=pl.String)
VAR_RNA_LENGTH = Var(id="rna_length", name="Length (nuc)", dtype=pl.Int64)
VAR_RUN_IDX = Var(id="run_idx", name="Run Index", dtype=pl.Int64)

# Dependent variables
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
VAR_NODES = Var(id="nodes", name="Nodes", dtype=pl.Int64)
VAR_EXPANSIONS = Var(id="expansions", name="Expansions", dtype=pl.Int64)
VAR_STRUCS_PER_SEC = Var(
    id="strucs_per_sec", name="Structures per second", dtype=pl.Float64, derived=True
)
VAR_BASES_PER_BYTE = Var(id="bases_per_byte", name="Bases per byte", dtype=pl.Float64, derived=True)

PACKAGE_VARS: list[str] = [VAR_PACKAGE_NAME.id, VAR_CTD.id, VAR_ALGORITHM.id, VAR_BACKEND.id]

GROUP_VARS: list[str] = [
    VAR_COUNT_ONLY.id,
    VAR_DATASET.id,
    VAR_DELTA.id,
    VAR_ENERGY_MODEL.id,
    VAR_LONELY_PAIRS.id,
    VAR_SORTED_STRUCS.id,
    VAR_STRUCS.id,
    VAR_TIME_SECS.id,
]

INDEPENDENT_VARS: list[str] = [VAR_RNA_NAME.id, VAR_RNA_LENGTH.id, VAR_RUN_IDX.id]

DEPENDENT_VARS: list[str] = [
    VAR_OUTPUT_STRUCS.id,
    VAR_MAXRSS_BYTES.id,
    VAR_USER_SEC.id,
    VAR_SYS_SEC.id,
    VAR_REAL_SEC.id,
    VAR_STRUCS_PER_SEC.id,
    VAR_BASES_PER_BYTE.id,
]

STATS_DEPENDENT_VARS: list[str] = [*DEPENDENT_VARS, VAR_NODES.id, VAR_EXPANSIONS.id]


class SuboptPerfPlotter:
    df: pl.DataFrame
    is_stats: bool
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path, is_stats: bool) -> None:
        self.output_dir = output_dir
        self.is_stats = is_stats
        self.df = read_var_data(sys.modules[type(self).__module__], input_path)

        # Remove any rows with failed true.
        self.df = self.df.filter(~pl.col(VAR_FAILED.id))

        # Remove any rows with real time less than 0.1 seconds for numeric stability.
        self.df = self.df.filter(pl.col(VAR_REAL_SEC.id) > 0.1)

        # Add column for program identifier.
        self.df = self.df.with_columns(
            pl.format("{}-{}-{}-{}", *PACKAGE_VARS).alias(VAR_PACKAGE.id),
            (pl.col(VAR_OUTPUT_STRUCS.id) / pl.col(VAR_REAL_SEC.id)).alias(VAR_STRUCS_PER_SEC.id),
            (
                pl.col(VAR_OUTPUT_STRUCS.id)
                * pl.col(VAR_RNA_LENGTH.id)
                / pl.col(VAR_MAXRSS_BYTES.id)
            ).alias(VAR_BASES_PER_BYTE.id),
        )

        set_style()

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.png"

    def _plot_quantity(self, name: str) -> None:
        for group, df in self.df.group_by(GROUP_VARS):
            group_name = "_".join(str(x) for x in group)
            y_vars = [VAR_STRUCS_PER_SEC, VAR_BASES_PER_BYTE, VAR_MAXRSS_BYTES]
            for y_var in y_vars:
                f = plot_mean_quantity(df, VAR_PACKAGE, VAR_RNA_LENGTH, y_var)
                save_figure(f, self._path(f"{name}_{group_name}_{y_var.id}"))

    def _analyze_complexity(self) -> None:
        # Average all dependent variables.
        df = self.df.group_by(PACKAGE_VARS + GROUP_VARS + [VAR_RNA_LENGTH.id]).agg(
            pl.col(DEPENDENT_VARS).mean()
        )

        # Filter out rows with RNA length less than 100 to avoid noise.
        # Just a heuristic.
        df = df.filter(pl.col(VAR_RNA_LENGTH.id) >= 100)

        # Filter out rows with no structures generated.
        df = df.filter(pl.col(VAR_OUTPUT_STRUCS.id) > 0)

        for group, group_df in df.group_by(PACKAGE_VARS):
            for split_var in [VAR_DELTA, VAR_STRUCS]:
                split_df = group_df.filter(pl.col(split_var.id).str.len_chars() > 0)
                for dependent in [VAR_REAL_SEC, VAR_MAXRSS_BYTES]:
                    group_name = (
                        "_".join(str(x) for x in group) + f"_{split_var.id}" + f"_{dependent.id}"
                    )
                    print(group_name)
                    if len(split_df) == 0:
                        print("No data for this group.")
                        continue
                    fitter = ComplexityFitter(
                        df=split_df, xs=(VAR_RNA_LENGTH, split_var), y=dependent
                    )
                    name, result = fitter.fit()
                    print(f"Best model: {name}")
                    print(result.fit_report())
                    print()
                    f = fitter.plot(name)
                    f.show()
                    plt.show(block=True)

    def _analyze_stats(self) -> None:
        # Average all dependent variables.
        df = self.df.group_by(PACKAGE_VARS + GROUP_VARS + [VAR_RNA_LENGTH.id]).agg(
            pl.col(STATS_DEPENDENT_VARS).mean()
        )

        # Filter out rows with RNA length less than 100 to avoid noise.
        # Just a heuristic.
        df = df.filter(pl.col(VAR_RNA_LENGTH.id) >= 100)

        # Filter out rows with no structures generated.
        df = df.filter(pl.col(VAR_OUTPUT_STRUCS.id) > 0)

        for group, group_df in df.group_by(PACKAGE_VARS):
            for split_var in [VAR_DELTA, VAR_STRUCS]:
                split_df = group_df.filter(pl.col(split_var.id).str.len_chars() > 0)
                for dependent in [VAR_NODES, VAR_EXPANSIONS, VAR_OUTPUT_STRUCS]:
                    group_name = (
                        "_".join(str(x) for x in group) + f"_{split_var.id}" + f"_{dependent.id}"
                    )
                    print(group_name)
                    if len(split_df) == 0:
                        print("No data for this group.")
                        continue
                    fitter = ComplexityFitter(
                        df=split_df, xs=(VAR_RNA_LENGTH, split_var), y=dependent
                    )
                    name, result = fitter.fit()
                    print(f"Best model: {name}")
                    print(result.fit_report())
                    print()
                    f = fitter.plot(name)
                    f.show()
                    plt.show(block=True)

    def run(self) -> None:
        # self._plot_quantity("quantity")
        # self._analyze_complexity()
        if self.is_stats:
            self._analyze_stats()
        else:
            self._analyze_complexity()
