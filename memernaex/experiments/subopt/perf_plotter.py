# Copyright 2022 Eliot Courtney.
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


class SuboptPerfPlotter:
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.df = read_var_data(self.__class__, input_path)

        # Remove any rows with failed true.
        self.df = self.df.filter(~pl.col(VAR_FAILED.id))

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
        # Average all dependent variables by group.
        df = self.df.group_by(PACKAGE_VARS + GROUP_VARS).agg(pl.col(DEPENDENT_VARS).mean())
        print(df)

        for group, group_df in self.df.group_by(PACKAGE_VARS + [VAR_RNA_LENGTH.id]):
            # Split into two dfs - ones with "delta" non-empty and ones with "strucs" non-empty.
            df = group_df.filter(pl.col(VAR_DELTA.id).str.len_chars() > 0)
            group_name = "_".join(str(x) for x in group)
            if group_name != "memerna_d2_heuristic_True_t04_iterative_base_3000":
                continue
            print(group_name)
            fitter = ComplexityFitter(df=df, xs=(VAR_DELTA, VAR_RNA_LENGTH), y=VAR_REAL_SEC)
            name, result = fitter.fit()
            print(f"Best model: {name}")
            print(result.fit_report())
            print()
            f = fitter.plot(name)
            f.show()
            plt.show(block=True)
            break

    def run(self) -> None:
        # self._plot_quantity("quantity")
        self._analyze_complexity()
