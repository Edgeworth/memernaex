# Copyright 2023 Eliot Courtney.
from pathlib import Path

import polars as pl
from matplotlib import ticker
from rnapy.util.format import human_size
from scipy.stats import ttest_rel

from memernaex.analysis.data import Var
from memernaex.plot.plots import plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class FoldAccuracyPlotter:
    VAR_NAME = Var(id="name", name="Name", dtype=pl.String)
    VAR_FAMILY = Var(id="family", name="Family", dtype=pl.String)
    VAR_SENSITIVITY = Var(id="sensitivity", name="Sensitivity", dtype=pl.Float64)
    VAR_PPV = Var(id="ppv", name="Positive predictive value", dtype=pl.Float64)
    VAR_F1 = Var(id="f1", name="F1 score", dtype=pl.Float64)
    VAR_LENGTH = Var(id="length", name="Length (nuc)", dtype=pl.Int64)
    VAR_REAL_SEC = Var(id="real_sec", name="Wall time (s)", dtype=pl.Float64)
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
        self.df = pl.read_ndjson(input_path)
        self.output_dir = output_dir
        set_style()

    def _path(self, plot_name: str) -> Path:
        return self.output_dir / f"{plot_name}.png"

    def _plot_quantity(self, df: pl.DataFrame, dataset_name: str) -> None:
        y_vars = [self.VAR_REAL_SEC, self.VAR_MAXRSS_BYTES]
        for y_var in y_vars:
            f = plot_mean_quantity(df, self.VAR_PROGRAM, self.VAR_LENGTH, y_var)
            save_figure(f, self._path(f"{dataset_name}_{y_var.id}"))

    def _get_parent_rnas(self, df: pl.DataFrame) -> list[str]:
        domained = df.filter(df["name"].str.contains("(?i)domain"))["name"].to_list()
        parents = set()
        for name in domained:
            parent = "_".join(name.split("_")[:-1])
            if parent not in df["name"].to_list():
                raise ValueError(f"Parent {parent} not found in dataframe.")
            parents.add(parent)
        return list(parents)

    def _filter_df(self, df: pl.DataFrame) -> pl.DataFrame:
        parents = self._get_parent_rnas(df)
        return df.filter(~df["name"].is_in(parents))

    def run(self) -> None:
        for group, df in self.df.group_by("dataset"):
            dataset_name = str(group[0])
            print(f"Dataset: {dataset_name}")
            filtered_df = self._filter_df(df)
            for program_name, program_df in filtered_df.group_by("program"):
                print(f"dataset {dataset_name} program {program_name}")
                for var in ["ppv", "sensitivity", "f1"]:
                    means = program_df.group_by("family").mean()[var]
                    print(means)
                    print(means.mean())
                print()

            df1 = self._filter_df(df.filter(pl.col("program") == "memerna-t04p2-TODO"))
            df2 = self._filter_df(df.filter(pl.col("program") == "memerna-t22p2-TODO"))
            df1_by_family = dict(df1.group_by("family"))
            df2_by_family = dict(df2.group_by("family"))
            families = set(df1_by_family.keys()).intersection(df2_by_family.keys())
            for var in ["ppv", "sensitivity", "f1"]:
                print(f"paired t-tests for {var}:")
                for family in families:
                    d1 = df1_by_family[family][var].to_numpy()
                    d2 = df2_by_family[family][var].to_numpy()
                    t_statistic, p_value = ttest_rel(d1, d2)
                    print(f"Family {family}: t-statistic={t_statistic}, p-value={p_value}")
                print()
