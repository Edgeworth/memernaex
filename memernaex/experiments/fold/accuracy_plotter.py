# Copyright 2023 Eliot Courtney.
from pathlib import Path
from typing import ClassVar

import polars as pl
from scipy.stats import ttest_rel

from memernaex.plot.plots import Column, plot_mean_quantity
from memernaex.plot.util import save_figure, set_style


class FoldAccuracyPlotter:
    COLS: ClassVar[dict[str, Column]] = {
        "name": Column(idx="name", name="Name"),
        "family": Column(idx="family", name="Family"),
        "sensitivity": Column(idx="sensitivity", name="Sensitivity"),
        "ppv": Column(idx="ppv", name="Positive predictive value"),
        "f1": Column(idx="f1", name="F1 score"),
    }
    df: pl.DataFrame
    output_dir: Path

    def __init__(self, input_path: Path, output_dir: Path) -> None:
        self.df = pl.read_ndjson(input_path)
        self.output_dir = output_dir
        set_style()

    def _path(self, plot_name: str) -> Path:
        return self.output_dir / f"{plot_name}.png"

    def _plot_quantity(self, df: pl.DataFrame, dataset_name: str) -> None:
        for y in ["real_sec", "maxrss_bytes"]:
            f = plot_mean_quantity(df, "program", self.COLS["length"], self.COLS[y])
            save_figure(f, self._path(f"{dataset_name}_{y}"))

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
                for col in ["ppv", "sensitivity", "f1"]:
                    means = program_df.group_by("family").mean()[col]
                    print(means)
                    print(means.mean())
                print()

            df1 = self._filter_df(df.filter(pl.col("program") == "memerna-t04p2-TODO"))
            df2 = self._filter_df(df.filter(pl.col("program") == "memerna-t22p2-TODO"))
            df1_by_family = dict(df1.group_by("family"))
            df2_by_family = dict(df2.group_by("family"))
            families = set(df1_by_family.keys()).intersection(df2_by_family.keys())
            for col in ["ppv", "sensitivity", "f1"]:
                print(f"paired t-tests for {col}:")
                for family in families:
                    d1 = df1_by_family[family][col].to_numpy()
                    d2 = df2_by_family[family][col].to_numpy()
                    t_statistic, p_value = ttest_rel(d1, d2)
                    print(f"Family {family}: t-statistic={t_statistic}, p-value={p_value}")
                print()
