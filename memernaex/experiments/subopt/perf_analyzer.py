# Copyright 2022 Eliot Courtney.
from pathlib import Path

import polars as pl

from memernaex.analysis.complexity import ComplexityFitter
from memernaex.experiments.subopt.perf_common import (
    SUBOPT_PERF_COLS,
    SUBOPT_PERF_GROUP_VARS,
    load_subopt_perf_df,
)


class SuboptPerfAnalyzer:
    df: pl.DataFrame

    def __init__(self, input_path: Path) -> None:
        self.df = load_subopt_perf_df(input_path)

    def run(self) -> None:
        cols = SUBOPT_PERF_COLS
        for group, df in self.df.group_by(SUBOPT_PERF_GROUP_VARS):
            group_name = "_".join(str(x) for x in group)
            print(group_name)
            fitter = ComplexityFitter(df)
            fitter.fit(x="rna_length", y="real_sec")
            break
