# Copyright 2025 Eliot Courtney.
from pathlib import Path

import polars as pl
from matplotlib import ticker
from rnapy.util.format import human_size

from memernaex.plot.plots import Column

SUBOPT_PERF_COLS: dict[str, Column] = {
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

SUBOPT_PERF_GROUP_VARS: list[str] = [
    "count_only",
    "ctd",
    "dataset",
    "delta",
    "lonely_pairs",
    "sorted_strucs",
    "strucs",
    "time_secs",
]


def load_subopt_perf_df(input_path: Path) -> pl.DataFrame:
    df = pl.read_ndjson(input_path)

    # Add column for program identifier.
    df = df.with_columns(
        pl.format("{}-{}-{}-{}", "package_name", "ctd", "algorithm", "backend").alias("program"),
        (pl.col("output_strucs") / pl.col("real_sec")).alias("strucs_per_sec"),
        (pl.col("output_strucs") * pl.col("rna_length") / pl.col("maxrss_bytes")).alias(
            "bases_per_byte"
        ),
    )

    # Remove any rows with failed true.
    df = df.filter(~pl.col("failed"))
    return df
