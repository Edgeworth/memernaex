# Copyright 2025 Eliot Courtney.
from pathlib import Path

import cloup

from memernaex.experiments.subopt.perf_plotter import SuboptPerfPlotter


@cloup.command()
@cloup.option(
    "--input-path",
    type=cloup.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
)
@cloup.option(
    "--output-dir",
    type=cloup.Path(dir_okay=True, file_okay=False, exists=True, path_type=Path),
    required=True,
)
@cloup.option("--is-stats", is_flag=True, help="Whether input is stats about subopt.")
def plot_subopt_perf(input_path: Path, output_dir: Path, is_stats: bool) -> None:
    plotter = SuboptPerfPlotter(input_path, output_dir, is_stats)
    plotter.run()
