# Copyright 2025 Eliot Courtney.
from pathlib import Path

import cloup

from memernaex.experiments.subopt.perf_plotter import SuboptPerfPlotter


@cloup.command()
@cloup.option(
    "--input-dir",
    type=cloup.Path(dir_okay=True, file_okay=False, exists=True, path_type=Path),
    required=True,
)
@cloup.option(
    "--output-dir",
    type=cloup.Path(dir_okay=True, file_okay=False, exists=True, path_type=Path),
    required=True,
)
def plot_subopt_perf(input_dir: Path, output_dir: Path) -> None:
    plotter = SuboptPerfPlotter(input_dir, output_dir)
    plotter.run()
