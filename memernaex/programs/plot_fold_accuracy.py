# Copyright 2022 Eliot Courtney.
from pathlib import Path

import cloup

from memernaex.experiments.fold.accuracy_plotter import FoldAccuracyPlotter


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
def plot_fold_accuracy(input_dir: Path, output_dir: Path) -> None:
    plotter = FoldAccuracyPlotter(input_dir, output_dir)
    plotter.run()
