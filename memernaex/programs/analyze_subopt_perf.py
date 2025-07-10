# Copyright 2025 Eliot Courtney.
from pathlib import Path

import cloup

from memernaex.experiments.subopt.perf_analyzer import SuboptPerfAnalyzer


@cloup.command()
@cloup.option(
    "--input-path",
    type=cloup.Path(dir_okay=False, file_okay=True, exists=True, path_type=Path),
    required=True,
)
def analyze_subopt_perf(input_path: Path) -> None:
    analyzer = SuboptPerfAnalyzer(input_path)
    analyzer.run()
