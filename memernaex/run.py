#!/usr/bin/env python3
import logging

import click_log
import cloup
from dotenv import load_dotenv

from memernaex.programs.compare_partition import compare_partition
from memernaex.programs.crop_image import crop_image
from memernaex.programs.parse_rnastructure_datatables import parse_rnastructure_datatables
from memernaex.programs.plot_ensemble import plot_ensemble
from memernaex.programs.plot_fold_accuracy import plot_fold_accuracy
from memernaex.programs.plot_fold_perf import plot_fold_perf
from memernaex.programs.plot_subopt_perf import plot_subopt_perf

CONTEXT_SETTINGS = cloup.Context.settings(
    show_constraints=True,
    show_subcommand_aliases=True,
    show_default=True,
    formatter_settings=cloup.HelpFormatter.settings(theme=cloup.HelpTheme.dark()),
)

load_dotenv()
logger = logging.getLogger()
click_log.basic_config(logger)


@cloup.group(context_settings=CONTEXT_SETTINGS)
@click_log.simple_verbosity_option(logger)
def cli() -> None:
    pass


cli.section("Plots", plot_ensemble, plot_fold_accuracy, plot_fold_perf, plot_subopt_perf)
cli.section("Utilities", compare_partition, crop_image, parse_rnastructure_datatables)

if __name__ == "__main__":
    cli()
