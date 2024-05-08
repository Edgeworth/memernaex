#!/usr/bin/env python3
import logging

import click_log
import cloup
from dotenv import load_dotenv
from rnapy.programs.design import design

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


cli.section("Design", design)

if __name__ == "__main__":
    cli()
