# Copyright 2021 Eliot Courtney.
import decimal
from pathlib import Path

import click
import cloup


def _read_decimals(path: Path) -> list[decimal.Decimal]:
    try:
        parts = path.read_text().split()
    except OSError as exc:
        raise click.ClickException(f"Failed to read {path}.") from exc

    try:
        return [decimal.Decimal(i) for i in parts]
    except decimal.InvalidOperation as exc:
        raise click.ClickException(f"Invalid numeric value in {path}.") from exc


@cloup.command()
@cloup.argument(
    "p0",
    type=cloup.Path(dir_okay=False, exists=True, readable=True, resolve_path=True, path_type=Path),
)
@cloup.argument(
    "p1",
    type=cloup.Path(dir_okay=False, exists=True, readable=True, resolve_path=True, path_type=Path),
)
def compare_partition(p0: Path, p1: Path) -> None:
    vals0 = _read_decimals(p0)
    vals1 = _read_decimals(p1)

    if not vals0 or not vals1:
        raise click.ClickException("Input files must contain at least one value.")

    if len(vals0) != len(vals1):
        raise click.ClickException(f"Input lengths do not match: {len(vals0)} vs {len(vals1)}.")

    diffs = [a - b for a, b in zip(vals0, vals1, strict=True)]
    n = decimal.Decimal(len(diffs))
    rms = (sum(d * d for d in diffs) / n).sqrt()
    largest_diff = max(abs(d) for d in diffs)
    click.echo(f"rms: {rms:.20f}")
    click.echo(f"largest diff: {largest_diff:.20f}")
