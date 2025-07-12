from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import seaborn as sns
from bidict import bidict
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rnapy.util.util import stable_hash


@dataclass(frozen=True, eq=True, order=True, kw_only=True)
class Var:
    id: str
    name: str
    formatter: ticker.FuncFormatter | None = None


def set_style() -> None:
    sns.set_theme(context="notebook", palette="deep")


def get_subplot_grid(
    n: int, *, sharex: bool = False, sharey: bool = False, inches: float = 3.0
) -> tuple[Figure, list[Axes]]:
    splittings = [(0, 0), (1, 1), (1, 2), (2, 2), (2, 2), (2, 3), (2, 3), (3, 3), (3, 3), (3, 3)]

    factors = splittings[n]
    axes: Any
    if factors == (1, 1):
        f, axes = plt.subplots(n, sharey=sharey, sharex=sharex)
    else:
        f, axes = plt.subplots(factors[0], factors[1], sharey=sharey, sharex=sharex)
        axes = axes.flatten()
    if n == 1:
        axes = [axes]
    f.tight_layout()

    # Hide subplots that are not used
    for i in range(n, factors[0] * factors[1]):
        axes[i].clear()
        axes[i].set_axis_off()
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)

    f.set_size_inches(factors[1] * inches, factors[0] * inches)
    return f, axes


def get_color(name: str, palette: Sequence[Any] | None = None) -> Any:
    color_map: bidict = getattr(get_color, "color_map", bidict())
    get_color.color_map = color_map  # type: ignore[attr-defined]
    print(name, color_map.keys())
    if name in color_map:
        return color_map[name]

    if palette is None:
        palette = sns.color_palette("husl", n_colors=12)
        print(len(set(palette)))

    start_idx = stable_hash(name) % len(palette)
    idx = start_idx
    while palette[idx] in color_map.inverse:
        idx = (idx + 1) % len(palette)
        print(idx, start_idx)
        if idx == start_idx:
            raise ValueError(f"No free color available in the palette: {len(palette)}")

    color_map[name] = palette[idx]

    return color_map[name]


def get_marker(name: int | str) -> dict[str, Any]:
    markers = " ov^sp*+xD|"
    idx = stable_hash(name) % len(markers)
    return {"marker": markers[idx], "markersize": 5, "markevery": 5}


def save_figure(f: Figure, path: Path) -> None:
    f.tight_layout()
    f.savefig(path, dpi=300)
    plt.close(f)


def set_up_axis(ax: Axes, varz: tuple[Var, Var], legend: bool = True) -> None:
    if legend:
        ax.legend(loc="best", framealpha=0.5, fontsize="medium")
    ax.set_xlabel(varz[0].name)
    ax.set_ylabel(varz[1].name)
    if varz[0].formatter:
        ax.xaxis.set_major_formatter(varz[0].formatter)
    if varz[1].formatter:
        ax.yaxis.set_major_formatter(varz[1].formatter)


def set_up_figure(f: Figure, varz: tuple[Var, Var], legend: bool = True) -> None:
    f.suptitle(f"{varz[0]} vs {varz[1]}", y=1.00)
    for ax in f.get_axes():
        set_up_axis(ax, varz, legend)
