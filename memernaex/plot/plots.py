from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from bidict import bidict
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from rnapy.util.util import stable_hash

from memernaex.plot.util import get_marker, get_subplot_grid, set_up_figure


@dataclass
class Column:
    idx: str
    name: str
    formatter: ticker.FuncFormatter | None = None

    def __str__(self) -> str:
        return self.idx


def _color(name: str, palette: Sequence[Any] | None = None) -> Any:
    color_map: bidict = getattr(_color, "color_map", bidict())
    _color.color_map = color_map  # type: ignore[attr-defined]
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


def plot_mean_quantity(
    df: pl.DataFrame, split_col: str, xcol: Column, ycols: list[Column] | Column
) -> Figure:
    if not isinstance(ycols, list):
        ycols = [ycols]
    f, ax = plt.subplots(1)

    for idx, split_name in enumerate(sorted(df[split_col].unique().to_list())):
        split_df = df.filter(pl.col(split_col) == split_name)

        for ycol in ycols:
            x, y = xcol.idx, ycol.idx
            agg_df = (
                split_df.group_by(x)
                .agg(pl.mean(y).alias(y), pl.min(y).alias("low"), pl.max(y).alias("high"))
                .sort(x)
            )

            sns.lineplot(
                data=agg_df,
                x=x,
                y=y,
                label=split_name,
                ax=ax,
                color=_color(split_name),
                **get_marker(idx),
            )
            ax.fill_between(
                agg_df[x], agg_df["low"], agg_df["high"], alpha=0.2, color=_color(split_name)
            )

    set_up_figure(f, names=(xcol.name, ycols[0].name))
    if ycols[0].formatter:
        ax.yaxis.set_major_formatter(ycols[0].formatter)
    ax.legend(loc="best", framealpha=0.5)
    return f


def plot_mean_log_quantity(
    df: pl.DataFrame,
    split_col: str,
    xcol: Column,
    ycol: Column,
    logx: bool = True,
    logy: bool = True,
) -> Figure:
    ep = 1e-2

    splits = sorted(df[split_col].unique().to_list())
    f, axes = get_subplot_grid(len(splits), sharex=True, sharey=True)
    for i, split_name in enumerate(splits):
        x, y = xcol.idx, ycol.idx

        df_model = (
            df.filter(pl.col(split_col) == split_name)
            .group_by(x)
            .agg(pl.mean(y))
            .filter(pl.col(y) > ep)
        )
        if logx:
            df_model = df_model.with_columns(pl.col(x).log10())
        if logy:
            df_model = df_model.with_columns(pl.col(y).log10())

        mod = smf.ols(f"{y} ~ {x}", data=df_model.to_pandas())
        res = mod.fit()

        b, a = res.params.tolist()
        sign = "-" if b < 0 else "+"
        label = f"{split_name}\n${a:.5f}x {sign} {abs(b):.2f}$\n$R^2 = {res.rsquared:.3f}$"

        sns.regplot(
            data=df_model.to_pandas(),
            x=x,
            y=y,
            label=label,
            fit_reg=False,
            ax=axes[i],
            color=_color(split_name),
        )
        sm.graphics.abline_plot(
            model_results=res,
            ax=axes[i],
            c=(0, 0, 0, 0.8),
            color=_color(split_name),
            **get_marker(i),
        )

    names = [xcol.name, ycol.name]
    if logx:
        names[0] = f"log({names[0]})"
    if logy:
        names[1] = f"log({names[1]})"
    set_up_figure(f, names=(names[0], names[1]))

    return f
