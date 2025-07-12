import dataclasses

import polars as pl
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from memernaex.plot.util import Var, get_color, get_marker, get_subplot_grid, set_up_figure


def plot_mean_quantity(
    df: pl.DataFrame, group_var: Var, x: Var, ys: tuple[Var, ...] | Var
) -> Figure:
    if not isinstance(ys, tuple):
        ys = (ys,)
    f, ax = plt.subplots(1)

    for group, group_df in df.group_by(group_var.id):
        group_name = str(group[0])
        for y in ys:
            agg_df = (
                group_df.group_by(x.id)
                .agg(
                    pl.mean(y.id).alias(y.id), pl.min(y.id).alias("low"), pl.max(y.id).alias("high")
                )
                .sort(x.id)
            )

            sns.lineplot(
                data=agg_df,
                x=x.id,
                y=y.id,
                label=group,
                ax=ax,
                color=get_color(group_name),
                **get_marker(group_name),
            )
            ax.fill_between(
                agg_df[x.id], agg_df["low"], agg_df["high"], alpha=0.2, color=get_color(group_name)
            )

    set_up_figure(f, varz=(x, ys[0]))
    return f


def plot_mean_log_quantity(
    df: pl.DataFrame, group_var: Var, x: Var, y: Var, logx: bool = True, logy: bool = True
) -> Figure:
    ep = 1e-2
    group_count = df.select(pl.col(group_var.id).n_unique()).item()
    f, axes = get_subplot_grid(group_count, sharex=True, sharey=True)

    if logx:
        x = dataclasses.replace(x, name=f"log({x.name})")
    if logy:
        y = dataclasses.replace(y, name=f"log({y.name})")

    for i, (group, group_df) in enumerate(df.group_by(group_var.id)):
        group_name = str(group[0])
        df_model = group_df.group_by(x.id).agg(pl.mean(y.id)).filter(pl.col(y.id) > ep)
        if logx:
            df_model = df_model.with_columns(pl.col(x.id).log10())
        if logy:
            df_model = df_model.with_columns(pl.col(y.id).log10())

        mod = smf.ols(f"{y} ~ {x}", data=df_model.to_pandas())
        res = mod.fit()

        b, a = res.params.tolist()
        sign = "-" if b < 0 else "+"
        label = f"{group_name}\n${a:.5f}x {sign} {abs(b):.2f}$\n$R^2 = {res.rsquared:.3f}$"

        sns.regplot(
            data=df_model.to_pandas(),
            x=x.id,
            y=y.id,
            label=label,
            fit_reg=False,
            ax=axes[i],
            color=get_color(group_name),
        )
        sm.graphics.abline_plot(
            model_results=res,
            ax=axes[i],
            c=(0, 0, 0, 0.8),
            color=get_color(group_name),
            **get_marker(group_name),
        )

    set_up_figure(f, varz=(x, y))
    return f
