import logging
import re
from collections.abc import Callable
from typing import Any, cast

import lmfit
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

from memernaex.analysis.data import Var
from memernaex.plot.util import set_up_figure_2d, set_up_figure_3d

log = logging.getLogger(__name__)


def _model_constant(x: tuple[npt.NDArray[np.float64], ...], b0: float) -> Any:
    return b0 * np.ones_like(x[0].astype(float))


def _generate_model_func(expr: str, var_names: tuple[str, ...]) -> Callable[..., Any]:
    if expr == "1":
        return _model_constant

    terms = expr.split("+")
    idents = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)

    coeffs = [f"a{i}" for i in range(len(terms))]
    params = sorted({v for v in idents if v not in var_names and v != "log"})
    params.extend(coeffs)
    params.append("b0")

    var_map = {name: f"x[{i}]" for i, name in enumerate(var_names)}
    func_body_parts = []
    for i, term in enumerate(terms):
        term_code = term.strip()
        for var, code in var_map.items():
            term_code = term_code.replace(var, code)

        term_code = re.sub(r"log\((.*?)\)", r"np.log(\1)", term_code)

        term_code = term_code.replace("^", "**")
        func_body_parts.append(f"{coeffs[i]} * ({term_code})")

    func_body = " + ".join(func_body_parts) + " + b0"

    params_with_types = [f"{p}: float" for p in params]
    signature = (
        "def model_func(x: tuple[npt.NDArray[np.float64], ...], "
        f"{', '.join(params_with_types)}) -> Any: return {func_body}"
    )
    exec_scope = {"np": np, "npt": npt, "Any": Any}
    exec(signature, exec_scope)  # noqa: S102
    return cast(Callable[..., Any], exec_scope["model_func"])


_MODELS_EXPRESSIONS_1D = ["1", "log(n)", "n*log(n)", "n", "n^2", "n^3", "n^2*log(n)", "n^c"]
_MODELS_EXPRESSIONS_2D = [
    "n+m",
    "n*m",
    "n*m+n+m",
    "n^3+n^2+n+m",
    "n*m+n^3+n^2+n+m*log(m)+m",
    "n*m+n^3+n^2+n+m",
    "n^2*m",
    "n*m^2",
    "k^n*m",
]


class ComplexityFitter:
    df: pl.DataFrame
    xs: tuple[Var, ...]
    y: Var
    results: dict[str, lmfit.model.ModelResult]

    def __init__(self, *, df: pl.DataFrame, xs: tuple[Var, ...] | Var, y: Var) -> None:
        self.df = df
        if isinstance(xs, Var):
            self.xs = (xs,)
        elif isinstance(xs, tuple) and len(xs) <= 2:
            self.xs = xs
        else:
            raise ValueError("Only 1D and 2D models are supported.")
        self.y = y
        self.results = {}

    @staticmethod
    def _best_model(
        results: dict[str, lmfit.model.ModelResult],
    ) -> tuple[str, lmfit.model.ModelResult]:
        best_name = min(results, key=lambda name: results[name].bic)
        return best_name, results[best_name]

    def _fitnd(
        self, *, model_expressions: list[str], xs: tuple[Var, ...], y: Var
    ) -> dict[str, lmfit.model.ModelResult]:
        results: dict[str, lmfit.model.ModelResult] = {}
        x_data = tuple(self.df[var.id].to_numpy() for var in xs)
        y_data = self.df[y.id].to_numpy()

        gen_var_names: tuple[str, ...]
        if len(xs) == 1:
            gen_var_names = ("n",)
        elif len(xs) == 2:
            gen_var_names = ("n", "m")
        else:
            raise ValueError("Only 1D and 2D models are supported.")

        for name in model_expressions:
            func = _generate_model_func(name, gen_var_names)
            model = lmfit.Model(func, independent_vars=["x"])
            params = model.make_params()

            for param_name in params:
                if param_name.startswith("a"):
                    params[param_name].set(value=1.0)
                elif param_name.startswith("b"):
                    params[param_name].set(value=0.0)
                elif param_name.startswith("k"):
                    params[param_name].set(value=1.0)
                else:
                    params[param_name].set(value=1.0)

            try:
                results[name] = model.fit(y_data, params, x=x_data)
            except Exception:
                log.exception(f"Error fitting model {name}")
                continue
        return results

    def _plot2d(self, result: lmfit.model.ModelResult) -> Figure:
        x0_data = self.df[self.xs[0].id].cast(pl.Float64).to_numpy()
        x1_data = self.df[self.xs[1].id].cast(pl.Float64).to_numpy()
        y_data = self.df[self.y.id].cast(pl.Float64).to_numpy()

        f = plt.figure()
        ax: Axes3D = cast(Axes3D, f.add_subplot(111, projection="3d"))

        x0_min, x0_max = float(x0_data.min()), float(x0_data.max())
        x1_min, x1_max = float(x1_data.min()), float(x1_data.max())
        x0_grid, x1_grid = np.meshgrid(
            np.linspace(x0_min, x0_max, 50), np.linspace(x1_min, x1_max, 50)
        )

        fit_y = cast(
            npt.NDArray[np.float64],
            result.model.eval(result.params, x=(x0_grid.ravel(), x1_grid.ravel())),
        )
        fit_y = fit_y.reshape(x0_grid.shape)

        ax.scatter(x0_data, x1_data, y_data, color="red", label="Data")
        ax.plot_surface(x0_grid, x1_grid, fit_y, color="blue", alpha=0.5, label="Fit")

        legend_elements = [
            Line2D(
                [0], [0], marker="o", color="w", label="Data", markerfacecolor="black", markersize=5
            ),
            Patch(facecolor="blue", alpha=0.7, label="Fit"),
        ]
        ax.legend(handles=legend_elements)

        set_up_figure_3d(f, varz=(self.xs[0], self.xs[1], self.y))
        return f

    def _plot1d(self, result: lmfit.model.ModelResult) -> Figure:
        f, ax = plt.subplots(1)
        result.plot()
        set_up_figure_2d(f, varz=(self.xs[0], self.y))
        return f

    def _fit1d(self) -> None:
        self.results = self._fitnd(model_expressions=_MODELS_EXPRESSIONS_1D, xs=self.xs, y=self.y)

    def _fit2d(self) -> None:
        self.results = self._fitnd(model_expressions=_MODELS_EXPRESSIONS_2D, xs=self.xs, y=self.y)

    def fit(self) -> tuple[str, lmfit.model.ModelResult]:
        if len(self.xs) == 1:
            self._fit1d()
        elif len(self.xs) == 2:
            self._fit2d()
        else:
            raise ValueError("Only 1D and 2D models are supported.")
        return self._best_model(self.results)

    def plot(self, model_name: str) -> Figure:
        model = self.results[model_name]
        if len(self.xs) == 1:
            return self._plot1d(model)
        if len(self.xs) == 2:
            return self._plot2d(model)
        raise ValueError("Only 1D and 2D models are supported.")
