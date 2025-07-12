from typing import Any

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


def _model_constant(x: tuple[npt.NDArray, ...], b: float) -> Any:
    return b * np.ones_like(x[0].astype(float))


def _model_log_n(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * np.log(x[0]) + b


def _model_n_log_n(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] * np.log(x[0]) + b


def _model_linear(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] + b


def _model_n_squared(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] ** 2 + b


def _model_n_cubed(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] ** 3 + b


def _model_n_squared_log_n(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * (x[0] ** 2) * np.log(x[0]) + b


def _model_polynomial(x: tuple[npt.NDArray, ...], a: float, b: float, c: float) -> Any:
    return a * x[0] ** c + b


_MODELS_FUNCS_1D = {
    "1": _model_constant,
    "log(n)": _model_log_n,
    "n*log(n)": _model_n_log_n,
    "n": _model_linear,
    "n^2": _model_n_squared,
    "n^3": _model_n_cubed,
    "n^2*log(n)": _model_n_squared_log_n,
    "n^c": _model_polynomial,
}


def _model_n_plus_m(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * (x[0] + x[1]) + b


def _model_n_times_m(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] * x[1] + b


def _model_n_squared_times_m(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * (x[0] ** 2) * x[1] + b


def _model_n_times_m_squared(x: tuple[npt.NDArray, ...], a: float, b: float) -> Any:
    return a * x[0] * (x[1] ** 2) + b


_MODELS_FUNCS_2D = {
    "n+m": _model_n_plus_m,
    "n*m": _model_n_times_m,
    "n^2*m": _model_n_squared_times_m,
    "n*m^2": _model_n_times_m_squared,
}


class ComplexityFitter:
    df: pl.DataFrame
    xs: tuple[Var, ...]
    y: Var
    results: dict[str, lmfit.model.ModelResult]

    def __init__(self, *, df: pl.DataFrame, xs: tuple[Var, ...] | Var, y: Var) -> None:
        self.df = df
        if isinstance(xs, Var):
            self.xs = (xs,)
        elif isinstance(xs, tuple) and len(xs) == 2:
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
        self, *, model_funcs: dict[str, Any], xs: tuple[Var, ...], y: Var
    ) -> dict[str, lmfit.model.ModelResult]:
        results: dict[str, lmfit.model.ModelResult] = {}
        x_data = tuple(self.df[var.id].to_numpy() for var in xs)
        y_data = self.df[y.id].to_numpy()
        for name, func in model_funcs.items():
            model = lmfit.Model(func, independent_vars=["x"])
            params = model.make_params()

            for param_name in params:
                params[param_name].set(value=1.0)
                if param_name == "a":
                    params[param_name].set(min=0.0)

            results[name] = model.fit(y_data, params, x=x_data)
        return results

    def _plot2d(self, result: lmfit.model.ModelResult) -> Figure:
        x0_data = self.df[self.xs[0].id].cast(pl.Float64).to_numpy()
        x1_data = self.df[self.xs[1].id].cast(pl.Float64).to_numpy()
        y_data = self.df[self.y.id].cast(pl.Float64).to_numpy()

        f = plt.figure()
        ax: Axes3D = f.add_subplot(111, projection="3d")

        x0_min, x0_max = float(x0_data.min()), float(x0_data.max())
        x1_min, x1_max = float(x1_data.min()), float(x1_data.max())
        x0_grid, x1_grid = np.meshgrid(
            np.linspace(x0_min, x0_max, 50), np.linspace(x1_min, x1_max, 50)
        )

        fit_y = result.model.eval(result.params, x=(x0_grid.ravel(), x1_grid.ravel()))
        fit_y = fit_y.reshape(x0_grid.shape)

        ax.scatter(x0_data, x1_data, y_data, color="red", label="Data")
        ax.plot_surface(x0_grid, x1_grid, fit_y, color="blue", alpha=0.5, label="Fit")
        # ax.set_box_aspect((np.ptp(x0_data), np.ptp(x1_data), np.ptp(y_data)))

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
        self.results = self._fitnd(model_funcs=_MODELS_FUNCS_1D, xs=self.xs, y=self.y)

    def _fit2d(self) -> None:
        self.results = self._fitnd(model_funcs=_MODELS_FUNCS_2D, xs=self.xs, y=self.y)

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
