from typing import Any

import lmfit
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from memernaex.plot.util import Var, set_up_figure


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

    def __init__(self, *, df: pl.DataFrame, xs: tuple[Var, ...] | Var, y: Var) -> None:
        self.df = df
        if isinstance(xs, Var):
            self.xs = (xs,)
        elif isinstance(xs, tuple) and len(xs) == 2:
            self.xs = xs
        else:
            raise ValueError("xs must be a string or a tuple of two strings.")
        self.y = y

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

    def _plot2d(self) -> Figure:
        f, ax = plt.subplots(1)

        return f

    def _plot1d(self) -> Figure:
        f, ax = plt.subplots(1)

        set_up_figure(f, varz=(self.xs[0], self.y))
        return f

    def _fit1d(self, *, x: Var, y: Var) -> tuple[str, lmfit.model.ModelResult]:
        results = self._fitnd(model_funcs=_MODELS_FUNCS_1D, xs=(x,), y=y)
        return self._best_model(results)

    def _fit2d(self, *, xs: tuple[Var, Var], y: Var) -> tuple[str, lmfit.model.ModelResult]:
        results = self._fitnd(model_funcs=_MODELS_FUNCS_2D, xs=xs, y=y)
        return self._best_model(results)

    def fit(self) -> tuple[str, lmfit.model.ModelResult]:
        if len(self.xs) == 1:
            return self._fit1d(x=self.xs[0], y=self.y)
        if len(self.xs) == 2:
            return self._fit2d(xs=self.xs, y=self.y)
        raise ValueError("xs must be a string or a tuple of two strings.")
