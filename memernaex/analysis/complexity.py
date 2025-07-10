from typing import Any

import lmfit
import numpy as np
import numpy.typing as npt
import polars as pl


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
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def _best_model(
        self, results: dict[str, lmfit.model.ModelResult]
    ) -> tuple[str, lmfit.model.ModelResult]:
        best_name = min(results, key=lambda name: results[name].bic)
        return best_name, results[best_name]

    def _fitnd(
        self, *, model_funcs: dict[str, Any], xs: tuple[str, ...], y: str
    ) -> dict[str, lmfit.model.ModelResult]:
        results: dict[str, lmfit.model.ModelResult] = {}
        x_data = tuple(self.df[col].to_numpy() for col in xs)
        y_data = self.df[y].to_numpy()
        for name, func in model_funcs.items():
            model = lmfit.Model(func, independent_vars=["x"])
            params = model.make_params()

            for param_name in params:
                params[param_name].set(value=1.0)
                if param_name == "a":
                    params[param_name].set(min=0.0)

            results[name] = model.fit(y_data, params, x=x_data)
        return results

    def fit1d(self, *, x: str, y: str) -> tuple[str, lmfit.model.ModelResult]:
        results = self._fitnd(model_funcs=_MODELS_FUNCS_1D, xs=(x,), y=y)
        return self._best_model(results)

    def fit2d(self, *, xs: tuple[str, str], y: str) -> tuple[str, lmfit.model.ModelResult]:
        results = self._fitnd(model_funcs=_MODELS_FUNCS_2D, xs=xs, y=y)
        return self._best_model(results)
