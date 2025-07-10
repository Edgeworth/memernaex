from typing import Any

import lmfit
import numpy as np
import numpy.typing as npt
import polars as pl


def _model_constant(x: npt.NDArray, b: float) -> Any:
    return b * np.ones_like(x.astype(float))


def _model_log_n(x: npt.NDArray, a: float, b: float) -> Any:
    return a * np.log(x) + b


def _model_n_log_n(x: npt.NDArray, a: float, b: float) -> Any:
    return a * x * np.log(x) + b


def _model_polynomial(x: npt.NDArray, a: float, b: float) -> Any:
    return a * x + b


_MODELS_FUNCS = {
    "constant": _model_constant,
    "log_n": _model_log_n,
    "n_log_n": _model_n_log_n,
    "polynomial": _model_polynomial,
}


class ComplexityFitter:
    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def fit(self, *, x: str, y: str) -> None:
        for name, func in _MODELS_FUNCS.items():
            print(name)
            model = lmfit.Model(func)
            params = model.make_params()

            for param_name in params:
                params[param_name].set(value=1.0)
                if param_name == "a":
                    params[param_name].set(min=0.0)

            print(model)
            print(params)
            result = model.fit(self.df[y], params, x=self.df[x])
            print(result.fit_report())
