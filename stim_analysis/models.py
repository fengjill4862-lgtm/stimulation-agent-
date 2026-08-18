"""Amplitude-response models (spec section 7.2).

Per channel and metric, on Block 1 (single phase width): linear A = m*I + c,
linear through the origin A = m*I (the artifact signature), and a sigmoid
A = Amax / (1 + exp(-(I - I50)/k)); compared by AIC/BIC (Gaussian residuals),
with a bootstrap CI on I50 (stratified resampling of trials within each
amplitude).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit


@dataclass(frozen=True)
class ModelFit:
    name: str
    params: dict[str, float] = field(default_factory=dict)
    rss: float = float("nan")
    n: int = 0
    k: int = 0
    aic: float = float("nan")
    bic: float = float("nan")
    converged: bool = False

    def predict(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, dtype=float)
        if self.name == "linear":
            return self.params["m"] * current + self.params["c"]
        if self.name == "linear_origin":
            return self.params["m"] * current
        if self.name == "sigmoid":
            return sigmoid(current, self.params["amax"], self.params["i50"], self.params["k"])
        raise ValueError(self.name)


def aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian log-likelihood AIC/BIC from the residual sum of squares."""
    if n <= 0 or not np.isfinite(rss) or rss <= 0:
        return float("nan"), float("nan")
    ll_term = n * np.log(rss / n)
    return float(ll_term + 2 * k), float(ll_term + k * np.log(n))


def sigmoid(current: np.ndarray, amax: float, i50: float, k: float) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    z = np.clip(-(current - i50) / max(k, 1e-9), -500, 500)
    return amax / (1.0 + np.exp(z))


def fit_linear(current: np.ndarray, response: np.ndarray) -> ModelFit:
    x = np.asarray(current, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.size < 3:
        return ModelFit("linear")
    A = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(np.sum((y - A @ beta) ** 2))
    aic, bic = aic_bic(rss, x.size, 2)
    return ModelFit("linear", {"m": float(beta[0]), "c": float(beta[1])}, rss, int(x.size), 2, aic, bic, True)


def fit_linear_origin(current: np.ndarray, response: np.ndarray) -> ModelFit:
    x = np.asarray(current, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.size < 2 or not np.any(x != 0):
        return ModelFit("linear_origin")
    m = float(np.sum(x * y) / np.sum(x * x))
    rss = float(np.sum((y - m * x) ** 2))
    aic, bic = aic_bic(rss, x.size, 1)
    return ModelFit("linear_origin", {"m": m}, rss, int(x.size), 1, aic, bic, True)


def fit_sigmoid(
    current: np.ndarray,
    response: np.ndarray,
    *,
    i50_grid: np.ndarray | None = None,
    p0_seed: tuple[float, float, float] | None = None,
    maxfev: int = 4000,
) -> ModelFit:
    """Bounded least squares; multi-start over I50 unless ``p0_seed`` is given."""
    x = np.asarray(current, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.size < 4 or np.unique(x).size < 3:
        return ModelFit("sigmoid")
    y_max = float(np.max(y)) if np.max(y) > 0 else 1.0
    lo_x, hi_x = float(np.min(x[x > 0])) if np.any(x > 0) else 1.0, float(np.max(x))
    bounds = ([0.0, lo_x / 2.0, 1.0], [10.0 * y_max, 2.0 * hi_x, 500.0])
    starts: list[list[float]] = []
    if p0_seed is not None:
        starts.append([
            float(np.clip(p0_seed[0], bounds[0][0], bounds[1][0])),
            float(np.clip(p0_seed[1], bounds[0][1], bounds[1][1])),
            float(np.clip(p0_seed[2], bounds[0][2], bounds[1][2])),
        ])
    else:
        grid = i50_grid if i50_grid is not None else np.geomspace(lo_x, hi_x, 6)
        for i50_start in grid:
            for k_start in (max(1.0, (hi_x - lo_x) / 10.0), max(1.0, (hi_x - lo_x) / 3.0)):
                starts.append([min(max(y_max, 1e-9), bounds[1][0]), float(np.clip(i50_start, bounds[0][1], bounds[1][1])), float(np.clip(k_start, 1.0, 500.0))])
    best: ModelFit | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for p0 in starts:
            try:
                params, _cov = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            except Exception:
                continue
            pred = sigmoid(x, *params)
            rss = float(np.sum((y - pred) ** 2))
            if best is None or rss < best.rss:
                aic, bic = aic_bic(rss, x.size, 3)
                best = ModelFit("sigmoid", {"amax": float(params[0]), "i50": float(params[1]), "k": float(params[2])}, rss, int(x.size), 3, aic, bic, True)
    return best if best is not None else ModelFit("sigmoid")


def bootstrap_i50(
    current: np.ndarray,
    response: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    seed_fit: ModelFit | None = None,
) -> tuple[float, float, float]:
    """Stratified (within amplitude) bootstrap CI of I50; returns (lo, hi, converged fraction)."""
    x = np.asarray(current, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.size < 4 or n <= 0:
        return float("nan"), float("nan"), 0.0
    levels = np.unique(x)
    index_by_level = {level: np.flatnonzero(x == level) for level in levels}
    seed = None
    if seed_fit is not None and seed_fit.converged:
        seed = (seed_fit.params["amax"], seed_fit.params["i50"], seed_fit.params["k"])
    draws: list[float] = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(index_by_level[level], size=index_by_level[level].size, replace=True) for level in levels])
        # one warm start from the full-data solution keeps 1000 refits affordable
        fit = fit_sigmoid(x[idx], y[idx], p0_seed=seed, maxfev=600) if seed is not None else fit_sigmoid(x[idx], y[idx])
        if fit.converged:
            draws.append(fit.params["i50"])
    if not draws:
        return float("nan"), float("nan"), 0.0
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), len(draws) / n


def compare_models(fits: list[ModelFit]) -> dict[str, object]:
    valid = [f for f in fits if f.converged and np.isfinite(f.aic)]
    if not valid:
        return {"preferred": "", "artifact_candidate": False}
    best_aic = min(valid, key=lambda f: f.aic)
    best_bic = min(valid, key=lambda f: f.bic)
    by_name = {f.name: f for f in valid}
    out: dict[str, object] = {"preferred": best_aic.name, "preferred_bic": best_bic.name}
    for f in valid:
        out[f"delta_aic_{f.name}"] = float(f.aic - best_aic.aic)
        out[f"delta_bic_{f.name}"] = float(f.bic - best_bic.bic)
    # Linear-through-origin winning (or within 2 AIC of the best) = artifact signature.
    origin = by_name.get("linear_origin")
    out["artifact_candidate"] = bool(origin is not None and (origin.aic - best_aic.aic) <= 2.0)
    sig = by_name.get("sigmoid")
    lin = by_name.get("linear")
    out["sigmoid_beats_linear_by_aic"] = float(lin.aic - sig.aic) if (sig is not None and lin is not None) else float("nan")
    return out


def fit_amplitude_response(
    trials: pd.DataFrame,
    metric: str,
    *,
    bootstrap_n: int,
    rng: np.random.Generator,
    by: tuple[str, ...] = ("channel",),
    current_col: str = "amplitude_uA",
) -> pd.DataFrame:
    """All three fits + comparison + I50 CI per group (channel) for one metric."""
    rows: list[dict[str, object]] = []
    if trials.empty or metric not in trials:
        return pd.DataFrame(rows)
    for key, group in trials.groupby(list(by)):
        sub = group.dropna(subset=[metric, current_col])
        sub = sub[np.isfinite(sub[metric])]
        x = sub[current_col].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(by, key_tuple))
        row.update({"metric": metric, "n_trials": int(x.size), "n_amplitudes": int(np.unique(x).size)})
        if x.size < 4 or np.unique(x).size < 3:
            row.update({"preferred": "insufficient", "artifact_candidate": False})
            rows.append(row)
            continue
        lin = fit_linear(x, y)
        origin = fit_linear_origin(x, y)
        sig = fit_sigmoid(x, y)
        row.update(
            {
                "linear_m": lin.params.get("m", float("nan")), "linear_c": lin.params.get("c", float("nan")), "linear_aic": lin.aic, "linear_bic": lin.bic,
                "origin_m": origin.params.get("m", float("nan")), "origin_aic": origin.aic, "origin_bic": origin.bic,
                "sigmoid_amax": sig.params.get("amax", float("nan")), "sigmoid_i50": sig.params.get("i50", float("nan")), "sigmoid_k": sig.params.get("k", float("nan")),
                "sigmoid_aic": sig.aic, "sigmoid_bic": sig.bic, "sigmoid_converged": sig.converged,
            }
        )
        row.update(compare_models([lin, origin, sig]))
        lo, hi, frac = bootstrap_i50(x, y, bootstrap_n, rng, seed_fit=sig)
        row.update({"i50_ci_low": lo, "i50_ci_high": hi, "i50_bootstrap_converged_fraction": frac})
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "ModelFit",
    "aic_bic",
    "bootstrap_i50",
    "compare_models",
    "fit_amplitude_response",
    "fit_linear",
    "fit_linear_origin",
    "fit_sigmoid",
    "sigmoid",
]
