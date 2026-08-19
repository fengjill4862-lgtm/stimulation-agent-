"""Summaries and the log-log slope tests.

Every summary point is a per-trial distribution: median, IQR and a percentile
bootstrap CI of the median.  Slopes are OLS of log10(y) on log10(x) over
per-trial data with a cluster bootstrap by run (trials are resampled within
each run, runs are kept), plus Theil-Sen on the per-run medians as a check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from filter_diag.common import bootstrap_median_ci
from bw_sweep.config import SweepConfig


def fraction_ci(flags: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of a boolean array with a percentile bootstrap CI."""
    v = np.asarray(flags, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, v.size, size=(n, v.size))
    draws = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "q25": float("nan"), "q75": float("nan")}
    med, lo, hi = bootstrap_median_ci(v, n_boot, rng)
    q25, q75 = np.percentile(v, [25, 75])
    return {"n": int(v.size), "median": med, "ci_low": lo, "ci_high": hi, "q25": float(q25), "q75": float(q75)}


@dataclass
class SlopeResult:
    arm: str
    slope: float = float("nan")
    intercept: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    theil_slope: float = float("nan")
    theil_low: float = float("nan")
    theil_high: float = float("nan")
    slope1_intercept_log10: float = float("nan")  # best fit with slope fixed at 1: log10 y = log10 x + b
    slope1_ratio: float = float("nan")  # 10**b: recovery / tau_nominal
    n_runs_used: int = 0
    n_trials_used: int = 0
    used_run_ids: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)  # run_id -> reason
    verdict: str = "insufficient"
    per_channel: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def describe(self) -> str:
        if self.n_runs_used < 2:
            return f"arm {self.arm}: slope not estimable ({self.n_runs_used} usable runs; excluded {self.excluded})"
        return (
            f"arm {self.arm}: log-log slope {self.slope:.2f} [95% CI {self.ci_low:.2f}, {self.ci_high:.2f}] "
            f"over {self.n_runs_used} runs / {self.n_trials_used} trials (Theil-Sen {self.theil_slope:.2f} "
            f"[{self.theil_low:.2f}, {self.theil_high:.2f}]); slope-1 fit ratio recovery/tau = {self.slope1_ratio:.2f}"
            + (f"; excluded {self.excluded}" if self.excluded else "")
        )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    denominator = float(np.sum((x - xm) ** 2))
    if denominator <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - xm) * (y - ym)) / denominator)
    return slope, float(ym - slope * xm)


def slope_verdict(lo: float, hi: float, cfg: SweepConfig) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "insufficient"
    contains_one = lo <= 1.0 <= hi
    contains_zero = lo <= 0.0 <= hi
    inside_one = (1.0 - cfg.slope_one_band) <= lo and hi <= (1.0 + cfg.slope_one_band)
    inside_zero = -cfg.slope_flat_band <= lo and hi <= cfg.slope_flat_band
    if inside_one or (contains_one and not contains_zero and (hi - lo) < cfg.slope_ci_width_max):
        return "~1"
    if inside_zero or (contains_zero and not contains_one):
        return "flat"
    return "intermediate"


def loglog_slope(
    trials: pd.DataFrame,
    *,
    arm: str,
    x_col: str,
    y_col: str,
    cfg: SweepConfig,
    rng: np.random.Generator,
    floor_by_run: dict[str, float] | None = None,
) -> SlopeResult:
    """OLS log10(y) ~ log10(x) on per-trial rows, cluster bootstrap by run.

    Runs whose median y is censored (>= ``censor_ms``) or on the hardware
    floor (<= floor + ``floor_margin_ms``) are excluded and listed.
    """
    result = SlopeResult(arm=arm)
    if trials.empty:
        return result
    frame = trials[np.isfinite(trials[x_col]) & np.isfinite(trials[y_col]) & (trials[x_col] > 0) & (trials[y_col] > 0)]
    usable_runs: list[str] = []
    for run_id, group in frame.groupby("run_id", sort=False):
        med = float(np.median(group[y_col]))
        floor = float(floor_by_run.get(run_id, 0.0)) if floor_by_run else 0.0
        if y_col == "recovery_ms" and med >= cfg.censor_ms:
            result.excluded[run_id] = f"censored (median {med:.0f} ms)"
        elif y_col == "recovery_ms" and med <= floor + cfg.floor_margin_ms:
            result.excluded[run_id] = f"at hardware floor (median {med:.2f} ms, floor {floor:.2f} ms)"
        elif y_col == "recovery_ms" and "local_drift_uV" in group and float((group["local_drift_uV"].abs() > cfg.local_drift_max_uV).mean()) > cfg.drifting_fraction_max:
            drifting = 100 * float((group["local_drift_uV"].abs() > cfg.local_drift_max_uV).mean())
            result.excluded[run_id] = f"baseline still moving before the pulse ({drifting:.0f}% of trials drift > {cfg.local_drift_max_uV:g} uV across the centre window; median {med:.0f} ms)"
        elif len(group) < 3:
            result.excluded[run_id] = f"only {len(group)} trials"
        else:
            usable_runs.append(run_id)
    frame = frame[frame["run_id"].isin(usable_runs)]
    result.used_run_ids = usable_runs
    result.n_runs_used = len(usable_runs)
    result.n_trials_used = int(len(frame))
    if len({round(v, 6) for v in frame[x_col].unique()}) < 2:
        return result
    lx = np.log10(frame[x_col].to_numpy(dtype=float))
    ly = np.log10(frame[y_col].to_numpy(dtype=float))
    result.slope, result.intercept = _ols(lx, ly)
    # cluster bootstrap: resample trials within each run
    groups = [np.flatnonzero(frame["run_id"].to_numpy() == run_id) for run_id in usable_runs]
    draws = np.full(cfg.bootstrap_n, np.nan)
    for b in range(cfg.bootstrap_n):
        idx = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in groups])
        draws[b], _ = _ols(lx[idx], ly[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size:
        result.ci_low, result.ci_high = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    # Theil-Sen on per-run medians (log10)
    med_x, med_y = [], []
    for run_id in usable_runs:
        g = frame[frame["run_id"] == run_id]
        med_x.append(np.log10(float(np.median(g[x_col]))))
        med_y.append(np.log10(float(np.median(g[y_col]))))
    if len(med_x) >= 3:
        try:
            ts = sps.theilslopes(med_y, med_x)
            result.theil_slope, result.theil_low, result.theil_high = float(ts[0]), float(ts[2]), float(ts[3])
        except Exception:
            pass
    elif len(med_x) == 2:
        result.theil_slope = float((med_y[1] - med_y[0]) / (med_x[1] - med_x[0])) if med_x[1] != med_x[0] else float("nan")
    result.slope1_intercept_log10 = float(np.mean(ly - lx))
    result.slope1_ratio = float(10 ** result.slope1_intercept_log10)
    result.verdict = slope_verdict(result.ci_low, result.ci_high, cfg)
    if "channel" in frame:
        for channel, g in frame.groupby("channel", sort=False):
            if len({round(v, 6) for v in g[x_col].unique()}) < 2:
                continue
            s, _ = _ols(np.log10(g[x_col].to_numpy(dtype=float)), np.log10(g[y_col].to_numpy(dtype=float)))
            gl = [np.flatnonzero(g["run_id"].to_numpy() == r) for r in g["run_id"].unique()]
            gx = np.log10(g[x_col].to_numpy(dtype=float))
            gy = np.log10(g[y_col].to_numpy(dtype=float))
            d = []
            for b in range(min(cfg.bootstrap_n, 400)):
                idx = np.concatenate([gg[rng.integers(0, gg.size, size=gg.size)] for gg in gl])
                d.append(_ols(gx[idx], gy[idx])[0])
            d = np.asarray(d)
            d = d[np.isfinite(d)]
            lo, hi = (np.percentile(d, [2.5, 97.5]) if d.size else (float("nan"), float("nan")))
            result.per_channel[str(channel)] = (s, float(lo), float(hi))
    return result


@dataclass
class AdditiveFit:
    """recovery = a * tau + b on per-trial rows (linear in the parameters), cluster bootstrap by run."""

    arm: str
    a: float = float("nan")
    b_ms: float = float("nan")
    a_ci: tuple[float, float] = (float("nan"), float("nan"))
    b_ci: tuple[float, float] = (float("nan"), float("nan"))
    r2_medians: float = float("nan")  # R2 of the fit against the per-run medians
    n_runs: int = 0

    def describe(self, linear_prediction: float | None = None) -> str:
        if self.n_runs < 2 or not np.isfinite(self.a):
            return ""
        text = (f"additive model recovery = a*tau + b: a = {self.a:.2f} [{self.a_ci[0]:.2f}, {self.a_ci[1]:.2f}], "
                f"b = {self.b_ms:.0f} ms [{self.b_ci[0]:.0f}, {self.b_ci[1]:.0f}] (R2 vs run medians {self.r2_medians:.2f}, {self.n_runs} runs)")
        if linear_prediction is not None and np.isfinite(linear_prediction):
            text += f"; a linear high-pass recovering from the rail predicts a = ln(rail/threshold) = {linear_prediction:.2f}"
        return text


def additive_fit(trials: pd.DataFrame, *, arm: str, x_col: str, y_col: str, run_ids: list[str], cfg: SweepConfig, rng: np.random.Generator) -> AdditiveFit:
    result = AdditiveFit(arm=arm)
    frame = trials[trials["run_id"].isin(run_ids) & np.isfinite(trials[x_col]) & np.isfinite(trials[y_col])]
    result.n_runs = int(frame["run_id"].nunique())
    if result.n_runs < 2:
        return result
    x = frame[x_col].to_numpy(dtype=float)
    y = frame[y_col].to_numpy(dtype=float)
    result.a, result.b_ms = _ols(x, y)
    groups = [np.flatnonzero(frame["run_id"].to_numpy() == r) for r in run_ids if (frame["run_id"] == r).any()]
    draws_a, draws_b = [], []
    for _ in range(cfg.bootstrap_n):
        idx = np.concatenate([g[rng.integers(0, g.size, size=g.size)] for g in groups])
        a, b = _ols(x[idx], y[idx])
        draws_a.append(a)
        draws_b.append(b)
    da, db = np.asarray(draws_a), np.asarray(draws_b)
    da, db = da[np.isfinite(da)], db[np.isfinite(db)]
    if da.size and db.size:
        result.a_ci = tuple(float(v) for v in np.percentile(da, [2.5, 97.5]))
        result.b_ci = tuple(float(v) for v in np.percentile(db, [2.5, 97.5]))
    med = frame.groupby("run_id").agg(x=(x_col, "median"), y=(y_col, "median"))
    pred = result.a * med["x"] + result.b_ms
    ss_res = float(np.sum((med["y"] - pred) ** 2))
    ss_tot = float(np.sum((med["y"] - med["y"].mean()) ** 2))
    result.r2_medians = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return result


__all__ = ["AdditiveFit", "SlopeResult", "additive_fit", "fraction_ci", "loglog_slope", "slope_verdict", "summarize"]
