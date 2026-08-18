"""Statistics helpers (spec section 8).

Per-trial values everywhere; bootstrap CIs (vectorised); pairing only through
an inner join on keys so baseline/post lists can never be compacted
independently; log-scale checks; fake-onset drawing for the shuffle control;
channel-as-random-effect model (statsmodels MixedLM) with an OLS fallback.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:  # optional: channel random effect
    import statsmodels.formula.api as smf  # type: ignore

    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - depends on the environment
    smf = None
    HAS_STATSMODELS = False


# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[..., np.ndarray] = np.median,
    n: int = 1000,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` (must accept ``axis``)."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0 or n <= 0:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, data.size, size=(n, data.size))
    draws = statistic(data[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_paired_db_ci(
    baseline: np.ndarray, post: np.ndarray, n: int = 1000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """CI of mean 10log10(post/base) over resampled PAIRS (same estimator as F5)."""
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(post, dtype=float)
    valid = np.isfinite(b) & np.isfinite(p) & (b > 0) & (p > 0)
    b, p = b[valid], p[valid]
    if b.size == 0 or n <= 0:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, b.size, size=(n, b.size))
    draws = np.mean(10.0 * np.log10(p[idx] / b[idx]), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_independent_db_ci(
    a: np.ndarray, b: np.ndarray, n: int = 1000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """CI of 10log10(mean(b*)/mean(a*)) with independent resampling."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    y = y[np.isfinite(y) & (y > 0)]
    if x.size == 0 or y.size == 0 or n <= 0:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    xs = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    ys = y[rng.integers(0, y.size, size=(n, y.size))].mean(axis=1)
    draws = 10.0 * np.log10(ys / xs)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def median_with_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float, int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = bootstrap_ci(data, np.median, n, rng)
    return float(np.median(data)), lo, hi, int(data.size)


# -----------------------------------------------------------------------------
# Pairing
# -----------------------------------------------------------------------------


def paired_frame(
    a: pd.DataFrame,
    b: pd.DataFrame,
    keys: Sequence[str] = ("run_id", "channel", "event_number"),
    suffixes: tuple[str, str] = ("_a", "_b"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner join on ``keys``; also returns the keys dropped from either side.

    This is the only pairing mechanism in the package: an epoch missing from
    one window is removed from both (spec section 8, pitfall 4).
    """
    keys = list(keys)
    for name, frame in (("a", a), ("b", b)):
        if not frame.empty and frame.duplicated(keys).any():
            raise ValueError(f"paired_frame: duplicate keys in frame {name}")
    merged = a.merge(b, on=keys, how="inner", suffixes=suffixes)
    only_a = a[keys].merge(merged[keys], on=keys, how="left", indicator=True)
    only_b = b[keys].merge(merged[keys], on=keys, how="left", indicator=True)
    dropped = pd.concat(
        [
            only_a[only_a["_merge"] == "left_only"][keys].assign(missing_in="b"),
            only_b[only_b["_merge"] == "left_only"][keys].assign(missing_in="a"),
        ],
        ignore_index=True,
    )
    return merged, dropped


# -----------------------------------------------------------------------------
# Distribution checks
# -----------------------------------------------------------------------------


def lognormal_check(values: np.ndarray, name: str = "") -> dict[str, object]:
    """Shapiro-Wilk on raw vs log values, skew, and a recommendation."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    out: dict[str, object] = {"metric": name, "n": int(data.size)}
    positive = data[data > 0]
    if data.size < 3:
        out.update({"shapiro_p_raw": float("nan"), "shapiro_p_log": float("nan"), "skew_raw": float("nan"), "skew_log": float("nan"), "recommended": "insufficient"})
        return out
    sample = data if data.size <= 5000 else np.random.default_rng(0).choice(data, 5000, replace=False)
    try:
        p_raw = float(sp_stats.shapiro(sample).pvalue)
    except Exception:
        p_raw = float("nan")
    if positive.size >= 3:
        logs = np.log10(positive if positive.size <= 5000 else np.random.default_rng(0).choice(positive, 5000, replace=False))
        try:
            p_log = float(sp_stats.shapiro(logs).pvalue)
        except Exception:
            p_log = float("nan")
        skew_log = float(sp_stats.skew(logs))
    else:
        p_log, skew_log = float("nan"), float("nan")
    skew_raw = float(sp_stats.skew(sample))
    if positive.size < data.size:
        recommended = "raw (non-positive values present)"
    elif np.isfinite(p_log) and (p_log > p_raw or abs(skew_log) < abs(skew_raw)):
        recommended = "log"
    else:
        recommended = "raw"
    out.update({"shapiro_p_raw": p_raw, "shapiro_p_log": p_log, "skew_raw": skew_raw, "skew_log": skew_log, "n_nonpositive": int(data.size - positive.size), "recommended": recommended})
    return out


def qq_points(values: np.ndarray, log: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if log:
        data = np.log10(data[data > 0])
    if data.size < 3:
        return np.zeros(0), np.zeros(0)
    (theoretical, ordered), _ = sp_stats.probplot(data, dist="norm")
    return np.asarray(theoretical), np.asarray(ordered)


# -----------------------------------------------------------------------------
# Shuffle control helpers
# -----------------------------------------------------------------------------


def clean_intervals(
    onset_samples: np.ndarray,
    blank_end_samples: np.ndarray,
    n_samples: int,
    *,
    guard_samples: int = 0,
    start_sample: int = 0,
) -> list[tuple[int, int]]:
    """Sample intervals between pulses that are free of artifact.

    ``blank_end_samples[k]`` is the sample where pulse k's blank ends (absolute).
    """
    onsets = np.asarray(onset_samples, dtype=np.int64)
    ends = np.asarray(blank_end_samples, dtype=np.int64)
    order = np.argsort(onsets)
    onsets, ends = onsets[order], ends[order]
    intervals: list[tuple[int, int]] = []
    previous_end = start_sample
    for onset, end in zip(onsets, ends):
        a = int(previous_end + guard_samples)
        b = int(onset - guard_samples)
        if b > a:
            intervals.append((a, b))
        previous_end = max(previous_end, int(end))
    a = int(previous_end + guard_samples)
    b = int(n_samples - guard_samples)
    if b > a:
        intervals.append((a, b))
    return intervals


def draw_fake_onsets(
    intervals: Sequence[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
    *,
    min_separation: int = 1,
    lower: int | None = None,
    upper: int | None = None,
) -> np.ndarray:
    """Uniform draws over the union of ``intervals`` with rejection for separation."""
    usable = []
    for a, b in intervals:
        if lower is not None:
            a = max(a, lower)
        if upper is not None:
            b = min(b, upper)
        if b > a:
            usable.append((a, b))
    if not usable or n <= 0:
        return np.zeros(0, dtype=np.int64)
    lengths = np.array([b - a for a, b in usable], dtype=float)
    weights = lengths / lengths.sum()
    chosen: list[int] = []
    attempts = 0
    while len(chosen) < n and attempts < 50 * n:
        attempts += 1
        k = int(rng.choice(len(usable), p=weights))
        a, b = usable[k]
        candidate = int(rng.integers(a, b))
        if all(abs(candidate - c) >= min_separation for c in chosen):
            chosen.append(candidate)
    return np.array(sorted(chosen), dtype=np.int64)


# -----------------------------------------------------------------------------
# Cross-channel model
# -----------------------------------------------------------------------------


def fit_cross_channel_model(
    frame: pd.DataFrame,
    response: str,
    fixed: Sequence[str],
    group: str = "channel",
    *,
    use_statsmodels: bool = True,
    bootstrap_n: int = 200,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """log(response) ~ fixed effects with ``group`` as a random intercept.

    Uses statsmodels MixedLM when available; otherwise OLS with group fixed
    effects and a trial bootstrap. Returns one row per term with CI + method.
    Values must be positive; the response and every fixed term named ``log_*``
    are already log10 transformed by the caller.
    """
    data = frame.dropna(subset=[response, *fixed, group]).copy()
    rows: list[dict[str, object]] = []
    if data.empty or data[group].nunique() < 2:
        return pd.DataFrame(rows)
    method = "mixedlm_random_intercept" if (use_statsmodels and HAS_STATSMODELS) else "ols_group_fixed_effects_bootstrap"
    if method.startswith("mixedlm"):
        try:
            import warnings

            formula = f"{response} ~ " + " + ".join(fixed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # singular random-effect covariance with few groups is expected
                model = smf.mixedlm(formula, data, groups=data[group])
                fit = model.fit(reml=True, method=["lbfgs", "powell"], maxiter=500)
            conf = fit.conf_int()
            for term in fit.fe_params.index:
                rows.append(
                    {
                        "response": response,
                        "term": term,
                        "estimate": float(fit.fe_params[term]),
                        "ci_low": float(conf.loc[term, 0]),
                        "ci_high": float(conf.loc[term, 1]),
                        "p_value": float(fit.pvalues[term]) if term in fit.pvalues else float("nan"),
                        "group_variance": float(np.asarray(fit.cov_re).ravel()[0]) if fit.cov_re.size else float("nan"),
                        "n_obs": int(fit.nobs),
                        "n_groups": int(data[group].nunique()),
                        "converged": bool(getattr(fit, "converged", True)),
                        "method": method,
                    }
                )
            return pd.DataFrame(rows)
        except Exception as exc:  # fall through to OLS
            method = f"ols_group_fixed_effects_bootstrap (mixedlm failed: {type(exc).__name__})"
    # OLS with group dummies
    groups = sorted(data[group].unique())
    def design(sub: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(sub))]
        cols += [sub[term].to_numpy(dtype=float) for term in fixed]
        for g in groups[1:]:
            cols.append((sub[group] == g).to_numpy(dtype=float))
        return np.column_stack(cols)
    y = data[response].to_numpy(dtype=float)
    X = design(data)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rng = rng if rng is not None else np.random.default_rng(0)
    draws = []
    for _ in range(max(0, bootstrap_n)):
        idx = rng.integers(0, len(data), size=len(data))
        sub = data.iloc[idx]
        try:
            b, *_ = np.linalg.lstsq(design(sub), sub[response].to_numpy(dtype=float), rcond=None)
            draws.append(b)
        except Exception:
            continue
    draws_arr = np.array(draws) if draws else np.zeros((0, len(beta)))
    names = ["Intercept", *fixed]
    for index, term in enumerate(names):
        lo, hi = (np.percentile(draws_arr[:, index], [2.5, 97.5]) if draws_arr.size else (float("nan"), float("nan")))
        rows.append(
            {
                "response": response, "term": term, "estimate": float(beta[index]),
                "ci_low": float(lo), "ci_high": float(hi), "p_value": float("nan"),
                "group_variance": float("nan"), "n_obs": int(len(data)), "n_groups": len(groups),
                "converged": True, "method": method,
            }
        )
    return pd.DataFrame(rows)


def spearman_ci(x: np.ndarray, y: np.ndarray, n: int = 1000, rng: np.random.Generator | None = None) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 3:
        return {"rho": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "n": int(a.size)}
    rho, p = sp_stats.spearmanr(a, b)
    rng = rng if rng is not None else np.random.default_rng(0)
    draws = []
    for _ in range(max(0, n)):
        idx = rng.integers(0, a.size, size=a.size)
        if np.unique(a[idx]).size < 2 or np.unique(b[idx]).size < 2:
            continue
        draws.append(sp_stats.spearmanr(a[idx], b[idx]).statistic)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan")))
    return {"rho": float(rho), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p), "n": int(a.size)}


__all__ = [
    "HAS_STATSMODELS",
    "bootstrap_ci",
    "bootstrap_independent_db_ci",
    "bootstrap_paired_db_ci",
    "clean_intervals",
    "draw_fake_onsets",
    "fit_cross_channel_model",
    "lognormal_check",
    "median_with_ci",
    "paired_frame",
    "qq_points",
    "spearman_ci",
]
