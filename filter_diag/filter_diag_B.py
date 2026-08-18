"""Test B: fit a single exponential to every post-rail tail; test tau invariance."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.stats import HAS_STATSMODELS
from stim_analysis.validate import railed_mask
from filter_diag.common import DiagRun, TailFit, bootstrap_median_ci, fit_exponential_tail, rail_exit_ms

R2_MIN = 0.9


def fit_run_tails(run: DiagRun, cfg: AnalysisConfig, *, end_ms: float = 800.0, r2_min: float = R2_MIN) -> pd.DataFrame:
    """One row per (channel, epoch): tail fit + the trial's recovery metrics."""
    rows: list[dict[str, object]] = []
    t_ms = run.epochs.t_ms
    core = run.epochs.core
    fs = run.epochs.sample_rate_hz
    trials = run.trials.set_index(["channel", "event_number"]) if not run.trials.empty else None
    kept_numbers = run.epochs.event_numbers[run.epochs.kept]
    for channel in run.channels:
        if channel not in run.baseline_mean:
            continue
        ep = run.channel_epochs(channel, centred=True)
        rail = run.rails.get(channel)
        railed = np.stack([railed_mask(row, rail, cfg, fs) for row in run.channel_epochs(channel, centred=False)]) if (rail is not None and rail.is_railed) else None
        for i, number in enumerate(kept_numbers):
            x = ep[i]
            exit_ms, sign, was_railed = rail_exit_ms(x, t_ms, None if railed is None else railed[i], core)
            fit = fit_exponential_tail(x, t_ms, exit_ms=exit_ms, excursion_sign=sign, end_ms=end_ms)
            trial = trials.loc[(channel, int(number))] if trials is not None and (channel, int(number)) in trials.index else None
            rows.append(
                {
                    "session": run.session,
                    "run_id": run.run_id,
                    "channel": channel,
                    "event_number": int(number),
                    "amplitude_uA": run.amplitude_uA,
                    "phase_us": run.phase_us,
                    "is_stim_contact": channel == run.stim_channel,
                    "impedance_kohm": float(trial["impedance_kohm"]) if trial is not None else float("nan"),
                    "distance_um": float(trial["distance_um"]) if trial is not None else float("nan"),
                    "recovery_ms": float(trial["recovery_ms"]) if trial is not None else float("nan"),
                    "censored": bool(trial["censored"]) if trial is not None else False,
                    "rail_ms": float(trial["rail_ms"]) if trial is not None else 0.0,
                    "threshold_uV": float(trial["threshold_uV"]) if trial is not None else float("nan"),
                    "baseline_sd_uV": float(trial["baseline_sd_uV"]) if trial is not None else float("nan"),
                    "peak_abs_uV": float(trial["peak_abs_uV"]) if trial is not None else float("nan"),
                    "exit_ms": exit_ms,
                    "was_railed": was_railed,
                    "excursion_sign": sign,
                    "A_uV": fit.A_uV,
                    "tau_ms": fit.tau_ms,
                    "C_uV": fit.C_uV,
                    "r2": fit.r2,
                    "n_points": fit.n_points,
                    "tail_sign": fit.tail_sign,
                    "fit_ok": bool(fit.converged and np.isfinite(fit.r2) and fit.r2 >= r2_min),
                }
            )
    return pd.DataFrame(rows)


def _impedance_tertile(z: pd.Series) -> pd.Series:
    valid = z.dropna()
    if valid.empty or valid.nunique() < 3:
        return pd.Series(["all"] * len(z), index=z.index)
    q1, q2 = np.nanpercentile(valid, [33.3, 66.7])
    return pd.Series(np.where(z <= q1, "low", np.where(z <= q2, "mid", "high")), index=z.index)


def tau_summary(fits: pd.DataFrame, cfg: AnalysisConfig, rng: np.random.Generator, tau_dsp_ms: float, *, exclude_stim_contact: bool = True) -> pd.DataFrame:
    """Median tau_fit with bootstrap CI, pooled and by covariate; deviation from tau_dsp."""
    good = fits[fits["fit_ok"]]
    if exclude_stim_contact:
        good = good[~good["is_stim_contact"]]
    rows: list[dict[str, object]] = []

    def add(group_name: str, level: object, sub: pd.DataFrame) -> None:
        med, lo, hi = bootstrap_median_ci(sub["tau_ms"].to_numpy(), cfg.bootstrap_n, rng)
        rows.append({"grouping": group_name, "level": level, "n_fits": int(len(sub)), "median_tau_ms": med, "ci_low": lo, "ci_high": hi, "iqr_low": float(np.nanpercentile(sub["tau_ms"], 25)) if len(sub) else float("nan"), "iqr_high": float(np.nanpercentile(sub["tau_ms"], 75)) if len(sub) else float("nan"), "ratio_to_tau_dsp": med / tau_dsp_ms if np.isfinite(med) else float("nan"), "within_20pct_of_dsp": bool(np.isfinite(med) and abs(med / tau_dsp_ms - 1.0) <= 0.2), "same_sign_fraction": float((sub["tail_sign"] > 0).mean()) if len(sub) else float("nan")})

    add("pooled", "all", good)
    for session, sub in good.groupby("session"):
        add("session", session, sub)
    for channel, sub in good.groupby(["session", "channel"]):
        add("session_channel", f"{channel[0]}:{channel[1]}", sub)
    for amp, sub in good.groupby(["session", "amplitude_uA"]):
        add("session_amplitude_uA", f"{amp[0]}:{amp[1]:g}", sub)
    good = good.assign(_ztertile=_impedance_tertile(good["impedance_kohm"]))
    for tert, sub in good.groupby("_ztertile"):
        add("impedance_tertile", tert, sub)
    for dist, sub in good.groupby("distance_um"):
        add("distance_um", f"{dist:g}", sub)
    stim = fits[fits["fit_ok"] & fits["is_stim_contact"]]
    if not stim.empty:
        for session, sub in stim.groupby("session"):
            add("stim_contact", session, sub)
    out = pd.DataFrame(rows)
    out["tau_dsp_ms"] = tau_dsp_ms
    return out


def invariance_model(fits: pd.DataFrame, *, exclude_stim_contact: bool = True) -> pd.DataFrame:
    """log(tau_fit) ~ log(current) + log(impedance) + distance + is_dead (OLS, HC1 CIs), plus per-covariate % shift."""
    good = fits[fits["fit_ok"] & np.isfinite(fits["tau_ms"]) & (fits["tau_ms"] > 0)].copy()
    if exclude_stim_contact:
        good = good[~good["is_stim_contact"]]
    good = good[np.isfinite(good["impedance_kohm"]) & (good["impedance_kohm"] > 0) & np.isfinite(good["distance_um"])]
    if len(good) < 20:
        return pd.DataFrame([{"term": "insufficient", "n": int(len(good))}])
    good["log_tau"] = np.log(good["tau_ms"])
    good["log_current"] = np.log(good["amplitude_uA"])
    good["log_impedance"] = np.log(good["impedance_kohm"])
    good["distance_mm"] = good["distance_um"] / 1000.0
    good["is_dead"] = (good["session"] != "live").astype(float)
    terms = ["log_current", "log_impedance", "distance_mm"]
    if good["is_dead"].nunique() > 1:
        terms.append("is_dead")
    rows: list[dict[str, object]] = []
    if HAS_STATSMODELS:
        import statsmodels.formula.api as smf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("log_tau ~ " + " + ".join(terms), data=good).fit(cov_type="HC1")
            conf = fit.conf_int()
            for term in fit.params.index:
                est = float(fit.params[term])
                rows.append({"term": term, "estimate_log": est, "ci_low": float(conf.loc[term, 0]), "ci_high": float(conf.loc[term, 1]), "p_value": float(fit.pvalues[term]), "pct_shift_per_unit": (np.exp(est) - 1.0) * 100.0 if term != "Intercept" else float("nan"), "n": int(fit.nobs), "r2": float(fit.rsquared), "method": "ols_hc1"})
            # channel random intercept as a check
            try:
                mixed = smf.mixedlm("log_tau ~ " + " + ".join(terms), good, groups=good["session"] + ":" + good["channel"]).fit(reml=True)
                mconf = mixed.conf_int()
                for term in mixed.fe_params.index:
                    rows.append({"term": term, "estimate_log": float(mixed.fe_params[term]), "ci_low": float(mconf.loc[term, 0]), "ci_high": float(mconf.loc[term, 1]), "p_value": float(mixed.pvalues[term]), "pct_shift_per_unit": (np.exp(float(mixed.fe_params[term])) - 1.0) * 100.0 if term != "Intercept" else float("nan"), "n": int(mixed.nobs), "r2": float("nan"), "method": "mixedlm_channel_intercept"})
            except Exception as exc:  # pragma: no cover
                rows.append({"term": "mixedlm_failed", "method": str(exc)[:80]})
    else:
        X = np.column_stack([np.ones(len(good))] + [good[t].to_numpy(dtype=float) for t in terms])
        beta, *_ = np.linalg.lstsq(X, good["log_tau"].to_numpy(dtype=float), rcond=None)
        for name, est in zip(["Intercept", *terms], beta):
            rows.append({"term": name, "estimate_log": float(est), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"), "pct_shift_per_unit": (np.exp(est) - 1.0) * 100.0 if name != "Intercept" else float("nan"), "n": int(len(good)), "r2": float("nan"), "method": "ols_numpy"})
    # Practical covariate shift across the observed range: median tau at the covariate's 10th vs 90th percentile prediction
    out = pd.DataFrame(rows)
    ols = out[out["method"] == "ols_hc1"] if "method" in out else out
    for term in terms:
        r = ols[ols["term"] == term]
        if r.empty or term == "is_dead":
            continue
        span = float(np.nanpercentile(good[term], 90) - np.nanpercentile(good[term], 10))
        out.loc[out["term"] == term, "pct_shift_over_observed_range"] = (np.exp(float(r["estimate_log"].iloc[0]) * span) - 1.0) * 100.0
    return out


def figure_tau(fits: pd.DataFrame, tau_dsp_ms: float, cfg: AnalysisConfig, ctx) -> object:
    from stim_analysis.figures import FOOTER_IN, _style_axes, amplitude_palette, build_caption, channel_palette, finish_figure, plt

    good = fits[fits["fit_ok"] & ~fits["is_stim_contact"]]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2 + FOOTER_IN))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=(FOOTER_IN + 0.5) / (8.2 + FOOTER_IN), hspace=0.42, wspace=0.28)
    lim = (5.0, 3000.0)
    bins = np.geomspace(lim[0], lim[1], 60)
    # pooled by session
    ax = axes[0, 0]
    for session, sub in good.groupby("session"):
        ax.hist(np.clip(sub["tau_ms"], *lim), bins=bins, alpha=0.5, label=f"{session} (n={len(sub)}, median {np.median(sub['tau_ms']):.0f} ms)")
    ax.axvline(tau_dsp_ms, color="#d62728", lw=1.2, ls="--", label=f"DSP tau {tau_dsp_ms:.1f} ms")
    ax.axvline(133.0, color="#ff7f0e", lw=0.8, ls=":", label="spec 133 ms")
    ax.set_xscale("log"); ax.set_xlim(lim); ax.set_xlabel("tau_fit (ms, log)"); ax.set_ylabel("fits"); ax.set_title("tau_fit, live vs dead", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    # by amplitude
    ax = axes[0, 1]
    rng = np.random.default_rng(0)
    for session, sub in good.groupby("session"):
        marker = "o" if session == "live" else "s"
        for amp, s2 in sub.groupby("amplitude_uA"):
            x = amp * (1 + rng.uniform(-0.06, 0.06, len(s2)))
            ax.scatter(x, np.clip(s2["tau_ms"], *lim), s=5, alpha=0.25, marker=marker, color="#1f77b4" if session == "live" else "#2ca02c", linewidths=0)
            ax.scatter([amp], [np.clip(np.median(s2["tau_ms"]), *lim)], s=28, marker=marker, color="black", zorder=4)
    ax.axhline(tau_dsp_ms, color="#d62728", lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlim(cfg.lim_amplitude_uA); ax.set_xlabel("current (uA, log)"); ax.set_ylabel("tau_fit (ms, log)"); ax.set_title("tau_fit vs current (o live, s dead; black = median)", fontsize=9)
    # by impedance
    ax = axes[0, 2]
    for session, sub in good.groupby("session"):
        marker = "o" if session == "live" else "s"
        z = sub["impedance_kohm"].to_numpy(dtype=float)
        ax.scatter(z * (1 + rng.uniform(-0.03, 0.03, len(sub))), np.clip(sub["tau_ms"], *lim), s=5, alpha=0.25, marker=marker, color="#1f77b4" if session == "live" else "#2ca02c", linewidths=0)
        for channel, s2 in sub.groupby("channel"):
            ax.scatter([np.nanmedian(s2["impedance_kohm"])], [np.clip(np.median(s2["tau_ms"]), *lim)], s=28, marker=marker, color="black", zorder=4)
    ax.axhline(tau_dsp_ms, color="#d62728", lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlim(cfg.lim_impedance_kohm); ax.set_xlabel("impedance (kOhm, log)"); ax.set_ylabel("tau_fit (ms, log)"); ax.set_title("tau_fit vs impedance (black = channel median)", fontsize=9)
    # by distance
    ax = axes[1, 0]
    for session, sub in good.groupby("session"):
        marker = "o" if session == "live" else "s"
        d = sub["distance_um"].to_numpy(dtype=float)
        ax.scatter(d + rng.uniform(-60, 60, len(sub)), np.clip(sub["tau_ms"], *lim), s=5, alpha=0.25, marker=marker, color="#1f77b4" if session == "live" else "#2ca02c", linewidths=0)
        for dist, s2 in sub.groupby("distance_um"):
            ax.scatter([dist], [np.clip(np.median(s2["tau_ms"]), *lim)], s=28, marker=marker, color="black", zorder=4)
    ax.axhline(tau_dsp_ms, color="#d62728", lw=1.0, ls="--")
    ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlim(cfg.lim_distance_um); ax.set_xlabel("distance from stim contact (um)"); ax.set_ylabel("tau_fit (ms, log)"); ax.set_title("tau_fit vs distance", fontsize=9)
    # tail sign
    ax = axes[1, 1]
    labels, same, opp = [], [], []
    for (session, amp), sub in good.groupby(["session", "amplitude_uA"]):
        labels.append(f"{session[:4]} {amp:g}"); same.append(float((sub["tail_sign"] > 0).mean())); opp.append(float((sub["tail_sign"] < 0).mean()))
    x = np.arange(len(labels))
    ax.bar(x, same, color="#1f77b4", label="same sign as excursion (slow input / analog)"); ax.bar(x, opp, bottom=same, color="#d62728", label="opposite sign (DSP/HP overshoot signature)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, fontsize=6); ax.set_ylim(0, 1); ax.set_ylabel("fraction of fits"); ax.set_title("tail sign relative to the rail excursion", fontsize=9); ax.legend(fontsize=6, frameon=False, loc="lower right")
    # R2 / rejection
    ax = axes[1, 2]
    allfits = fits[~fits["is_stim_contact"]]
    ax.hist(np.clip(allfits["r2"].fillna(0), -0.5, 1), bins=50, color="0.5")
    ax.axvline(R2_MIN, color="#d62728", ls="--", lw=1.0)
    rej = float((~allfits["fit_ok"]).mean()) if len(allfits) else float("nan")
    ax.set_xlabel("R^2 of the single-exponential fit"); ax.set_ylabel("fits"); ax.set_title(f"fit quality: {rej:.0%} rejected (R^2 < {R2_MIN})", fontsize=9)
    for a in axes.ravel():
        _style_axes(a)
    fig.suptitle(f"Test B: single-exponential tail fits from rail exit + 2 ms to +800 ms; predicted DSP tau = {tau_dsp_ms:.1f} ms (dashed red); stim contacts excluded", fontsize=10)
    finish_figure(fig, build_caption(ctx))
    return fig


__all__ = ["R2_MIN", "figure_tau", "fit_run_tails", "invariance_model", "tau_summary"]
