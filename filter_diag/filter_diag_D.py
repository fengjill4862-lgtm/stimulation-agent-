"""Test D: live vs post-mortem, matched conditions (same current, same width, single pulses)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from filter_diag.common import bootstrap_median_ci


def _diff_ci(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """median(b) - median(a) with independent bootstrap CI (unpaired: different animals)."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    ia = rng.integers(0, a.size, size=(n, a.size))
    ib = rng.integers(0, b.size, size=(n, b.size))
    draws = np.median(b[ib], axis=1) - np.median(a[ia], axis=1)
    return float(np.median(b) - np.median(a)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def live_vs_dead(fits: pd.DataFrame, cfg: AnalysisConfig, rng: np.random.Generator, *, live: str = "live", dead: str = "dead") -> pd.DataFrame:
    """Per matched (amplitude, phase): recovery, tau_fit, rail duration for live and dead + differences."""
    rows: list[dict[str, object]] = []
    base = fits[~fits["is_stim_contact"]]
    live_conds = set(zip(base[base["session"] == live]["amplitude_uA"].round(1), base[base["session"] == live]["phase_us"].round(0)))
    dead_conds = set(zip(base[base["session"] == dead]["amplitude_uA"].round(1), base[base["session"] == dead]["phase_us"].round(0)))
    matched = sorted(live_conds & dead_conds)
    for amp, phase in matched:
        a = base[(base["session"] == live) & (base["amplitude_uA"].round(1) == amp) & (base["phase_us"].round(0) == phase)]
        b = base[(base["session"] == dead) & (base["amplitude_uA"].round(1) == amp) & (base["phase_us"].round(0) == phase)]
        for metric, col, sub_a, sub_b in (
            ("recovery_ms", "recovery_ms", a, b),
            ("tau_fit_ms", "tau_ms", a[a["fit_ok"]], b[b["fit_ok"]]),
            ("rail_ms", "rail_ms", a, b),
        ):
            va, vb = sub_a[col].to_numpy(dtype=float), sub_b[col].to_numpy(dtype=float)
            med_a, lo_a, hi_a = bootstrap_median_ci(va, cfg.bootstrap_n, rng)
            med_b, lo_b, hi_b = bootstrap_median_ci(vb, cfg.bootstrap_n, rng)
            diff, dlo, dhi = _diff_ci(va, vb, cfg.bootstrap_n, rng)
            rows.append(
                {
                    "amplitude_uA": amp, "phase_us": phase, "metric": metric,
                    "n_live": int(np.isfinite(va).sum()), "n_dead": int(np.isfinite(vb).sum()),
                    "live_median": med_a, "live_ci_low": lo_a, "live_ci_high": hi_a,
                    "dead_median": med_b, "dead_ci_low": lo_b, "dead_ci_high": hi_b,
                    "dead_minus_live": diff, "diff_ci_low": dlo, "diff_ci_high": dhi,
                    "ratio_dead_over_live": med_b / med_a if (np.isfinite(med_a) and med_a > 0) else float("nan"),
                    "indistinguishable_95ci": bool(np.isfinite(dlo) and np.isfinite(dhi) and dlo <= 0.0 <= dhi),
                    "live_impedance_kohm_median": float(np.nanmedian(a["impedance_kohm"])) if len(a) else float("nan"),
                    "dead_impedance_kohm_median": float(np.nanmedian(b["impedance_kohm"])) if len(b) else float("nan"),
                    "note": "unpaired: different animals, different impedances and placement",
                }
            )
    return pd.DataFrame(rows)


def figure_live_dead(table: pd.DataFrame, fits: pd.DataFrame, cfg: AnalysisConfig, ctx) -> object:
    from stim_analysis.figures import FOOTER_IN, _style_axes, build_caption, finish_figure, plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8 + FOOTER_IN))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=(FOOTER_IN + 0.5) / (4.8 + FOOTER_IN), wspace=0.3)
    lims = {"recovery_ms": cfg.lim_recovery_ms, "tau_fit_ms": (5.0, 3000.0), "rail_ms": (0.1, 100.0)}
    titles = {"recovery_ms": "recovery time", "tau_fit_ms": "tau_fit", "rail_ms": "rail duration"}
    rng = np.random.default_rng(0)
    base = fits[~fits["is_stim_contact"]]
    for ax, metric in zip(axes, ("recovery_ms", "tau_fit_ms", "rail_ms")):
        col = "recovery_ms" if metric == "recovery_ms" else ("tau_ms" if metric == "tau_fit_ms" else "rail_ms")
        lim = lims[metric]
        for session, color, marker, dx in (("live", "#1f77b4", "o", 0.94), ("dead", "#2ca02c", "s", 1.06)):
            sub = base[base["session"] == session]
            if metric == "tau_fit_ms":
                sub = sub[sub["fit_ok"]]
            for amp, s2 in sub.groupby("amplitude_uA"):
                vals = s2[col].to_numpy(dtype=float)
                if metric == "rail_ms":
                    vals = vals[vals > 0]
                if vals.size == 0:
                    continue
                x = amp * dx * (1 + rng.uniform(-0.03, 0.03, vals.size))
                ax.scatter(x, np.clip(vals, *lim), s=5, alpha=0.25, color=color, marker=marker, linewidths=0)
                ax.scatter([amp * dx], [np.clip(np.median(vals), *lim)], s=30, color="black", marker=marker, zorder=4)
        rows = table[table["metric"] == metric]
        for _, r in rows.iterrows():
            ax.annotate(f"d-l {r['dead_minus_live']:+.0f}\n[{r['diff_ci_low']:.0f},{r['diff_ci_high']:.0f}]", (r["amplitude_uA"], lim[1] * 0.55), fontsize=5.8, ha="center", color="0.3")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lim); ax.set_xlim(cfg.lim_amplitude_uA)
        ax.set_xlabel("current (uA, log)"); ax.set_ylabel(f"{titles[metric]} (ms, log)"); ax.set_title(f"{titles[metric]}: live (o, blue) vs post-mortem (s, green); black = median", fontsize=9)
        _style_axes(ax)
    fig.suptitle("Test D: live vs post-mortem at matched current and width (single pulses; unpaired, different animals; stim contacts excluded)", fontsize=10)
    finish_figure(fig, build_caption(ctx))
    return fig


__all__ = ["figure_live_dead", "live_vs_dead"]
