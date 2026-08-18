"""Test F: remove the filter offline and re-measure recovery.

F1 -- fit and subtract: remove each epoch's fitted exponential (Test B) from
     rail exit onward, re-run compute_recovery.
F2 -- inverse filter: apply the exact inverse of the verified DSP recursion to
     the raw continuous channel, follow with a gentle causal 0.1 Hz high-pass
     (the inverse restores DC drift), re-epoch, re-run compute_recovery.
     Exact only where the recorded signal was not clamped: during a rail the
     DSP's input is unknown, so railed epochs are a lower bound and are
     reported separately from non-railed ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import baseline_stats, extract_epochs, gap_starts, window_slice
from stim_analysis.recovery import compute_recovery
from stim_analysis.validate import railed_mask
from filter_diag.common import DiagRun, gentle_highpass, intan_dsp_inverse
from scipy import signal


def f1_subtract(run: DiagRun, fits_run: pd.DataFrame, cfg: AnalysisConfig, *, examples: dict | None = None, example_key: str | None = None) -> pd.DataFrame:
    """Subtract A exp(-(t - t0)/tau) + C from t >= t0 (t0 = exit + 2 ms) per epoch; recompute recovery."""
    rows: list[pd.DataFrame] = []
    t_ms = run.epochs.t_ms
    core = run.epochs.core
    fs = run.epochs.sample_rate_hz
    kept_numbers = run.epochs.event_numbers[run.epochs.kept]
    fits = fits_run.set_index(["channel", "event_number"])
    for channel in run.channels:
        if channel not in run.baseline_mean:
            continue
        ep = run.channel_epochs(channel, centred=True)
        cleaned = ep.copy()
        used = np.zeros(ep.shape[0], dtype=bool)
        for i, number in enumerate(kept_numbers):
            key = (channel, int(number))
            if key not in fits.index:
                continue
            f = fits.loc[key]
            if not bool(f["fit_ok"]):
                continue
            t0 = float(f["exit_ms"]) + 2.0
            sl = window_slice(t_ms, t0, float("inf"))
            tt = t_ms[sl] - t0
            cleaned[i, sl] = ep[i, sl] - (float(f["A_uV"]) * np.exp(-tt / float(f["tau_ms"])) + float(f["C_uV"]))
            used[i] = True
        sd = run.baseline_sd[channel]
        rail = run.rails.get(channel)
        railed = np.stack([railed_mask(row, rail, cfg, fs) for row in run.channel_epochs(channel, centred=False)]) if (rail is not None and rail.is_railed) else None
        frame = compute_recovery(cleaned, t_ms, fs, sd, cfg, railed=railed, core=core, event_numbers=kept_numbers)
        frame.insert(0, "channel", channel)
        frame.insert(0, "run_id", run.run_id)
        frame.insert(0, "session", run.session)
        frame["method"] = "F1_fit_subtract"
        frame["fit_used"] = used
        frame["is_stim_contact"] = channel == run.stim_channel
        rows.append(frame)
        if examples is not None and example_key is not None and used.any() and f"{example_key}:{channel}" not in examples:
            i = int(np.flatnonzero(used)[0])
            examples[f"{example_key}:{channel}"] = {"t_ms": t_ms[core], "before": ep[i, core].copy(), "after_f1": cleaned[i, core].copy(), "channel": channel, "session": run.session, "run_id": run.run_id, "amplitude_uA": run.amplitude_uA}
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def f2_inverse(run: DiagRun, raw_uV: np.ndarray, channels: list[str], onset_samples: np.ndarray, event_numbers: np.ndarray, timestamps: np.ndarray, cfg: AnalysisConfig, *, k: int, gentle_hz: float = 0.1, freeze_at_full_scale: bool = True, examples: dict | None = None, example_key: str | None = None) -> pd.DataFrame:
    """Inverse DSP on the continuous channel, gentle causal HP, re-epoch, recompute recovery."""
    fs = run.epochs.sample_rate_hz
    rows: list[pd.DataFrame] = []
    restored = np.empty_like(raw_uV, dtype=np.float64)
    b, a = signal.butter(1, gentle_hz, btype="highpass", fs=fs)
    for c_index, channel in enumerate(channels):
        x = raw_uV[c_index].astype(np.float64)
        inv = intan_dsp_inverse(x, fs, k=k, initial_mean=0.0, freeze_at_full_scale=freeze_at_full_scale)
        zi = signal.lfilter_zi(b, a) * inv[0]
        y, _ = signal.lfilter(b, a, inv, zi=zi)
        restored[c_index] = y
    epochs = extract_epochs(restored, fs, onset_samples, event_numbers, cfg, run_id=run.run_id, channels=channels, gap_sample_starts=gap_starts(timestamps))
    kept_numbers = epochs.event_numbers[epochs.kept]
    for c_index, channel in enumerate(channels):
        if channel not in run.baseline_mean:
            continue
        ep = epochs.raw[c_index][epochs.kept].astype(np.float64)
        mean, sd = baseline_stats(ep, epochs.t_ms, cfg.baseline_ms)
        centred = ep - mean[:, None]
        # railed samples in the ORIGINAL recording (the reconstruction is a lower bound there)
        rail = run.rails.get(channel)
        orig = run.channel_epochs(channel, centred=False)
        railed = np.stack([railed_mask(row, rail, cfg, fs) for row in orig]) if (rail is not None and rail.is_railed) else None
        frame = compute_recovery(centred, epochs.t_ms, fs, sd, cfg, railed=railed, core=epochs.core, event_numbers=kept_numbers)
        frame.insert(0, "channel", channel)
        frame.insert(0, "run_id", run.run_id)
        frame.insert(0, "session", run.session)
        frame["method"] = "F2_inverse_dsp_freeze" if freeze_at_full_scale else "F2_inverse_dsp_track"
        frame["epoch_was_railed"] = (railed[:, epochs.core].any(axis=1) if railed is not None else np.zeros(ep.shape[0], dtype=bool))
        frame["is_stim_contact"] = channel == run.stim_channel
        rows.append(frame)
        if examples is not None and example_key is not None:
            key = f"{example_key}:{channel}"
            if key in examples and "after_f2" not in examples[key]:
                examples[key]["after_f2"] = centred[0, epochs.core].copy()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def before_after_table(before: pd.DataFrame, f1: pd.DataFrame, f2: pd.DataFrame, cfg: AnalysisConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Median recovery before vs after F1 / F2 per session x channel x amplitude (stim contacts flagged)."""
    from filter_diag.common import bootstrap_median_ci

    rows: list[dict[str, object]] = []
    keys = ["session", "run_id", "channel"]
    info = before.groupby(keys).agg(amplitude_uA=("amplitude_uA", "first"), is_stim_contact=("is_stim_contact", "first"), n=("recovery_ms", "size")).reset_index()
    for _, r in info.iterrows():
        sel = (before["session"] == r["session"]) & (before["run_id"] == r["run_id"]) & (before["channel"] == r["channel"])
        b_vals = before[sel]["recovery_ms"].to_numpy(dtype=float)
        row = {"session": r["session"], "run_id": r["run_id"], "channel": r["channel"], "amplitude_uA": r["amplitude_uA"], "is_stim_contact": bool(r["is_stim_contact"]), "n": int(r["n"])}
        med, lo, hi = bootstrap_median_ci(b_vals, cfg.bootstrap_n, rng)
        row.update({"before_median_ms": med, "before_ci_low": lo, "before_ci_high": hi})
        for name, frame in (("f1", f1), ("f2", f2)):
            if frame.empty:
                continue
            s2 = (frame["session"] == r["session"]) & (frame["run_id"] == r["run_id"]) & (frame["channel"] == r["channel"])
            sub = frame[s2]
            if name == "f1":
                sub = sub[sub["fit_used"]]
            vals = sub["recovery_ms"].to_numpy(dtype=float)
            med2, lo2, hi2 = bootstrap_median_ci(vals, cfg.bootstrap_n, rng)
            row.update({f"{name}_median_ms": med2, f"{name}_ci_low": lo2, f"{name}_ci_high": hi2, f"{name}_n": int(vals.size), f"{name}_fraction_of_before": med2 / med if (np.isfinite(med) and med > 0) else float("nan")})
            if name == "f2" and "epoch_was_railed" in sub:
                nr = sub[~sub["epoch_was_railed"]]["recovery_ms"].to_numpy(dtype=float)
                rr = sub[sub["epoch_was_railed"]]["recovery_ms"].to_numpy(dtype=float)
                row.update({"f2_nonrailed_median_ms": float(np.median(nr)) if nr.size else float("nan"), "f2_nonrailed_n": int(nr.size), "f2_railed_median_ms": float(np.median(rr)) if rr.size else float("nan"), "f2_railed_n": int(rr.size)})
        rows.append(row)
    return pd.DataFrame(rows)


def figure_examples(examples: dict, cfg: AnalysisConfig, ctx, *, max_panels: int = 8) -> object:
    from stim_analysis.figures import FOOTER_IN, _style_axes, build_caption, finish_figure, plt

    keys = list(examples)[:max_panels]
    n = max(1, len(keys))
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows + FOOTER_IN + 0.8), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=1 - 0.7 / (3.0 * n_rows + FOOTER_IN + 0.8), bottom=(FOOTER_IN + 0.45) / (3.0 * n_rows + FOOTER_IN + 0.8), hspace=0.45, wspace=0.28)
    for ax, key in zip(axes.ravel(), keys):
        ex = examples[key]
        t = ex["t_ms"]
        w = (t >= -100) & (t <= 800)
        ax.plot(t[w], np.clip(ex["before"][w], -7000, 7000), color="0.2", lw=0.8, label="recorded (centred)")
        if "after_f1" in ex:
            ax.plot(t[w], np.clip(ex["after_f1"][w], -7000, 7000), color="#d62728", lw=0.8, alpha=0.9, label="F1: fitted exponential subtracted")
        if "after_f2" in ex:
            ax.plot(t[w], np.clip(ex["after_f2"][w], -7000, 7000), color="#2ca02c", lw=0.8, alpha=0.9, label="F2: inverse DSP + 0.1 Hz HP")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlim(-100, 800); ax.set_ylim(-1500, 1500)
        ax.set_title(f"{ex['session']} {ex['run_id']} {ex['channel']} {ex['amplitude_uA']:g} uA", fontsize=8)
        ax.set_xlabel("time from pulse (ms)", fontsize=7); ax.set_ylabel("uV (y-limits +/-1500)", fontsize=7)
        ax.legend(fontsize=5.8, frameon=False)
        _style_axes(ax)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Test F: example epochs before and after removing the filter (same axes; values beyond +/-1500 uV are off-scale, not clipped in the data)", fontsize=9.5)
    finish_figure(fig, build_caption(ctx))
    return fig


__all__ = ["before_after_table", "f1_subtract", "f2_inverse", "figure_examples"]
