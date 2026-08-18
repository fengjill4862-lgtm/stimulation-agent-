"""Artifact recovery time per trial (spec section 4) -- the gating analysis.

Computed on raw, baseline-mean-subtracted, unfiltered epochs:

    baseline_sd   = SD(signal[-500 ms : -50 ms])
    threshold     = max(k * baseline_sd, floor_uV)
    recovery_time = last t > 0 where |signal| > threshold, before the signal
                    stays below threshold for >= quiet_ms
    rail_ms       = time the signal sits at the empirical rail level after 0

Then per (run, channel) condition: quantiles, the derived post-window start,
which trials survive, and the verdict (early_ok / late_only / unusable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig
from stim_analysis.epoch import window_slice


def compute_recovery(
    centred: np.ndarray,
    t_ms: np.ndarray,
    sample_rate_hz: float,
    baseline_sd: np.ndarray,
    cfg: AnalysisConfig,
    *,
    railed: np.ndarray | None = None,
    core: slice | None = None,
    event_numbers: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-epoch recovery metrics for one channel of one run.

    ``centred`` is (E, S) raw minus the per-epoch baseline mean; ``railed`` is
    an optional (E, S) boolean mask of samples at the rail level.
    """
    x = np.asarray(centred, dtype=np.float64)
    n_epochs, n_samples = x.shape
    sd = np.asarray(baseline_sd, dtype=np.float64)
    core = core if core is not None else slice(0, n_samples)
    t_core = t_ms[core]
    post = window_slice(t_core, 0.0, float("inf"))
    post_abs = slice(core.start + post.start, core.start + post.stop)
    t_post = t_ms[post_abs]
    epoch_end_ms = float(t_core[-1]) if t_core.size else float(cfg.epoch_ms[1])
    quiet = max(1, int(round(cfg.quiet_ms * 1.0e-3 * sample_rate_hz)))
    numbers = (
        np.asarray(event_numbers, dtype=np.int64)
        if event_numbers is not None
        else np.arange(1, n_epochs + 1, dtype=np.int64)
    )

    rows: list[dict[str, object]] = []
    for index in range(n_epochs):
        threshold = max(cfg.threshold_k * float(sd[index]), cfg.threshold_floor_uV)
        if not np.isfinite(threshold):
            threshold = cfg.threshold_floor_uV
        segment = x[index, post_abs]
        above = np.abs(segment) > threshold
        n_post = segment.size
        recovery_ms = 0.0
        quiet_start_ms = 0.0
        censored = False
        if n_post == 0:
            censored = True
            recovery_ms = epoch_end_ms
            quiet_start_ms = float("nan")
        elif not np.any(above):
            recovery_ms = 0.0
            quiet_start_ms = float(t_post[0])
        else:
            if n_post >= quiet:
                # number of "above" samples in each length-`quiet` window starting at i
                csum = np.concatenate(([0], np.cumsum(above)))
                counts = csum[quiet:] - csum[:-quiet]  # length n_post - quiet + 1
                quiet_starts = np.flatnonzero(counts == 0)
            else:
                quiet_starts = np.zeros(0, dtype=np.int64)
            if quiet_starts.size == 0:
                censored = True
                recovery_ms = epoch_end_ms
                quiet_start_ms = float("nan")
            else:
                tau = int(quiet_starts[0])
                quiet_start_ms = float(t_post[tau])
                before = np.flatnonzero(above[:tau])
                recovery_ms = float(t_post[before[-1]]) if before.size else 0.0
        peak_index = int(np.argmax(np.abs(segment))) if n_post else 0
        peak_abs = float(np.abs(segment[peak_index])) if n_post else float("nan")
        peak_ms = float(t_post[peak_index]) if n_post else float("nan")
        if railed is not None:
            rail_samples = int(np.count_nonzero(np.asarray(railed)[index, post_abs]))
        else:
            rail_samples = 0
        rows.append(
            {
                "event_number": int(numbers[index]),
                "baseline_sd_uV": float(sd[index]),
                "threshold_uV": float(threshold),
                "recovery_ms": float(recovery_ms),
                "censored": bool(censored),
                "quiet_start_ms": quiet_start_ms,
                "rail_ms": rail_samples * 1.0e3 / sample_rate_hz,
                "peak_abs_uV": peak_abs,
                "peak_ms": peak_ms,
                "n_above_samples": int(np.count_nonzero(above)),
            }
        )
    return pd.DataFrame(rows)


def mark_baseline_contamination(
    trials: pd.DataFrame, onset_s_by_event: dict[int, float], cfg: AnalysisConfig
) -> pd.DataFrame:
    """Flag epochs whose baseline window overlaps the previous pulse's recovery.

    With a ~1 s inter-pulse interval, the -500..-50 ms baseline of pulse n is
    the +500..+950 ms tail of pulse n-1 (spec section 5). Contaminated when
    ``recovery(n-1) > IPI - |baseline_start|``.
    """
    out = trials.copy()
    out["baseline_contaminated"] = False
    if out.empty or "recovery_ms" not in out:
        return out
    order = sorted(onset_s_by_event.items())
    previous: dict[int, tuple[int, float]] = {}
    for (prev_event, prev_onset), (event, onset) in zip(order[:-1], order[1:]):
        previous[event] = (prev_event, (onset - prev_onset) * 1.0e3)
    recovery_by_event = dict(zip(out["event_number"], out["recovery_ms"]))
    contaminated = []
    for event in out["event_number"]:
        item = previous.get(int(event))
        if item is None:
            contaminated.append(False)
            continue
        prev_event, ipi_ms = item
        prev_recovery = recovery_by_event.get(prev_event, float("nan"))
        limit = ipi_ms - abs(cfg.baseline_ms[0])
        contaminated.append(bool(np.isfinite(prev_recovery) and prev_recovery > limit))
    out["baseline_contaminated"] = contaminated
    return out


def condition_windows(trials: pd.DataFrame, cfg: AnalysisConfig, floor_ms_by_run: dict[str, float]) -> pd.DataFrame:
    """Per (run_id, channel): recovery quantiles, post window, verdict."""
    rows: list[dict[str, object]] = []
    if trials.empty:
        return pd.DataFrame(rows)
    epoch_end = float(cfg.epoch_ms[1])
    for (run_id, channel), group in trials.groupby(["run_id", "channel"], sort=False):
        recovery = group["recovery_ms"].to_numpy(dtype=float)
        censored = group["censored"].to_numpy(dtype=bool)
        contaminated = (
            group["baseline_contaminated"].to_numpy(dtype=bool)
            if "baseline_contaminated" in group
            else np.zeros(recovery.size, dtype=bool)
        )
        n = recovery.size
        floor = float(floor_ms_by_run.get(run_id, 1.0))
        q = float(np.quantile(recovery, cfg.recovery_quantile)) if n else float("nan")
        p50 = float(np.median(recovery)) if n else float("nan")
        p25, p75 = (np.percentile(recovery, [25, 75]) if n else (float("nan"), float("nan")))
        p90 = float(np.percentile(recovery, 90)) if n else float("nan")
        vmax = float(np.max(recovery)) if n else float("nan")
        post_start = max(q + cfg.blank_margin_ms, floor) if np.isfinite(q) else float("nan")
        post_start = min(post_start, epoch_end) if np.isfinite(post_start) else post_start
        post_end = min(post_start + cfg.post_length_ms, epoch_end) if np.isfinite(post_start) else float("nan")
        blank_end = recovery + cfg.blank_margin_ms
        usable = (~censored) & (~contaminated)
        retained_early = usable & (blank_end <= post_start) if np.isfinite(post_start) else np.zeros(n, bool)
        retained_late = usable & (blank_end < cfg.late_ms[0])
        n_retained = int(np.count_nonzero(retained_early))
        early_possible = bool(
            np.isfinite(post_start)
            and post_start <= cfg.post_start_max_ms
            and post_end > post_start
            and n_retained >= cfg.min_trials
        )
        late_possible = bool(np.isfinite(q) and (q + cfg.blank_margin_ms) < cfg.late_ms[0] and int(np.count_nonzero(retained_late)) >= cfg.min_trials)
        if early_possible and p50 <= cfg.early_verdict_ms:
            verdict = "early_ok"
        elif late_possible:
            verdict = "late_only"
        else:
            verdict = "unusable"
        rows.append(
            {
                "run_id": run_id,
                "channel": channel,
                "n_trials": int(n),
                "n_censored": int(np.count_nonzero(censored)),
                "n_baseline_contaminated": int(np.count_nonzero(contaminated)),
                "median_recovery_ms": p50,
                "q25_recovery_ms": float(p25),
                "q75_recovery_ms": float(p75),
                "p90_recovery_ms": p90,
                "max_recovery_ms": vmax,
                "recovery_quantile_used": cfg.recovery_quantile,
                "quantile_recovery_ms": q,
                "median_rail_ms": float(np.median(group["rail_ms"])) if n else float("nan"),
                "hardware_floor_ms": floor,
                "min_blank_ms": post_start,
                "post_start_ms": post_start,
                "post_end_ms": post_end,
                "n_retained_early": n_retained,
                "n_rejected_early": int(n - n_retained),
                "n_retained_late": int(np.count_nonzero(retained_late)),
                "early_possible": early_possible,
                "late_possible": late_possible,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def mark_retained(trials: pd.DataFrame, windows: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Add retained_early / retained_late / reject_reason / blank_end_ms per trial."""
    out = trials.copy()
    if out.empty:
        for column in ("blank_end_ms", "post_start_ms", "post_end_ms", "retained_early", "retained_late", "reject_reason"):
            out[column] = []
        return out
    lookup = windows.set_index(["run_id", "channel"])[["post_start_ms", "post_end_ms"]] if not windows.empty else None
    post_start = np.full(len(out), np.nan)
    post_end = np.full(len(out), np.nan)
    if lookup is not None:
        keys = list(zip(out["run_id"], out["channel"]))
        for index, key in enumerate(keys):
            if key in lookup.index:
                post_start[index] = lookup.loc[key, "post_start_ms"]
                post_end[index] = lookup.loc[key, "post_end_ms"]
    out["blank_end_ms"] = out["recovery_ms"] + cfg.blank_margin_ms
    out["post_start_ms"] = post_start
    out["post_end_ms"] = post_end
    censored = out["censored"].to_numpy(dtype=bool)
    contaminated = (
        out["baseline_contaminated"].to_numpy(dtype=bool)
        if "baseline_contaminated" in out
        else np.zeros(len(out), dtype=bool)
    )
    usable = (~censored) & (~contaminated)
    early = usable & (out["blank_end_ms"].to_numpy() <= post_start)
    late = usable & (out["blank_end_ms"].to_numpy() < cfg.late_ms[0])
    out["retained_early"] = early
    out["retained_late"] = late
    reasons = np.where(
        censored, "censored",
        np.where(contaminated, "baseline_contaminated", np.where(early, "", "recovery_after_post_start")),
    )
    out["reject_reason"] = reasons
    return out


def recovery_summary_table(trials: pd.DataFrame, windows: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """table03: median recovery per (channel x amplitude x phase) with derived windows."""
    if windows.empty:
        return pd.DataFrame()
    info = runs.set_index("run_id")
    columns = ["block", "amplitude_uA_data", "phase_us_data", "charge_nC_per_phase", "included", "n_detected"]
    columns = [c for c in columns if c in info.columns]
    merged = windows.merge(info[columns], left_on="run_id", right_index=True, how="left")
    merged = merged.rename(columns={"amplitude_uA_data": "amplitude_uA", "phase_us_data": "phase_us"})
    front = ["channel", "amplitude_uA", "phase_us", "run_id", "block", "included"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[[c for c in front if c in merged.columns] + rest]
    return merged.sort_values(["channel", "amplitude_uA", "phase_us"], na_position="last").reset_index(drop=True)


__all__ = [
    "compute_recovery",
    "condition_windows",
    "mark_baseline_contamination",
    "mark_retained",
    "recovery_summary_table",
]
