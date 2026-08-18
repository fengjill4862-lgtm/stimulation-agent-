"""Per-trial metrics on blanked-then-filtered epochs (spec sections 5, 6, 7.1).

Every function returns long-form pandas frames keyed by ``event_number`` so
that baseline / post windows are always paired by key (never by position).
Spectral quantities use full-rate data; nothing here touches display
envelopes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stim_analysis.config import Band
from stim_analysis.epoch import FilterDesign, filter_epochs, window_slice


def cycles_in_window(band: Band, window_ms: tuple[float, float]) -> float:
    """How many cycles of the band's lowest frequency fit in the window."""
    return (window_ms[1] - window_ms[0]) * 1e-3 * band.low_hz


def band_power_per_trial(
    blanked: np.ndarray,
    t_ms: np.ndarray,
    bands: Sequence[Band],
    windows: Mapping[str, tuple[float, float]],
    designs: Mapping[str, FilterDesign],
    *,
    event_numbers: np.ndarray | None = None,
    core: slice | None = None,
) -> pd.DataFrame:
    """Mean squared band-filtered voltage per (event, band, window).

    ``blanked`` is (E, S) float64, padded; each band is filtered from the
    blanked padded epoch and the padding is trimmed via ``core`` afterwards.
    """
    x = np.asarray(blanked, dtype=np.float64)
    n_epochs = x.shape[0]
    numbers = (
        np.asarray(event_numbers, dtype=np.int64)
        if event_numbers is not None
        else np.arange(1, n_epochs + 1, dtype=np.int64)
    )
    core = core if core is not None else slice(0, x.shape[1])
    t_core = t_ms[core]
    rows: list[dict[str, object]] = []
    for band in bands:
        design = designs[band.name]
        filtered = filter_epochs(x, design)[:, core]
        power = np.square(filtered)
        for window_name, (start_ms, end_ms) in windows.items():
            if not (np.isfinite(start_ms) and np.isfinite(end_ms)) or end_ms <= start_ms:
                continue
            sl = window_slice(t_core, start_ms, end_ms)
            if sl.stop <= sl.start:
                continue
            means = power[:, sl].mean(axis=1)
            cycles = cycles_in_window(band, (start_ms, end_ms))
            for index in range(n_epochs):
                rows.append(
                    {
                        "event_number": int(numbers[index]),
                        "band": band.name,
                        "window": window_name,
                        "window_start_ms": float(start_ms),
                        "window_end_ms": float(end_ms),
                        "power_uV2": float(means[index]),
                        "n_samples": int(sl.stop - sl.start),
                        "cycles_in_window": cycles,
                    }
                )
    return pd.DataFrame(rows)


def response_amplitude_per_trial(
    broadband: np.ndarray,
    t_ms: np.ndarray,
    windows: Mapping[str, tuple[float, float]],
    *,
    event_numbers: np.ndarray | None = None,
    core: slice | None = None,
) -> pd.DataFrame:
    """RMS and peak |amplitude| of the broadband-filtered epoch per window."""
    x = np.asarray(broadband, dtype=np.float64)
    n_epochs = x.shape[0]
    numbers = (
        np.asarray(event_numbers, dtype=np.int64)
        if event_numbers is not None
        else np.arange(1, n_epochs + 1, dtype=np.int64)
    )
    core = core if core is not None else slice(0, x.shape[1])
    t_core = t_ms[core]
    data = x[:, core]
    rows: list[dict[str, object]] = []
    for window_name, (start_ms, end_ms) in windows.items():
        if not (np.isfinite(start_ms) and np.isfinite(end_ms)) or end_ms <= start_ms:
            continue
        sl = window_slice(t_core, start_ms, end_ms)
        if sl.stop <= sl.start:
            continue
        segment = data[:, sl]
        rms = np.sqrt(np.mean(np.square(segment), axis=1))
        peak = np.max(np.abs(segment), axis=1)
        for index in range(n_epochs):
            rows.append(
                {
                    "event_number": int(numbers[index]),
                    "window": window_name,
                    "window_start_ms": float(start_ms),
                    "window_end_ms": float(end_ms),
                    "rms_uV": float(rms[index]),
                    "peak_abs_uV": float(peak[index]),
                }
            )
    return pd.DataFrame(rows)


def paired_change_db(
    long_frame: pd.DataFrame,
    *,
    post: str = "early",
    base: str = "baseline",
    value: str = "power_uV2",
    keys: Sequence[str] = ("event_number", "band"),
) -> pd.DataFrame:
    """10 log10(post / base) per key; rows exist only where BOTH windows exist."""
    if long_frame.empty:
        return pd.DataFrame(columns=[*keys, f"{value}_{base}", f"{value}_{post}", "base_value", "post_value", "base_window", "post_window", "db"])
    a = long_frame[long_frame["window"] == base][[*keys, value]].rename(columns={value: f"{value}_{base}"})
    b = long_frame[long_frame["window"] == post][[*keys, value]].rename(columns={value: f"{value}_{post}"})
    merged = a.merge(b, on=list(keys), how="inner")
    merged["base_value"] = merged[f"{value}_{base}"]
    merged["post_value"] = merged[f"{value}_{post}"]
    merged["base_window"] = base
    merged["post_window"] = post
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["db"] = 10.0 * np.log10(merged["post_value"] / merged["base_value"])
    merged.loc[~np.isfinite(merged["db"]), "db"] = np.nan
    return merged


def first_vs_last(frame: pd.DataFrame, value: str, n: int, *, by: Sequence[str] = ()) -> pd.DataFrame:
    """Median of the first n vs last n trials (by event_number) per group."""
    rows: list[dict[str, object]] = []
    groups = frame.groupby(list(by)) if by else [((), frame)]
    for key, group in groups:
        ordered = group.sort_values("event_number")
        if len(ordered) < 2 * n:
            first = ordered.iloc[: max(1, len(ordered) // 2)]
            last = ordered.iloc[max(1, len(ordered) // 2) :]
        else:
            first = ordered.iloc[:n]
            last = ordered.iloc[-n:]
        row: dict[str, object] = {}
        if by:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(by, key_tuple)))
        row.update(
            {
                "n_first": int(len(first)),
                "n_last": int(len(last)),
                f"{value}_first_median": float(np.nanmedian(first[value])) if len(first) else float("nan"),
                f"{value}_last_median": float(np.nanmedian(last[value])) if len(last) else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def noise_floor_table(
    baseline_amplitude: pd.DataFrame,
    baseline_power: pd.DataFrame,
    impedance_kohm: Mapping[str, float],
    *,
    window: str = "late",
) -> pd.DataFrame:
    """Per-channel noise floor from the no-stim run (spec section 6).

    Uses the broadband RMS and per-band power of the no-stim pseudo-epochs;
    ``rms_per_sqrt_kohm`` normalises by the thermal-noise expectation.
    """
    rows: list[dict[str, object]] = []
    if baseline_amplitude.empty:
        return pd.DataFrame(rows)
    amp = baseline_amplitude[baseline_amplitude["window"] == window]
    for channel, group in amp.groupby("channel"):
        z = float(impedance_kohm.get(channel, float("nan")))
        rms = float(np.nanmedian(group["rms_uV"]))
        row: dict[str, object] = {
            "channel": channel,
            "impedance_kohm": z,
            "n_pseudo_epochs": int(len(group)),
            "broadband_rms_uV_median": rms,
            "broadband_rms_uV_q25": float(np.nanpercentile(group["rms_uV"], 25)),
            "broadband_rms_uV_q75": float(np.nanpercentile(group["rms_uV"], 75)),
            "rms_per_sqrt_kohm": rms / np.sqrt(z) if np.isfinite(z) and z > 0 else float("nan"),
        }
        if not baseline_power.empty:
            pw = baseline_power[(baseline_power["channel"] == channel) & (baseline_power["window"] == window)]
            for band, sub in pw.groupby("band"):
                row[f"power_uV2_{band}_median"] = float(np.nanmedian(sub["power_uV2"]))
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "band_power_per_trial",
    "cycles_in_window",
    "first_vs_last",
    "noise_floor_table",
    "paired_change_db",
    "response_amplitude_per_trial",
]
