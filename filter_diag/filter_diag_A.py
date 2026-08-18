"""Test A: read the actual filter settings from every RHS header, per run, per session."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from filter_diag.common import SessionSelection, dsp_k_from_cutoff, dsp_tau_s


def settings_table(selections: list[SessionSelection]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sel in selections:
        included_ids = {v.run_id for v in sel.included}
        for v in sel.all_validations:
            record = sel.records.get(v.run_id)
            h = record.header if record is not None else None
            fs = float(h.sample_rate_hz) if h else float("nan")
            fc = float(h.actual_dsp_cutoff_hz) if h else float("nan")
            k = dsp_k_from_cutoff(fs, fc) if (h and np.isfinite(fc) and fc > 0) else None
            rows.append(
                {
                    "session": sel.session,
                    "run_id": v.run_id,
                    "run_folder": Path(v.run_folder).name,
                    "included": v.run_id in included_ids,
                    "exclusion_reason": v.exclusion_reason,
                    "block": v.block,
                    "amplitude_uA": v.amplitude_uA_data,
                    "phase_us": v.phase_us_data,
                    "sample_rate_hz": fs,
                    "dsp_enabled": bool(h.dsp_enabled) if h else None,
                    "desired_dsp_cutoff_hz": h.desired_dsp_cutoff_hz if h else float("nan"),
                    "actual_dsp_cutoff_hz": fc,
                    "dsp_k": k,
                    "tau_dsp_ms_from_cutoff": 1e3 / (2 * math.pi * fc) if (np.isfinite(fc) and fc > 0) else float("nan"),
                    "tau_dsp_ms_from_k": dsp_tau_s(fs, k) * 1e3 if k else float("nan"),
                    "desired_lower_bandwidth_hz": h.desired_lower_bandwidth_hz if h else float("nan"),
                    "actual_lower_bandwidth_hz": h.actual_lower_bandwidth_hz if h else float("nan"),
                    "tau_analog_ms": 1e3 / (2 * math.pi * h.actual_lower_bandwidth_hz) if (h and h.actual_lower_bandwidth_hz > 0) else float("nan"),
                    "actual_lower_settle_bandwidth_hz": h.actual_lower_settle_bandwidth_hz if h else float("nan"),
                    "desired_upper_bandwidth_hz": h.desired_upper_bandwidth_hz if h else float("nan"),
                    "actual_upper_bandwidth_hz": h.actual_upper_bandwidth_hz if h else float("nan"),
                    "notch_filter_mode": h.notch_filter_mode if h else None,
                    "amp_settle_mode": h.amp_settle_mode if h else None,
                    "charge_recovery_mode": h.charge_recovery_mode if h else None,
                    "post_amp_settle_us": record.settings.post_amp_settle_us if (record and record.settings) else float("nan"),
                    "enable_amp_settle": record.settings.enable_amp_settle if (record and record.settings) else None,
                    "stim_channel": v.stim_channel_data,
                }
            )
    return pd.DataFrame(rows)


def settings_summary(table: pd.DataFrame) -> dict[str, object]:
    """Are the DSP settings identical across runs and sessions? A difference would be a natural experiment."""
    if table.empty:
        return {"identical_dsp_across_sessions": None}
    ok = table[table["sample_rate_hz"].notna()]
    per_session = ok.groupby("session")["actual_dsp_cutoff_hz"].agg(lambda s: sorted(set(np.round(s.dropna(), 6))))
    all_cutoffs = sorted(set(np.round(ok["actual_dsp_cutoff_hz"].dropna(), 6)))
    return {
        "dsp_cutoffs_by_session_hz": {k: [float(x) for x in v] for k, v in per_session.items()},
        "all_dsp_cutoffs_hz": [float(x) for x in all_cutoffs],
        "identical_dsp_across_sessions": len(all_cutoffs) == 1,
        "dsp_enabled_everywhere": bool(ok["dsp_enabled"].all()),
        "sample_rates_hz": sorted(set(ok["sample_rate_hz"].dropna())),
        "tau_dsp_ms": float(ok["tau_dsp_ms_from_k"].median()),
        "analog_lower_bandwidth_hz": sorted(set(np.round(ok["actual_lower_bandwidth_hz"].dropna(), 4))),
        "natural_experiment": "no: identical DSP cutoff in every run of both sessions" if len(all_cutoffs) == 1 else f"YES: cutoffs differ ({all_cutoffs}) -- recovery should scale with tau",
    }


__all__ = ["settings_summary", "settings_table"]
