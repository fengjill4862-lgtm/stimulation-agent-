"""Phase-3 self-test checks, called from ``stim_analysis.selftest``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from stim_analysis.config import AnalysisConfig, Band
from stim_analysis.epoch import design_bandpass
from stim_analysis.metrics import band_power_per_trial, paired_change_db, response_amplitude_per_trial
from stim_analysis.models import bootstrap_i50, compare_models, fit_linear, fit_linear_origin, fit_sigmoid, sigmoid
from stim_analysis.stats import (
    HAS_STATSMODELS,
    bootstrap_ci,
    bootstrap_paired_db_ci,
    clean_intervals,
    draw_fake_onsets,
    fit_cross_channel_model,
    lognormal_check,
    paired_frame,
)


def run(tmp: Path, c) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # ---- pairing: an epoch dropped from one window vanishes from both -----------------
    a = pd.DataFrame({"run_id": "r", "channel": "A", "event_number": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"run_id": "r", "channel": "A", "event_number": [1, 2, 4, 5], "y": [1.0, 2.0, 4.0, 5.0]})
    merged, dropped = paired_frame(a, b)
    c.ok(list(merged["event_number"]) == [1, 2, 4] and set(dropped["event_number"]) == {3, 5}, "paired_frame: inner join on keys, both dropped keys reported")
    long = pd.DataFrame(
        {
            "event_number": [1, 1, 2, 3, 3],
            "band": ["delta"] * 5,
            "window": ["baseline", "early", "baseline", "baseline", "early"],
            "power_uV2": [1.0, 10.0, 1.0, 2.0, 2.0],
        }
    )
    db = paired_change_db(long)
    c.ok(list(db["event_number"]) == [1, 3] and abs(db["db"].iloc[0] - 10.0) < 1e-9 and abs(db["db"].iloc[1]) < 1e-9, "paired_change_db: only events with both windows; 10 dB and 0 dB exact")

    # ---- bootstrap sanity ------------------------------------------------------------------
    values = rng.normal(5.0, 1.0, size=400)
    lo, hi = bootstrap_ci(values, np.mean, 1000, rng)
    sample_mean = float(values.mean())
    expected_width = 2 * 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    c.ok(lo < sample_mean < hi and abs((hi - lo) - expected_width) < 0.35 * expected_width, f"bootstrap_ci brackets the sample mean with ~1.96 SE half-width ({lo:.2f}, {hi:.2f}; width {hi - lo:.3f} vs {expected_width:.3f})")
    base = np.exp(rng.normal(0, 0.3, 300))
    post = base * 10 ** (0.3) * np.exp(rng.normal(0, 0.05, 300))  # +3 dB paired
    lo, hi = bootstrap_paired_db_ci(base, post, 1000, rng)
    db_pairs = 10 * np.log10(post / base)
    c.ok(lo < db_pairs.mean() < hi and abs((lo + hi) / 2 - 3.0) < 0.1 and (hi - lo) < 0.1, f"paired dB bootstrap CI centred on +3 dB, brackets the sample mean ({lo:.3f}, {hi:.3f})")

    # ---- band power / response amplitude on a known sinusoid --------------------------------
    fs = 30000.0
    t = np.arange(int(2.5 * fs)) / fs
    t_ms = (t - 1.1) * 1e3  # -1100 .. +1400 ms
    core = slice(int(0.5 * fs), int(2.0 * fs))
    x = 100.0 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz, 100 uV amplitude -> power 5000 uV^2 in alpha
    epochs = np.stack([x, x])
    bands = (Band("alpha", 8.0, 12.0), Band("gamma", 30.0, 80.0))
    designs = {band.name: design_bandpass(fs, band.low_hz, band.high_hz, 4, True) for band in bands}
    windows = {"baseline": (-500.0, -50.0), "early": (100.0, 400.0)}
    pw = band_power_per_trial(epochs, t_ms, bands, windows, designs, core=core)
    alpha = pw[(pw["band"] == "alpha") & (pw["window"] == "early")]["power_uV2"].iloc[0]
    gamma = pw[(pw["band"] == "gamma") & (pw["window"] == "early")]["power_uV2"].iloc[0]
    c.ok(abs(alpha - 5000.0) / 5000.0 < 0.05 and gamma < 50.0, f"band power: 10 Hz sinusoid -> alpha {alpha:.0f} uV^2 (expect 5000), gamma {gamma:.1f}")
    amp = response_amplitude_per_trial(epochs, t_ms, windows, core=core)
    rms = amp[amp["window"] == "early"]["rms_uV"].iloc[0]
    c.ok(abs(rms - 100 / np.sqrt(2)) < 3.0, f"response amplitude: RMS of 100 uV sinusoid = {rms:.1f} (expect 70.7)")

    # ---- shuffle helpers ---------------------------------------------------------------------
    onsets = np.arange(30000, 30000 * 20, 30000)
    ends = onsets + 6000
    intervals = clean_intervals(onsets, ends, 30000 * 21, guard_samples=600)
    c.ok(len(intervals) == 20 and all(b - a == 30000 - 6000 - 1200 for a, b in intervals[1:-1]) and intervals[0] == (600, 29400), "clean_intervals: gaps between blank end and next pulse minus guards")
    fake = draw_fake_onsets(intervals, 15, rng, min_separation=1000)
    inside = all(any(a <= f < b for a, b in intervals) for f in fake)
    c.ok(fake.size == 15 and inside and np.all(np.diff(fake) >= 1000), "draw_fake_onsets: inside clean intervals, separated")

    # ---- shuffle null: pure noise -> ~0 dB, but only with equal-length windows ----------------
    # Log-power estimates from windows of different length have different degrees
    # of freedom, so mean 10log10(post/base) is biased on pure noise. The pipeline
    # therefore pairs every post window with an equal-length baseline slice.
    unequal = {"baseline": (-500.0, -50.0), "early": (100.0, 400.0)}
    equal = {"baseline": (-350.0, -50.0), "early": (100.0, 400.0)}
    results = {}
    for label, wins in (("unequal", unequal), ("equal", equal)):
        n_ci_contains_zero = 0
        n_total = 0
        medians = []
        for _ in range(20):
            noise = rng.normal(0, 20, size=(40, t.size))
            pwn = band_power_per_trial(noise, t_ms, bands, wins, designs, core=core)
            for band in bands:
                sub = pwn[pwn["band"] == band.name]
                dbn = paired_change_db(sub, keys=("event_number", "band"))
                lo, hi = bootstrap_paired_db_ci(dbn["base_value"].to_numpy(), dbn["post_value"].to_numpy(), 300, rng)
                n_total += 1
                n_ci_contains_zero += int(lo <= 0.0 <= hi)
                medians.append(float(np.nanmean(dbn["db"])))
        results[label] = (n_ci_contains_zero, n_total, float(np.mean(medians)))
    ku, nu, mu = results["unequal"]
    ke, ne, me = results["equal"]
    c.ok(mu < -0.2, f"unequal windows (450 vs 300 ms) bias the mean dB on pure noise ({mu:+.2f} dB, {ku}/{nu} CIs contain 0) -- the reason for equal-length pairing")
    c.ok(ke / ne >= 0.85 and abs(me) < 0.15, f"equal-length windows: {ke}/{ne} paired dB CIs contain 0 on pure noise, mean {me:+.2f} dB")

    # ---- models: sigmoid recovered, artifact detected ------------------------------------------
    current = np.repeat([10, 20, 30, 50, 70, 100, 150, 200, 250], 40).astype(float)
    y_sig = sigmoid(current, 300.0, 80.0, 20.0) + rng.normal(0, 25, current.size)
    fits = [fit_linear(current, y_sig), fit_linear_origin(current, y_sig), fit_sigmoid(current, y_sig)]
    cmp_sig = compare_models(fits)
    sig = fits[2]
    lo, hi, frac = bootstrap_i50(current, y_sig, 300, rng, seed_fit=sig)
    c.ok(cmp_sig["preferred"] == "sigmoid" and abs(sig.params["i50"] - 80) < 8 and lo < 80 < hi and frac > 0.9, f"models: sigmoid preferred, I50 {sig.params['i50']:.1f} [{lo:.1f}, {hi:.1f}] (true 80), converged {frac:.2f}")
    y_art = 2.0 * current + rng.normal(0, 25, current.size)
    fits_art = [fit_linear(current, y_art), fit_linear_origin(current, y_art), fit_sigmoid(current, y_art)]
    cmp_art = compare_models(fits_art)
    c.ok(bool(cmp_art["artifact_candidate"]), f"models: pure linear-through-origin data flagged artifact_candidate (preferred {cmp_art['preferred']})")

    # ---- log-normal check --------------------------------------------------------------------
    lognormal = np.exp(rng.normal(3.0, 0.8, 500))
    check = lognormal_check(lognormal, "test")
    c.ok(check["recommended"] == "log" and check["shapiro_p_log"] > check["shapiro_p_raw"], f"lognormal_check recommends log for log-normal data (p_log {check['shapiro_p_log']:.3f} vs p_raw {check['shapiro_p_raw']:.1e})")

    # ---- cross-channel model -------------------------------------------------------------------
    rows = []
    for ch, z in zip("ABCDEFGH", np.geomspace(30, 300, 8)):
        for amp_ in (10, 20, 50, 100, 200):
            for _ in range(15):
                resp = 1.0 * np.log10(amp_) - 0.5 * np.log10(z) + rng.normal(0, 0.1) + {"A": 0.1, "B": -0.1}.get(ch, 0.0)
                rows.append({"channel": ch, "log_amplitude": np.log10(amp_), "log_impedance": np.log10(z), "log_response": resp})
    frame = pd.DataFrame(rows)
    fit = fit_cross_channel_model(frame, "log_response", ("log_amplitude", "log_impedance"), "channel", bootstrap_n=50, rng=rng)
    slope = float(fit[fit["term"] == "log_amplitude"]["estimate"].iloc[0])
    method = str(fit["method"].iloc[0])
    c.ok(abs(slope - 1.0) < 0.1, f"cross-channel model: amplitude slope {slope:.2f} (true 1.0) via {method}{'' if HAS_STATSMODELS else ' (statsmodels absent)'}")

    # ---- full pipeline, stage=all, synthetic session -----------------------------------------
    from stim_analysis import selftest as st
    from stim_analysis.pipeline import run_session

    fs = 30000.0
    parent = tmp / "session"
    parent.mkdir()
    names = ["A-024", "A-025", "A-026", "A-027"]
    n = 128 * 6000
    onsets = 30000 + np.arange(20) * 30001

    def make_run(folder_name: str, amplitude: float, phase: float, n_pulses: int, artifact_uV: float, tau_s: float, response_uV: float = 0.0) -> None:
        folder = parent / folder_name
        folder.mkdir()
        uv = rng.normal(0, 15, size=(4, n))
        # a modest 15 Hz background so band power is not pure white noise
        tt = np.arange(n) / fs
        uv += 8.0 * np.sin(2 * np.pi * 15.37 * tt + rng.uniform(0, 2 * np.pi))
        words = np.zeros((4, n), dtype=np.uint16)
        if n_pulses:
            words[3] = st.biphasic_words(n, fs, onsets[:n_pulses], amplitude_uA=amplitude, phase_us=phase)
            for onset in onsets[:n_pulses]:
                seg = slice(int(onset), min(n, int(onset) + 30000))
                t_rel = tt[seg] - tt[int(onset)]
                for ch in range(4):
                    uv[ch, seg] += -artifact_uV * (0.4 + 0.2 * ch) * np.exp(-t_rel / tau_s)
                    if response_uV:
                        uv[ch, seg] += response_uV * np.exp(-((t_rel - 0.25) / 0.05) ** 2) * np.sin(2 * np.pi * 10.0 * t_rel)
        st.write_synthetic_rhs(folder / f"{folder_name.split(' ')[0]}.rhs", sample_rate_hz=fs, channel_names=names, amp_codes=st.uv_to_code(uv), stim_words=words, impedances_ohms=[29e3, 240e3, 164e3, 88e3])
        (folder / "settings.xml").write_text(st._settings_xml("A-027", amplitude, phase, n_pulses, fs))

    make_run("recording_260101_100000", 0, 0, 0, 0, 0.03)
    make_run("step1_260101_100100 A-027 stim cathodic first 200us 10uA 1Hz 20 pulses 999ms RP", 10, 200, 20, 3000, 0.02)
    make_run("step1_260101_100200 A-027 stim cathodic first 200us 20uA 1Hz 20 pulses 999ms RP", 20, 200, 20, 6000, 0.03)
    make_run("step1_260101_100300 A-027 stim cathodic first 200us 50uA 1Hz 20 pulses 999ms RP", 50, 200, 20, 8000, 0.03, response_uV=60.0)
    make_run("step2_260101_100400 A-027 stim cathodic first 300us 50uA 1Hz 20 pulses 999ms RP", 50, 300, 20, 8000, 0.03)
    cfg = dataclasses.replace(AnalysisConfig(), trace_amplitudes_uA=(10.0, 20.0), min_trials=10, bootstrap_n=50, shuffle_n_events=20)
    result = run_session(parent, cfg, stage="all")
    expected_figs = {"fig04_bandpower_vs_amplitude", "fig04b_bandpower_heatmap", "fig05_linear_vs_sigmoid", "fig06_spatial_decay", "fig07_compliance_pulses_vs_charge", "fig08_shuffle_control_bandpower_vs_amplitude", "figS1_lognormal_qq", "figS2_within_block_drift", "figS3_impedance_drift", "figS4_noise_floor"}
    c.ok(expected_figs <= set(result.figures), f"stage=all: secondary figures present ({sorted(expected_figs - set(result.figures))} missing)")
    expected_tables = {"trials_bandpower", "trials_response_amplitude", "trials_paired_db", "comparisons_within_epoch", "comparisons_block_vs_baseline", "comparisons_across_amplitude", "drift_first_vs_last", "charge_dependence", "spatial_decay", "compliance_characterisation", "models_amplitude_response", "channel_model", "lognormal_checks", "noise_floor", "shuffle_control"}
    c.ok(expected_tables <= set(result.tables), f"stage=all: secondary tables present ({sorted(expected_tables - set(result.tables))} missing)")
    within = result.tables["comparisons_within_epoch"]
    c.ok(not within.empty and (within["n_pairs"] > 0).any() and within["ci_low_db"].notna().any(), "comparison (a) has paired trials with CIs")
    blockb = result.tables["comparisons_block_vs_baseline"]
    c.ok(not blockb.empty and blockb["baseline_run_id"].iloc[0] == "260101_100000", "comparison (b) uses the no-stim baseline run")
    paired = result.tables["trials_paired_db"]
    early = paired[(paired["window"] == "early") & (paired["retained"])]
    c.ok(not early.empty and early.groupby(["run_id", "channel", "event_number"]).ngroups >= 20, "paired dB per trial exists for retained early windows")
    shuffle = result.tables["shuffle_control"]
    c.ok(not shuffle.empty and shuffle["mode"].isin(["baseline_run", "within_run"]).all() and result.metadata.get("shuffle_mode_used") in ("baseline_run", "within_run"), f"shuffle control table present (mode {result.metadata.get('shuffle_mode_used')})")
    base_sh = shuffle[shuffle["mode"] == "baseline_run"]
    frac_zero = float(base_sh["ci_contains_zero"].mean()) if not base_sh.empty else float("nan")
    c.ok(np.isfinite(frac_zero) and frac_zero >= 0.8, f"baseline-run shuffle: {frac_zero:.0%} of CIs contain 0 dB")
    c.ok("filter_designs" in result.metadata and any("sos" in d for fsd in result.metadata["filter_designs"].values() for d in fsd.values()), "metadata records the filter designs (SOS, zero-phase flag)")
    c.ok(all("Blanking:" in cap and "Filter:" in cap for cap in result.captions.values()), "every secondary caption states blanking and filter")
    noise = result.tables["noise_floor"]
    c.ok(not noise.empty and noise["broadband_rms_uV_median"].between(5, 40).all(), f"noise floor from the no-stim run is sensible ({noise['broadband_rms_uV_median'].round(1).tolist()} uV RMS)")
