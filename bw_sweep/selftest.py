"""Self-test for bw_sweep (no real data):

    /usr/local/bin/python3 -m bw_sweep.selftest

1. Fixed threshold: through the real ``compute_recovery`` the threshold column
   is exactly 100 uV for baseline SDs of 1, 10 and 100 uV; a first-order
   high-pass tail of amplitude A recovers at tau*ln(A/100) within one sample;
   the log-log slope over synthetic tau values is 1.00 +- 0.02.
2. Tail fit recovers a known tau (R2 > 0.99); a plateau-then-drop shape gives
   R2 < 0.9.  Spec rail mask counts exactly the clipped samples.
3. End to end on a synthetic sweep written as RHS folders (arm A/B/C headers,
   two shared runs, a duplicated 500 Hz run): arm assignment, one-knob checks,
   censoring / floor exclusion, verdict lines, figures and tables; a run with a
   different stim amplitude fails the one-knob check naming the field.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from stim_analysis import selftest as st
from stim_analysis.epoch import window_slice
from stim_analysis.recovery import compute_recovery
from filter_diag.common import fit_exponential_tail
from bw_sweep.config import RAIL_LEVEL_UV, SweepConfig, tau_ms_from_hz
from bw_sweep.load import discover_sweep, one_knob_check
from bw_sweep.config import ARMS
from bw_sweep.metrics import spec_rail_mask
from bw_sweep.stats import loglog_slope


class Check(st.Check):
    pass


FS = 30000.0


def _epoch_axis(fs: float = FS) -> tuple[np.ndarray, slice]:
    n0 = int(round(0.6 * fs))
    n1 = int(round(0.9 * fs))
    t_ms = np.arange(-n0, n1) * 1e3 / fs
    return t_ms, slice(0, t_ms.size)


def check_fixed_threshold(c: Check) -> None:
    cfg = SweepConfig()
    acfg = cfg.analysis_config()
    c.ok(acfg.threshold_k == 0.0 and acfg.threshold_floor_uV == 100.0, "sweep config: threshold_k = 0, floor = 100 uV")
    t_ms, core = _epoch_axis()
    rng = np.random.default_rng(1)
    A = 5000.0
    for sd in (1.0, 10.0, 100.0):
        x = rng.normal(0, sd, size=(3, t_ms.size)) + np.where(t_ms > 0, -A * np.exp(-t_ms / 50.0), 0.0)
        frame = compute_recovery(x, t_ms, FS, np.full(3, sd), acfg, core=core)
        c.ok(np.allclose(frame["threshold_uV"], 100.0), f"threshold is 100 uV with baseline SD {sd:g} uV (3*SD would be {3 * sd:g})")
    rows = []
    for run_index, tau in enumerate((2.0, 5.0, 10.0, 50.0, 200.0)):
        x = rng.normal(0, 1.0, size=(20, t_ms.size)) + np.where(t_ms > 0, -A * np.exp(-t_ms / tau), 0.0)
        frame = compute_recovery(x, t_ms, FS, np.full(20, 1.0), acfg, core=core)
        expected = tau * math.log(A / 100.0)
        med = float(frame["recovery_ms"].median())
        # noise of 1 uV extends the last crossing by ~ noise / slope = tau/100 ms per uV
        c.ok(abs(med - expected) <= 1e3 / FS + 3.0 * tau / 100.0, f"tau {tau:g} ms: recovery {med:.3f} ms vs tau*ln(A/100) = {expected:.3f} ms")
        for _, r in frame.iterrows():
            rows.append({"run_id": f"r{run_index}", "channel": "A-001", "tau_nominal_ms": tau, "recovery_ms": r["recovery_ms"], "is_stim_contact": False})
    trials = pd.DataFrame(rows)
    slope = loglog_slope(trials, arm="synthetic", x_col="tau_nominal_ms", y_col="recovery_ms", cfg=cfg.with_(bootstrap_n=300), rng=np.random.default_rng(0), floor_by_run={})
    c.ok(abs(slope.slope - 1.0) < 0.02 and slope.verdict == "~1", f"synthetic log-log slope {slope.slope:.3f} [{slope.ci_low:.3f}, {slope.ci_high:.3f}] -> {slope.verdict}")
    # censoring + floor exclusion
    rows2 = list(rows)
    for tau, label in ((3000.0, "long"), (0.3, "short")):
        x = rng.normal(0, 1.0, size=(20, t_ms.size)) + np.where(t_ms > 0, -A * np.exp(-t_ms / tau), 0.0)
        frame = compute_recovery(x, t_ms, FS, np.full(20, 1.0), acfg, core=core)
        for _, r in frame.iterrows():
            rows2.append({"run_id": label, "channel": "A-001", "tau_nominal_ms": tau, "recovery_ms": r["recovery_ms"], "is_stim_contact": False})
    slope2 = loglog_slope(pd.DataFrame(rows2), arm="synthetic", x_col="tau_nominal_ms", y_col="recovery_ms", cfg=cfg.with_(bootstrap_n=200), rng=np.random.default_rng(0), floor_by_run={"short": 1.5})
    c.ok("long" in slope2.excluded and "censored" in slope2.excluded["long"] and "short" in slope2.excluded and "floor" in slope2.excluded["short"], f"censored and at-floor runs excluded from the slope: {slope2.excluded}")
    c.ok(abs(slope2.slope - 1.0) < 0.02, f"slope unchanged after exclusions: {slope2.slope:.3f}")
    drifting = pd.DataFrame(rows2)
    drifting["local_drift_uV"] = np.where(drifting["run_id"] == "r4", 40.0, 2.0)
    slope3 = loglog_slope(drifting, arm="synthetic", x_col="tau_nominal_ms", y_col="recovery_ms", cfg=cfg.with_(bootstrap_n=100), rng=np.random.default_rng(0), floor_by_run={"short": 1.5})
    c.ok("r4" in slope3.excluded and "still moving" in slope3.excluded["r4"] and "r3" not in slope3.excluded, f"run whose pre-pulse baseline drifts > {cfg.local_drift_max_uV:g} uV is excluded: {slope3.excluded.get('r4')}")
    # local centring: mean and drift of a ramp
    ramp = np.tile(np.linspace(-100.0, 100.0, t_ms.size), (2, 1))
    from bw_sweep.metrics import local_centre
    off, drift = local_centre(ramp, t_ms, (-50.0, -5.0))
    expected_off = float(np.interp(-27.5, t_ms, ramp[0]))
    expected_drift = 45.0 * 200.0 / (t_ms[-1] - t_ms[0])
    c.ok(abs(off[0] - expected_off) < 0.05 and abs(drift[0] - expected_drift) < 0.05, f"local_centre on a ramp: offset {off[0]:.2f} (exp {expected_off:.2f}), drift {drift[0]:.2f} uV (exp {expected_drift:.2f})")


def check_fit_and_rail(c: Check) -> None:
    t_ms, _ = _epoch_axis()
    rng = np.random.default_rng(2)
    for tau in (5.0, 30.0, 150.0):
        x = np.where(t_ms > 0, -3000.0 * np.exp(-t_ms / tau), 0.0) + rng.normal(0, 5.0, t_ms.size)
        fit = fit_exponential_tail(x, t_ms, exit_ms=0.0, excursion_sign=-1.0, start_offset_ms=2.0, end_ms=800.0)
        c.ok(abs(fit.tau_ms / tau - 1.0) < 0.05 and fit.r2 > 0.99 and fit.tail_sign > 0, f"tail fit tau {tau:g} -> {fit.tau_ms:.2f} ms, R2 {fit.r2:.4f}, same sign")
    plateau = np.where((t_ms > 0) & (t_ms < 300.0), -3000.0, 0.0) + rng.normal(0, 5.0, t_ms.size)
    fit = fit_exponential_tail(plateau, t_ms, exit_ms=0.0, excursion_sign=-1.0)
    c.ok(fit.r2 < 0.9, f"plateau-then-drop (non-exponential) tail: R2 {fit.r2:.3f} < 0.9")
    x = rng.normal(0, 3.0, size=(4, t_ms.size)).astype(np.float32)
    x[0, 18000:18030] = 6388.98
    x[1, 18000:18010] = -6388.98
    x[2, 18000:18010] = 6300.0
    mask = spec_rail_mask(x, RAIL_LEVEL_UV)
    counts = mask.sum(axis=1)
    c.ok(list(counts) == [30, 10, 0, 0], f"spec rail mask counts {list(counts)} (expected [30, 10, 0, 0]) at level {RAIL_LEVEL_UV} uV")


# -----------------------------------------------------------------------------
# synthetic sweep
# -----------------------------------------------------------------------------

DESIGN = [
    # (folder, lower_hz, dsp_enabled, dsp_hz, upper_hz)
    ("analogsweep_260101_100000", 0.0945, False, 0.146, 7603.8),
    ("analogsweep_260101_100100", 1.0977, False, 0.146, 7603.8),
    ("analogsweep_260101_100200", 9.924, False, 0.146, 7603.8),
    ("analogsweep_260101_100300", 29.94, False, 0.146, 7603.8),
    ("analogsweep_260101_100400", 97.81, False, 0.146, 7603.8),
    ("analogsweep_260101_100500", 324.0, False, 0.146, 7603.8),
    ("DSPsweep_260101_100600", 0.0945, True, 0.2914, 7603.8),
    ("DSPsweep_260101_100700", 0.0945, True, 1.1658, 7603.8),
    ("DSPsweep_260101_100800", 0.0945, True, 4.665, 7603.8),
    ("DSPsweep_260101_100900", 0.0945, True, 18.69, 7603.8),
    ("DSPsweep_260101_101000", 0.0945, True, 151.6, 7603.8),
    ("upperanalogbandwidth_260101_101100", 1.0977, False, 151.6, 3008.2),
    ("upperanalogbandwidth_260101_101200", 1.0977, False, 151.6, 999.4),
    ("upperanalogbandwidth_260101_101300", 1.0977, False, 151.6, 499.9),
    ("upperanalogbandwidth_260101_101400", 1.0977, False, 151.6, 299.9),
    ("upperanalogbandwidth_260101_101500", 1.0977, False, 151.6, 499.9),
]


def write_synthetic_sweep(root: Path, *, amplitude_override: dict[str, float] | None = None, n_pulses: int = 6, seed: int = 3) -> None:
    """16 small RHS runs whose artifacts are pure exponentials with the run's own tau_nominal."""
    rng = np.random.default_rng(seed)
    names = ["A-024", "A-025", "A-026", "A-027"]
    n = 128 * int(math.ceil((2.0 + n_pulses + 1.0) * FS / 128))  # 2 s lead-in, 1 s per pulse, 1 s tail
    onsets = int(2.0 * FS) + np.arange(n_pulses) * int(FS)
    t = np.arange(n) / FS
    for folder_name, lower, dsp_on, dsp_hz, upper in DESIGN:
        folder = root / folder_name
        folder.mkdir(parents=True)
        fc = max(lower, dsp_hz if dsp_on else 0.0)
        tau_s = tau_ms_from_hz(fc) * 1e-3
        noise = 5.0 * math.sqrt(upper / 7603.8)
        uv = rng.normal(0, noise, size=(4, n))
        words = np.zeros((4, n), dtype=np.uint16)
        amplitude = (amplitude_override or {}).get(folder_name, 100.0)
        words[3] = st.biphasic_words(n, FS, onsets, amplitude_uA=amplitude, phase_us=200.0, step_uA=0.5)
        rec_peak = 4000.0 * (upper / 7603.8) ** 0.3
        stim_peak = 20000.0 * (upper / 7603.8) ** 0.7  # rails on the stim contact, shorter at narrower bandwidth
        for onset in onsets:
            seg = slice(int(onset), n)  # tails run to the end of the record (they overlap at long tau, as in the real recordings)
            tt = t[seg] - t[int(onset)]
            for ch in range(3):
                uv[ch, seg] += -rec_peak * (0.6 + 0.2 * ch) * np.exp(-tt / tau_s)
            uv[3, seg] += -stim_peak * np.exp(-tt / max(tau_s, 0.004))
        codes = st.uv_to_code(uv)
        st.write_synthetic_rhs(
            folder / f"{folder_name}.rhs", sample_rate_hz=FS, channel_names=names, amp_codes=codes, stim_words=words,
            impedances_ohms=[157e3, 176e3, 185e3, 123e3], stim_step_uA=0.5,
            dsp_enabled=dsp_on, dsp_cutoff_hz=dsp_hz, lower_bw_hz=lower, upper_bw_hz=upper,
        )
        xml = st._settings_xml("A-027", amplitude, 200.0, n_pulses, FS).replace('StimStepSizeMicroAmps="2"', 'StimStepSizeMicroAmps="0.5"')
        (folder / "settings.xml").write_text(xml)


def check_end_to_end(tmp: Path, c: Check) -> None:
    from bw_sweep.run import run_sweep, write_result

    root = tmp / "sweep"
    write_synthetic_sweep(root)
    cfg = SweepConfig().with_(root=root, bootstrap_n=150)
    sweep = discover_sweep(root, cfg)
    c.ok(len(sweep.runs) == 16, f"synthetic sweep: {len(sweep.runs)} runs discovered")
    arms = {r.folder.name: r.arms for r in sweep.runs}
    c.ok(arms["analogsweep_260101_100000"] == ["A", "B"] and arms["analogsweep_260101_100100"] == ["A", "C"], f"shared runs land in two arms: {arms['analogsweep_260101_100000']}, {arms['analogsweep_260101_100100']}")
    c.ok(all(len(sweep.arm_runs(a)) == 6 for a in "ABC"), f"6 runs per arm: {[len(sweep.arm_runs(a)) for a in 'ABC']}")
    reps = {r.folder.name: r.replicate for r in sweep.runs if r.replicate}
    c.ok(set(reps) == {"upperanalogbandwidth_260101_101300", "upperanalogbandwidth_260101_101500"} and {v["C"] for v in reps.values()} == {"a", "b"}, f"duplicate 500 Hz runs tagged as replicates: {reps}")
    c.ok(sweep.all_ok, f"one-knob checks pass: {[ (k, v.ok) for k, v in sweep.checks.items()]}")
    off = sweep.by_run_id("260101_100000")
    c.ok(off.arm_label("B") == "off" and abs(off.tau_nominal_ms - 1684) < 5, f"Arm B off point: label {off.arm_label('B')!r}, tau_nominal {off.tau_nominal_ms:.0f} ms (analog pole)")
    k12 = sweep.by_run_id("260101_100700")
    c.ok(k12.dsp_k == 12 and abs(k12.tau_nominal_ms - 136.5) < 0.5 and abs(k12.tau_dsp_from_k_ms - 136.53) < 0.1 and k12.dominant_pole == "dsp", f"DSP k=12 run: k {k12.dsp_k}, tau {k12.tau_nominal_ms:.1f} ms, from k {k12.tau_dsp_from_k_ms:.1f} ms")

    result = run_sweep(root=root, output_dir=tmp / "out", cfg=cfg, quiet=True)
    trials = result.trials
    c.ok(len(trials) == 16 * 4 * 6, f"{len(trials)} trial rows (16 runs x 4 channels x 6 pulses)")
    c.ok(np.allclose(trials["threshold_uV"], 100.0), "every trial row has threshold 100 uV")
    for arm in ("A", "B"):
        s = result.slopes[arm]
        c.ok(s.verdict == "~1" and abs(s.slope - 1.0) < 0.05, f"arm {arm}: slope {s.slope:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] -> {s.verdict}; excluded {s.excluded}")
    sA = result.slopes["A"]
    c.ok(any("censored" in v for v in sA.excluded.values()) and any("floor" in v for v in sA.excluded.values()), f"arm A: the 0.1 Hz run is censored and the 324 Hz run sits on the floor: {sA.excluded}")
    t1 = result.tables["table1_summary_per_run"]
    c.ok(len(t1) == 32 and set(t1["contacts"]) == {"recording", "stim"}, f"table1: {len(t1)} rows (16 runs x 2 contact groups)")
    pc = result.tables["table1b_summary_per_run_channel"]
    stim_c = pc[(pc["channel"] == "A-027") & pc["arms"].str.contains("C")].sort_values("upper_hz")
    c.ok(stim_c["median_rail_fs_ms"].is_monotonic_increasing and stim_c["median_rail_fs_ms"].iloc[-1] > 0, f"arm C stim contact rail duration grows with upper bandwidth: {list(np.round(stim_c['median_rail_fs_ms'], 2))}")
    rec_c = pc[(pc["channel"] == "A-024") & pc["arms"].str.contains("C")]
    c.ok((rec_c["frac_zero_rail"] >= 0.95).all(), "arm C recording contact never rails")
    v = result.verdict
    c.ok(v is not None and len(v.lines) == 3 and v.lines[0].startswith("Arm A") and v.lines[1].startswith("Arm B") and v.lines[2].startswith("Arm C"), "verdict has one line per arm")
    c.ok("~1" in v.lines[0] and "~1" in v.lines[1], f"verdict lines: {v.lines[0][:80]} | {v.lines[1][:80]}")
    c.ok(v.recommendation.startswith("Recommendation:") and "analogsweep_260101_100500" in v.recommendation, f"recommendation picks the shortest-recovery run: {v.recommendation[:120]}")
    c.ok({"fig0_median_traces", "fig1_recovery_vs_tau_armA", "fig2_recovery_vs_tau_armB", "fig3_armC_rail_and_peak", "fig4_tau_fit_vs_tau_nominal", "fig5_r2_per_arm", "fig6_noise_vs_bandwidth"} <= set(result.figures), f"figures rendered: {sorted(result.figures)}")
    taus = result.slopes["all_taufit"]
    c.ok(abs(taus.slope - 1.0) < 0.1, f"tau_fit vs tau_nominal slope {taus.slope:.3f} on informative fits")
    inf = trials[trials["fit_informative"] & ~trials["is_stim_contact"] & (trials["recovery_ms"] < 850)]
    ratio = (inf["tau_fit_ms"] / inf["tau_nominal_ms"]).median()
    c.ok(abs(ratio - 1.0) < 0.1 and (inf["r2"] > 0.95).mean() > 0.9, f"informative fits: tau_fit/tau_nominal median {ratio:.3f}, {(100 * (inf['r2'] > 0.95).mean()):.0f}% with R2 > 0.95")
    noise = pc[pc["channel"] == "A-024"]
    hi = noise[noise["upper_hz"] > 7000]["median_prestim_sd_uV"].mean()
    lo = noise[noise["upper_hz"] < 400]["median_prestim_sd_uV"].mean()
    c.ok(hi > lo * 1.5, f"pre-train noise SD tracks the synthetic bandwidth ({hi:.2f} vs {lo:.2f} uV)")
    written = write_result(result)
    c.ok(all(p.exists() for p in written) and (tmp / "out" / "verdict.txt").exists() and (tmp / "out" / "table0_settings_per_run.csv").exists(), f"{len(written)} files written atomically")

    # broken set: one arm-A run at a different stim amplitude
    root2 = tmp / "sweep_broken"
    write_synthetic_sweep(root2, amplitude_override={"analogsweep_260101_100200": 50.0}, n_pulses=3)
    sweep2 = discover_sweep(root2, cfg)
    check = sweep2.checks["A"]
    c.ok(not check.ok and any("amplitude_uA" in m for m in check.messages), f"one-knob check fails and names the field: {check.messages[:1]}")
    # a run whose upper bandwidth differs falls out of arm A rather than contaminating it
    run = sweep2.by_run_id("260101_100300")
    run.upper_hz = 3000.0
    from bw_sweep.load import assign_arms
    assign_arms(sweep2.runs, cfg)
    c.ok(run.arms == [] and any("no arm" in w for w in run.warnings), f"upper 3000 Hz inside the analog sweep matches no arm: {run.warnings}")
    # keep_duplicate policy
    sweep3 = discover_sweep(root, cfg.with_(keep_duplicate="first"))
    c.ok(len(sweep3.arm_runs("C")) == 5 and any("dropped" in n for n in sweep3.notes), "keep_duplicate=first drops one 500 Hz replicate from arm C")


def main() -> int:
    started = time.time()
    c = Check()
    with tempfile.TemporaryDirectory(prefix="bw_sweep_selftest_") as tmpdir:
        tmp = Path(tmpdir)
        try:
            print("1. fixed threshold + scaling")
            check_fixed_threshold(c)
            print("2. tail fit + rail mask")
            check_fit_and_rail(c)
            print("3. end to end on a synthetic sweep")
            check_end_to_end(tmp, c)
        except AssertionError as exc:
            print(f"\nFAILED: {exc}")
            return 1
        except Exception:
            traceback.print_exc()
            return 2
    print(f"\nall {c.n} checks passed in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
