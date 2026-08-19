# Bandwidth Sweep Analysis -- Plan (session 20260818 re stim in vitro filter settings)

Written 2026-08-18. Status: **implemented as `bw_sweep/` and run on the sweep
folder on 2026-08-18 (user go-ahead); outputs in
`<root>/bandwidth_sweep/`.** Section 9 lists what the build changed relative
to sections 1-7, section 10 the results. Verified with
`python3 -m bw_sweep.selftest` (45 synthetic checks) before the run.
Everything in sections 1-8 that is stated as a data fact was read from the
`.rhs` headers / `settings.xml` and a read-only `validate_run` pass with the
*existing* loader.

Question: does stimulation-artifact recovery scale with the amplifier's
low-frequency time constant (a linear high-pass step response), or is it a
nonlinear saturation recovery independent of the nominal cutoff? The in vivo
diagnosis (session 260817, `filter_diagnosis/verdict.txt`) concluded
"electrode / analog front-end dominated; DSP shortens rather than causes it".
This sweep is the direct experimental test.

---

## 1. Data inventory (verified from headers)

Folder: `/Users/jf/SynologyDrive/Research/Stimulation/20260818 re stim in vitro filter settings`
(working copy; the Endovascular copy stays pristine). 16 run folders, 22 `.rhs`
files (Intan splits at 60 s; `read_rhs_run` concatenates a folder). All runs:
30 kS/s, 4 amplifier channels **A-021, A-022, A-025, A-026**, stim on **A-026**,
100 uA biphasic negative-first, 200 us/phase, 100 us interphase, 50 pulses at
1.000 s period, amp-settle on (0 us pre / 1000 us post, settle bandwidth
1000 Hz), charge recovery on (0-1000 us). Every run: 50/50 pulses detected,
no compliance bit, no timestamp gaps. Header impedance (1 kHz, PBS):
A-021 157 kOhm, A-022 176, A-025 185, A-026 123 (identical in every header --
measured once). In vivo 260817 for comparison: 180-285 kOhm on A-025..A-030.

| arm | folder | actual lower (Hz) | DSP | actual DSP cutoff (Hz) | actual upper (Hz) | tau_nominal | note |
|---|---|---|---|---|---|---|---|
| A (+B off) | analogsweep_193534 | 0.095 | off | -- | 7603.8 | 1675 ms | **shared: Arm B "off" point** |
| A (+C 7500) | analogsweep_193705 | 1.098 | off | -- | 7603.8 | 145 ms | **shared: Arm C 7500 Hz point** |
| A | analogsweep_193807 | 9.924 | off | -- | 7603.8 | 16.0 ms | |
| A | analogsweep_193906 | 29.945 | off | -- | 7603.8 | 5.3 ms | |
| A | analogsweep_194009 | 97.814 | off | -- | 7603.8 | 1.63 ms | |
| A | analogsweep_194120 | 324.014 | off | -- | 7603.8 | 0.49 ms | (desired 300) |
| B k=14 | DSPsweep_194325 | 0.095 | on | 0.291 | 7603.8 | 547 ms | |
| B k=12 | DSPsweep_194430 | 0.095 | on | 1.166 | 7603.8 | 136.5 ms | **= in vivo 260817 setting** |
| B k=10 | DSPsweep_194547 | 0.095 | on | 4.665 | 7603.8 | 34.1 ms | |
| B k=8 | DSPsweep_194648 | 0.095 | on | 18.687 | 7603.8 | 8.5 ms | |
| B k=5 | DSPsweep_194807 | 0.095 | on | 151.589 | 7603.8 | 1.05 ms | |
| C 3000 | upperanalogbandwidth_195019 | 1.098 | off | -- | 3008.2 | 145 ms | |
| C 1000 | upperanalogbandwidth_195141 | 1.098 | off | -- | 999.4 | 145 ms | |
| C 500a | upperanalogbandwidth_195249 | 1.098 | off | -- | 499.9 | 145 ms | **duplicate 500 Hz** |
| C 300 | upperanalogbandwidth_195354 | 1.098 | off | -- | 299.9 | 145 ms | |
| C 500b | upperanalogbandwidth_195458 | 1.098 | off | -- | 499.9 | 145 ms | **duplicate 500 Hz** (longer, 74 s) |

Findings that shape the plan:

1. **Shared runs.** Arm B "off" and Arm C "7500 Hz" are not separate
   recordings; they are `analogsweep_193534` and `analogsweep_193705`. The
   loader must assign one run to several arms (a run -> list-of-arms map), and
   the settings table must say so.
2. **Duplicate 500 Hz in Arm C** (`195249`, `195458`). Default: treat as
   replicates (both plotted, both in table1, pooled for the 500 Hz summary
   point, and a replicate-agreement line in the log). Please confirm, or tell
   me one was a mistake (e.g. intended 300 Hz re-do) and I will drop it.
3. **One-knob check** must ignore `actual_dsp_cutoff` when `dsp_enabled` is
   false (Arm A headers carry 0.146 Hz, Arm C headers 151.6 Hz -- desired
   values left in the GUI, no effect). Effective high-pass poles per run:
   analog pole always; DSP pole only when enabled. `lower_settle_bw` (1000 Hz)
   and the amp-settle/charge-recovery windows are constant across all 16 runs.
4. **Railing in PBS is rare on the recording contacts.** Empirical rail
   estimator flags A-026 (stim contact) in 12/16 runs, A-025 in 3, A-022 in 3,
   A-021 never. So Arm C's "does rail duration reach zero" must be answered
   per channel with the stim contact separate; for A-021 it is already zero
   at 7500 Hz. Peak_uV (first 5 ms) is the informative Arm C metric on the
   non-railed contacts.
5. **Censoring / baseline contamination at long tau.** Epoch is -600..+900 ms
   with 1 s IPI. If recovery really scales as tau*ln(A0/100 uV) (~4.2 tau for
   a rail-level excursion), the 0.095 Hz (tau 1.7 s) and DSP 0.29 Hz (0.55 s)
   points would be censored at 900 ms and their -500..-50 ms baselines are the
   previous pulse's tail. Handled by: censored counts in table1, censored
   medians drawn as lower-bound arrows at 900 ms and excluded from slope
   fits, `mark_baseline_contamination` reused, and a clean pre-train noise
   floor per run (section 3) so fig6 is not driven by tail contamination.
6. **Fit window vs short tau.** The spec's fit window starts at rail exit
   + 2 ms; for tau_nominal < ~4 ms (A 98 Hz, A 324 Hz, B k=5) the tail is gone
   before the window opens and tau_fit is meaningless (will sit at the 2 ms
   bound). Keep the spec window, add `fit_informative = tau_nominal >= 4 ms`,
   show uninformative fits hollow in fig4/fig5, and report %R2<0.9 both over
   all fits and over informative fits.
7. **Hardware floor.** Pulse 0.5 ms + amp settle 1 ms + charge recovery
   1 ms => recovery cannot be below ~2.5 ms in any run. Runs whose median
   sits at the floor are also excluded from slope fits (they are the "flat
   at floor" regime, reported as such).

---

## 2. Definitions (exact, per epoch x channel)

Epoching, centring and every metric run on **raw, unfiltered** data.
Epoch -600..+900 ms around the detected pulse onset (existing `epoch_ms`),
baseline -500..-50 ms, `centred = raw - baseline_mean` (existing
`extract_epochs`, `baseline_stats`).

| column | definition | source |
|---|---|---|
| `threshold_uV` | **100.0 for every trial of every run** | `compute_recovery` with `cfg = replace(AnalysisConfig(), threshold_k=0.0, threshold_floor_uV=100.0)` -> `max(0*sd, 100) = 100`; asserted `== 100.0` on every output row |
| `recovery_ms` | last t > 0 with abs(centred) > 100 uV before >= 20 ms (`quiet_ms`) continuously below; censored at 900 ms | `compute_recovery` (imported, unchanged) |
| `rail_fs_ms` | post-onset samples with abs(raw) >= 6388.9 uV (ADC code +-32764, the observed saturation) x 1000/fs | new, 3 lines |
| `rail_emp_ms` | existing empirical rail estimate (`estimate_rail` / `railed_mask`), for cross-check | `compute_recovery(railed=...)` -> `rail_ms` |
| `peak_uV` | max abs(centred) in 0..5 ms | new: `window_slice(t, 0, 5)` |
| `exit_ms` | last spec-railed sample after 0, else time of abs peak | `filter_diag.common.rail_exit_ms` (fed the spec rail mask) |
| `tau_fit_ms`, `A_uV`, `C_uV`, `R2`, `tail_sign` | soft-L1 fit V = A exp(-t/tau) + C on [exit+2 ms, 800 ms], decimate 10, tau bounds 2-5000 ms; `tail_sign = sign(A)*sign(excursion)` | `filter_diag.common.fit_exponential_tail` |
| `baseline_sd_uV` | SD of raw in -500..-50 ms | `compute_recovery` -> `baseline_sd_uV` |
| `baseline_contaminated` | previous pulse's recovery reaches into this baseline | `stim_analysis.recovery.mark_baseline_contamination` |
| `is_stim_contact` | channel == A-026 | settings.xml / header |

Per run:

- `tau_nominal_ms = 1000 / (2 pi f_c)`, `f_c = max(actual_lower_bandwidth,
  actual_dsp_cutoff if dsp_enabled)`; also stored: both poles, `dsp_k`
  (`dsp_k_from_cutoff`), `tau_dsp_from_k = 2^k/fs` (cross-check, matches to
  <1 %), and the *second* pole's tau (Arm B keeps the 0.095 Hz analog pole,
  tau 1.7 s, underneath the DSP -- worth having in table1).
- `prestim_sd_uV` per channel: SD of raw from (recording start + 1 s) to
  (first onset - 0.5 s) -- the clean noise floor of the setting; every run
  has 3-10 s of pre-train data. Fig6 shows both this and the median
  `baseline_sd_uV`.
- Impedance per channel from the header (PBS), plus a fixed reference column
  with the in vivo 260817 header values for the same contact numbers.

---

## 3. Code plan (new package `bw_sweep/`, CLI first)

Mirrors `filter_diag/`: a package + `python3 -m bw_sweep.run`, atomic writes,
`--dry-run`, self-test with no data. No notebook function unless you want one
(a Function 7 hook is a thin wrapper later).

```
bw_sweep/
  __init__.py       __version__
  config.py         SweepConfig: fixed threshold cfg (built from AnalysisConfig via
                    dataclasses.replace), arm definitions + expected nominal values,
                    rail level 6388.9 uV, fit window (2, 800 ms), fit_informative_tau_ms=4,
                    floor_ms, bootstrap_n=1000, seed=0, ALL fixed axis limits (sec. 5)
  load.py           discover_runs(root) -> [SweepRun]: header table, arm assignment
                    (folder prefix analogsweep/DSPsweep/upperanalogbandwidth + header),
                    shared-run duplication into arms, duplicate-setting detection,
                    one_knob_check(arm) -> pass/fail with the offending fields
  metrics.py        per_run_trials(run) -> DataFrame (one row per channel x epoch,
                    columns of sec. 2). Reuses load_run, extract_epochs, baseline_stats,
                    compute_recovery, rail_exit_ms, fit_exponential_tail,
                    mark_baseline_contamination. New code: spec rail mask, peak_5ms,
                    prestim_sd, tau_nominal.
  stats.py          median + bootstrap CI (reuse bootstrap_median_ci), loglog_slope()
                    with cluster bootstrap by run (resample trials within run, refit),
                    Theil-Sen cross-check, R2<0.9 fraction with CI, same-sign fraction
  figures.py        fig0-fig6 (sec. 5), fixed limits only, captions via
                    stim_analysis.figures.build_caption / finish_figure
  verdict.py        decision rules (sec. 6) -> one line per arm + recommendation
  run.py            CLI: --root DIR --out DIR --dry-run --keep-duplicate {both,a,b}
                    --exclude-stim-contact/--include-stim-contact --fast
  selftest.py       synthetic checks, no data (sec. 7)
```

Reuse map (import, do not reimplement):
`stim_analysis.load_rhs.load_run`, `stim_analysis.epoch.extract_epochs /
baseline_stats / window_slice / gap_starts`, `stim_analysis.recovery.compute_recovery /
mark_baseline_contamination`, `stim_analysis.validate.validate_run / estimate_rail /
railed_mask`, `stim_analysis.stats.bootstrap_ci`, `stim_analysis.figures.*`,
`filter_diag.common.fit_exponential_tail / rail_exit_ms / bootstrap_median_ci /
dsp_k_from_cutoff / dsp_tau_s / ADC_FULL_SCALE_UV`, `rhs_files.atomic_write_all`,
`rhs_reader.read_rhs_header`.

No change to `stim_analysis/` is required: `AnalysisConfig.validate()` (which
rejects `threshold_k <= 0`) is only called by the CLI/notebook front ends and
`pipeline.run`, not by `compute_recovery`. The sweep config is built with
`dataclasses.replace` and never validated; the fixed-threshold assertion on
the output rows is the guard. (Alternative if you prefer: relax `validate` to
allow `k == 0` when `floor > 0` -- one line; I lean towards not touching the
spec-v2 package.)

Outputs (all under `<root>/bandwidth_sweep/`, written atomically at the end):
`table0_settings_per_run.csv`, `table1_summary_per_run.csv`,
`trials_per_epoch.csv` (3200 rows), `fig0..fig6*.png` + `captions.txt`,
`verdict.txt`, `metadata.json` (config, versions, git commit, run list,
one-knob check results, duplicate handling), `log.txt`.

---

## 4. Statistics and decision rules

- Every summary point = per-trial distribution: median, IQR, 95 % bootstrap
  CI of the median (n=1000, seed 0), per-trial dots jittered. Recording
  contacts (A-021/022/025) pooled but colour-coded; stim contact A-026 as its
  own marker/panel, excluded from slopes by default.
- **Slope (Arms A, B):** OLS of log10(recovery) on log10(tau_nominal) over
  per-trial data; CI by cluster bootstrap (resample trials within each run,
  1000x). Fit uses only runs whose median is neither censored (>= 850 ms)
  nor at the hardware floor (<= floor + 1 ms); excluded runs are listed in
  the caption. Also per-channel slopes and Theil-Sen as robustness. Also
  reported: slope-1 line with free intercept (least squares) and its
  intercept ratio recovery/tau_nominal, and the linear-filter prediction
  recovery = tau * ln(6389/100) as a dashed reference.
  Rule: `slope ~ 1` if the 95 % CI contains 1 and excludes 0 (and CI width
  < 1); `flat` if the CI contains 0 and excludes 1; otherwise `intermediate`
  with the numbers.
- **Arm C rail -> 0:** per channel and run, median `rail_fs_ms` with CI and
  the fraction of trials with `rail_fs_ms == 0`. "Reaches zero" when the
  fraction of zero-rail trials is >= 0.95 at some upper bandwidth for that
  channel (report the highest bandwidth at which this first holds). Peak_uV
  vs upper bandwidth alongside, with the 6389 uV rail marked.
- **R2:** fraction R2 < 0.9 per arm with CI (all fits / informative fits).
- **tau_fit vs tau_nominal:** ratio and log-log slope, informative fits only.
- **Baseline SD:** median `baseline_sd_uV` and `prestim_sd_uV` per run x
  channel with CI.

## 5. Figures (all limits fixed in `SweepConfig`, none data-derived)

| fig | content | axes (fixed) |
|---|---|---|
| fig0 (QC) | median centred trace per run, one panel per arm, recording contacts + stim contact dashed, -5..300 ms | y +-7000 uV and a +-500 uV zoom row; x linear |
| fig1 | Arm A: recovery vs tau_nominal, per-trial dots, median +- IQR, CI bars; identity line, best slope-1 line, tau*ln(63.9) dashed; censored medians as up-arrows at 900 | x 0.1-10000 ms log; y 0.5-1000 ms log |
| fig2 | Arm B, same as fig1 (off point = shared run) | same |
| fig3 | Arm C: (a) rail_fs_ms vs upper BW per channel, (b) peak_uV vs upper BW with 6389 uV line; the 7500 point is the shared run; duplicate 500 Hz shown as two markers | x 100-10000 Hz log; rail y 0-50 ms linear; peak y 100-10000 uV log |
| fig4 | tau_fit vs tau_nominal, all arms, identity; hollow = uninformative | both 0.1-10000 ms log |
| fig5 | R2 ECDF per arm + vertical line at 0.9; second row per-arm box of R2 by run | x 0-1 |
| fig6 | baseline_sd_uV (median, IQR) and prestim_sd_uV vs the varied knob, three panels (lower BW / DSP cutoff / upper BW), per channel | y 0.1-1000 uV log; x per arm log |

Every PNG gets the standard caption block (session, config hash, threshold =
100 uV fixed, n trials, exclusions).

## 6. Verdict lines and recommendation

Printed and stored in `verdict.txt`, one line per arm using the rules above:

- Arm A: `slope = s [lo, hi]` -> "~1: low-frequency analog pole sets
  recovery; fix = higher analog cutoff" / "flat: nonlinear saturation
  recovery, independent of nominal cutoff" / intermediate.
- Arm B: slope -> "flat: DSP exonerated" / "~1: DSP cutoff sets recovery"
  / intermediate; plus the k=12 point compared with the in vivo 260817
  medians (same instrument setting, PBS vs tissue).
- Arm C: "rail duration reaches 0 at <= X Hz on channels [...]; never on
  [...]" and the peak_uV trend.
- Recommendation: among all 16 settings, the one with the lowest median
  recovery (recording contacts) subject to upper BW >= 300 Hz and baseline
  noise acceptable. **Acceptable = `prestim_sd_uV` <= 2x the lowest
  prestim SD across the sweep** (default; you may prefer an absolute
  number -- say so). Table of the Pareto front (recovery vs prestim SD)
  printed so the choice is auditable.

## 7. Verification (no real data)

`python3 -m bw_sweep.selftest`:
1. Synthetic first-order high-pass step response through the *real*
   `compute_recovery` with the sweep config: threshold column is exactly
   100 uV for baseline SDs of 1, 10, 100 uV; recovery = tau*ln(A/100) within
   1 sample; slope over synthetic tau in {2, 5, 10, 50, 200 ms} = 1.00 +- 0.02.
2. Synthetic exponential tail + noise -> `tau_fit` within 5 %, R2 > 0.99;
   a saturating (non-exponential) synthetic gives R2 < 0.9.
3. Clipped synthetic -> `rail_fs_ms` counts exactly the clipped samples;
   unclipped -> 0.
4. Fake header set -> arm assignment, shared-run duplication, duplicate
   detection, one-knob check pass, and a deliberately broken set (upper BW
   changed inside Arm A) -> one-knob check fails with the field named.
5. `py_compile` all modules; `--dry-run` on a scratch copy of two run
   folders under the job tmp dir (writes nothing to the NAS).

Then, **only after you say go**: `python3 -m bw_sweep.run --root ".../20260818 re stim in vitro filter settings"`
writes `<root>/bandwidth_sweep/` (working copy only, never the Endovascular
folder). Runtime estimate: 16 runs x 4 ch x 50 epochs, well under a minute.

## 8. Decisions I need from you

1. Duplicate Arm C 500 Hz runs: keep both as replicates (default) or drop one?
2. Stim contact A-026: separate panel/marker and excluded from slopes
   (default) -- OK?
3. "Acceptable baseline SD" for the recommendation: 2x the sweep minimum
   (default) or an absolute value in uV?
4. Notebook Function 7 wrapper wanted, or CLI only (default)?
5. Go-ahead to implement, then a second go-ahead before running on the data.

Status 2026-08-18: 1, 2 and 4 taken as the defaults; 3 changed (see 9.4);
implementation go-ahead given; **the run on the data still needs your go**.

## 9. Implementation notes and deviations (2026-08-18)

What the build changed relative to the sections above, and why. All of it is
in `bw_sweep/config.py` as named parameters.

1. **Rail level is 6388.9 uV, not 6389.5.** The converted data saturate at
   ADC code +-32764 = +-6388.98 uV (checked on a railed run), so the "+-6389
   uV rail" is `|raw| >= 6388.9`.
2. **Epochs are centred on a local pre-pulse window (-50..-5 ms), not on
   the -500..-50 ms baseline mean.** With a 1 s IPI the recording contacts'
   tails outlast the interval (sawtooth), so the spec baseline mean sits on
   the previous tail and turns a censored recovery into a spurious
   mean-crossing (668 ms instead of >= 900 ms on the scratch copy of the
   analog 1 Hz run). The spec window still gives `baseline_sd_uV` and
   `baseline_contaminated`; the spec-centred recovery is kept as
   `recovery_spec_centred_ms` next to the primary `recovery_ms`, and two
   drift diagnostics are reported: `baseline_drift_uV` (local mean minus spec
   mean) and `local_drift_uV` (linear drift of raw across the local window).
3. **Slope-fit exclusions.** A run is left out of the log-log fit (and drawn
   hollow with its reason) when its median recovery is censored (>= 850 ms),
   at the hardware floor (<= floor + 1 ms), or when > 50 % of its trials have
   |local_drift| > 25 uV (the pre-pulse level is still moving, so a fixed
   100 uV recovery is not defined). Verdict rule: "~1" if the 95 % CI lies
   within 1 +- 0.25 or contains 1 (not 0) with width < 1; "flat" if it lies
   within 0 +- 0.25 or contains 0 (not 1); otherwise "intermediate".
4. **Recommendation noise rule** is relative to the *reference setting*
   (analog 0.1 Hz / DSP k = 12 / 7500 Hz -- the in-vivo setting, present in
   the sweep): noise SD <= 2x its pre-train SD, plus upper >= 300 Hz and
   < 50 % censored. "2x the sweep minimum" would always force the narrowest
   upper bandwidth (noise ~ sqrt(BW)).
5. **Pre-train noise floor** uses [start + 1 s, first pulse - 0.5 s]; when
   the lead-in is shorter than 1.5 s it uses the last 0.5 s before the gap
   (past the amplifier's start-up transient). `prestim_seconds` is in table1.
6. `stim_analysis/selftest.py::write_synthetic_rhs` gained optional
   bandwidth/DSP keyword arguments (defaults unchanged) so the sweep self-test
   can write RHS files with Arm A/B/C headers. Nothing else in
   `stim_analysis/` or `filter_diag/` changed.
7. Package layout as planned plus `summary.py` (table1 / per-channel table);
   outputs `table0_settings_per_run`, `table1_summary_per_run`,
   `table1b_summary_per_run_channel`, `table2_recommendation_pareto`,
   `trials_per_epoch`, `fig0..fig6`, `captions.txt`, `verdict.txt`,
   `metadata.json`, `log.txt` under `<root>/bandwidth_sweep/`.
8. Added after the first real run: clean-segment noise SD (pre-train, or
   post-train 2 s after the last pulse when there was no lead-in) and its
   > 5 Hz component (`clean_sd_*` columns; fig6 crosses/plus signs); the
   recommendation uses the clean-segment SD. Verdict lines carry
   per-channel slopes, an additive fit recovery = a*tau + b (cluster
   bootstrap), a shortest-tau floor clause, and a sensitivity fit without
   the drift rule.

## 10. Results (run 2026-08-18, `bandwidth_sweep/` in the sweep folder)

16 runs, 3192 epochs (4 channels x 50 pulses, 2 epochs dropped at file
edges), threshold fixed at 100 uV, all one-knob checks pass. Verdict lines
are in `verdict.txt`; the numbers below are recording-contact medians unless
stated (A-021 is 2.5 mm from the stim contact A-026, A-022 2 mm, A-025
0.5 mm).

**Arm A (analog lower cutoff).** Median recovery 0.095 Hz 871 ms (45 %
censored), 1.1 Hz censored (>= 900), 9.9 Hz 164, 29.9 Hz 65, 97.8 Hz 63,
324 Hz 52 ms. Pooled log-log slope 0.65 [0.57, 0.74] over the four
uncensored runs -> intermediate, but the pooling hides two regimes:
A-021 (far contact) follows the pole with slope 1.19 [1.11, 1.26]
(2 ms at 324 Hz, 21 ms at 98 Hz, 56 ms at 30 Hz, 126 ms at 10 Hz), while
A-025 (next to the stim contact) sits on a floor of 94-300 ms whatever the
cutoff (slope 0.50) and A-022 in between (52-153 ms, slope 0.28). So the
low-frequency analog pole does set the recovery of the far contact, but
the near contacts carry an additional, amplitude-dependent component of
~50-300 ms that no high-pass setting removes.

**Arm B (DSP cutoff).** off 871 (censored), k=14 856 (censored), k=12
498 (drift-flagged), k=10 169, k=8 88, k=5 44 ms. Slope 0.32 [0.28, 0.36]
over k=10/8/5, 0.42 [0.39, 0.44] with k=12 included. The additive model
fits the four DSP-on runs almost exactly: recovery = 2.87 [2.69, 3.04] x
tau + 82 ms (R2 vs run medians 0.98); a linear high-pass recovering from
the rail predicts 4.16 x tau. tau_fit ~= tau_nominal for k=10 and k=12
(ratio 0.92 and 1.26; 54-64 % of tails exponential with R2 >= 0.9), i.e.
in the DSP-on runs the tail *is* the DSP step response plus a fixed
~80 ms floor. The DSP is therefore not "exonerated": its cutoff sets the
linear part of the recovery (k=12 -> ~500 ms in PBS at 100 uA) and it is
the knob that shortens it. The same setting in vivo (260817) gave 174 ms
median at lower currents.

**Arm C (analog upper cutoff).** Recording contacts do not rail at any
setting except A-021 at 7500 Hz (26 % of trials; zero at <= 3008 Hz); the
0-5 ms peak on the recording contacts falls 8722 -> 391 uV from 7604 to
300 Hz. The stim contact rails at every setting and rails *longer* at
narrower bandwidth (25 ms at 7604 Hz -> 60 ms at 300 Hz). Recovery on
this arm is 465-900 ms because the analog pole is 1 Hz throughout; the
post-train baselines drift by hundreds of uV (electrode polarisation).

**Noise (fig6).** The broadband noise (> 5 Hz SD of the stimulation-free
segment) is 3.4-4.5 uV on every channel at every setting; raw SD
differences (up to ~1 mV) are sub-5 Hz drift that the low corners let
through. Baseline SD in the -500..-50 ms window is tail-contaminated at
long tau (245-940 uV) and equals the noise floor (3.7-5 uV) at short tau.

**Recommendation printed:** DSP 152 Hz (k=5), analog 0.1 Hz, upper
7604 Hz: 44 ms median [43, 52] (per channel 34 / 47 / 139 ms), noise
3.4 uV; runner-up analog 324 Hz / DSP off: 52 ms (2 / 52 / 94 ms). Both
remove everything below ~150-300 Hz (spikes only). If LFP is needed, the
ladder is DSP k=8 (18.7 Hz): 88 ms (72 / 80 / 216); k=10 (4.7 Hz):
169 ms (165 / 130 / 228); k=12 (1.17 Hz, the in-vivo setting): 498 ms.

**Caveats.** 1 s IPI: recoveries >= ~800 ms are censored and their
baselines sit on the previous tail (local centring + drift flags handle
this; the spec-centred recovery is in table1 for comparison). PBS at
100 uA into 123-185 kOhm; the near-contact floor is amplitude-dependent
and will differ in vivo -- the *scaling* with the cutoff is what transfers.
