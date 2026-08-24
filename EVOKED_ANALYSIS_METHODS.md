# Function 7 Evoked-Response Analysis: Methods Reference

Authoritative description of the `evoked_sweep/` analysis as of 2026-08-22.
Audience: the stimulation agent and any future session extending Function 7.
Code lives in `/Users/jf/Claude/Stimulation agent/evoked_sweep/`; UI in
`wideband_function7_evoked_ui.py`; CLI in `run_evoked_sweep.py`. Every claim
below is enforced by `python3 -m evoked_sweep.selftest` (125 synthetic checks).

Design principles carried throughout: measurements win over labels, labels are
checks; suspicious data is **flagged, never corrected**; runs.csv columns are
append-only; the UI parses nothing (`config_from_text_fields` is the only text
parser); preview writes nothing, Save Bundle writes atomically.

---

## 1. Inputs and session layouts

Recordings are Intan RHD (`rhd_reader.py`), 20 kHz, one `.rhd` per ~60 s.
Three layouts are discovered by `naming.discover_runs` (a "run" = any folder
directly containing `.rhd` files, found by rglob at any depth):

1. **Config-folder sessions** (20260819 style): `<date>/<wiring folder>/<run>`,
   wiring parsed from the folder name ("stim 1 stim ground 2 recording ground 3"),
   amplitude from legacy run names (§2). Nested amplitude folders
   (`-0_05mA/<run>`) inherit the parent's amplitude (`amplitude_from_parent`).
2. **Wiring-folder sessions**: whole-session mode pointed directly at one
   wiring folder; every run below it inherits that wiring.
3. **Flat protocol sessions** (20260821 style): runs at the session root with
   protocol-named files (§2); wiring label comes from the `wiring_label`
   config/UI field, else the session folder name.

Baseline classification: a name containing "baseline" is always a baseline;
a run with protocol evidence (protocol name or `*.onset.txt`) never is; bare
runs at a session root without either are baselines (legacy rule). Baselines
are excluded from stimulus analysis and counted in a verdict note.

Wiring filters (whole-session): `wiring_include` / `wiring_exclude` —
comma-separated case-insensitive substrings of the config folder name
(include first, then exclude, e.g. exclude `artifact`). Excluded runs are
listed in a verdict note, never silently dropped.

## 2. Amplitude and protocol parsing (`naming.py`)

**Legacy names** (20260819): leading token before "mA"; underscore = decimal
point (`-0_02mA` → −0.02), bare leading zero = tenths (`-02mA` → −0.2).
Unsigned names parse positive with `sign_assumed`. A trailing standalone
integer is a replicate index. Caveat flags parsed from the name: `unsure`,
`late_start`, `interval_noted`, `com_port_noted`.

**Protocol names** (20260821):
`<A>mA_<B>mA_pulsewidth<W>s_interval<I>s_pulsenumber<N>` on the run folder,
the renamed first `.rhd`, or the `.onset.txt`. Amplitude = leading phase A
(signed; the sign is later overridden by the scope, §4). Sets per-run
overrides `expected_pulses_run`, `pulse_width_s_run`, `interval_s_run` which
replace the config's protocol values for that run (`pipeline._effective_config`),
with `max_period_s` raised to 2×(interval+width) for Keithley serial overhead
and `envelope_ms` coarsened to ≥2 ms for wide pulses. The legacy parser is
shielded from protocol names (they would misparse as 0.0 mA).

**Ground truth on Keithley widths** (bench-verified): the 08/19 monophasic
pulses were genuinely 5–7.5 ms (resistor test, 2026-08-22). The slow biphasic
protocol commands 0.3 s/phase and delivers ~0.36–0.5 s phases, ~0.9 s
envelope, ~5.5 s period vs 4.8 s labelled — serial overhead stretches every
timescale; always trust the measured value.

## 3. Channel selection and geometry

`load.healthy_channels`: keep channels with header impedance ≤ 1 MΩ
(`healthy_impedance_max_ohms`); floating contacts read ~10 MΩ. Explicit
channel lists override. Contact positions (`naming.contact_positions_um`):
trailing channel index × `contact_pitch_um` (500), zeroed at the lowest
healthy index; `contact_order` overrides the index order. Positions are
RELATIVE — the stim-site offset was never measured. Shown in fig1 legends and
fig5 titles, and as `contact_position_um` in runs.csv/peaks.csv.
**Shorted-array caveat** (20260821): all recording channels tied together →
one signal on N channels (correlations ≥0.999); spatial columns are
meaningless there and the channels are redundant copies.

## 4. Stimulus timing — three sources, in priority order

runs.csv column `timing_source` records which was used per run.

### 4a. Oscilloscope sync (`scope_sync.py`) — "scope"
Used when the session has an `oscilloscope/` folder (auto-detected; or
`scope_dir` config) and the run has a `*.onset.txt`
(`pulse_onset_epoch_s=<unix epoch of first pulse command>`).
Scope captures: ASCII, one per run, filename = start epoch; columns
`timestamp time voltage1 current_mA1`; **column 1 is absolute epoch per
sample**; sampling is bursty (dt 6 µs–16 ms) so nothing may assume a uniform
grid; files up to 256 MB are streamed with seek-bisection on the monotone
timestamps. The `current_mA1` column is `voltage1 × 200` with an arbitrary
probe scale — it is the **voltage across the electrode**, used for TIMING and
SHAPE only, never amplitude.

Pipeline per run:
1. `read_onset_epoch` + `run_start_epoch` (folder name, 1 s resolution) +
   `find_scope_capture` (span containment; 0-byte decoys skipped).
2. `read_scope_pulse_epochs`: deviation from median, threshold =
   max(0.5×p99.5, 6×MAD) (noise-only → zero events); events grouped by TIME
   gaps (1 s); each event's edge walked backward to the ramp base (low floor
   max(3×MAD, 5%×p99.5)) because a current step across the electrode RC makes
   a voltage ramp whose half-max crossing sits well inside the pulse.
3. `_segment_pulses`: 5 ms uniform resample, smooth, phases from the voltage
   slope — first extremum = phase-1 end, last opposite-slope span = phase 2,
   envelope = onset→second extremum; `lead_sign` from which extremum comes
   first. Envelope and phase-1 are reliable; the IPD/phase-2 boundary is soft
   (steep interphase decay). Medians across pulses → `PulseTrain.width_ms`
   (= envelope), `phase1_s`, `ipd_s`, `phase2_s`, `lead_sign`.
4. Fine alignment: naive onsets (scope epoch − folder-start epoch) slid
   ±`scope_align_window_s` (5 s) against the amplifier robust-z envelope,
   scored by window-averaged EDGE CONTRAST (mean z in +20..+180 ms minus
   −180..−20 ms, via prefix sums); confidence `align_z` = peak vs the curve
   outside its own ±width neighbourhood; rejected below `min_comb_z` (5) →
   comb fallback with reason. `clock_offset_s = −δ*`: host epoch of Intan
   sample t = folder_start + t + clock_offset_s.
5. **Polarity override**: if `lead_sign` contradicts the label's sign, the
   amplitude sign is flipped (magnitude kept), flag `polarity_from_scope`,
   listed in the verdict. Needed because the control script always writes the
   anodic-first name.
6. **Envelope override**: for scope-timed runs the measured envelope replaces
   `pulse_width_ms` in the effective config, so the during/post split, peak
   windows and coupling evidence all use ground truth.

Known gap: runs whose stimulus is invisible in the amplifier (1–2 µA) fail
step 4 and fall back to comb, which then mis-measures the width — their
windows are wrong. Planned fix (not yet built): a `scope_unconfirmed` tier
applying scope times with the session-median clock offset. The control script
will also start writing `recording_start_epoch_s` (field name TBD by the
user); the reader should learn it when it appears.

### 4b. Comb recovery (`pulses.recover_pulses`) — "comb"
For sessions/runs without scope data (all of 20260819). RMS envelope
(`envelope_ms` bins) → robust z (MAD) → max across channels → autocorrelation
period candidates (bounded by `min_period_s`/`max_period_s`) → vectorised comb
scan → snap to fine peaks (wide pulses ≥50 ms snap to the leading half-max
edge, not the plateau argmax) → ≤3 least-squares refits. Confidence `comb_z`
(SD the winning alignment beats all others by; <5 ⇒ unreliable). Checks vs
protocol → issues → `timing_ok`. Whole-session second pass: periods
established by confident runs (≥2 agreeing) re-time the weak runs
(scope-timed runs are skipped; they also contribute period priors).

### 4c. Effective config
`analyse_run` computes a per-run effective config (protocol overrides + scope
envelope) used by every downstream stage.

## 5. Per-pulse response metrics (`metrics.evoked_deflection`)

Epochs: −20 ms .. `response_window_ms` (UI "Resp win"; capped at period−5 ms)
around each onset; baseline = mean of the 20 ms pre-onset window; all metrics
on the baseline-centred epoch; raw µV, no filtering.
- Legacy (never redefined): `evoked_pp_uV` = p-p over the whole window;
  `peak_latency_ms` = argmax |response| (lands on the onset coupling spike by
  design); `post_pulse_fraction` = energy after nominal width / total.
- Window split: during = [0, effective width + `post_pulse_guard_ms`),
  post = the rest → `evoked_pp_during_uV`, `evoked_pp_post_uV(+iqr)`,
  `post_peak_latency_ms`; per-epoch `baseline_sd_uV`.
- Gap baseline: p-p over 150–250 ms post-onset; when the response window
  swallows that (slow protocols) it auto-relocates to the last 100 ms before
  the next pulse. `snr_vs_gap` = pp / gap p-p. `pre_train_pp_uV` = p-p of the
  2 s before the train.
- The centred epoch stack and mean waveform are retained for peaks/figures.

## 6. Per-peak analysis (`peaks.analyse_channel_peaks`)

On the mean waveform, smoothed with a **Gaussian** low-pass (−3 dB at
`peak_lowpass_hz`, 500; 0 disables). Gaussian, not Butterworth: a Butterworth
rings ~10% around pulse edges and the detector reports the ringing as peaks;
a Gaussian's step response is monotone and cannot invent one.
Detection: `scipy.signal.find_peaks` on the windowed mean and its negation,
window = [effective width + guard, response end]; prominence threshold =
`peak_prominence_k` (3) × max(baseline MAD of the smoothed mean, trial-SD/√n)
— the MAD catches structure averaging didn't remove, the SEM is the
statistical floor; floor 2 µV. Keep the `max_peaks` (5) most prominent,
sort by latency, label N1/P1/N2… by polarity+order. `edge_suspect` = within
`edge_flag_ms` (1.5) of the MEASURED off-edge (reported, never dropped).
Per peak, per pulse: extremum of matching polarity within
±`peak_search_half_ms` (3) on the smoothed epoch stack → median/IQR
amplitude, latency jitter (SD), presence (|median| > `presence_k`×trial SD),
adaptation (median last-10 / first-10), amplitude slope per pulse index.

## 7. Band power (`metrics.band_power`)

Bands (config): delta 1–4, theta 4–8, beta 12–30, gamma 30–100, high
100–300 Hz. Two estimators, cross-checking each other:
- **Comb-excluded**: Welch (nperseg 4×fs) over the train window vs an
  equal-length pre-train window; every bin within max(0.5 Hz, 2 df) of any
  harmonic of the pulse rate excluded before integrating (a 5 ms pulse comb
  reaches past 200 Hz — notching a few harmonics is not enough);
  dB = 10 log10(train/base) of mean in-band PSD.
- **Gap-based**: PSD from the quiet stretch [response window, period−5 ms]
  after each onset vs matched pre-train chunks; valid only for bands with ≥3
  cycles per gap — the cutoff is `gap_minimum_hz` = 3/gap_duration (columns +
  fig3 annotation + dynamic verdict caveat). At 259 ms period this blanks
  delta/theta/beta; at 5.5 s everything is valid.
Amplifier caveat: DSP high-pass 0.777 Hz — nothing below ~0.8 Hz is real.

## 8. Post-train change (`metrics.post_train_change`)

10 s window starting 2 s after train end vs matched pre-train window: slow
level delta, dominant rhythm 0.2–20 Hz before/after, broadband SD ratio (dB).

## 9. Artifact / coupling evidence (`artifact.py`)

Within-run (per channel): `fast_latency` (legacy latency ≤1.5 ms),
`stops_with_pulse` (tail fraction ≤0.30), `suspicion` = their mean; window-
split additions: `fast_latency_post` (post-window latency ≤1.5 ms past the
pulse — the onset spike cannot win this), `coupling_ratio` (during/post p-p),
`post_response_detected` (any clean post-pulse peak). Sweep-level (≥3 points
per wiring/channel): linear fit of p-p vs |I| — "coupling-like" if r²>0.95
with near-zero intercept; polarity asymmetry when both polarities exist.
**Decade-mislabel detector**: Theil–Sen fit in log-log (median pairwise
slope — one 10× point cannot drag it); a point is suspect when its residual
exceeds max(0.3, 0.5|slope|) log-units AND shifting its current by exactly
±1 decade lands it within 0.15 of the trend; ≥4 points and ≥3 distinct
levels required; a RUN is flagged when ≥half its tested channels agree on
the shift direction → `decade_suspect(_note)` columns, verdict section,
open "10x?" markers in fig1. Interpretation caveat: instrument instability
(saturation, sensitivity change) also fires it — the flag reports "off the
shared curve", the cause needs the scope/notes.

## 10. Outputs (all through `pipeline.render_outputs`, atomic on save)

`<target>/evoked_sweep/`: `verdict.txt` (timing incl. source/clock/shape/
polarity lines, warnings, dose-response, decade section, coupling, peak
summary, dynamic caveats, notes), `runs.csv` (one row per run×channel; legacy
46 columns + appended: window split, positions, gap cutoff, decade, timing
source/clock/scope columns, per-run protocol), `peaks.csv` (long format, one
row per run×channel×peak), `conditions.csv` (sweep fits), figs:
1 dose-response (log-log, color=channel+position, marker=polarity, open+10x?
= decade suspect) · 2 mean waveforms · 3 band power (Hz-ranged titles;
filled=comb, open ▲=gap, grey ties pair them; invalid-band note) · 4 artifact
scatter · 5 peak-annotated waveforms (grey=all epochs, bands=nominal pulse+
guard, red dashes=measured off-edge, dotted=detection threshold, ▲▼=peaks,
open=edge?) · 6 per-peak dose-response (color=label, fill=present,
square=edge?). UI single-run/session modes preview identically; the timing
table's `src` column shows scope/comb per run.

## 11. Session-specific interpretation notes

- **20260819**: comb-timed; 57 runs, 5 wirings; pulse 5–7.5 ms verified by
  resistor; P1 ≈ +360 µV at 13 ms is the benchmark response; decade flags
  there traced to real `-02`/`-0_02` naming mixups.
- **20260821**: scope-timed; shorted array (one signal ×5 channels); slow
  biphasic ~0.9 s envelope at ~5.5 s; two runs polarity-corrected from the
  scope; structural limit — any short-latency EP (like the 13 ms P1) occurs
  INSIDE the pulse and is unrecoverable without artifact subtraction; the
  post-pulse window rides the electrode discharge tail, so treat detected
  peaks there as candidates (tail-subtraction, latency-vs-current stability,
  and anodic-vs-cathodic comparison are the discriminators). Electrode
  interface changed ~4× after the 100 µA run (scope voltage drop at fixed
  current) — check impedance before/after in future sessions.

## 12. Verification

`cd "/Users/jf/Claude/Stimulation agent" && /usr/local/bin/python3 -m evoked_sweep.selftest`
(125 checks: naming/protocol/geometry, comb recovery incl. slow biphasic,
window split, peaks, band power incl. gap validity, decade detector, wiring
scope/filter, full scope-sync round trip with planted clock offset and
polarity flip). Reload chain: `python3 wideband_main_ui.py`. CLI parity:
`run_evoked_sweep.py --help`.
