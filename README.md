# Intan RHS stimulation analysis (notebook agent)

Python/Jupyter workflow for multichannel Intan RHS recordings with electrical
stimulation. The user interface is `Plot_All_Channel_Data_Wideband.ipynb`;
the implementation lives in the `.py` files next to it. Full technical detail
and history: `STIM_AGENT_HANDOFF.md`. (`agent.md` describes the separate MATLAB
spike-sorting workflow.)

## Working agreements

- **Do not run data analysis on newly revised code without asking the user
  first.** After a code change, verify with the synthetic self-test
  (`/usr/local/bin/python3 -m stim_analysis.selftest`), `py_compile`, the
  reload-chain check and scratch copies of a small session. Running any
  function, the CLI or the batch runner on real recordings -- and writing into
  data folders -- is the user's call.
- Preview never writes. Save buttons write only the preview they hold, into
  the selected data folder, atomically (temp file + replace).
- Interpreter: `/usr/local/bin/python3` (3.12) and the `py312-rhs` kernel;
  bare `python3` is anaconda 3.9 and must not be used.
- Layering: notebook -> `wideband_main_ui.py` (ordered reload only) ->
  `wideband_functionN_ui.py` (widgets only) -> `plot_rhs_*` / `rhs_*` /
  `stim_analysis/` (parsing, numerics, output paths).

## Functions

| Function | Purpose | UI module | Numerics |
| --- | --- | --- | --- |
| 0 | Rename RHS folders from the recorded stim waveform | `wideband_function0_ui.py` | `rename_rhs_folders_by_stim_waveform.py` |
| 2 | Continuous traces: raw, or bandpass + amplitude window, optional ignore-stim | `wideband_function2_ui.py` | `plot_rhs_raw_wideband_with_stim_legend.py` + `plot_rhs_filtered_wideband.py` |
| 3 | Stim-triggered event grid (quick look; epoch -> blank -> filter) | `wideband_function3_ui.py` | `plot_rhs_stim_triggered_events.py` |
| 5 | Pre/post neuromodulation band power (event-locked: Function 6 or the batch runner) | `wideband_function5_power_ui.py` | `plot_rhs_power_analysis.py` |
| 6 | Session-level stimulation analysis (Spec v2) | `wideband_function6_session_ui.py` | `stim_analysis/` + `run_stim_analysis.py` |

Functions 1 and 4 were folded into Function 2 on 2026-08-20: Raw mode is the
old Function 1, Filtered + "Ignore stim" is the old Function 4.

File formats: Functions 2, 3 and 5 read both `.rhs` and `.rhd` folders
(decided per folder in `rhs_stim.folder_recording_format`). RHD recordings
carry no stim channel, so Functions 3 and 5 recover the Keithley pulse train
from the amplifier trace (`rhd_timing.py`, Function 7's comb fit) and run on a
synthetic unit-amplitude stim proxy -- timing is real, amplitude is not.
Function 6 stays RHS-only: its validation reads the RHS stim marker. Function
0 renames RHS folders only. The batch runner is RHS-only for now.

Shared: `wideband_ui_common.py` (widgets, previews, atomic saves),
`rhs_stim.py` (channel reading, stim-channel resolution), `rhs_reader.py`
(multi-channel RHS readers), `rhs_naming.py`, `rhs_files.py`.
Batch: `batch_run_wideband_main_ui.py` runs the raw / filtered / events /
response / power outputs for every run below a parent folder (CLI mode names
unchanged; `--power-mode event` remains the scripted home of event-locked
power).

## Function 6: session-level stimulation analysis (Analysis Spec v2)

Operates on a **session parent folder** (one sub-folder per RHS run) and
answers the gating question first -- how much of each record is usable and
what analysis window survives the stimulation artifact -- then runs the
secondary analyses on the conditions that survive.

Stages (each one gates the next):

1. **validate** -- every run: sample rate from the header, pulses commanded
   (`settings.xml`: `NumberOfStimPulses` x trains) vs detected on the stim
   marker, compliance from data (pulse-count drop or the RHS compliance bit),
   empirical rail level and % samples railed per channel, metadata
   cross-checks (data wins over `settings.xml` over folder name), block
   assignment (baseline / block1 amplitude ladder / block2 width sweep /
   block3 paired pulses), exclusions. `table01_validation.csv` exists before
   any analysis runs; a broken run is listed as `error`, never skipped silently.
2. **recovery** (spec section 4) -- raw epochs (-600..+900 ms, padded), per
   trial: `threshold = max(3 x baseline SD, 100 uV)`, recovery = last sample
   above threshold before a >= 20 ms quiet run, rail duration, censoring,
   baseline contamination by the previous pulse. Per channel x amplitude: the
   derived post-window start (P90 of recovery + margin, configurable), retained
   trials, verdict `early_ok` / `late_only` / `unusable`. Figures 1-3, tables
   1-3.
3. **all** -- epoch -> blank -> filter (neighbouring pulses inside the filter
   pad blanked too) -> per-trial band power and RMS; comparisons (a) within
   epoch, (b) block vs no-stim baseline, (c) across amplitude; first-vs-last
   drift; charge dependence; spatial decay; compliance characterisation;
   linear vs through-origin vs sigmoid fits with AIC/BIC and I50 bootstrap;
   channel random-effect model with impedance covariate; log-normal checks;
   shuffle control; figures 4-8 and S1-S4; metadata JSON with the exact
   filter designs.

Rules baked in: the continuous trace is never filtered; high-pass >= 1 Hz;
fixed axis and colour limits everywhere; every caption states n trials
retained/rejected, blanking window and filter; baseline and post windows are
paired by event id and cropped to equal length (unequal lengths bias mean dB
on pure noise); the shuffle control must come out null.

How to run:

```bash
# notebook: Block 6 cell -> Validate -> Generate Preview -> Save Bundle
/usr/local/bin/python3 run_stim_analysis.py "<session folder>" --stage validate
/usr/local/bin/python3 run_stim_analysis.py "<session folder>" --stage recovery
/usr/local/bin/python3 run_stim_analysis.py "<session folder>" --stage all [--bootstrap N] [--no-per-run] [--dry-run]
/usr/local/bin/python3 -m stim_analysis.selftest        # synthetic-data checks, writes nothing real
```

Outputs: `<session folder>/stim_analysis/` (tables, figures, per-trial CSVs,
`stim_analysis_metadata.json`, `run_log.txt`) plus per-run CSVs next to each
run's `.rhs` files (untick the box / `--no-per-run` to skip).

Package layout: `stim_analysis/{config, load_rhs, validate, epoch, recovery,
metrics, stats, models, figures, secondary, pipeline, selftest}.py`. All text
parsing lives in `config.py`; all output paths in `pipeline.render_outputs`;
`pipeline.run_session` never writes.

## Session 260817 (as analysed 2026-08-18)

18 runs validated, 11 included (500/800 us and all Block 3 excluded for
compliance, one truncated `.rhs` reported), baseline auto-selected. No channel
x amplitude condition reaches `early_ok`: median recovery is 64-490 ms
everywhere, including 10 uA, so every cell is `late_only` or `unusable`; the
baseline-run shuffle null is clean. That is the spec's section 12 outcome:
this electrode/impedance configuration cannot support evoked-potential
measurement; late-window and block-level measures are what remain. Outputs
were written to `.../Jill/20260817 re stim Noah 1/stim_analysis/`.

## Environment

`/usr/local/bin/python3` 3.12 with numpy, scipy, matplotlib, pandas,
statsmodels, ipywidgets 8; select the `py312-rhs` kernel in VS Code or
JupyterLab (`/usr/local/bin/python3 -m jupyterlab`).

## Filter diagnosis (`filter_diag/`)

Tests whether the 100-500 ms artifact "recovery" is the step response of the
Intan DSP high-pass (1.166 Hz, tau = 2^12/fs = 136.5 ms at 30 kHz) combined with
ADC railing, per the Filter Diagnosis Spec: A settings table, B exponential tail
fits and tau invariance, C synthetic step through the instrument chain (rail
modes freeze / track / spec) with the unchanged recovery algorithm, D live vs
post-mortem, E additive decomposition, F fit-and-subtract and inverse-DSP
removal, one-line verdict.

```bash
/usr/local/bin/python3 -m filter_diag.selftest        # synthetic checks (no data)
/usr/local/bin/python3 -m filter_diag.run_all --dry-run   # both sessions, writes nothing
/usr/local/bin/python3 -m filter_diag.run_all             # writes <live>/filter_diagnosis/
```

Defaults: live `~/SynologyDrive/Research/Stimulation/20260817 re stim Noah 1`,
dead `.../20260816 re stim Yun dead rat`; compliance-flagged, paired-pulse and
no-stim runs excluded; A-031 excluded; stim contacts reported separately. Do
not run it on recordings without asking (working agreement above).

## Bandwidth sweep (`bw_sweep/`)

In vitro PBS sweep of session `20260818 re stim in vitro filter settings`:
does artifact recovery scale with the amplifier's low-frequency time constant?
Arm A analog lower cutoff (0.1-300 Hz, DSP off), Arm B DSP cutoff (off, k =
14/12/10/8/5), Arm C analog upper cutoff (7500-300 Hz). Arms are assigned from
the RHS headers, so the two shared recordings (0.1 Hz/DSP off = Arm A + Arm B
"off"; 1 Hz/7500 Hz = Arm A + Arm C 7500) and the duplicated 500 Hz run are
handled explicitly, and a one-knob check per arm is written to table0.
Recovery uses the unchanged `compute_recovery` with a **fixed 100 uV
threshold** (`threshold_k = 0`), epochs centred on the local -50..-5 ms
pre-pulse window (the spec's -500..-50 ms baseline still gives baseline SD; the
spec-centred recovery is kept as a secondary column). Outputs: fig0-fig6,
table0/table1/table1b/table2, per-epoch CSV, verdict.txt (one line per arm +
setting recommendation), metadata.json. Plan and design notes:
`BANDWIDTH_SWEEP_PLAN.md`.

```bash
/usr/local/bin/python3 -m bw_sweep.selftest         # synthetic checks (no data)
/usr/local/bin/python3 -m bw_sweep.run --dry-run    # computes, writes nothing
/usr/local/bin/python3 -m bw_sweep.run              # writes <root>/bandwidth_sweep/
```

Default root: `~/SynologyDrive/Research/Stimulation/20260818 re stim in vitro
filter settings`. Do not run it on recordings without asking (working
agreement above).

Run on 2026-08-18 (outputs in `<root>/bandwidth_sweep/`, details in
`BANDWIDTH_SWEEP_PLAN.md` section 10): the far recording contact follows the
high-pass pole (Arm A per-channel slope 1.19), the contact next to the stim
site sits on a 94-300 ms floor at any cutoff; the DSP-on runs are described
by recovery = 2.9 x tau + 82 ms (R2 0.98), so the DSP cutoff sets the linear
part of the recovery and is the knob that shortens it (k=12 ~500 ms -> k=5
44 ms in PBS at 100 uA); recording contacts do not rail below 3 kHz upper
bandwidth; broadband noise is 3.4-4.5 uV at every setting.
