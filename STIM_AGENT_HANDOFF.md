# Stimulation Analysis Agent Handoff

Last reviewed: 2026-08-18  
Workspace: `/Users/jf/Claude/Matlab code`  
Git snapshot at handoff: branch `worktree-fn3-blank-space` (Function 6 /
`stim_analysis` package added 2026-08-18; the same files are applied to `main`)  
Worktree status before creating this handoff: clean

## Purpose

This document is the continuation context for the Python/Jupyter Intan RHS
stimulation-analysis workflow. The main user interface is a VS Code/Jupyter
notebook. The user should be able to paste any RHS session folder from anywhere
on the computer, generate inline previews, and save outputs manually into that
same selected RHS session folder.

The workflow is for multichannel neural recordings where stimulation may occur
on only one recorded amplifier channel. It supports raw wideband inspection,
bandpass/amplitude views, stimulation-triggered response plots, response-only
plots, folder naming from the stimulation waveform, and LFP/field-potential
power analysis.

The existing `agent.md` describes a separate MATLAB spike-sorting workflow. Do
not replace it with this file.

## Start Here

Main notebook:

`/Users/jf/Claude/Matlab code/Plot_All_Channel_Data_Wideband.ipynb`

Open it in VS Code as a Jupyter notebook, select the Python 3 kernel, and run the
launcher cell below the desired Markdown section. The launcher source is hidden
by notebook metadata so the notebook behaves like a compact UI. The actual code
lives in helper `.py` files.

The notebook currently contains Functions 0 through 5:

| Function | UI purpose | UI module | Numerics |
| --- | --- | --- | --- |
| 0 | Preview and apply RHS folder renames from the recorded stim waveform | `wideband_function0_ui.py` | `rename_rhs_folders_by_stim_waveform.py` |
| 2 | Continuous traces: raw wideband, or bandpass + amplitude filtered, optional ignore-stim (absorbed the old Functions 1 and 4 on 2026-08-20) | `wideband_function2_ui.py` | `plot_rhs_raw_wideband_with_stim_legend.py` + `plot_rhs_filtered_wideband.py` |
| 3 | Plot stim-triggered response events, three events per row (epoch -> blank -> filter) | `wideband_function3_ui.py` | `plot_rhs_stim_triggered_events.py` |
| 5 | Pre/post neuromodulation band-power analysis (event-locked power: Function 6 or `batch_run_wideband_main_ui.py --power-mode event`) | `wideband_function5_power_ui.py` | `plot_rhs_power_analysis.py` |
| 6 | Session-level stimulation analysis (Spec v2): validation, artifact recovery gating, secondary analyses over every run of a session | `wideband_function6_session_ui.py` | `stim_analysis/` package + `run_stim_analysis.py` CLI |

Shared across the UI modules: `wideband_ui_common.py` (widget factories, preview
rendering, atomic saves, error markup), `rhs_stim.py` (channel reading and
stim-channel resolution) and `rhs_reader.py` (multi-channel RHS readers: the
batch runner's reader moved here verbatim, plus a fast structured-dtype reader
used by `stim_analysis`).

`wideband_main_ui.py` is the notebook launcher layer. Its public functions are:

```python
show_function0_rename_rhs_folders(globals())
show_function1_raw_wideband(globals())
show_function2_bandpass_filtered(globals())
show_function3_stim_triggered_events(globals())
show_function4_recorded_response_only(globals())
show_function5_power_analysis(globals())
show_function6_session_analysis(globals())
```

`wideband_main_ui.py` contains no function-specific code. It exists to **own
ordered module reloading**, and that is the only reason the layer is there.

Until 2026-08-17 the UI code for Functions 0, 1, 2 and 4 was stored as Python
source escaped into single-line string literals (`_BLOCK_N_SOURCE`) and run
through `exec()`. That is gone; each function is now a real module. A dead
`_BLOCK_3_SOURCE` was deleted at the same time.

## Layering Rule

Four layers, each with one job. The third row is the one that matters most,
because violating it is what let the batch runner drift away from the notebook.

| Layer | May do | Must not do |
| --- | --- | --- |
| Notebook | display cells, call one launcher | hold any logic |
| `wideband_main_ui.py` | ordered reload, dispatch | anything function-specific |
| `wideband_functionN_ui.py` | read widget values, display results | parse, name files, branch on data |
| `plot_rhs_*` / `rhs_*` | parse, compute, decide output paths | know that widgets exist |

Function 5 is the model to copy: its output paths come back *from* the analysis
(`result.png_path`, `result.csv_path`) rather than being computed in widget code.
The UI asks; the helper decides.

## Reloading

The notebook cells call `importlib.reload(wideband_main_ui)`, which does **not**
reload submodules. `wideband_main_ui._RELOAD_CHAIN` therefore reloads the
dependency graph explicitly on every launch, **leaves first**. That order is
load-bearing: reloading a module re-executes its `from X import y` lines against
`sys.modules`, so a dependent reloaded before its dependency silently re-imports
the stale symbol and your edit appears to do nothing.

Three things worth knowing:

- Loading a **new data file needs no reload** -- that is just pasting a path into
  the widget. Reload is only for code edits.
- Package modules use dotted names in the chain (`stim_analysis.config`, ...);
  `check_reload_chain()` maps them to `stim_analysis/config.py`.
- Adding a new helper module means adding it to `_RELOAD_CHAIN` **in the right
  position**. Getting this wrong is not a silent staleness bug: reloading a
  module re-executes its `from X import y` lines, so if `X` has not been
  reloaded yet, any name newly added to `X` raises
  `ImportError: cannot import name ...` from inside the launcher. This happened
  on 2026-08-17 when `rhs_naming` was appended after the `plot_rhs_*` modules
  that import it. Run `python3 wideband_main_ui.py` after any change to the
  chain -- `check_reload_chain()` parses the imports and reports any module
  listed before something it depends on, or missing from the chain entirely.
- `importlib.reload` cannot fix objects already built from old class
  definitions. If behavior ever looks impossible, restart the kernel; that
  remains the ground truth.

## Environment

The correct interpreter is **`/usr/local/bin/python3` (3.12.0)**. Packages
observed at this handoff:

```text
numpy 2.4.6
scipy 1.18.0
matplotlib 3.11.0
ipywidgets 8.1.8
IPython 9.14.1
jupyter installed
```

Install the required packages on another environment with:

```bash
python3 -m pip install numpy scipy matplotlib ipywidgets ipython jupyter
```

VS Code also needs its Python and Jupyter extensions.

### Two interpreters live on this machine -- pick deliberately

```text
/usr/local/bin/python3        3.12.0   numpy 2.4.6   scipy 1.18.0   matplotlib 3.11.0   ipywidgets 8.1.8
/Users/jf/opt/anaconda3/bin/python3   3.9.7    numpy 1.20.3  scipy 1.7.1    matplotlib 3.4.3    ipywidgets 7.6.5
```

Bare `python3` on PATH resolves to **anaconda 3.9.7**, not the 3.12 environment.
Two consequences:

1. **Notebook kernel.** Select the `Python 3.12 (RHS analysis)` kernel
   (kernelspec `py312-rhs`). The anaconda kernel has ipywidgets 7.6.5, and VS
   Code's renderer targets ipywidgets 8 -- widgets silently render as nothing,
   with no error in the cell. If the UI ever appears blank, check the kernel
   before suspecting the code.
### Running the notebook in JupyterLab instead of VS Code

The VS Code install on this machine is ~2 years old (its Jupyter extension is
pinned at `2023.11`, which bundles the ipywidgets **7**-era renderer). Against an
ipywidgets 8 kernel that renderer produces **no widgets and no error**. A CDN
workaround is set in VS Code user settings
(`jupyter.widgetScriptSources: ["jsdelivr.com","unpkg.com"]`), which works but
needs internet and only applies to webviews created after the setting.

JupyterLab 4.6.3 is installed in the 3.12 environment and needs no workaround:

```bash
cd "/Users/jf/Claude/Matlab code"
/usr/local/bin/python3 -m jupyterlab
```

Do **not** run bare `jupyter lab`. `jupyter`, `jupyter-lab` and `pip3` on PATH
all resolve to anaconda 3.9.7, which has ipywidgets 7.6.5 and would reproduce
the blank-widget problem. Install into this environment with
`/usr/local/bin/python3 -m pip install ...`, never bare `pip3`.

Inside either front end, select the **`py312-rhs`** kernel
("Python 3.12 (RHS analysis)").

2. **Batch runner.** Invoke it as `/usr/local/bin/python3
   batch_run_wideband_main_ui.py ...`, not bare `python3`. The two stacks are 11
   scipy minor versions and a matplotlib generation apart, so the same session
   analyzed under each will not produce identical PNGs. Do not compare batch
   output produced under one interpreter with notebook output produced under the
   other.

The UI uses Matplotlib's noninteractive `Agg` backend and renders PNG bytes into
`ipywidgets.Image`, so plots appear inline rather than in pop-up windows. It
stores Matplotlib caches under the system temporary folder because the default
Matplotlib configuration directory may not be writable.

## Data Locations Used During Development

These are examples and are not hard requirements:

```text
/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260617 stimulation Pt
/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260624 pt stimulation
/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260715 ic implant re
```

Important neuromodulation example:

```text
20260624 pt stimulation_260715_175113 A-012 stim cathodic first 200us 10uA 20Hz 20 pulses 49ms RP
```

That session contains a few minutes of recording, about 15 minutes of
stimulation, and another few minutes of recording. It is appropriate for the
Function 5 pre/post mode because there is clean data on both sides of the
stimulation block.

Example that produced a short-pre-period warning:

```text
20260624 pt stimulation_260715_164756 A-012 stim cathodic first 200us 5uA 0.0987Hz 5 pulses 10135ms RP
```

Data are not stored in this code repository. Do not move or rename data unless
the user explicitly uses Function 0 and confirms the previewed rename plan.

## Core RHS Parsing

`plot_rhs_raw_wideband_with_stim_legend.py` contains the low-level Python RHS
reader used by the plotting and power helpers.

- It reads Intan RHS headers and amplifier channel names.
- Amplifier values are converted to microvolts with
  `0.195 * (uint16_value - 32768)`.
- `stim_data` is decoded from the RHS stim bitfield into signed commanded
  current in microamps using the header's stimulation step size.
- Files in a session folder are sorted and concatenated.
- Concatenated files must use the same amplifier sample rate.
- Timestamp discontinuities are counted and reported by Function 1.
- `read_rhs_amplifier_channel_names(folder)` returns recorded channels.
- `resolve_channel_selection(text, folder)` accepts `all`, one channel such as
  `A-014`, a range such as `A-014-16`, or comma-separated channels.
- `parse_time_window(text)` accepts `all` or absolute recording times such as
  `10-20 s`.
- Long traces use a min/max display envelope. This reduces plotted points but
  does not filter or modify the underlying signal.

The first channel with nonzero decoded `stim_data` is treated as the stimulation
channel. Plot labels append `*` to that channel where stimulation marking is
enabled.

## Function 0: Rename RHS Folders

Paste either a parent folder containing immediate RHS session subfolders or one
RHS session folder. Always use **Preview Rename Plan** before **Apply Renames**.

The generated suffix has this shape:

```text
A-026 stim cathodic first 100us 200uA 1Hz 30 pulses 999ms RP comp
```

It includes:

- stimulation channel;
- first-phase polarity;
- first-phase duration;
- first-phase amplitude;
- within-train pulse frequency when inferable;
- pulses in the first detected train;
- inferred refractory/silent interval (`RP`) when inferable;
- `comp` if the RHS compliance-limit bit was observed.

Important limitation: `RP` is inferred from the median silent sample interval
between adjacent pulses in the first detected train. It is not read from an
explicit Intan configuration field. A single isolated pulse cannot provide this
interval. Treat the generated title as a waveform-derived summary and review it
before renaming.

The helper strips a previously generated suffix before creating a new one and
avoids collisions by adding `__2`, `__3`, and so on.

## Function 2: Continuous Traces (Raw / Filtered)

Absorbed the old Functions 1 and 4 on 2026-08-20. Launcher:
`show_function2_continuous_traces`. All loading goes through
`rhs_stim.read_selected_channels`, which reads `.rhs` per channel or an entire
`.rhd` run in one pass (`rhs_stim.folder_recording_format` decides). On RHD the
stim triple carries a shared all-zeros `stim_uA` and the status line says the
format has no stim channel rather than implying an empty one.

Controls:

- `Mode`: `Raw wideband` (default) or `Filtered`.
- `Ignore stim (response only)` checkbox: Filtered mode only; skips the stim
  lookup entirely (the old Function 4). Toggling it swaps the amplitude seed
  `-100 - 100 uV` <-> `-500 - 500 uV` only while the field still holds the
  other state's default.
- `Channels`: `all`, explicit channels, or channel ranges.
- `Bandpass` / `Amplitude`: Filtered mode only; `all` or a numeric range, and
  a signed uV window that also pins the y-axis.
- `Time (s)`: `all` or an absolute range such as `0-60`.
- `Max points`: display-envelope limit; default `600000`.
- `Pulse`: Raw mode only; which decoded pulse to use in the caption.

Behavior per mode:

- **Raw**: no filter; stacked channels; biphasic pulse caption (amplitude and
  first-phase duration); status line reports displayed samples, stim pulses
  and timestamp gaps. Output `raw_wideband_*.png`
  (`default_output_path`).
- **Filtered + stim**: zero-phase Butterworth for numeric bands; only samples
  inside the amplitude window shown; stim channel starred; status reports RMS
  and % amplitude-selected, plus the sub-1 Hz high-pass warning. Output
  `filtered_wideband_*.png` (`default_filtered_output_path`).
- **Filtered + ignore stim**: same plot with `stim_channel_name=None`; output
  `recorded_response_*.png` with a time-window label
  (`default_response_output_path`).
- All modes resolve the stim channel with the all-recorded-channel fallback
  (selected channels first, then every other recorded channel) unless Ignore
  stim is checked.
- Preview generation does not save; **Save PNG** writes into the selected RHS
  folder.

## Function 3: Stim-Triggered Response Events

Active UI implementation: `wideband_function3_ui.py`.

Controls and current defaults:

```text
Channels: A-014 (accepts all and ranges)
Bandpass: all
Amplitude: -100 - 100 uV
Pre time (ms): 100
Post time (ms): 500
Train gap (ms): 10
Max points: 200000
Show stim current: checked
```

Behavior:

- It searches selected channels first, then every other recorded channel until
  it finds nonzero `stim_data`. Therefore neural-response channels can be shown
  without including the stimulation channel in `Channels`.
- `Train gap (ms)` groups pulse onsets into one stimulation train/event. For a
  100 Hz train, pulses are about 10 ms apart; a gap threshold of 12 ms groups the
  pulses into one event.
- Each event uses the exact decoded RHS stim trigger as `0 s`.
- `Pre time (ms) = 100` includes 100 ms before the trigger.
- `Post time (ms) = 500` includes 0 to 500 ms after the trigger.
- `Post time (ms) = 20-300` keeps the same trigger alignment but blanks the first
  20 ms of response and shows through 300 ms.
- A light grey vertical line marks the trigger at `0 s`.
- Selected response channels are stacked within each event.
- If **Show stim current** is enabled, the matching red stim-current waveform is
  plotted below all response channels for that event on the same x-axis.
- If it is disabled, no stim-current row is drawn and the filename gains
  `_noStim`.
- Channel labels use `*` for the detected stimulation channel when that channel
  is displayed.
- Event titles omit `Event 00x` and `s from stim`; they retain onset and duration
  information without overlapping the next panel.
- One combined PNG contains all events, arranged three events per row.
- Preview generation does not save; **Save PNG** writes that combined image into
  the selected RHS folder (via `rhs_files.atomic_write_bytes`).
- **Filter pipeline (since 2026-08-20)**: numeric bands are applied per event
  in the epoch -> blank -> filter order (`epoch_filter_channel_data`): each
  event's display window is cut with 500 ms padding (reflect-padded at
  recording edges), stim pulses (-1/+5 ms) and the response-blank window are
  linearly interpolated away, the epoch is bandpassed zero-phase, and the
  padding is trimmed. The continuous trace is never filtered. `Bandpass=all`
  remains a bit-identical pass-through. Every Function 3 PNG with a numeric
  band differs from pre-2026-08-20 output by design.
- **RHD folders**: no stim channel exists, so when `resolve_stim_channel`
  returns None on an `.rhd` folder the UI recovers the Keithley train from the
  amplifier trace (`rhd_timing.recover_stim_proxy`, Function 7's comb fit)
  using the `RHD pulses` / `RHD width (ms)` fields, and drives the event grid
  from a synthetic unit-amplitude stim proxy labelled `recovered`. The stim
  row then shows unit rectangles: timing is real, amplitude is not. The status
  line reports pulses, period, width and comb-z confidence.

## Function 5: Power Analysis

Active UI implementation: `wideband_function5_power_ui.py`. Numerical and plot
logic: `plot_rhs_power_analysis.py`.

The purpose is LFP/field-potential band-power analysis. Lack of visible spikes
above 200 Hz does not make lower-frequency power analysis meaningless, but the
result is evidence about field-potential power, not spike firing or unit
activity.

The UI runs pre/post mode only (the event-locked mode was retired from the UI
on 2026-08-20; its numerics remain in `plot_rhs_power_analysis.py` for
`batch_run_wideband_main_ui.py --power-mode event` and as import donors to
`stim_analysis/`). On `.rhd` folders the first/last-stim split is driven by
the trace-recovered proxy (`rhd_timing`, same `RHD pulses` / `RHD width (ms)`
fields as Function 3). Controls:

```text
Channels: all
Scale (dB): 3
Bands:
  delta 0.5-4
  theta 4-8
  alpha 8-12
  beta 13-30
  gamma 30-80
```

The stimulation channel search examines selected channels first and then all
other recorded channels. Power can therefore be computed for non-stimulation
channels alone while still using the correct stimulation timing.

### Pre/Post neuromodulation mode

Defaults:

```text
Window (s): 10
Step (s): 5
State guard (s): 1
```

The mode detects the first and last nonzero `stim_data` samples and defines:

```text
pre  = recording start through (first stim - state guard)
post = (last stim + state guard) through recording end
```

For each channel and band, it bandpasses the clean pre and post signals, squares
the filtered voltage, computes sliding-window mean power, and reports:

```text
power change (dB) = 10 * log10(mean post power / mean pre power)
```

The CSV includes clean boundaries, window counts, mean power in `uV^2`, dB
change, percent change, and a bootstrap confidence interval. Pre and post
sliding windows are treated as independent samples for this confidence
interval. The random generator has a fixed seed for reproducibility.

Terms:

- `Window (s)`: duration of each power estimate. A 10 s window contains about
  five cycles at 0.5 Hz and is much more defensible for delta than a 1 s window.
- `Step (s)`: how far the analysis advances between window starts. A 10 s window
  and 5 s step gives 50% overlap.
- `State guard (s)`: extra data discarded immediately before the first stim and
  after the last stim so transition/artifact samples do not enter the clean
  states. It is not a software state-machine setting.
- `Scale (dB)`: fixed symmetric heatmap color range. With `3`, colors run from
  -3 to +3 dB; larger values are clipped visually but remain unchanged in CSV.

The error

```text
Pre-stim recording is shorter than one power window. Reduce Window size (s) or State guard (s).
```

means that `(first stim time - state guard) < window size`. Reducing the guard
can recover transition-adjacent data; reducing the window changes frequency
resolution and reliability. Do not recommend a 1 s window for the default
0.5-4 Hz delta band merely to suppress this error. Prefer a session with at
least 10-20 s of clean baseline, use a longer valid window for low frequencies,
or omit delta when only short windows are possible.

The 15-minute stimulation example has many overlapping technical windows before
and after stimulation. That is useful within-session averaging, but it remains
one biological session, not multiple independent experiments or animals.
Overlapping windows are also not statistically independent replicates.

### Event-locked power (not in the UI)

For event-locked band power use Function 6 (equal-length pairing,
recovery-derived blanking, shuffle control), or the scripted
`batch_run_wideband_main_ui.py --power-mode event`, which still calls
`analyze_event_locked_power` (defaults: baseline -500 to -50 ms, post 1100 to
1600 ms, train gap 60 ms, blank -10 to 50 ms).

### Power outputs

Heatmaps use a fixed `coolwarm` dB scale and star the stimulation channel. The
button saves both one PNG and one CSV into the selected data folder. Filename
prefixes are `power_prepost_...` from the UI, `power_event_...` from the batch
runner's event mode. Nothing is written during preview generation.

## Function 7: Evoked-Response Sweep

The complete, current methodology -- three timing sources (oscilloscope sync,
comb recovery, per-run protocol overrides), the during/post window split,
Gaussian-smoothed per-peak analysis, dual band-power estimators with the
3-cycles gap rule, coupling and decade-mislabel evidence, the full runs.csv /
peaks.csv / figure glossary, and session-specific interpretation notes for
20260819 and 20260821 -- is maintained in **`EVOKED_ANALYSIS_METHODS.md`**.
Read that file before extending `evoked_sweep/`; it is the authoritative
reference and is kept in lockstep with the 125-check
`python3 -m evoked_sweep.selftest`.

## Function 6: Session Stimulation Analysis (Spec v2)

Active UI implementation: `wideband_function6_session_ui.py`. Numerics:
`stim_analysis/` package. Headless CLI: `run_stim_analysis.py`. Spec: the
user's "Endovascular Stimulation -- Analysis Spec v2" (session 260817).

Unlike Functions 1-5 this operates on a **session parent folder** (one
sub-folder per RHS run) and answers one question first: how much of each record
is usable and what analysis window survives the stimulation artifact. Only then
does it run secondary analyses on the conditions that survive.

### Package layout

```text
stim_analysis/config.py     AnalysisConfig: every parameter and every fixed plot
                            limit; config_from_text_fields() is the ONLY place
                            that parses CLI/widget text (Layering Rule)
stim_analysis/load_rhs.py   folder-name + settings.xml parsers, load_run,
                            stim events from the decoded marker (per pulse:
                            amplitude, width, polarity, compliance bit, amp
                            settle), header impedance, contact geometry
stim_analysis/validate.py   commanded (NumberOfStimPulses x trains) vs detected
                            pulses, compliance from data, empirical rail per
                            channel, block assignment, exclusions, table 1
stim_analysis/epoch.py      epoch -> blank -> filter primitives (padded
                            extraction, baseline stats, blank windows, SOS
                            design identical to bandpass_filter_wideband)
stim_analysis/recovery.py   per-trial recovery time / rail duration on RAW
                            epochs; per-condition windows (P90 rule); verdicts
stim_analysis/metrics.py    per-trial band power / RMS on blanked-then-filtered
                            epochs; paired dB keyed by event id; noise floor
stim_analysis/stats.py      vectorised bootstrap CIs, paired_frame (inner join),
                            log-normal checks, fake onsets, MixedLM / OLS model
stim_analysis/models.py     linear / through-origin / sigmoid fits, AIC/BIC,
                            stratified I50 bootstrap
stim_analysis/figures.py    fixed-scale figures with self-describing captions
stim_analysis/secondary.py  stage "all": comparisons (a) (b) (c), drift, charge,
                            spatial, compliance, channel model, shuffle, figs 4-8
stim_analysis/pipeline.py   run_session() -- never writes -- and
                            render_outputs()/write_outputs() (paths decided here)
stim_analysis/selftest.py   synthetic-data checks (writes only under a temp dir)
```

Rules: the package never imports ipywidgets; matplotlib Agg; results and
output paths come back from the pipeline; ASCII in source and labels.

### Stages and gating

1. **validate** -- every run below the parent is loaded and validated:
   sample rate from the header (30 kHz for session 260817, not the spec's
   20 kHz), commanded vs detected pulses, compliance (pulse-count drop or the
   RHS compliance bit), empirical rail level and % railed per channel, metadata
   cross-checks (data wins over settings.xml over folder name), block
   assignment (baseline / block1 amplitude ladder / block2 width sweep /
   block3 paired pulses / other), exclusions. `table01_validation.csv` exists
   before any analysis; the CLI writes it immediately; a run that fails does
   not silently proceed (status `error`, listed).
2. **recovery** (spec section 4, the gating analysis) -- raw epochs
   (-600..+900 ms, +/-500 ms pad), per trial: `threshold = max(3 x baseline SD,
   100 uV)`, `recovery = last t > 0 above threshold before a >= 20 ms quiet
   run`, rail duration, censoring, baseline contamination by the previous
   pulse. Per channel x run: quantiles, `post_start = max(P90(recovery) +
   5 ms, hardware floor)` (`recovery_quantile` = 0.5 reproduces a median rule),
   retained trials, verdict `early_ok` / `late_only` (median > 50 ms) /
   `unusable`. Figures 1, 2, 2b, 3, 3b, S5 and tables 1-3.
3. **all** -- epoch -> blank -> filter -> per-trial metrics; comparisons
   (a) within-epoch, (b) block vs no-stim baseline, (c) across amplitude;
   first-vs-last drift; charge dependence; spatial decay; compliance
   characterisation; amplitude-response models with impedance covariate
   (statsmodels MixedLM, channel random effect); log-normal checks; shuffle
   control; figures 4-8 and S1-S4; metadata JSON with the exact filter designs.

### Design decisions that go beyond the letter of the spec (all stated in captions/metadata)

- **Equal-length window pairing.** The shuffle-null self-test caught a
  -0.4 dB bias on pure noise when a 450 ms baseline is compared with a 300 ms
  post window (different degrees of freedom of the log-power estimate). Every
  post window is paired with an equal-length slice at the END of the baseline.
- **Neighbouring pulses inside the filter pad are blanked** through their own
  measured recovery; at a 1 s IPI the previous pulse sits in the -1100 ms pad
  and the next in the +1400 ms pad, and filtfilt would otherwise ring them
  into the baseline / late windows.
- **`baseline_contaminated`** (previous pulse's recovery reaches into
  -500..-50 ms) is a rejection reason; the trial is dropped from both windows.
- **Shuffle control** has two modes: within-run fake onsets in clean
  inter-pulse intervals (starves at a 1 s IPI because a 1.5 s epoch almost
  always overlaps a real pulse) and no-stim-run pseudo-onsets (jittered).
  Figure 8 uses within-run only when every condition keeps >= `min_trials`
  fake epochs, else the baseline-run variant, and says which.
- **High-pass < 1 Hz is rejected** by `AnalysisConfig.validate()`.
- Figure 3 is truly raw (no baseline subtraction) so rail lines are exact;
  the zoom variant marks off-scale samples in red instead of clipping.

### Data notes for session 260817

- Runs are under `.../SynologyDrive-Endovascular/Jill/20260817 re stim Noah 1/`
  (already renamed by Function 0), NOT `Noah/Noah_260817_surgery/`, which holds
  RHD recordings, tifs and Intan impedance CSVs that list A-024..A-031 as
  disabled/open (unusable). Impedances come from the RHS header (Port A,
  29-256 kOhm; A-031 is a 3.8 MOhm outlier).
- `settings.xml`: `NumberOfStimPulses=50`, `PulseTrainPeriod=999990 us`,
  `RefractoryPeriod=1000 us` -- the folder-name "999ms RP" is the inter-event
  interval, as the spec warns. `SaveDCAmplifierWaveforms=False`, so the rail is
  found empirically (ADC full scale +/-6390 uV; the "+/-2000 uV rail" was a
  plot limit). `step2_*` folders contain `step1_*.rhs` files -- glob, never
  trust stems. `step 3_260817_142832` has a truncated .rhs (reported as error).
- Full session (18 runs, 11 included) runs in about 2 minutes with
  bootstrap 1000. Result: no channel x amplitude condition reaches `early_ok`;
  every cell is `late_only` or `unusable` (median recovery 64-490 ms), i.e.
  the spec's section 12 outcome.

### Outputs

Bundle in `<parent>/stim_analysis/` (or `--output-dir`): `table01_validation.csv`
(+ `_rail_long`), `table02_impedance_per_run.csv`, `table03_recovery_summary.csv`,
per-trial CSVs (`trials_recovery`, `trials_bandpower`, `trials_response_amplitude`,
`trials_paired_db`), `comparisons_*.csv`, `models_amplitude_response.csv`,
`channel_model.csv`, `lognormal_checks.csv`, `noise_floor.csv`,
`shuffle_control.csv`, `fig01`..`fig08` + `figS1`..`figS5` PNGs,
`figures_index.csv`, `stim_analysis_metadata.json`, `run_log.txt`. Per run:
`stim_analysis_<run_id>_{validation,rail,recovery_trials,condition_windows}.csv`
next to the .rhs files (untick the box / `--no-per-run` to skip).

### Controls (Function 6)

Session folder; Baseline run / Stim channel (`auto`); Stage; HP/LP/order,
zero-phase, Pad; Epoch / Baseline / Late (ms); k x SD, Floor, Quiet, Recovery
quantile, Post length, Blank margin; Bands (Function 5 grammar); Bootstrap,
Seed, shuffle; Trace uA (figure 3 amplitudes); Impedance CSV (optional
override); Output folder; per-run CSV checkbox. **Validate** and **Generate
Preview** write nothing; **Save Bundle** writes atomically.

CLI equivalents: `/usr/local/bin/python3 run_stim_analysis.py "<parent>"
[--stage validate|recovery|all] [--runs SUBSTR ...] [--bootstrap N]
[--output-dir DIR] [--no-per-run] [--dry-run]`.

## Save and UI Invariants

Preserve these behaviors in future edits:

- The notebook is the main user interface; helper files hold the implementation.
- Notebook launcher code should remain hidden while widget output remains shown.
- Folder fields must be pasteable, including paths with spaces.
- `all` must resolve to all amplifier channels actually recorded in the selected
  RHS folder, not a hardcoded 32-channel list.
- Figures are embedded in VS Code/Jupyter, not displayed in Matplotlib pop-ups.
- Generating a preview must not automatically save anything.
- A save button writes only the preview currently held by that function.
- Output files go directly into the selected RHS session folder, not the code
  folder and not a global analysis-output parent folder.
- Saving in one data folder must never delete or move outputs from another data
  folder.
- Use atomic temporary-file replacement when saving a generated output.
- Do not restore grey vertical time-grid lines.
- Use ASCII in source and labels (`uV`, `uA`, `+/-`) unless an existing file
  requires Unicode.

Notebook launcher cells currently carry all of these hiding fields/tags:

```json
{
  "jupyter": {"source_hidden": true, "outputs_hidden": false},
  "inputHidden": true,
  "tags": ["hide-input", "hide-source", "hide_input"],
  "source_hidden": true,
  "inputCollapsed": true,
  "collapsed": true,
  "hide_input": true
}
```

If VS Code shows the code despite this metadata, reopen the notebook and collapse
the cell input. VS Code versions can interpret hiding metadata differently.

## Batch Runner

`batch_run_wideband_main_ui.py` runs plot functions over all immediate RHS
session folders below a parent folder. Available names are:

```text
raw filtered events response power
```

Example:

```bash
python3 batch_run_wideband_main_ui.py \
  "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260715 ic implant re" \
  --functions raw filtered events response power \
  --channels all \
  --bandpass all \
  --amplitude "-500-500" \
  --power-mode prepost \
  --skip-existing
```

Use `python3 batch_run_wideband_main_ui.py --help` for all controls. Function 0
renaming is deliberately not part of batch plot generation. The batch runner
writes a `batch_main_ui_outputs.csv` summary in the chosen parent folder.

## Verification Completed

Static checks:

```bash
python3 -m py_compile *.py
python3 -m json.tool Plot_All_Channel_Data_Wideband.ipynb > /dev/null
```

Behavioral check used for the 2026-08-17 refactor, and the one to repeat before
any substantial change. Copy one short session to scratch, run the batch runner
from the old and new code, and byte-compare:

```bash
python3 batch_run_wideband_main_ui.py "<scratch parent>" \
  --functions raw filtered events response power \
  --channels all --bandpass all --amplitude="-500-500" --power-mode prepost
python3 batch_run_wideband_main_ui.py "<scratch parent>" \
  --functions power --channels all --power-mode event
```

Note `--amplitude="-500-500"` must use the `=` form; argparse otherwise reads
the leading minus as a flag.

Against `20260816 re stim yun/step2_260816_192732` (20 MB, 6 channels A-002 to
A-007, 719,616 samples, 23.99 s, 0 timestamp gaps, stim on A-007, 38 events),
all five PNGs and the power CSV were byte-identical before and after, and the
summary CSV differed only in its `elapsed_s` timing column.

The batch runner does not exercise the widget layer. That needs a driven UI
pass: for each of the six functions, generate a preview, confirm **nothing is
written during preview**, then Save and confirm the file lands in the selected
RHS folder. All six were confirmed this way on the session above. Function 5
Pre/Post mode correctly reports the short-pre-stim error on that session, which
has only a few seconds of baseline; its event mode succeeds.

Note the data under `SynologyDrive-Endovascular` are **dataless cloud
placeholders** (`stat -f %b` reports 0 blocks). Reading one triggers a NAS
download, so prefer a small session for smoke tests.

The `stim_analysis` package has a synthetic-data self-test (no recordings
needed, writes only under a temp dir):

```bash
/usr/local/bin/python3 -m stim_analysis.selftest
```

It writes an RHS file that mirrors `_read_header` field for field and checks
reader equivalence, event/compliance detection, rail levels, analytic recovery
times, filter-design equality, paired dropping, the shuffle null (equal vs
unequal windows), models, the mixed model, and a whole synthetic session
through `run_session` (75 checks, ~15 s). The Function 6 widget tree was also
driven headlessly (Validate/Preview write nothing, Save writes the bundle).
The rest of the code base still has no automated tests.

## Current Research Interpretation

The user has endovascular ring-electrode recordings in an intracortical setup
with approximately 400 um end-to-end electrode spacing. Some sessions show no
obvious spike-band activity above 200 Hz. The intended interpretation is:

- power changes below 200 Hz may still be biologically meaningful as LFP or
  field-potential changes;
- raw traces contain large transients and stimulation artifacts, so power
  results require clean windows or explicit artifact blanking;
- common-mode movement, amplifier recovery, state changes, and line noise can
  mimic neural power changes;
- conclusions should be supported by spatial/channel consistency, artifact
  controls, repeated sessions, and preferably sham/no-stim comparisons;
- multiple windows from one session improve estimation precision but do not
  increase the biological sample size.

## Bandwidth sweep (`bw_sweep/`, 2026-08-18)

CLI-only package for the in vitro bandwidth sweep (session `20260818 re stim
in vitro filter settings`); see `BANDWIDTH_SWEEP_PLAN.md` for the plan, the
data facts, and the implementation notes / deviations. It reuses
`stim_analysis` (loader, epochs, `compute_recovery` with a fixed 100 uV
threshold via `threshold_k = 0`, rail estimator) and `filter_diag.common`
(exponential tail fit, rail exit, bootstrap, DSP k). Verify with
`python3 -m bw_sweep.selftest` (45 synthetic checks, ~6 s); the real run is
`python3 -m bw_sweep.run` (working agreement 0 applies). Not wired into the
notebook; a Function 7 wrapper would be a thin call to
`bw_sweep.run.run_sweep` + `write_result`.

## Known Technical Limitations and Good Next Improvements

0. ~~The batch runner and the notebook disagree on Function 3 filenames.~~
   **Fixed 2026-08-17.** `batch_run_wideband_main_ui.py` dropped its private
   `event_window_label` and `parse_post_window_ms` and now imports
   `event_window_label` / `parse_post_time_ms` from
   `plot_rhs_stim_triggered_events`. Both front ends emit
   `pre100ms_post500ms` for `pre=100, post=500`, and `--skip-existing` now
   recognises notebook-produced PNGs. The shared parser accepts the union of the
   two old grammars: `all`/`end` (from the UI) and en/em/minus dashes (from
   batch). **Batch PNGs written before this date used the old
   `pre100ms_post0to500ms` form and will be regenerated once.**

0b. ~~The batch runner has no stim-channel fallback.~~ **Fixed 2026-08-17.**
   All three batch sites call `rhs_stim.resolve_stim_channel`; the filtered-plot
   site passes `fallback=False` to keep Function 2's selected-channels-only
   semantics. A session with stim on an unselected channel is now analyzed by
   batch instead of being skipped.

0c. ~~Filename conventions differ between functions.~~ **Fixed 2026-08-17.**
   Standardized on the stripped-dash form: Function 5 now writes
   `A002_to_A007` like everything else, and band labels map decimals to `p`
   (`delta_0p5-4Hz`). The dash inside a frequency range is preserved, since it
   separates the two frequencies -- `rhs_naming.band_token` exists for exactly
   that distinction. **Power PNG/CSV files written before this date used
   `A-002_to_A-007` and `0.5-4Hz` and will be regenerated once.**

1. ~~Functions 1 and 2 lack the all-recorded-channel stim fallback.~~
   **Fixed 2026-08-17.** Both now call `rhs_stim.resolve_stim_channel` with the
   fallback enabled, as do the batch raw and filtered plots. (Function 1 has
   since become Function 2's Raw mode, 2026-08-20.)
   `plot_raw_channels_with_stim_pulse` gained optional `stim_channel_name` /
   `stim_uA` parameters so Function 1 can draw the biphasic pulse caption from a
   stim channel that is not itself displayed. Status text distinguishes the two
   cases: `A-007 *` when the stim channel is shown, `A-007 (not displayed)` when
   it was found elsewhere in the folder, so the message never implies a star
   that is not drawn.

2. Function 5 would benefit from displaying detected first-stim time, last-stim
   time, clean pre duration, and clean post duration before analysis. This would
   make `Window`/`State guard` errors self-explanatory.
3. Function 5 could warn when the selected power window contains too few cycles
   of the lowest requested frequency.
4. The pre/post bootstrap treats sliding windows as samples even when they
   overlap. For inferential statistics across experiments, aggregate at the
   session/animal level rather than treating overlapping windows as independent
   biological replicates.
5. Full-session multichannel reads can consume substantial RAM because channels
   are loaded separately. Future optimization could add shared-file parsing or
   chunked power computation without changing the UI contract.
7. ~~Function 3 still filters the continuous trace before epoching (spec v2
   pitfall 1).~~ **Resolved 2026-08-20.** `epoch_filter_channel_data` now cuts
   each event with 500 ms padding, blanks pulses and the response-blank
   window, filters per epoch, and trims. Every Function 3 PNG with a numeric
   band (and the batch byte-compare baseline) changed by design;
   `Bandpass=all` output is unchanged.
8. Function 5's event mode compares a 450 ms baseline with a 500 ms post
   window; log-power estimates of unequal length carry a small mean-dB bias on
   noise (see the equal-length pairing in Function 6). The mode was retired
   from the Function 5 UI on 2026-08-20; the numerics are unchanged and remain
   reachable through the batch runner's `--power-mode event`.
6. The hardcoded default paths in launcher helpers are only initial text-box
   seeds. A future cleanup could centralize defaults, but arbitrary pasted paths
   must remain supported.

## Instructions for the Next Codex Agent

0. **Do not run data analysis on newly revised code without asking the user
   first.** Verify code changes with the synthetic self-test, `py_compile`, the
   reload-chain check and scratch copies; running any function, the CLI or the
   batch runner on real recordings (and writing into data folders) is the
   user's call. See also `README.md`.
1. Read this file and `Plot_All_Channel_Data_Wideband.ipynb` Markdown before
   changing the workflow.
2. Inspect `git status` first and preserve unrelated user changes.
3. Follow the Layering Rule above. If you find yourself parsing a string or
   building a filename inside a `wideband_function*_ui.py`, it belongs one layer
   down -- that is precisely how the batch runner and the notebook drifted apart.
4. When changing a behavior, update both the active helper and the notebook
   Markdown explanation.
5. Adding a helper module means adding it to `_RELOAD_CHAIN` in
   `wideband_main_ui.py`, leaves-first, or your edits will not take effect in
   the notebook.
6. Keep manual-save and same-folder output behavior intact: preview must never
   write, and saves go to the selected RHS folder via the atomic temp+replace
   helpers in `wideband_ui_common.py`.
7. Validate with `py_compile`, notebook JSON parsing, and the byte-comparison
   smoke test described under Verification Completed.
9. For Function 6, keep all parsing in `stim_analysis/config.py` and all output
   paths in `stim_analysis/pipeline.render_outputs`; run the self-test after any
   change and, for real data, `run_stim_analysis.py --stage validate` first.
8. Prefer editing one function's UI module over touching `wideband_ui_common.py`;
   a change there affects all six.

A useful opening prompt in the new account is:

```text
Read /Users/jf/Claude/Matlab code/STIM_AGENT_HANDOFF.md and continue the Intan RHS stimulation-analysis workflow from that exact state. Inspect git status and the active helper files before editing.
```
