# Stimulation Analysis Agent Handoff

Last reviewed: 2026-08-17  
Workspace: `/Users/jf/Claude/Matlab code`  
Git snapshot at handoff: branch `refactor/extract-ui-blocks`, after the
Functions 0/1/2/4 extraction  
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
| 1 | Plot raw wideband data for one, many, or all recorded channels | `wideband_function1_ui.py` | `plot_rhs_raw_wideband_with_stim_legend.py` |
| 2 | Plot bandpass-filtered samples inside a signed amplitude and time window | `wideband_function2_ui.py` | `plot_rhs_filtered_wideband.py` |
| 3 | Plot stim-triggered response events, three events per row | `wideband_function3_ui.py` | `plot_rhs_stim_triggered_events.py` |
| 4 | Plot recorded response only, without requiring or showing stim current | `wideband_function4_ui.py` | `plot_rhs_filtered_wideband.py` |
| 5 | Pre/post neuromodulation or stim-triggered band-power analysis | `wideband_function5_power_ui.py` | `plot_rhs_power_analysis.py` |

Shared across the UI modules: `wideband_ui_common.py` (widget factories, preview
rendering, atomic saves, error markup) and `rhs_stim.py` (channel reading and
stim-channel resolution).

`wideband_main_ui.py` is the notebook launcher layer. Its public functions are:

```python
show_function0_rename_rhs_folders(globals())
show_function1_raw_wideband(globals())
show_function2_bandpass_filtered(globals())
show_function3_stim_triggered_events(globals())
show_function4_recorded_response_only(globals())
show_function5_power_analysis(globals())
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
- Adding a new helper module means adding it to `_RELOAD_CHAIN` in the right
  position, or edits to it will not take effect.
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

## Function 1: Raw Wideband

Controls:

- `RHS folder`: pasteable session folder path.
- `Channels`: `all`, explicit channels, or channel ranges.
- `Time (s)`: `all` or an absolute range such as `0-60`.
- `Max points`: display-envelope limit; default `600000`.
- `Pulse`: which decoded pulse to use in the waveform caption; default `1`.

Behavior:

- No bandpass filter is applied.
- Multiple selected channels are stacked vertically.
- There are no vertical time-grid lines.
- The stimulation channel is marked with `*` when it is found among the
  selected channels.
- The upper-right caption shows a small thin red biphasic pulse with only its
  amplitude (for example `50uA`) and first-phase duration (for example `0.1ms`).
- The caption is raised above the traces to avoid covering recorded data.
- Preview generation does not write a file.
- **Save PNG** writes the current preview into the selected RHS folder.

Current limitation: Function 1 searches for nonzero `stim_data` among the
selected channels. Enter `all` when the stimulation channel is not known. It
does not currently perform the all-recorded-channel fallback used by Functions
3 and 5.

## Function 2: Bandpass and Amplitude View

Controls:

- `Channels`: supports `all`, explicit channels, and ranges.
- `Bandpass`: `all` or a numeric range such as `0.1-150`, `200-400`, or
  `400-6000 Hz`.
- `Amplitude`: a signed range such as `-100 - 100 uV` or `-100 - 200 uV`.
- `Time (s)`: `all` or an absolute recording range.
- `Max points`: drawing limit.

Behavior:

- Numeric bands use a zero-phase Butterworth bandpass.
- `all` keeps all recorded frequencies.
- Only samples inside the signed amplitude window are displayed.
- The y-axis is set to the selected amplitude limits and labeled
  `filtered amplitude (uV)`.
- The former grey full-bandpass trace and both legend captions were removed.
- There are no vertical time-grid lines.
- The stimulation channel is starred when found among selected channels.
- Preview generation does not save; **Save PNG** writes into the selected RHS
  folder.

Current limitation: like Function 1, Function 2 finds stimulation among the
selected channels rather than scanning unselected recorded channels.

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
  the selected RHS folder.

## Function 4: Recorded Response Only

This is the non-stimulation display path. It plots only recorded amplifier
responses over an absolute recording-time range.

Controls:

```text
Channels: supports all, explicit channels, and ranges
Bandpass: all or numeric range
Amplitude: signed uV range
Time (s): all or absolute recording range
Max points: default 600000
```

It does not search for `stim_data`, mark a stimulation channel, or draw a
stim-current row. It reuses the Function 2 filtering and plotting code. The
preview remains inline and is saved only when **Save PNG** is clicked.

## Function 5: Power Analysis

Active UI implementation: `wideband_function5_power_ui.py`. Numerical and plot
logic: `plot_rhs_power_analysis.py`.

The purpose is LFP/field-potential band-power analysis. Lack of visible spikes
above 200 Hz does not make lower-frequency power analysis meaningless, but the
result is evidence about field-potential power, not spike firing or unit
activity.

Common controls:

```text
Mode: Pre/Post neuromodulation (default) or Stim-triggered events
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

### Stim-triggered events mode

Defaults:

```text
Baseline (ms): -500 to -50
Post (ms): 1100 to 1600
Train gap (ms): 60
Blank (ms): -10 to 50
```

One or more post windows can be entered on separate lines or separated by
semicolons. The analysis:

1. Groups pulses into train/events using `Train gap`.
2. Extracts each event with filter padding.
3. Blanks and linearly interpolates around every pulse using the `Blank` range.
4. Bandpass-filters the cleaned epoch.
5. Computes baseline and post mean power for each event.
6. Computes paired event-level dB changes and bootstrap confidence intervals.

This mode is for transient responses associated with repeated stim events. It
is not a substitute for pre/post neuromodulation analysis when the question is a
sustained state change after a long stimulation block.

### Power outputs

Heatmaps use a fixed `coolwarm` dB scale and star the stimulation channel. The
button saves both one PNG and one CSV into the selected data folder. Filename
prefixes are:

```text
power_prepost_...
power_event_...
```

Nothing is written during preview generation.

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

There is still no automated test suite.

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

## Known Technical Limitations and Good Next Improvements

0. **The batch runner and the notebook disagree on Function 3 filenames.** For
   `pre=100, post=(0,500)` the notebook writes `pre100ms_post500ms` and
   `batch_run_wideband_main_ui.py` writes `pre100ms_post0to500ms`. Both label
   builders now sit in the codebase deliberately: the correct one is
   `plot_rhs_stim_triggered_events.event_window_label`, and the batch copy is
   kept only because reconciling them renames existing batch output files. This
   also silently defeats `--skip-existing` against notebook-produced PNGs.
   Fixing it means deleting `batch_run_wideband_main_ui.event_window_label` and
   importing the shared one.

0b. The batch runner also has **no stim-channel fallback**: `run_event_plot` and
   `run_power_analysis` bail with "no nonzero stim_data" where the notebook
   would scan the remaining recorded channels. `rhs_stim.resolve_stim_channel`
   is the shared implementation; the batch runner simply does not call it yet.

0c. Filename conventions still differ between functions: Function 5 keeps the
   dash (`A-014`) while everything else strips it (`A014`), and `.` becomes `p`
   in the events/response names but not in Function 2's. Unifying renames output
   files, so it needs its own decision.

1. Functions 1 and 2 should eventually gain the same all-recorded-channel stim
   fallback already used by Functions 3 and 5. Until then, use `Channels = all`
   when the stim channel must be identified in those plots. This is now a
   one-argument change: they call `rhs_stim.resolve_stim_channel(...,
   fallback=False)`; Functions 3 and 5 pass `fallback=True`.
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
6. The hardcoded default paths in launcher helpers are only initial text-box
   seeds. A future cleanup could centralize defaults, but arbitrary pasted paths
   must remain supported.

## Instructions for the Next Codex Agent

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
8. Prefer editing one function's UI module over touching `wideband_ui_common.py`;
   a change there affects all six.

A useful opening prompt in the new account is:

```text
Read /Users/jf/Claude/Matlab code/STIM_AGENT_HANDOFF.md and continue the Intan RHS stimulation-analysis workflow from that exact state. Inspect git status and the active helper files before editing.
```
