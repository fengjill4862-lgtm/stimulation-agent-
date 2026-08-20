"""Function 7 widget UI: evoked-response sweep for externally-stimulated RHD runs.

Widgets only. Text fields go to ``evoked_sweep.config.config_from_text_fields``
(the one place that parses), the pipeline decides every output path, and the
Save button writes exactly what the preview holds. Preview never writes.

Function 7 exists because session 20260819 was stimulated by a Keithley 2400 and
recorded on an RHD controller: the files are .rhd rather than .rhs, and nothing
recorded the trigger, so pulse times are recovered from the amplifier trace.
"""

from __future__ import annotations

import html
import traceback
from collections.abc import MutableMapping
from pathlib import Path

from wideband_ui_common import (
    ERROR_COLOR,
    NO_PREVIEW_TARGET,
    default_data_dir,
    display,
    error_html,
    folder_path_from_text,
    folder_textarea,
    generate_button,
    save_button,
    status_html,
    target_label_html,
    widgets,
)
from evoked_sweep.config import config_from_text_fields
from evoked_sweep.pipeline import render_outputs, run_session, run_single, write_outputs
from evoked_sweep.summary import verdict_text

EXAMPLE_SESSION_DIR = Path("/Users/jf/SynologyDrive/Research/Stimulation/20260819")

PREVIEW_FIGURES = (
    "fig1_dose_response.png",
    "fig2_pulse_waveforms.png",
    "fig3_band_power.png",
    "fig4_artifact_evidence.png",
)


def _session_seed(namespace: MutableMapping[str, object] | None) -> Path:
    seed = default_data_dir(
        namespace,
        (
            "EVOKED_DEFAULT_SESSION_DIR",
            "SESSION_DEFAULT_PARENT_DIR",
            "DEFAULT_DATA_DIR",
        ),
        fallback=EXAMPLE_SESSION_DIR,
    )
    if namespace is not None:
        namespace["EVOKED_DEFAULT_SESSION_DIR"] = seed
    return seed


def _timing_table(result) -> str:
    header = (
        f"{'run':38s} {'mA':>7s} {'ch':>6s} {'n':>3s} {'T ms':>7s} "
        f"{'start s':>8s} {'w ms':>6s} {'p-p uV':>9s} {'lat ms':>7s} {'ok':>5s}"
    )
    lines = [header, "-" * len(header)]
    for row in result.rows:
        amplitude = row.get("amplitude_mA")
        lines.append(
            f"{str(row['run'])[:38]:38s} "
            f"{('' if amplitude is None else f'{amplitude:+.3g}'):>7s} "
            f"{row['channel']:>6s} {row['n_pulses']:>3d} "
            f"{row['pulse_period_ms']:>7.1f} {row['train_start_s']:>8.2f} "
            f"{row['pulse_width_ms']:>6.1f} {row['evoked_pp_uV']:>9.1f} "
            f"{row['peak_latency_ms']:>7.2f} {str(row['timing_ok']):>5s}"
        )
    return "\n".join(lines)


def show_function7_evoked_response(namespace: MutableMapping[str, object] | None = None) -> None:
    """Display Function 7 widgets inside VS Code/Jupyter."""
    seed = _session_seed(namespace)

    folder_text = folder_textarea(str(seed), description="Folder", description_width="90px")
    folder_text.placeholder = "Session folder for a whole sweep, or one run folder for a quick check"

    mode_dropdown = widgets.Dropdown(
        options=[("Whole session", "session"), ("Single run (quick check)", "single")],
        value="single",
        description="Mode",
        layout=widgets.Layout(width="290px"),
        style={"description_width": "50px"},
    )
    channels_text = widgets.Text(
        value="all",
        description="Channels",
        placeholder="all = every contact under 1 Mohm",
        layout=widgets.Layout(width="300px"),
        style={"description_width": "75px"},
    )
    pulses_int = widgets.IntText(
        value=50, description="Pulses", layout=widgets.Layout(width="150px"), style={"description_width": "60px"}
    )
    width_float = widgets.FloatText(
        value=5.0, description="Width (ms)", layout=widgets.Layout(width="180px"), style={"description_width": "85px"}
    )
    max_runs_int = widgets.IntText(
        value=0, description="Max runs", layout=widgets.Layout(width="170px"), style={"description_width": "75px"}
    )

    preview_button = generate_button("Generate Preview", width="170px")
    save_btn = save_button("Save Bundle", width="130px")
    status = status_html()
    target_label = target_label_html()
    progress = widgets.HTML(value="")
    output = widgets.Output()

    held: dict[str, object] = {"result": None}

    display(
        widgets.VBox(
            [
                folder_text,
                widgets.HBox([mode_dropdown, channels_text, max_runs_int]),
                widgets.HBox([pulses_int, width_float]),
                widgets.HTML(
                    value=(
                        "<b>Evoked response to external (Keithley) stimulation, from RHD "
                        "recordings.</b> Nothing recorded the trigger, so pulse onsets are "
                        "recovered by fitting the known train -- 50 pulses, 5 ms wide -- to the "
                        "amplifier trace, and every run reports whether that fit is trustworthy. "
                        "Measures per-pulse deflection, band power (train vs baseline) and "
                        "post-train change, and flags deflections that look like stimulus "
                        "coupling rather than a response. Single run mode checks one folder in "
                        "seconds; whole session reads about 1.9 GB. Preview writes nothing; "
                        "Save Bundle writes the whole output set atomically."
                    )
                ),
                widgets.HBox([preview_button, save_btn, target_label]),
                status,
                progress,
                output,
            ]
        )
    )

    def report_progress(message: str) -> None:
        progress.value = f"<span style='color:#555'>{html.escape(message.splitlines()[0][:160])}</span>"

    def on_preview(_button=None) -> None:
        held["result"] = None
        save_btn.disabled = True
        target_label.value = NO_PREVIEW_TARGET
        output.clear_output()
        progress.value = ""

        folder = folder_path_from_text(folder_text.value)
        if not folder.exists() or not folder.is_dir():
            status.value = error_html(f"Folder not found: {folder}")
            return

        single = mode_dropdown.value == "single"
        try:
            config = config_from_text_fields(
                session_folder=str(folder),
                channels=channels_text.value,
                expected_pulses=str(pulses_int.value),
                pulse_width_ms=str(width_float.value),
                single_run="1" if single else "",
                max_runs=str(max_runs_int.value or ""),
            )
        except ValueError as exc:
            status.value = error_html(str(exc))
            return

        status.value = f"Analysing {html.escape(folder.name)} ..."
        try:
            runner = run_single if single else run_session
            result = runner(config, progress=report_progress)
        except Exception as exc:  # show the failure inline rather than a traceback dump
            status.value = error_html(f"{type(exc).__name__}: {exc}")
            with output:
                display(
                    widgets.HTML(
                        value=(
                            f"<pre style='font-size:10px;color:{ERROR_COLOR}'>"
                            f"{html.escape(traceback.format_exc()[-3000:])}</pre>"
                        )
                    )
                )
            return

        if not result.rows:
            status.value = error_html(
                "No run produced a measurement. The verdict below says why."
            )
            with output:
                display(
                    widgets.HTML(value=f"<pre style='font-size:11px'>{html.escape(verdict_text(result))}</pre>")
                )
            return

        held["result"] = result
        items = render_outputs(result)
        figures = {path.name: payload for path, payload in items if path.suffix == ".png"}

        with output:
            display(widgets.HTML(value="<b>Recovered timing and per-pulse response</b>"))
            display(
                widgets.HTML(value=f"<pre style='font-size:11px'>{html.escape(_timing_table(result))}</pre>")
            )
            for name in PREVIEW_FIGURES:
                png = figures.get(name)
                if png is None:
                    continue
                display(widgets.HTML(value=f"<b>{html.escape(name)}</b>"))
                display(widgets.Image(value=png, format="png", layout=widgets.Layout(width="100%")))
            display(widgets.HTML(value="<b>Verdict</b>"))
            display(
                widgets.HTML(value=f"<pre style='font-size:11px'>{html.escape(verdict_text(result))}</pre>")
            )

        save_btn.disabled = False
        target_label.value = f"<b>Target:</b> {html.escape(str(result.config.output_dir))} ({len(items)} files)"
        unreliable = sum(
            1 for run in result.runs if run.train is not None and not run.train.ok
        )
        status.value = (
            f"{len(result.analysed)} of {len(result.runs)} run(s) analysed, "
            f"{len(result.rows)} row(s); {unreliable} with unreliable timing. Nothing was written."
        )

    def on_save(_button=None) -> None:
        result = held.get("result")
        if result is None:
            status.value = error_html("Generate a preview before saving.")
            return
        try:
            paths = write_outputs(result)
        except Exception as exc:
            status.value = error_html(f"Save failed: {type(exc).__name__}: {exc}")
            return
        target_label.value = f"<b>Saved:</b> {html.escape(str(result.config.output_dir))} ({len(paths)} files)"
        status.value = f"Saved {len(paths)} file(s) into <b>{html.escape(str(result.config.output_dir))}</b>."

    preview_button.on_click(on_preview)
    save_btn.on_click(on_save)


__all__ = ["show_function7_evoked_response"]
