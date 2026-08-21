"""Function 6 widget UI: session-level stimulation analysis (Analysis Spec v2).

Widgets only. Text fields go to ``stim_analysis.config.config_from_text_fields``
(the one place that parses), the pipeline decides every output path, and the
Save button writes exactly what the preview holds. Preview never writes.
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
from stim_analysis.config import DEFAULT_BANDS_TEXT, config_from_text_fields
from stim_analysis.pipeline import render_outputs, run_session, write_outputs
from stim_analysis.validate import format_validation_ascii

EXAMPLE_SESSION_PARENT_DIR = Path(
    "/Users/jf/Library/CloudStorage/SynologyDrive-Endovascular/Jill/20260817 re stim Noah 1"
)

HEADLINE_FIGURES = (
    "fig01_recovery_vs_amplitude",
    "fig02_recovery_vs_impedance",
    "fig02b_recovery_vs_distance",
    "fig03_raw_traces_low_amplitudes",
    "fig03b_raw_traces_low_amplitudes_zoom",
    "fig04_bandpower_vs_amplitude",
    "fig04b_bandpower_heatmap",
    "fig05_linear_vs_sigmoid",
    "fig06_spatial_decay",
    "fig07_compliance_pulses_vs_charge",
    "fig08_shuffle_control_bandpower_vs_amplitude",
    "figS5_recovery_all_runs_incl_excluded",
)


def _session_seed(namespace: MutableMapping[str, object] | None) -> Path:
    seed = default_data_dir(
        namespace,
        (
            "SESSION_DEFAULT_PARENT_DIR",
            "RESPONSE_DEFAULT_DATA_DIR",
            "EVENT_DEFAULT_DATA_DIR",
            "FILTER_DEFAULT_DATA_DIR",
            "DEFAULT_DATA_DIR",
        ),
        fallback=EXAMPLE_SESSION_PARENT_DIR,
    )
    # Earlier functions seed a *run* folder; the session parent is one level up.
    if seed.exists() and list(seed.glob("*.rhs")):
        seed = seed.parent
    if namespace is not None:
        namespace["SESSION_DEFAULT_PARENT_DIR"] = seed
    return seed


def show_function6_session_analysis(namespace: MutableMapping[str, object] | None = None) -> None:
    """Display Function 6 widgets inside VS Code/Jupyter."""
    seed = _session_seed(namespace)

    parent_text = folder_textarea(str(seed), description="Session folder", description_width="110px")
    parent_text.placeholder = "Paste the parent folder that holds one sub-folder per RHS run"
    baseline_text = widgets.Text(value="auto", description="Baseline run", placeholder="auto or a folder-name substring", layout=widgets.Layout(width="330px"), style={"description_width": "90px"})
    stim_channel_text = widgets.Text(value="auto", description="Stim channel", placeholder="auto or A-030", layout=widgets.Layout(width="230px"), style={"description_width": "90px"})
    stage_dropdown = widgets.Dropdown(
        options=[("Validate only", "validate"), ("Recovery (spec section 4)", "recovery"), ("Full (all sections)", "all")],
        value="recovery",
        description="Stage",
        layout=widgets.Layout(width="260px"),
        style={"description_width": "50px"},
    )
    hp_float = widgets.FloatText(value=1.0, description="HP (Hz)", layout=widgets.Layout(width="150px"), style={"description_width": "60px"})
    lp_float = widgets.FloatText(value=150.0, description="LP (Hz)", layout=widgets.Layout(width="150px"), style={"description_width": "60px"})
    order_int = widgets.IntText(value=4, description="Order", layout=widgets.Layout(width="130px"), style={"description_width": "50px"})
    zero_phase_check = widgets.Checkbox(value=True, description="zero-phase (sosfiltfilt)", indent=False, layout=widgets.Layout(width="200px"))
    pad_float = widgets.FloatText(value=500.0, description="Pad (ms)", layout=widgets.Layout(width="150px"), style={"description_width": "65px"})
    epoch_text = widgets.Text(value="-600 to 900", description="Epoch (ms)", layout=widgets.Layout(width="220px"), style={"description_width": "85px"})
    baseline_win_text = widgets.Text(value="-500 to -50", description="Baseline (ms)", layout=widgets.Layout(width="230px"), style={"description_width": "95px"})
    late_text = widgets.Text(value="400 to 900", description="Late (ms)", layout=widgets.Layout(width="200px"), style={"description_width": "70px"})
    k_float = widgets.FloatText(value=3.0, description="k x SD", layout=widgets.Layout(width="140px"), style={"description_width": "55px"})
    floor_float = widgets.FloatText(value=100.0, description="Floor (uV)", layout=widgets.Layout(width="160px"), style={"description_width": "75px"})
    quiet_float = widgets.FloatText(value=20.0, description="Quiet (ms)", layout=widgets.Layout(width="160px"), style={"description_width": "75px"})
    quantile_float = widgets.FloatText(value=0.9, description="Recovery quantile", layout=widgets.Layout(width="220px"), style={"description_width": "125px"})
    post_len_float = widgets.FloatText(value=300.0, description="Post length (ms)", layout=widgets.Layout(width="200px"), style={"description_width": "115px"})
    margin_float = widgets.FloatText(value=5.0, description="Blank margin (ms)", layout=widgets.Layout(width="210px"), style={"description_width": "125px"})
    bands_text = widgets.Textarea(value=DEFAULT_BANDS_TEXT.replace("; ", "\n"), description="Bands", layout=widgets.Layout(width="360px", height="96px"), style={"description_width": "50px"})
    bootstrap_int = widgets.IntText(value=1000, description="Bootstrap", layout=widgets.Layout(width="170px"), style={"description_width": "75px"})
    seed_int = widgets.IntText(value=0, description="Seed", layout=widgets.Layout(width="130px"), style={"description_width": "45px"})
    shuffle_check = widgets.Checkbox(value=True, description="shuffle control", indent=False, layout=widgets.Layout(width="150px"))
    trace_amps_text = widgets.Text(value="10 20 30", description="Trace uA", placeholder="amplitudes shown in figure 3", layout=widgets.Layout(width="200px"), style={"description_width": "70px"})
    impedance_csv_text = widgets.Text(value="", description="Impedance CSV", placeholder="optional explicit Intan export; default = RHS header", layout=widgets.Layout(width="100%"), style={"description_width": "110px"})
    output_dir_text = widgets.Text(value="", description="Output folder", placeholder="blank = <session folder>/stim_analysis", layout=widgets.Layout(width="100%"), style={"description_width": "110px"})
    per_run_check = widgets.Checkbox(value=True, description="also write per-run CSVs into each run folder", indent=False, layout=widgets.Layout(width="360px"))
    dpi_int = widgets.IntText(value=220, description="DPI", layout=widgets.Layout(width="130px"), style={"description_width": "40px"})

    validate_button = generate_button("Validate", width="120px")
    validate_button.icon = "check"
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
                parent_text,
                widgets.HBox([stage_dropdown, baseline_text, stim_channel_text]),
                widgets.HTML(
                    value=(
                        "<b>Session-level analysis (Spec v2).</b> Validate lists every run "
                        "(commanded vs detected pulses, compliance, rail %) before anything else runs. "
                        "Recovery measures the artifact recovery time per trial and derives the usable "
                        "window per channel x amplitude. Full adds the secondary analyses on the "
                        "conditions that survive. Preview writes nothing; Save Bundle writes the whole "
                        "output set atomically."
                    )
                ),
                widgets.HBox([hp_float, lp_float, order_int, zero_phase_check, pad_float]),
                widgets.HBox([epoch_text, baseline_win_text, late_text]),
                widgets.HBox([k_float, floor_float, quiet_float, quantile_float]),
                widgets.HBox([post_len_float, margin_float, trace_amps_text]),
                widgets.HBox([bands_text, widgets.VBox([widgets.HBox([bootstrap_int, seed_int]), widgets.HBox([shuffle_check, dpi_int])])]),
                impedance_csv_text,
                output_dir_text,
                per_run_check,
                widgets.HBox([validate_button, preview_button, save_btn, target_label]),
                status,
                progress,
                output,
            ]
        )
    )

    def build_config():
        return config_from_text_fields(
            epoch=epoch_text.value,
            baseline=baseline_win_text.value,
            late=late_text.value,
            bands=bands_text.value,
            highpass_hz=hp_float.value,
            lowpass_hz=lp_float.value,
            filter_order=order_int.value,
            zero_phase=zero_phase_check.value,
            threshold_k=k_float.value,
            threshold_floor_uV=floor_float.value,
            quiet_ms=quiet_float.value,
            recovery_quantile=quantile_float.value,
            post_length_ms=post_len_float.value,
            bootstrap_n=bootstrap_int.value,
            seed=seed_int.value,
            shuffle=shuffle_check.value,
            stim_channel=stim_channel_text.value,
            trace_amplitudes=trace_amps_text.value,
            filter_pad_ms=pad_float.value,
            blank_margin_ms=margin_float.value,
            dpi=dpi_int.value,
        )

    def optional_text(value: str) -> str | None:
        cleaned = value.strip().strip('"').strip("'")
        if not cleaned or cleaned.lower() == "auto":
            return None
        return cleaned

    def report_progress(message: str) -> None:
        progress.value = f"<span style='color:#555'>{html.escape(message.splitlines()[0][:160])}</span>"

    def show_validation(frame) -> None:
        with output:
            display(widgets.HTML(value="<b>Validation table (before any analysis)</b>"))
            display(widgets.HTML(value=f"<pre style='font-size:11px'>{html.escape(format_validation_ascii(frame))}</pre>"))

    def run(stage: str):
        held["result"] = None
        save_btn.disabled = True
        target_label.value = NO_PREVIEW_TARGET
        output.clear_output()
        progress.value = ""
        parent = folder_path_from_text(parent_text.value)
        if not parent.exists() or not parent.is_dir():
            status.value = error_html(f"Session folder not found: {parent}")
            return None
        # RHS-only by design: the whole validation stage (commanded vs detected
        # pulses, compliance, amplitude ladder) reads the RHS stim marker, which
        # .rhd files do not have.
        has_rhs = bool(
            list(parent.glob("*.rhs")) or any(child.glob("*.rhs") for child in parent.iterdir() if child.is_dir())
        )
        has_rhd = bool(
            list(parent.glob("*.rhd")) or any(child.glob("*.rhd") for child in parent.iterdir() if child.is_dir())
        )
        if not has_rhs and has_rhd:
            status.value = error_html(
                "This is an RHD session. RHD recordings carry no stim marker to "
                "validate against, so Function 6 is RHS-only -- use Function 7 "
                "for Keithley-stimulated RHD sweeps."
            )
            return None
        try:
            cfg = build_config()
        except ValueError as exc:
            status.value = error_html(str(exc))
            return None
        output_dir = optional_text(output_dir_text.value)
        impedance_csv = optional_text(impedance_csv_text.value)
        status.value = f"Running stage <b>{stage}</b> on {html.escape(parent.name)} ..."
        try:
            result = run_session(
                parent,
                cfg,
                stage=stage,
                baseline_run=optional_text(baseline_text.value),
                output_dir=Path(output_dir) if output_dir else None,
                impedance_csv=Path(impedance_csv) if impedance_csv else None,
                progress=report_progress,
                on_validation=show_validation,
            )
        except Exception as exc:  # show the failure inline rather than a traceback dump
            status.value = error_html(f"{type(exc).__name__}: {exc}")
            with output:
                display(widgets.HTML(value=f"<pre style='font-size:10px;color:{ERROR_COLOR}'>{html.escape(traceback.format_exc()[-3000:])}</pre>"))
            return None
        result.write_per_run = bool(per_run_check.value)
        if result.exit_code == 2:
            status.value = error_html("; ".join(result.warnings) or "hard failure")
            return None
        return result

    def on_validate(_button=None) -> None:
        result = run("validate")
        if result is None:
            return
        n_runs = len(result.validation)
        n_included = int(result.validation["included"].sum()) if n_runs else 0
        status.value = (
            f"Validated {n_runs} run(s), {n_included} included; baseline run "
            f"{html.escape(str(result.baseline_run_id))}. Nothing was written."
        )
        if result.warnings:
            with output:
                display(widgets.HTML(value="<b>Warnings</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings) + "</ul>"))

    def on_preview(_button=None) -> None:
        result = run(stage_dropdown.value)
        if result is None:
            return
        held["result"] = result
        items = render_outputs(result)
        with output:
            if "table03_recovery_summary" in result.tables and not result.tables["table03_recovery_summary"].empty:
                t3 = result.tables["table03_recovery_summary"]
                cols = [c for c in ("channel", "amplitude_uA", "phase_us", "block", "n_trials", "median_recovery_ms", "p90_recovery_ms", "post_start_ms", "n_retained_early", "verdict") if c in t3.columns]
                display(widgets.HTML(value="<b>Table 3: median recovery per channel x amplitude (derived window and verdict)</b>"))
                display(widgets.HTML(value=f"<pre style='font-size:11px'>{html.escape(t3[cols].round(1).to_string(index=False))}</pre>"))
            for stem in HEADLINE_FIGURES:
                png = result.figures.get(stem)
                if png is None:
                    continue
                display(widgets.HTML(value=f"<b>{html.escape(result.figure_titles.get(stem, stem))}</b> ({stem}.png)"))
                display(widgets.Image(value=png, format="png", layout=widgets.Layout(width="100%")))
            if result.warnings:
                display(widgets.HTML(value="<b>Warnings</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings) + "</ul>"))
        save_btn.disabled = False
        target_label.value = f"<b>Target:</b> {html.escape(str(result.output_dir))} ({len(items)} files)"
        status.value = (
            f"Stage <b>{result.stage}</b> done in {result.elapsed_s:.0f} s: {len(result.figures)} figure(s), "
            f"{len(result.tables)} table(s); {int(result.validation['included'].sum())} of {len(result.validation)} runs included. "
            "Nothing was written."
        )

    def on_save(_button=None) -> None:
        result = held.get("result")
        if result is None:
            status.value = error_html("Generate a preview before saving.")
            return
        result.write_per_run = bool(per_run_check.value)
        try:
            paths = write_outputs(result)
        except Exception as exc:
            status.value = error_html(f"Save failed: {type(exc).__name__}: {exc}")
            return
        target_label.value = f"<b>Saved:</b> {html.escape(str(result.output_dir))} ({len(paths)} files)"
        status.value = f"Saved {len(paths)} file(s); cross-run outputs in <b>{html.escape(str(result.output_dir))}</b>."

    validate_button.on_click(on_validate)
    preview_button.on_click(on_preview)
    save_btn.on_click(on_save)


__all__ = ["show_function6_session_analysis"]
