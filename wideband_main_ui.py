#!/usr/bin/env python3
"""Widget launchers for Plot_All_Channel_Data_Wideband.ipynb.

The notebook keeps only tiny launcher cells visible; each function's real
implementation lives in its own `wideband_functionN_ui.py` module.

This module exists for one reason: **it owns ordered module reloading.** The
notebook cells call `importlib.reload(wideband_main_ui)`, which does not reload
submodules, so without the reload chain below an edit to a UI or helper module
would not take effect until the kernel was restarted.

Reload order is load-bearing. Reloading a module re-executes its top-level code,
including its `from X import y` lines, and those resolve against `sys.modules`.
If `X` was not reloaded first, the dependent module re-imports the *stale* `y`
and the edit silently appears to do nothing. `_RELOAD_CHAIN` is therefore listed
leaves-first and must stay that way.

Note that loading a new *data file* needs no reload at all; that is just pasting
a path into the widget. Reload is only for code edits. And `importlib.reload`
cannot fix objects already built from old class definitions, so a kernel restart
remains the ground truth if behavior ever looks impossible.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from collections.abc import MutableMapping

# Leaves first. Every entry is reloaded, in this order, before the UI module for
# the function being launched.
_RELOAD_CHAIN = (
    # Pure leaves: standard library only.
    "rhs_naming",
    "rhs_files",
    # Depend on the leaves above, and on each other in this order.
    "plot_rhs_raw_wideband_with_stim_legend",
    "rhs_reader",
    "rhd_reader",
    "plot_rhs_filtered_wideband",
    "plot_rhs_stim_triggered_events",
    "plot_rhs_power_analysis",
    "rename_rhs_folders_by_stim_waveform",
    "rhs_stim",
    "wideband_ui_common",
    # stim_analysis package (Function 6), leaves first: config -> data -> stages.
    "stim_analysis.config",
    "stim_analysis.load_rhs",
    "stim_analysis.validate",
    "stim_analysis.epoch",
    "stim_analysis.recovery",
    "stim_analysis.metrics",
    "stim_analysis.stats",
    "stim_analysis.models",
    "stim_analysis.figures",
    "stim_analysis.secondary",
    "stim_analysis.pipeline",
    # evoked_sweep package (Function 7), leaves first: config -> data -> stages.
    # figures imports pipeline, and summary imports pipeline, so pipeline is
    # reloaded before either of them.
    "evoked_sweep.config",
    "evoked_sweep.naming",
    "evoked_sweep.load",
    "evoked_sweep.pulses",
    "evoked_sweep.metrics",
    "evoked_sweep.peaks",
    "evoked_sweep.artifact",
    "evoked_sweep.pipeline",
    "evoked_sweep.figures",
    "evoked_sweep.summary",
    # Stim-timing recovery for RHD viewers (Functions 3 and 5); imports from
    # evoked_sweep, so it reloads after the whole package.
    "rhd_timing",
)

_LAUNCHERS = {
    "show_function0_rename_rhs_folders": "wideband_function0_ui",
    "show_function2_continuous_traces": "wideband_function2_ui",
    "show_function3_stim_triggered_events": "wideband_function3_ui",
    "show_function5_power_analysis": "wideband_function5_power_ui",
    "show_function6_session_analysis": "wideband_function6_session_ui",
    "show_function7_evoked_response": "wideband_function7_evoked_ui",
}


def _launch(function_name: str, namespace: MutableMapping[str, object] | None) -> None:
    """Reload the dependency chain leaves-first, then run one function's UI."""
    for module_name in _RELOAD_CHAIN:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)

    ui_module_name = _LAUNCHERS[function_name]
    ui_module = sys.modules.get(ui_module_name)
    if ui_module is None:
        ui_module = importlib.import_module(ui_module_name)
    else:
        ui_module = importlib.reload(ui_module)

    getattr(ui_module, function_name)(namespace)


def show_function0_rename_rhs_folders(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 0: Rename RHS folders by stim waveform."""
    _launch("show_function0_rename_rhs_folders", namespace)


def show_function2_continuous_traces(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 2: Continuous traces (raw or filtered, optional ignore-stim)."""
    _launch("show_function2_continuous_traces", namespace)


def show_function3_stim_triggered_events(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 3: Plot stim-triggered bandpass events."""
    _launch("show_function3_stim_triggered_events", namespace)


def show_function5_power_analysis(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 5: Pre/post neuromodulation power analysis."""
    _launch("show_function5_power_analysis", namespace)


def show_function6_session_analysis(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 6: Session-level stimulation analysis (Spec v2)."""
    _launch("show_function6_session_analysis", namespace)


def show_function7_evoked_response(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 7: Evoked-response sweep (RHD, external stimulator)."""
    _launch("show_function7_evoked_response", namespace)


__all__ = [
    "show_function0_rename_rhs_folders",
    "show_function2_continuous_traces",
    "show_function3_stim_triggered_events",
    "show_function5_power_analysis",
    "show_function6_session_analysis",
    "show_function7_evoked_response",
]


def check_reload_chain() -> list[str]:
    """Verify _RELOAD_CHAIN is ordered leaves-first. Returns a list of problems.

    A module must appear after every project module it imports, or reloading it
    re-executes `from X import y` against a stale cached X and raises ImportError
    for anything newly added to X. Run this after adding or reordering modules:

        python3 wideband_main_ui.py
    """
    import re

    def module_path(name: str) -> pathlib.Path:
        # Dotted names are package modules: stim_analysis.config -> stim_analysis/config.py
        return pathlib.Path(name.replace(".", "/") + ".py")

    def project_imports(source: str) -> list[str]:
        found = re.findall(r"^from ([a-z_][a-z0-9_.]*) import", source, flags=re.M)
        found += re.findall(r"^import ([a-z_][a-z0-9_.]*)\s*$", source, flags=re.M)
        return found

    position = {name: index for index, name in enumerate(_RELOAD_CHAIN)}
    problems: list[str] = []
    for name, index in position.items():
        source = module_path(name).read_text()
        for dependency in project_imports(source):
            if dependency in position and position[dependency] > index:
                problems.append(
                    f"{name} (position {index}) imports {dependency} "
                    f"(position {position[dependency]}) -- dependency must come first"
                )
            elif dependency not in position and module_path(dependency).exists():
                problems.append(
                    f"{name} imports {dependency}, which is missing from _RELOAD_CHAIN"
                )
    for function_name, module_name in _LAUNCHERS.items():
        source = module_path(module_name).read_text()
        for dependency in project_imports(source):
            if dependency not in position and module_path(dependency).exists():
                problems.append(
                    f"{module_name} imports {dependency}, which is missing from _RELOAD_CHAIN"
                )
    return problems


if __name__ == "__main__":
    issues = check_reload_chain()
    if issues:
        print("_RELOAD_CHAIN problems:")
        for issue in issues:
            print(f"  {issue}")
        raise SystemExit(1)
    print(f"_RELOAD_CHAIN is correctly ordered ({len(_RELOAD_CHAIN)} modules).")
