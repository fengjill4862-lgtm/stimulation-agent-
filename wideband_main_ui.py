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
import sys
from collections.abc import MutableMapping

# Leaves first. Every entry is reloaded, in this order, before the UI module for
# the function being launched.
_RELOAD_CHAIN = (
    "plot_rhs_raw_wideband_with_stim_legend",
    "plot_rhs_filtered_wideband",
    "plot_rhs_stim_triggered_events",
    "plot_rhs_power_analysis",
    "rename_rhs_folders_by_stim_waveform",
    "wideband_ui_common",
)

_LAUNCHERS = {
    "show_function0_rename_rhs_folders": "wideband_function0_ui",
    "show_function1_raw_wideband": "wideband_function1_ui",
    "show_function2_bandpass_filtered": "wideband_function2_ui",
    "show_function3_stim_triggered_events": "wideband_function3_ui",
    "show_function4_recorded_response_only": "wideband_function4_ui",
    "show_function5_power_analysis": "wideband_function5_power_ui",
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


def show_function1_raw_wideband(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 1: Plot all channel data wideband."""
    _launch("show_function1_raw_wideband", namespace)


def show_function2_bandpass_filtered(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 2: Plot bandpass filtered wideband."""
    _launch("show_function2_bandpass_filtered", namespace)


def show_function3_stim_triggered_events(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 3: Plot stim-triggered bandpass events."""
    _launch("show_function3_stim_triggered_events", namespace)


def show_function4_recorded_response_only(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 4: Plot recorded response only."""
    _launch("show_function4_recorded_response_only", namespace)


def show_function5_power_analysis(namespace: MutableMapping[str, object] | None = None) -> None:
    """Launch Function 5: Pre/post and event-locked power analysis."""
    _launch("show_function5_power_analysis", namespace)


__all__ = [
    "show_function0_rename_rhs_folders",
    "show_function1_raw_wideband",
    "show_function2_bandpass_filtered",
    "show_function3_stim_triggered_events",
    "show_function4_recorded_response_only",
    "show_function5_power_analysis",
]
