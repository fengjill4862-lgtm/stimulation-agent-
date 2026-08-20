"""Evoked-response analysis for externally-stimulated RHD sessions.

Session 20260819 stimulates with a Keithley 2400 and records with an Intan RHD
controller, so there is no stim marker channel: pulse times are recovered from
the amplifier trace and amplitude comes from the folder name.

This package holds the numerics for Function 7. It never imports ipywidgets and
never displays anything -- the UI layer reads widget values and displays what
comes back, and every output path is decided here, not in widget code.
"""

from __future__ import annotations

__all__ = ["config", "naming", "load", "pulses", "metrics", "artifact", "pipeline"]
