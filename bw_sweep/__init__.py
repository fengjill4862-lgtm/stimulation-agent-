"""Bandwidth sweep analysis (session 20260818, PBS in vitro).

Does stimulation-artifact recovery scale with the amplifier's low-frequency
time constant?  Three arms, one knob each: A analog lower cutoff, B DSP
cutoff, C analog upper cutoff.  Recovery uses the unchanged
``stim_analysis.recovery.compute_recovery`` with a FIXED 100 uV threshold
(``threshold_k = 0``), because the baseline SD changes with bandwidth and a
k*SD threshold would make recovery times non-comparable across the sweep.

Modules: config, load (headers -> arms), metrics (per epoch x channel),
stats (bootstrap, log-log slopes), figures, verdict, run (CLI), selftest.
See BANDWIDTH_SWEEP_PLAN.md at the repo root.
"""

__version__ = "0.1.0"
