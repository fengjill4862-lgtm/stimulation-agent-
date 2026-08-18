"""Filter Diagnosis Spec: is the 100-500 ms "recovery" the Intan DSP high-pass step response?

Modules: common (DSP model, inverse, loading, tail fits), synthetic_step,
filter_diag_A ... filter_diag_F, run_all (verdict), selftest.

Reuses stim_analysis for loading, epoching, rail detection and -- unchanged --
the recovery-time algorithm (stim_analysis.recovery.compute_recovery).
"""

__version__ = "0.1.0"
