"""Session-level stimulation analysis (Endovascular Stimulation Analysis Spec v2).

The package answers one question first -- how much of an Intan RHS stimulation
session is usable, and what analysis window survives the artifact -- and only
then computes secondary metrics on the conditions that survive.

Modules (implementation order):

    config     AnalysisConfig: every parameter and every fixed plot limit
    load_rhs   folder-name and settings.xml parsers, run loading, stim events
    validate   event count, compliance, rail detection, blocks, exclusions
    epoch      epoch -> blank -> filter, in that order
    recovery   artifact recovery time per trial (the gating analysis)
    metrics    per-trial band power / response amplitude / paired dB
    stats      bootstrap, paired frames, log-normal checks, shuffle helpers
    models     linear vs sigmoid amplitude-response fits, AIC/BIC
    figures    fixed-scale figures with self-describing captions
    pipeline   run_session (never writes) + render_outputs / write_outputs
    selftest   synthetic-data checks:  python3 -m stim_analysis.selftest

Nothing in this package imports ipywidgets or IPython. Submodules are not
imported here so `wideband_main_ui._RELOAD_CHAIN` can reload them one by one.
"""

__version__ = "0.1.0"
