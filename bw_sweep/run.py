"""Run the bandwidth sweep analysis.

    /usr/local/bin/python3 -m bw_sweep.run [--root DIR] [--out DIR] [--dry-run]
        [--keep-duplicate both|first|last] [--include-stim-contact-in-slopes]
        [--fast] [--quiet]

Nothing is written until the end (atomic); ``--dry-run`` computes and writes
nothing.  Output folder default: <root>/bandwidth_sweep/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from rhs_files import atomic_write_all
from stim_analysis.config import config_to_dict
from stim_analysis.figures import CaptionContext, build_caption, figure_to_png_bytes, finish_figure
from bw_sweep import __version__
from bw_sweep.config import ARMS, SweepConfig, sweep_config_to_dict
from bw_sweep.figures import fig_arm_c, fig_noise, fig_r2, fig_recovery_vs_tau, fig_tau_fit, fig_traces
from bw_sweep.load import SweepSet, discover_sweep, format_settings_ascii, settings_table
from bw_sweep.metrics import RunMetrics, per_run_metrics
from bw_sweep.stats import SlopeResult, loglog_slope
from bw_sweep.summary import in_arm, table1, table_per_channel
from bw_sweep.verdict import Verdict, build_verdict


@dataclass
class SweepResult:
    output_dir: Path
    cfg: SweepConfig
    sweep: SweepSet | None = None
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, bytes] = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    slopes: dict[str, SlopeResult] = field(default_factory=dict)
    verdict: Verdict | None = None
    log: list[str] = field(default_factory=list)
    metrics: dict[str, RunMetrics] = field(default_factory=dict)

    def outputs(self) -> list[tuple[Path, bytes | str]]:
        items: list[tuple[Path, bytes | str]] = []
        for stem, frame in self.tables.items():
            items.append((self.output_dir / f"{stem}.csv", frame.to_csv(index=False)))
        for stem, png in self.figures.items():
            items.append((self.output_dir / f"{stem}.png", png))
        items.append((self.output_dir / "captions.txt", "\n\n".join(f"{s}.png\n{c}" for s, c in self.captions.items()) + "\n"))
        items.append((self.output_dir / "metadata.json", json.dumps(self.metadata, indent=2, default=_json_default)))
        items.append((self.output_dir / "verdict.txt", (self.verdict.text() if self.verdict else "") + "\n"))
        items.append((self.output_dir / "log.txt", "\n".join(self.log) + "\n"))
        return items


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _log(result: SweepResult, message: str, quiet: bool) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {message}"
    result.log.append(line)
    if not quiet:
        print(message, flush=True)


def _git_commit(repo_dir: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version(), "bw_sweep": __version__}
    for name in ("numpy", "scipy", "matplotlib", "pandas"):
        try:
            out[name] = getattr(__import__(name), "__version__", "?")
        except Exception:
            out[name] = None
    return out


def _add_figure(result: SweepResult, stem: str, fig, ctx: CaptionContext) -> None:
    caption = build_caption(ctx)
    # footer sized to the wrapped caption (finish_figure wraps at 165 chars per 16 in of width) plus room for the x-axis label
    width_in, _ = fig.get_size_inches()
    n_lines = len(textwrap.wrap(caption, width=int(165 * width_in / 16.0))) or 1
    finish_figure(fig, caption, footer_in=0.75 + 0.135 * n_lines)
    result.figures[stem] = figure_to_png_bytes(fig, dpi=result.cfg.dpi)
    result.captions[stem] = caption


def _ctx(result: SweepResult, note: str) -> CaptionContext:
    n = int(len(result.trials))
    censored = int(result.trials["censored"].sum()) if n else 0
    return CaptionContext(
        n_retained=n,
        n_rejected=0,
        reject_reasons={"censored_at_epoch_end": censored},
        blank_desc="none (raw, baseline-mean-subtracted)",
        filter_desc="none (raw wideband, instrument bandwidth per run)",
        epoch_desc=f"-600..+900 ms, baseline -500..-50 ms; recovery threshold FIXED {result.cfg.threshold_uV:g} uV; bootstrap n={result.cfg.bootstrap_n}, seed {result.cfg.seed}",
        note=note,
    )


def run_sweep(
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    cfg: SweepConfig | None = None,
    fast: bool = False,
    quiet: bool = False,
) -> SweepResult:
    started = time.time()
    cfg = cfg or SweepConfig()
    if root is not None:
        cfg = cfg.with_(root=Path(root))
    if fast:
        cfg = cfg.with_(bootstrap_n=200)
    output_dir = Path(output_dir) if output_dir is not None else Path(cfg.root) / cfg.output_subdir
    result = SweepResult(output_dir=output_dir, cfg=cfg)
    acfg = cfg.analysis_config()
    rng = np.random.default_rng(cfg.seed)
    _log(result, f"bandwidth sweep v{__version__}: root={cfg.root} out={output_dir} threshold fixed {cfg.threshold_uV:g} uV", quiet)

    # ---- headers -> arms --------------------------------------------------------------------
    sweep = discover_sweep(cfg.root, cfg)
    result.sweep = sweep
    for line in format_settings_ascii(sweep).splitlines():
        _log(result, line, quiet)
    result.tables["table0_settings_per_run"] = settings_table(sweep)
    result.metadata["one_knob_checks"] = {name: {"ok": c.ok, "n_runs": c.n_runs, "values": c.values, "messages": c.messages} for name, c in sweep.checks.items()}
    result.metadata["notes"] = list(sweep.notes)
    if not sweep.all_ok:
        _log(result, "WARNING: one-knob check failed for at least one arm -- see table0 / metadata", quiet)

    # ---- per run metrics --------------------------------------------------------------------
    frames: list[pd.DataFrame] = []
    traces: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    floors: dict[str, float] = {}
    run_meta: list[dict[str, object]] = []
    stim_channel: str | None = None
    for run in sweep.runs:
        if not run.arms:
            _log(result, f"skip {run.folder.name}: matches no arm", quiet)
            continue
        t0 = time.time()
        try:
            m = per_run_metrics(run, cfg, acfg)
        except Exception as exc:
            _log(result, f"ERROR {run.folder.name}: {type(exc).__name__}: {exc}", quiet)
            run_meta.append({"run_id": run.run_id, "folder": run.folder.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        result.metrics[run.run_id] = m
        frames.append(m.trials)
        traces[run.run_id] = (m.trace_t_ms, m.median_trace_uV)
        floors[run.run_id] = m.floor_ms
        stim_channel = stim_channel or run.stim_channel
        rec = m.trials[~m.trials["is_stim_contact"]] if not m.trials.empty else m.trials
        med = float(np.median(rec["recovery_ms"])) if not rec.empty else float("nan")
        _log(result, f"{run.folder.name}: {m.n_events} pulses, {m.n_kept} epochs kept, floor {m.floor_ms:.2f} ms, recording-contact median recovery {med:.1f} ms, prestim SD " + ", ".join(f"{c} {v:.2f}" for c, v in m.prestim_sd_uV.items()) + f" uV ({m.prestim_seconds:.1f} s), {time.time() - t0:.1f} s" + (f"; warnings: {m.warnings}" if m.warnings else ""), quiet)
        run_meta.append({"run_id": run.run_id, "folder": run.folder.name, "arms": run.arms, "n_events": m.n_events, "n_kept": m.n_kept, "floor_ms": m.floor_ms, "compliance_flag": m.compliance_flag, "prestim_sd_uV": m.prestim_sd_uV, "prestim_seconds": m.prestim_seconds, "rail_levels_uV": m.rail_levels, "warnings": m.warnings})
    result.metadata["runs"] = run_meta
    result.trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.trials.empty:
        _log(result, "no trials -- stopping", quiet)
        result.metadata["elapsed_s"] = time.time() - started
        return result
    if not np.allclose(result.trials["threshold_uV"].to_numpy(dtype=float), cfg.threshold_uV):
        raise AssertionError("threshold is not fixed on every trial")
    result.tables["trials_per_epoch"] = result.trials

    # ---- summaries --------------------------------------------------------------------------
    t1 = table1(result.trials, sweep, cfg, rng)
    pc = table_per_channel(result.trials, sweep, cfg, rng)
    result.tables["table1_summary_per_run"] = t1
    result.tables["table1b_summary_per_run_channel"] = pc

    # ---- slopes -----------------------------------------------------------------------------
    for arm in ("A", "B"):
        sub = result.trials[in_arm(result.trials, arm)]
        if cfg.exclude_stim_contact_from_slopes:
            sub = sub[~sub["is_stim_contact"]]
        result.slopes[arm] = loglog_slope(sub, arm=arm, x_col="tau_nominal_ms", y_col="recovery_ms", cfg=cfg, rng=rng, floor_by_run=floors)
        _log(result, result.slopes[arm].describe(), quiet)
        # tau_fit vs tau_nominal slope (informative fits only)
        inf = sub[sub["fit_informative"] & sub["fit_converged"]]
        result.slopes[f"{arm}_taufit"] = loglog_slope(inf, arm=f"{arm} tau_fit", x_col="tau_nominal_ms", y_col="tau_fit_ms", cfg=cfg, rng=rng)
    all_inf = result.trials[result.trials["fit_informative"] & result.trials["fit_converged"] & ~result.trials["is_stim_contact"]]
    result.slopes["all_taufit"] = loglog_slope(all_inf, arm="all tau_fit", x_col="tau_nominal_ms", y_col="tau_fit_ms", cfg=cfg, rng=rng)
    _log(result, result.slopes["all_taufit"].describe(), quiet)
    result.metadata["slopes"] = {k: {kk: vv for kk, vv in v.__dict__.items()} for k, v in result.slopes.items()}

    # ---- verdict ----------------------------------------------------------------------------
    result.verdict = build_verdict(sweep, t1, pc, result.slopes, cfg)
    result.tables["table2_recommendation_pareto"] = result.verdict.pareto
    for line in result.verdict.lines + [result.verdict.recommendation]:
        _log(result, line, quiet)

    # ---- figures ----------------------------------------------------------------------------
    _add_figure(result, "fig0_median_traces", fig_traces(sweep, traces, stim_channel, cfg), _ctx(result, "Median centred trace per run over kept epochs, decimated x10 for plotting."))
    for stem, arm in (("fig1_recovery_vs_tau_armA", "A"), ("fig2_recovery_vs_tau_armB", "B")):
        s = result.slopes[arm]
        _add_figure(result, stem, fig_recovery_vs_tau(result.trials, sweep, arm, t1, s, cfg, rng), _ctx(result, s.describe() + ". Black = median of the recording contacts with IQR (thin) and 95% bootstrap CI (thick); stim contact = hollow triangles, excluded from the slope."))
    _add_figure(result, "fig3_armC_rail_and_peak", fig_arm_c(result.trials, sweep, pc, cfg, rng), _ctx(result, result.verdict.arm_C))
    _add_figure(result, "fig4_tau_fit_vs_tau_nominal", fig_tau_fit(result.trials, sweep, t1, cfg, rng), _ctx(result, result.slopes["all_taufit"].describe()))
    _add_figure(result, "fig5_r2_per_arm", fig_r2(result.trials, sweep, cfg), _ctx(result, "ECDF of R2 per run (colour = knob value) and pooled (black); dotted curves = runs whose tau_nominal is below the fit window."))
    _add_figure(result, "fig6_noise_vs_bandwidth", fig_noise(pc, sweep, cfg), _ctx(result, "Circles/triangles: median baseline SD (-500..-50 ms) with CI; crosses: SD of the pre-train segment (clean noise floor)."))

    result.metadata.update({
        "version": __version__,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(cfg.root),
        "output_dir": str(output_dir),
        "config": sweep_config_to_dict(cfg),
        "analysis_config": config_to_dict(acfg),
        "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "versions": _versions(),
        "verdict": result.verdict.lines,
        "recommendation": result.verdict.recommendation,
        "elapsed_s": time.time() - started,
    })
    _log(result, f"done in {time.time() - started:.1f} s", quiet)
    return result


def write_result(result: SweepResult) -> list[Path]:
    items = result.outputs()
    atomic_write_all(items)
    return [p for p, _ in items]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="sweep folder (default: the 20260818 in vitro folder)")
    parser.add_argument("--out", default=None, help="output folder (default: <root>/bandwidth_sweep)")
    parser.add_argument("--keep-duplicate", choices=("both", "first", "last"), default="both", help="how to treat two runs with the same setting in one arm")
    parser.add_argument("--include-stim-contact-in-slopes", action="store_true")
    parser.add_argument("--fast", action="store_true", help="200 bootstrap resamples")
    parser.add_argument("--dry-run", action="store_true", help="compute, write nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    cfg = SweepConfig().with_(keep_duplicate=args.keep_duplicate, exclude_stim_contact_from_slopes=not args.include_stim_contact_in_slopes)
    result = run_sweep(root=Path(args.root) if args.root else None, output_dir=Path(args.out) if args.out else None, cfg=cfg, fast=args.fast, quiet=args.quiet)
    if args.dry_run:
        print(f"dry run: {len(result.outputs())} files would be written to {result.output_dir}")
        return 0
    written = write_result(result)
    print(f"wrote {len(written)} files to {result.output_dir}")
    return 0 if (result.sweep is not None and result.sweep.all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
