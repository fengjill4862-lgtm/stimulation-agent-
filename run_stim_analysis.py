#!/usr/bin/env python3
"""Headless session-level stimulation analysis (Analysis Spec v2).

Runs validation first, then artifact recovery (the gating analysis), then the
secondary analyses, over every RHS run folder below a session parent folder.

    /usr/local/bin/python3 run_stim_analysis.py "<session parent folder>" [options]

The validation table is printed and written before any analysis runs. Cross-run
outputs go to <parent>/stim_analysis/ (or --output-dir); per-run CSVs go next
to each run's .rhs files. Use --dry-run to compute everything without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stim_analysis.config import DEFAULT_BANDS_TEXT, config_from_text_fields
from stim_analysis.pipeline import STAGES, render_outputs, run_session, write_outputs
from stim_analysis.validate import format_validation_ascii


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parent", help="session parent folder holding one sub-folder per RHS run")
    parser.add_argument("--stage", choices=STAGES, default="all", help="stop after this stage (default all)")
    parser.add_argument("--baseline-run", default=None, help="substring of the no-stim baseline run folder (default: auto)")
    parser.add_argument("--stim-channel", default="auto", help="stim channel like A-030 (default auto-detect)")
    parser.add_argument("--channels", nargs="*", default=None, help="analysis channels (default: every recorded channel)")
    parser.add_argument("--output-dir", default=None, help="cross-run output folder (default <parent>/stim_analysis)")
    parser.add_argument("--runs", nargs="*", default=None, help="only run folders whose name/run_id contains one of these substrings")
    parser.add_argument("--impedance-csv", default=None, help="explicit Intan impedance CSV override (default: RHS header)")
    parser.add_argument("--manifest", default=None, help="CSV with run_id, block[, include] overriding block assignment")
    # epochs / filter
    parser.add_argument("--epoch-ms", default="-600 to 900")
    parser.add_argument("--baseline-ms", default="-500 to -50")
    parser.add_argument("--late-ms", default="400 to 900")
    parser.add_argument("--hp", type=float, default=1.0, help="high-pass Hz (>= 1)")
    parser.add_argument("--lp", type=float, default=150.0, help="low-pass Hz")
    parser.add_argument("--order", type=int, default=4, help="Butterworth order (SOS)")
    parser.add_argument("--causal", action="store_true", help="causal sosfilt instead of zero-phase sosfiltfilt")
    parser.add_argument("--pad-ms", type=float, default=500.0, help="filter padding per epoch side")
    # recovery
    parser.add_argument("--threshold-k", type=float, default=3.0)
    parser.add_argument("--threshold-floor-uv", type=float, default=100.0)
    parser.add_argument("--quiet-ms", type=float, default=20.0)
    parser.add_argument("--recovery-quantile", type=float, default=0.9, help="quantile of recovery that sets the post-window start (0.5 = median rule)")
    parser.add_argument("--blank-margin-ms", type=float, default=5.0)
    parser.add_argument("--post-length-ms", type=float, default=300.0)
    # bands / stats
    parser.add_argument("--bands", default=DEFAULT_BANDS_TEXT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="skip the shuffle control")
    parser.add_argument("--trace-amplitudes", default="10 20 30", help="amplitudes (uA) shown in figure 3")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--dry-run", action="store_true", help="compute everything, write nothing")
    parser.add_argument("--no-per-run", action="store_true", help="do not write per-run CSVs into the run folders")
    parser.add_argument("--quiet", action="store_true", help="less progress output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = config_from_text_fields(
            epoch=args.epoch_ms,
            baseline=args.baseline_ms,
            late=args.late_ms,
            bands=args.bands,
            highpass_hz=args.hp,
            lowpass_hz=args.lp,
            filter_order=args.order,
            zero_phase=not args.causal,
            threshold_k=args.threshold_k,
            threshold_floor_uV=args.threshold_floor_uv,
            quiet_ms=args.quiet_ms,
            recovery_quantile=args.recovery_quantile,
            post_length_ms=args.post_length_ms,
            bootstrap_n=args.bootstrap,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            stim_channel=args.stim_channel,
            trace_amplitudes=args.trace_amplitudes,
            filter_pad_ms=args.pad_ms,
            blank_margin_ms=args.blank_margin_ms,
            dpi=args.dpi,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parent = Path(args.parent).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    wrote_validation = {"done": False}

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    def on_validation(frame) -> None:
        # Spec section 2: the validation table exists before any analysis runs.
        print("\nVALIDATION TABLE (written before analysis):")
        print(format_validation_ascii(frame))
        print()
        if not args.dry_run:
            target = (output_dir or parent / cfg.output_subdir) / "table01_validation.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            from rhs_files import atomic_write_text

            atomic_write_text(target, frame.to_csv(index=False))
            wrote_validation["done"] = True
            print(f"wrote {target}\n")

    result = run_session(
        parent,
        cfg,
        stage=args.stage,
        baseline_run=args.baseline_run,
        output_dir=output_dir,
        run_filter=args.runs,
        impedance_csv=Path(args.impedance_csv) if args.impedance_csv else None,
        manifest=Path(args.manifest) if args.manifest else None,
        channels=args.channels or None,
        progress=progress if not args.quiet else None,
        on_validation=on_validation,
    )

    for line in result.summary_lines():
        print(line)
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.exit_code == 2:
        print("hard failure; nothing else written", file=sys.stderr)
        return 2
    result.write_per_run = not args.no_per_run
    if args.dry_run:
        print(f"dry run: {len(render_outputs(result))} files would be written under {result.output_dir}")
        return result.exit_code
    paths = write_outputs(result)
    print(f"wrote {len(paths)} files; cross-run outputs in {result.output_dir}")
    print(f"elapsed {result.elapsed_s:.1f} s")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
