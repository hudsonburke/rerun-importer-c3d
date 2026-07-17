#!/usr/bin/env python3
"""
Batch subcommand for rerun-importer-c3d.

Walks a directory of C3D files, groups them by subject, and produces one
``.rrd`` file per subject with all trials logged inside a single recording.

Usage:
    rerun-importer-c3d batch /path/to/c3d/root -o /path/to/output/
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import ezc3d
import numpy as np
import rerun as rr

from . import log_c3d, get_param_list, get_param_strings


def extract_subject(filepath: str) -> str:
    """Extract the subject name from a C3D file."""
    try:
        c3d = ezc3d.c3d(filepath)
        names = get_param_strings(c3d, ["SUBJECTS", "NAMES"])
        if names and names[0].strip():
            return names[0].strip()
        # Fall back to PROCESSING data
        proc = c3d["parameters"].get("PROCESSING", {})
        for pname, pval in proc.items():
            if isinstance(pval, dict) and pval.get("value") is not None:
                return pname
        return Path(filepath).stem.rsplit("_", 1)[0] if "_" in Path(filepath).stem else Path(filepath).stem
    except Exception:
        return Path(filepath).stem


def discover_c3d_files(root: str) -> list[str]:
    """Recursively find all .c3d files under *root*."""
    root_path = Path(root).resolve()
    files = []
    for entry in sorted(root_path.rglob("*.[cC][3 threeD][dD]")):
        if entry.is_file():
            files.append(str(entry))
    # Also try .c3d explicitly
    for entry in sorted(root_path.rglob("*.c3d")):
        if entry.is_file() and str(entry) not in files:
            files.append(str(entry))
    return sorted(set(files))


def batch_import(root: str, output_dir: str, min_body_measurements: int = 3) -> dict[str, str]:
    """
    Walk *root* for C3D files, group by subject, and write one ``.rrd`` per subject.

    Returns a mapping of subject_name → rrd_filepath.
    """
    files = discover_c3d_files(root)
    if not files:
        print(f"No .c3d files found under {root}", file=sys.stderr)
        return {}

    # Group by subject
    groups: dict[str, list[str]] = defaultdict(list)
    for fp in files:
        subject = extract_subject(fp)
        groups[subject].append(fp)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for subject, trials in sorted(groups.items()):
        print(f"\nSubject: {subject}  ({len(trials)} trials)")
        rrd_path = str((out_path / f"{subject}.rrd").resolve())

        # Use the subject name as the application ID
        rr.init(subject)
        recording = rr.RecordingStream(application_id=subject)

        # Stream directly to the .rrd file
        recording.save(rrd_path)

        # --- Log subject-level static data ---
        # Use the first trial that has PROCESSING params to extract body measurements
        body_measurements: dict[str, float] = {}
        subject_name_found: str | None = None
        for trial_fp in trials:
            try:
                c3d = ezc3d.c3d(trial_fp)
                # Get subject name
                names = get_param_strings(c3d, ["SUBJECTS", "NAMES"])
                if names and names[0].strip():
                    subject_name_found = names[0].strip()
                # Extract body measurements
                proc = c3d["parameters"].get("PROCESSING", {})
                for pname, pval in proc.items():
                    if pname == "__METADATA__" or not isinstance(pval, dict):
                        continue
                    value = pval.get("value")
                    if value is not None and hasattr(value, "__len__") and len(value) == 1:
                        v = value[0]
                        if v is not None:
                            try:
                                body_measurements[pname] = float(v)
                            except (TypeError, ValueError):
                                pass
            except Exception:
                continue
            if len(body_measurements) >= min_body_measurements:
                break

        # Log subject name as static data
        recording.log(
            f"/{subject}/info",
            rr.TextLog(subject_name_found or subject),
            static=True,
        )

        # Log body measurements once as static scalars at the subject level
        if body_measurements:
            bm_path = f"/{subject}/body_measurements"
            for pname, val in sorted(body_measurements.items()):
                safe = pname.replace("/", "_").replace(" ", "_")
                recording.log(
                    f"{bm_path}/{safe}",
                    rr.Scalars([val]),
                    static=True,
                )
            print(f"  Body measurements: {len(body_measurements)} params logged")

        # --- Log each trial ---
        for trial_idx, trial_fp in enumerate(trials):
            trial_name = Path(trial_fp).stem
            prefix = f"/{subject}/{trial_name}"
            print(f"  [{trial_idx + 1}/{len(trials)}] {trial_name}")

            # Use the per-file importer logic with trial-specific prefix
            log_c3d(trial_fp, prefix, recording)

        recording.flush()
        results[subject] = rrd_path
        print(f"  → {rrd_path}  ({len(trials)} trials)")

    total = sum(len(v) for v in groups.values())
    print(f"\nDone: {len(results)} subjects, {total} trials, {len(files)} C3D files")
    return results


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``batch`` subcommand."""
    parser = subparsers.add_parser(
        "batch",
        help="Batch-import C3D files grouped by subject into .rrd files",
        description="""\
Walk a directory of C3D files, group them by subject, and produce one .rrd file
per subject.  The resulting files can be opened in the Rerun Viewer or served via
``rerun server`` for cross-trial SQL queries.

Example:
    rerun-importer-c3d batch /data/c3d_root -o /data/rrd_output/
""",
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory to recursively search for .c3d files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Output directory for .rrd files (default: current dir)",
    )
    parser.add_argument(
        "--min-body-measurements",
        type=int,
        default=3,
        help="Min number of body measurement params before stopping scan (default: 3)",
    )
    parser.set_defaults(func=_run_batch)


def _run_batch(args: argparse.Namespace) -> None:
    """Execute the batch subcommand."""
    results = batch_import(args.root, args.output_dir, args.min_body_measurements)
    if results:
        print(f"\nProduced {len(results)} .rrd file(s)")
        print("To serve with the Rerun catalog server:")
        print(f"  rr.server.Server(datasets={{'biomechanics': '{args.output_dir}'}})")
        print("Or from the CLI:")
        print(f"  rerun server --datasets biomechanics={args.output_dir}")
        print()
        print("To open a subject's full recording:")
        print(f"  rerun {args.output_dir}/<subject_name>.rrd")
