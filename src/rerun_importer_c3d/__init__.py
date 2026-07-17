#!/usr/bin/env python3
"""
Rerun external importer for C3D biomechanics files.

Any executable on ``$PATH`` whose name starts with ``rerun-importer-`` is
automatically discovered by the Rerun Viewer as an external importer.

Install this package, then drag a ``.c3d`` file onto the Rerun Viewer (or run
``rerun trial.c3d``) to visualise marker trajectories, force plates, analog
signals, and events.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ezc3d
import numpy as np
import rerun as rr


# ---------------------------------------------------------------------------
# C3D helpers  (adapted from movedb-core)
# ---------------------------------------------------------------------------

def get_param_list(
    c3d: ezc3d.c3d, keys: list[str], default: list | None = None
) -> list | np.ndarray:
    """Navigate nested C3D parameters and return the ``value`` list."""
    param: dict = c3d.parameters
    for key in keys:
        param = param.get(key, {})
    value = param.get("value")
    if value is None:
        return default if default is not None else []
    return value


def get_param_strings(
    c3d: ezc3d.c3d, keys: list[str], default: list[str] | None = None
) -> list[str]:
    """Return a ``list[str]`` from a C3D parameter."""
    value = get_param_list(c3d, keys)
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.flat]
    if not value:
        return default if default is not None else []
    return [str(v) for v in value]


def get_param(c3d: ezc3d.c3d, keys: list[str], index: int = 0, default=None):
    """Get a single indexed value from a C3D parameter list."""
    param_list = get_param_list(c3d, keys)
    if not hasattr(param_list, "__len__") or len(param_list) == 0:
        return default
    if index < 0 or index >= len(param_list):
        return default
    value = param_list[index]
    return value if value is not None else default


def extract_analog_rate(c3d: ezc3d.c3d) -> float:
    """Analog rate, falling back to point_rate × ratio."""
    rate = float(get_param(c3d, ["ANALOG", "RATE"], default=0.0))
    if rate <= 0:
        point_rate = float(get_param(c3d, ["POINT", "RATE"], default=0.0))
        ratio = float(get_param(c3d, ["ANALOG", "RATIO"], default=1.0))
        rate = point_rate * ratio
    return rate


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def log_c3d(path: str, prefix: str, recording: rr.RecordingStream) -> None:
    """Parse a C3D file and log its contents to Rerun."""
    c3d = ezc3d.c3d(path, extract_forceplat_data=True)

    point_rate = float(get_param(c3d, ["POINT", "RATE"], default=0.0))
    analog_rate = extract_analog_rate(c3d)

    # ---- Extract marker data ----
    raw_points = c3d["data"]["points"]                     # (4, n_markers, n_frames)
    marker_names = get_param_strings(c3d, ["POINT", "LABELS"])
    n_markers = len(marker_names)
    n_frames = raw_points.shape[2] if raw_points.ndim >= 3 else 0

    # Residuals
    residuals = None
    try:
        raw_residuals = c3d["data"]["meta_points"]["residuals"]
        residuals = np.squeeze(raw_residuals, axis=0).T.astype(np.float64)  # (n_frames, n_markers)
    except (KeyError, TypeError):
        pass

    # Markers: (4, N, T)[:3] → (3, N, T) → transpose(2, 1, 0) → (T, N, 3)
    marker_data = np.transpose(raw_points[:3, :, :], (2, 1, 0)).astype(np.float64)

    # ---- Log markers frame by frame ----
    for frame_idx in range(n_frames):
        recording.set_time("frame", sequence=frame_idx)
        recording.set_time("time", duration=frame_idx / point_rate if point_rate > 0 else 0)

        positions = marker_data[frame_idx]  # (n_markers, 3)
        recording.log(
            f"{prefix}/markers",
            rr.Points3D(
                positions,
                labels=marker_names,
                radii=np.full(n_markers, 0.01),
            ),
        )

        # Per-marker residuals as scalar
        if residuals is not None and n_markers > 0:
            for m_idx, m_name in enumerate(marker_names):
                if not m_name.strip():
                    continue
                recording.log(
                    f"{prefix}/markers/{m_name}/residual",
                    rr.Scalars([float(residuals[frame_idx, m_idx])]),
                )

    # ---- Log analog channels as time series ----
    try:
        raw_analogs = c3d["data"]["analogs"]
        analog_data = raw_analogs[0, :, :].T.astype(np.float64)
        analog_names = get_param_strings(c3d, ["ANALOG", "LABELS"])
        analog_units = get_param_strings(c3d, ["ANALOG", "UNITS"])

        if len(analog_names) > 0 and analog_data.size > 0:
            n_analog_frames = analog_data.shape[0]
            for ch_idx, ch_name in enumerate(analog_names):
                if not ch_name.strip():
                    continue
                unit = analog_units[ch_idx] if ch_idx < len(analog_units) else ""
                safe_name = ch_name.replace("/", "_").replace(" ", "_").replace(".", "_")
                for a_frame in range(n_analog_frames):
                    recording.set_time("analog_frame", sequence=a_frame)
                    analog_time = a_frame / analog_rate if analog_rate > 0 else 0
                    recording.set_time("analog_time", duration=analog_time)
                    recording.log(
                        f"{prefix}/analogs/{safe_name}",
                        rr.Scalars([float(analog_data[a_frame, ch_idx])]),
                    )
    except (KeyError, IndexError):
        pass

    # ---- Log force plate geometry and forces ----
    try:
        platforms = c3d["data"]["platform"]
        if platforms:
            n_plates = len(platforms)
            fp_names = _find_forceplate_names(c3d, n_plates)

            for p_idx in range(n_plates):
                plat = platforms[p_idx]
                fp_name = fp_names[p_idx] if p_idx < len(fp_names) else f"FP_{p_idx}"
                fp_path = f"{prefix}/force_plates/{fp_name}"

                # Static geometry: corners
                corners = plat["corners"].astype(np.float64)  # (3, 4)
                corners_4 = corners.T  # (4, 3) — four 3D points
                line_strip = np.vstack([corners_4, corners_4[0:1]])
                recording.log(
                    f"{fp_path}/corners",
                    rr.LineStrips3D([line_strip]),
                    static=True,
                )

                # Static: origin
                origin = plat["origin"].astype(np.float64)
                recording.log(
                    f"{fp_path}/origin",
                    rr.Points3D([origin], radii=0.02, labels=[f"{fp_name} origin"]),
                    static=True,
                )

                # Per-frame: force, moment, COP
                forces = np.asarray(plat["force"]).T.astype(np.float64)
                moments = np.asarray(plat["moment"]).T.astype(np.float64)
                cop = np.asarray(plat["center_of_pressure"]).T.astype(np.float64)

                for f_idx in range(n_frames):
                    recording.set_time("frame", sequence=f_idx)
                    f_time = f_idx / point_rate if point_rate > 0 else 0
                    recording.set_time("time", duration=f_time)

                    recording.log(
                        f"{fp_path}/force",
                        rr.Arrows3D(
                            vectors=[forces[f_idx]],
                            origins=[cop[f_idx]],
                            radii=0.005,
                            labels=[f"Force ({f_idx})"],
                        ),
                    )

                    # COP as a point
                    recording.log(
                        f"{fp_path}/cop",
                        rr.Points3D([cop[f_idx]], radii=0.015, colors=[[255, 0, 0]]),
                    )

                    # Moment as arrow
                    recording.log(
                        f"{fp_path}/moment",
                        rr.Arrows3D(
                            vectors=[moments[f_idx]],
                            origins=[cop[f_idx]],
                            radii=0.003,
                            colors=[[0, 255, 0]],
                        ),
                    )
    except (KeyError, IndexError) as e:
        print(f"Skipping force plates: {e}", file=sys.stderr)

    # ---- Log subject body measurements (PROCESSING params) ----
    try:
        proc = c3d["parameters"].get("PROCESSING", {})
        if proc:
            subject_path = f"{prefix}/subject/body_measurements"
            # Determine subject name
            subjects_names = c3d["parameters"].get("SUBJECTS", {}).get("NAMES", {}).get("value", [])
            subject_name = str(subjects_names[0]) if subjects_names else "unknown"
            recording.log(
                f"{subject_path}/name",
                rr.TextLog(subject_name),
                static=True,
            )
            for pname, pval in proc.items():
                if pname == "__METADATA__" or not hasattr(pval, "get"):
                    continue
                value = pval.get("value")
                if value is not None and hasattr(value, "__len__") and len(value) == 1:
                    v = float(value[0]) if value[0] is not None else None
                    if v is not None:
                        safe = pname.replace("/", "_").replace(" ", "_")
                        recording.log(
                            f"{subject_path}/{safe}",
                            rr.Scalars([v]),
                            static=True,
                        )
    except (KeyError, IndexError):
        pass

    # ---- Log analog descriptions ----
    try:
        analog_names = get_param_strings(c3d, ["ANALOG", "LABELS"])
        analog_descs = get_param_strings(c3d, ["ANALOG", "DESCRIPTIONS"])
        if analog_descs:
            for ch_idx, ch_name in enumerate(analog_names):
                desc = analog_descs[ch_idx] if ch_idx < len(analog_descs) else ""
                if desc:
                    safe_name = ch_name.replace("/", "_").replace(" ", "_").replace(".", "_")
                    recording.log(
                        f"{prefix}/analogs/{safe_name}/description",
                        rr.TextLog(desc),
                        static=True,
                    )
    except (KeyError, IndexError):
        pass

    # ---- Log events ----
    try:
        event_labels = get_param_strings(c3d, ["EVENT", "LABELS"])
        event_contexts = get_param_strings(c3d, ["EVENT", "CONTEXTS"])
        times = get_param_list(c3d, ["EVENT", "TIMES"])

        if isinstance(times, np.ndarray) and times.ndim >= 2 and len(event_labels) > 0:
            for e_idx, e_label in enumerate(event_labels):
                if e_idx < times.shape[1]:
                    t_min, t_sec = times[:, e_idx]
                    event_time = float(t_min) * 60.0 + float(t_sec)
                    context = event_contexts[e_idx] if e_idx < len(event_contexts) else ""
                    frame_num = int(event_time * point_rate)

                    recording.set_time("frame", sequence=frame_num)
                    recording.set_time("time", duration=event_time)
                    recording.log(
                        f"{prefix}/events/{e_label}",
                        rr.TextLog(f"{context} — {e_label} @ {event_time:.3f}s"),
                    )
    except (KeyError, IndexError):
        pass

    # ---- Log metadata as text ----
    first_frame = int(c3d["header"]["points"]["first_frame"])
    last_frame = int(c3d["header"]["points"]["last_frame"])

    rr.log(
        f"{prefix}/info",
        rr.TextDocument(
            f"""# C3D File: {os.path.basename(path)}

- **Markers**: {n_markers} markers, {n_frames} frames
- **Point rate**: {point_rate} Hz
- **Analog rate**: {analog_rate} Hz
- **Frame range**: {first_frame} – {last_frame}
""",
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )


def _find_forceplate_names(c3d: ezc3d.c3d, n_platforms: int) -> list[str]:
    """Derive human-readable forceplate names from ANALOG:DESCRIPTIONS."""
    channel_arr = get_param_list(c3d, ["FORCE_PLATFORM", "CHANNEL"])
    analog_descriptions = get_param_strings(c3d, ["ANALOG", "DESCRIPTIONS"])

    fp_names: list[str] = []
    if analog_descriptions and isinstance(channel_arr, np.ndarray) and channel_arr.ndim == 2:
        first_channels = channel_arr[0, :]
        for idx in first_channels:
            adj = int(idx) - 1
            if 0 <= adj < len(analog_descriptions):
                name = analog_descriptions[adj].replace(" ", "_").replace("[", "").replace("]", "")
                if name and name not in fp_names:
                    fp_names.append(name)

    if len(fp_names) != n_platforms:
        fp_names = [f"FP_{i}" for i in range(n_platforms)]
    return fp_names


# ---------------------------------------------------------------------------
# CLI entry point (external importer protocol)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""\
External Rerun importer for C3D biomechanics data.

Any executable on $PATH whose name starts with ``rerun-importer-`` will be
discovered by the Rerun Viewer as an external importer.

This importer parses .c3d files and logs marker trajectories, force plate
data, analog signals, and events to the Rerun SDK, which streams them to
the Viewer via stdout.

Usage:
    rerun-importer-c3d path/to/trial.c3d

Or simply drag the .c3d file onto the Rerun Viewer.
""",
    )
    parser.add_argument("filepath", type=str, help="Path to the file to load")
    parser.add_argument("--application-id", type=str, help="Recommended ID for the application")
    parser.add_argument("--opened-application-id", type=str, help="Optional recommended ID for the application")
    parser.add_argument("--recording-id", type=str, help="Optional recommended ID for the recording")
    parser.add_argument("--opened-recording-id", type=str, help="Optional recommended ID for the recording")
    parser.add_argument("--entity-path-prefix", type=str, help="Optional prefix for all entity paths")
    parser.add_argument("--static", action="store_true", default=False, help="Optionally mark data as static")
    parser.add_argument("--time", type=str, action="append", help="Optional timestamps (e.g. `--time sim_time=1709203426`)")
    parser.add_argument("--sequence", type=str, action="append", help="Optional sequences (e.g. `--sequence sim_frame=42`)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Inform the Rerun Viewer that we don't support this file
    if not os.path.isfile(args.filepath) or not args.filepath.lower().endswith(".c3d"):
        sys.exit(rr.EXTERNAL_IMPORTER_INCOMPATIBLE_EXIT_CODE)

    app_id = args.application_id or args.filepath
    rr.init(app_id, recording_id=args.recording_id)
    recording = rr.get_global_data_recording()
    if recording is None:
        recording = rr.RecordingStream(application_id=app_id, recording_id=args.recording_id)

    # Stream data to stdout so the Rerun Viewer can ingest it
    recording.stdout()

    _set_time_from_args(args, recording)

    prefix = args.entity_path_prefix or Path(args.filepath).stem

    log_c3d(args.filepath, prefix, recording)


def _set_time_from_args(args: argparse.Namespace, recording: rr.RecordingStream) -> None:
    if args.static:
        return
    if args.time:
        for time_str in args.time:
            parts = time_str.split("=")
            if len(parts) == 2:
                recording.set_time_seconds(parts[0], seconds=float(parts[1]))
    if args.sequence:
        for seq_str in args.sequence:
            parts = seq_str.split("=")
            if len(parts) == 2:
                recording.set_time_sequence(parts[0], sequence=int(parts[1]))


if __name__ == "__main__":
    main()
