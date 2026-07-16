# rerun-importer-c3d

External [Rerun](https://rerun.io/) importer for C3D biomechanics data files.

Drop a `.c3d` file onto the Rerun Viewer (or run `rerun trial.c3d`) to instantly
visualise marker trajectories, force plate forces, analog signals, and events.

## How it works

Any executable on `$PATH` whose name starts with `rerun-importer-` is automatically
discovered by the Rerun Viewer as an [external importer](https://rerun.io/docs/concepts/logging-and-ingestion/importers/overview).

When you drag a `.c3d` file onto the Viewer (or use `rerun file.c3d`), the viewer
invokes this importer, which:

1. Parses the C3D file using [ezc3d](https://github.com/pyomeca/ezc3d)
2. Logs data to Rerun's SDK and streams it to the Viewer over stdout
3. Returns exit code 66 for non-C3D files so the viewer knows to try other importers

## Data logged

| Data          | Rerun entity          | Archetype               | Timeline                     |
|---------------|-----------------------|-------------------------|------------------------------|
| Markers       | `{trial}/markers`     | `rr.Points3D`           | `frame` (sequence)           |
| Marker residuals | `{trial}/markers/{name}/residual` | `rr.Scalars` | `frame` |
| Analog channels | `{trial}/analogs/{name}` | `rr.Scalars`       | `analog_frame` (sequence)    |
| Force plate geometry | `{trial}/force_plates/{fp}/corners` | `rr.LineStrips3D` (static) | — |
| Force plate forces | `{trial}/force_plates/{fp}/force` | `rr.Arrows3D` | `frame` |
| COP             | `{trial}/force_plates/{fp}/cop` | `rr.Points3D`      | `frame` |
| Events          | `{trial}/events/{label}` | `rr.TextLog`         | `frame` |
| File metadata   | `{trial}/info`        | `rr.TextDocument` (static) | — |

## Installation

Requires Python 3.10+.

```bash
# Install from source
pip install git+https://github.com/hudsonburke/rerun-importer-c3d.git

# Or from a local clone
git clone https://github.com/hudsonburke/rerun-importer-c3d.git
cd rerun-importer-c3d
pip install .
```

Make sure the `rerun-importer-c3d` script is on your `$PATH` (pip usually handles this):

```bash
which rerun-importer-c3d
# → /path/to/bin/rerun-importer-c3d
```

## Usage

With the importer installed and on `$PATH`:

```bash
# Open in the Rerun Viewer
rerun path/to/trial.c3d

# Or drag the .c3d file onto the Rerun Viewer window
```

You can also call it directly — it streams the RRD data to stdout:

```bash
rerun-importer-c3d path/to/trial.c3d > output.rrd
```

## Development

```bash
# Clone and set up
git clone https://github.com/hudsonburke/rerun-importer-c3d.git
cd rerun-importer-c3d
uv venv
uv pip install -e "."
```

### Testing

```bash
# Generate a test C3D file and run the importer
python test_data/generate_sample.py
rerun-importer-c3d test_data/sample_trial.c3d
```

## License

MIT
