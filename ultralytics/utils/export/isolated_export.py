#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Isolated export bridge: JSON argv -> YOLO.export -> JSON stdout.

Run inside an isolated export venv by run_isolated_export() (see ultralytics/engine/exporter.py) so a format whose
dependencies conflict with the base environment exports in its own interpreter without mutating it.

Usage:
    /opt/venvs/<env>/bin/python isolated_export.py '{"model": ".../model.pt", "format": "rknn", "imgsz": 32}'

Output:
    The FINAL stdout line is json.dumps({"success", "path", "error"}); all diagnostics go to stderr.
"""

import json
import sys
from pathlib import Path


def main(raw):
    """Export a model from a JSON spec and print the JSON result line; return the process exit code."""
    result = {"success": False, "path": None, "error": None}
    try:
        args = json.loads(raw)
        print(f"Export format: {args.get('format')}", file=sys.stderr)
        from ultralytics import YOLO

        args.pop("output_dir", None)  # caller sets cwd to the output dir; the export writes alongside the model
        exported_path = Path(YOLO(args.pop("model")).export(**args))
        print(f"Export completed: {exported_path}", file=sys.stderr)
        result["success"], result["path"] = True, str(exported_path)
    except Exception as e:
        import traceback

        print(f"Export failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        result["error"] = str(e)
    print(json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "path": None, "error": "Usage: isolated_export.py '<json_args>'"}))
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
