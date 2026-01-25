import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.slow
def test_pre_post_export_diagnosis_smoke(tmp_path):
    """
    Smoke test to ensure the pre/post export diagnostic script
    runs end-to-end and exits cleanly.
    """

    # Create a tiny dummy image
    img_path = tmp_path / "dummy.jpg"
    img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    img.save(img_path)

    # Path to the diagnostic script
    script_path = (
        Path(__file__).resolve().parents[1]
        / "diagnostic_tools"
        / "pre_post_export_diagnosis.py"
    )

    # Run the script as a subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--weights",
            "yolov8n.pt",
            "--source",
            str(img_path),
        ],
        capture_output=True,
        text=True,
    )

    # Assert clean exit
    assert result.returncode == 0, result.stderr
