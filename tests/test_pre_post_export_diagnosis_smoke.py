import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.slow
def test_pre_post_export_diagnosis_smoke(tmp_path):
    pytest.importorskip("onnxruntime")

    weights_path = Path("yolov8n.pt")
    if not weights_path.exists():
        pytest.skip("yolov8n.pt not available locally")

    img_path = tmp_path / "dummy.jpg"
    img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
    img.save(img_path)

    script_path = (
        Path(__file__).resolve().parents[1]
        / "diagnostic_tools"
        / "pre_post_export_diagnosis.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--weights",
            str(weights_path),
            "--source",
            str(img_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
