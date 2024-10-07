import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr

def check_train_batch_size(model, imgsz=640, amp=True, batch=-1):
    """
    Compute optimal YOLO training batch size by running autobatch in a separate process.

    Args:
        model (torch.nn.Module): YOLO model to check batch size for.
        imgsz (int, optional): Image size used for training. Defaults to 640.
        amp (bool, optional): Use automatic mixed precision if True. Defaults to True.
        batch (float, optional): Fraction of GPU memory to use. If -1, use default. Defaults to -1.

    Returns:
        (int): Optimal batch size computed using the autobatch() function.
    """
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz}")

    device = next(model.parameters()).device
    if device.type in {"cpu", "mps"}:
        LOGGER.info(f"{prefix}⚠️ CUDA not detected, using default CPU batch-size {DEFAULT_CFG.batch}")
        return DEFAULT_CFG.batch

    fraction = batch if 0.0 < batch < 1.0 else 0.60

    try:
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            torch.save(model, tmp_file.name)
            tmp_file_path = Path(tmp_file.name)

        # Prepare the Python code to be executed in the subprocess
        code = f"""
import torch
from ultralytics.utils.autobatch import autobatch
import json

try:
    model = torch.load('{tmp_file_path}', map_location='{device}')
    batch_size = autobatch(model, imgsz={imgsz}, fraction={fraction}, amp={amp})
    print(json.dumps({{"batch_size": int(batch_size)}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""

        # Run the code as a separate process
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

        # Parse the output
        try:
            output = json.loads(result.stdout)
            if "error" in output:
                raise RuntimeError(output["error"])
            batch_size = output["batch_size"]
            LOGGER.info(f"{prefix}Determined optimal batch size: {batch_size}")
            return batch_size
        except json.JSONDecodeError:
            LOGGER.error(f"{prefix}Failed to parse subprocess output. stdout: {result.stdout}, stderr: {result.stderr}")
            raise

    except subprocess.CalledProcessError as e:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Subprocess error: {e}")
        LOGGER.warning(f"{prefix}Subprocess stdout: {e.stdout}")
        LOGGER.warning(f"{prefix}Subprocess stderr: {e.stderr}")
    except Exception as e:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Error: {str(e)}")
    
    LOGGER.warning(f"{prefix}Using default batch-size {DEFAULT_CFG.batch}")
    return DEFAULT_CFG.batch

    finally:
        # Delete the temporary file
        tmp_file_path.unlink(missing_ok=True)
        torch.cuda.empty_cache()
