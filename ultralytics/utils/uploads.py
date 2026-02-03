# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Upload utilities for Ultralytics, mirroring downloads.py patterns."""

from __future__ import annotations

from pathlib import Path
from time import sleep

from ultralytics.utils import LOGGER, TQDM


def safe_upload(
    file: str | Path,
    url: str,
    headers: dict | None = None,
    retry: int = 2,
    timeout: int = 600,
    progress: bool = False,
) -> bool:
    """Upload a file to a URL with retry logic and optional progress bar.

    Args:
        file (str | Path): Path to the file to upload.
        url (str): The URL endpoint to upload the file to (e.g., signed GCS URL).
        headers (dict, optional): Additional headers to include in the request.
        retry (int, optional): Number of retry attempts on failure (default: 2 for 3 total attempts).
        timeout (int, optional): Request timeout in seconds.
        progress (bool, optional): Whether to display a progress bar during upload.

    Returns:
        (bool): True if upload succeeded, False otherwise.

    Examples:
        >>> from ultralytics.utils.uploads import safe_upload
        >>> success = safe_upload("model.pt", "https://storage.googleapis.com/...", progress=True)
    """
    import requests

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    file_size = file.stat().st_size
    desc = f"Uploading {file.name}"

    # Prepare headers
    upload_headers = {"Content-Type": "application/octet-stream", "Content-Length": str(file_size)}
    if headers:
        upload_headers.update(headers)

    last_error = None
    for attempt in range(retry + 1):
        try:
            with open(file, "rb") as f:
                if progress:
                    pbar = TQDM(total=file_size, desc=desc, unit="B", unit_scale=True, unit_divisor=1024)
                    data = (pbar.update(len(chunk)) or chunk for chunk in iter(lambda: f.read(65536), b""))
                else:
                    pbar, data = None, f

                requests.put(url, data=data, headers=upload_headers, timeout=timeout).raise_for_status()
                if pbar:
                    pbar.close()

            LOGGER.info(f"{desc} ({file_size / 1e6:.1f} MB) ✅")
            return True

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500 and status not in {408, 429}:
                LOGGER.warning(f"{desc} failed: {status} {e.response.reason}")
                return False
            last_error = f"HTTP {status}"
        except Exception as e:
            last_error = str(e)

        if attempt < retry:
            wait_time = 2 ** (attempt + 1)
            LOGGER.warning(f"{desc} failed ({last_error}), retrying {attempt + 1}/{retry} in {wait_time}s...")
            sleep(wait_time)

    LOGGER.warning(f"{desc} failed after {retry + 1} attempts: {last_error}")
    return False
