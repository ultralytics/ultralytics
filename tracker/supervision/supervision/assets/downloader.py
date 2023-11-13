import os
from hashlib import md5
from pathlib import Path
from shutil import copyfileobj
from typing import Union

from supervision.assets.list import VIDEO_ASSETS, VideoAssets

try:
    from requests import get
    from tqdm.auto import tqdm
except ImportError:
    raise ValueError(
        "\n"
        "Please install requests and tqdm to download assets \n"
        "or install supervision with assets \n"
        "pip install supervision[assets] \n"
        "\n"
    )


def is_md5_hash_matching(filename: str, original_md5_hash: str) -> bool:
    """
    Check if the MD5 hash of a file matches the original hash.

    Parameters:
        filename (str): The path to the file to be checked as a string.
        original_md5_hash (str): The original MD5 hash to compare against.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    if not os.path.exists(filename):
        return False

    with open(filename, "rb") as file:
        file_contents = file.read()
        computed_md5_hash = md5(file_contents).hexdigest()

    return computed_md5_hash == original_md5_hash


def download_assets(asset_name: Union[VideoAssets, str]) -> str:
    """
    Download a specified asset if it doesn't already exist or is corrupted.

    Parameters:
        asset_name (Union[VideoAssets, str]): The name or type of the asset to be
            downloaded.

    Returns:
        str: The filename of the downloaded asset.

    Example:
        ```python
        >>> from supervision.assets import download_assets, VideoAssets

        >>> download_assets(VideoAssets.VEHICLES)
        "vehicles.mp4"
        ```
    """

    filename = asset_name.value if isinstance(asset_name, VideoAssets) else asset_name

    if not Path(filename).exists() and filename in VIDEO_ASSETS:
        print(f"Downloading {filename} assets \n")
        response = get(VIDEO_ASSETS[filename][0], stream=True, allow_redirects=True)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))
        folder_path = Path(filename).expanduser().resolve()
        folder_path.parent.mkdir(parents=True, exist_ok=True)

        with tqdm.wrapattr(
            response.raw, "read", total=file_size, desc="", colour="#a351fb"
        ) as raw_resp:
            with folder_path.open("wb") as file:
                copyfileobj(raw_resp, file)

    elif Path(filename).exists():
        if not is_md5_hash_matching(filename, VIDEO_ASSETS[filename][1]):
            print("File corrupted. Re-downloading... \n")
            os.remove(filename)
            return download_assets(filename)

        print(f"{filename} asset download complete. \n")

    else:
        valid_assets = ", ".join(asset.value for asset in VideoAssets)
        raise ValueError(
            f"Invalid asset. It should be one of the following: {valid_assets}."
        )

    return filename
