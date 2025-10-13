# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
import shutil
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request

from ultralytics.utils import ASSETS_URL, LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

# Define Ultralytics GitHub assets maintained at https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = frozenset(
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolo11{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    + [f"yolo12{k}{suffix}.pt" for k in "nsmlx" for suffix in ("",)]  # detect models only currently
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yoloe-v8{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-11{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"sam2_{k}.pt" for k in "blst"]
    + [f"sam2.1_{k}.pt" for k in "blst"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + [
        "mobile_sam.pt",
        "mobileclip_blt.ts",
        "yolo11n-grayscale.pt",
        "calibration_image_sample_data_20x128x128x3_float32.npy.zip",
    ]
)
GITHUB_ASSETS_STEMS = frozenset(k.rpartition(".")[0] for k in GITHUB_ASSETS_NAMES)


def is_url(url: str | Path, check: bool = False) -> bool:
    """
    Validate if the given string is a URL and optionally check if the URL exists online.

    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.

    Returns:
        (bool): True for a valid URL. If 'check' is True, also returns True if the URL exists online.

    Examples:
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            r = request.urlopen(request.Request(url, method="HEAD"), timeout=3)
            return 200 <= r.getcode() < 400
        return True
    except Exception:
        return False


def delete_dsstore(path: str | Path, files_to_delete: tuple[str, ...] = (".DS_Store", "__MACOSX")) -> None:
    """
    Delete all specified system files in a directory.

    Args:
        path (str | Path): The directory path where the files should be deleted.
        files_to_delete (tuple): The files to be deleted.

    Examples:
        >>> from ultralytics.utils.downloads import delete_dsstore
        >>> delete_dsstore("path/to/dir")

    Notes:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(
    directory: str | Path,
    compress: bool = True,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    progress: bool = True,
) -> Path:
    """
    Zip the contents of a directory, excluding specified files.

    The resulting zip file is named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping.
        exclude (tuple, optional): A tuple of filename strings to be excluded.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the resulting zip file.

    Examples:
        >>> from ultralytics.utils.downloads import zip_directory
        >>> file = zip_directory("path/to/dir")
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Zip with progress bar
    files = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]  # files to zip
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files, desc=f"Zipping {directory} to {zip_file}...", unit="files", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file  # return path to zip file


def unzip_file(
    file: str | Path,
    path: str | Path | None = None,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    exist_ok: bool = False,
    progress: bool = True,
) -> Path:
    """
    Unzip a *.zip file to the specified path, excluding specified files.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str | Path): The path to the zipfile to be extracted.
        path (str | Path, optional): The path to extract the zipfile to.
        exclude (tuple, optional): A tuple of filename strings to be excluded.
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Examples:
        >>> from ultralytics.utils.downloads import unzip_file
        >>> directory = unzip_file("path/to/file.zip")
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # Decide to unzip directly or unzip into a directory
        unzip_as_dir = len(top_level_dirs) == 1  # (len(files) > 1 and not files[0].endswith("/"))
        if unzip_as_dir:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. extract coco8/ dir to ../datasets/
        else:
            # Zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. extract multiple files to ../datasets/coco8/

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f"Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="files", disable=not progress):
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path  # return unzip dir


def check_disk_space(
    file_bytes: int,
    path: str | Path = Path.cwd(),
    sf: float = 1.5,
    hard: bool = True,
) -> bool:
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        file_bytes (int): The file size in bytes.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    total, used, free = shutil.disk_usage(path)  # bytes
    if file_bytes * sf < free:
        return True  # sufficient space

    # Insufficient space
    text = (
        f"Insufficient free disk space {free >> 30:.3f} GB < {int(file_bytes * sf) >> 30:.3f} GB required, "
        f"Please free {int(file_bytes * sf - free) >> 30:.3f} GB additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link: str) -> tuple[str, str | None]:
    """
    Retrieve the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        url (str): Direct download URL for the Google Drive file.
        filename (str | None): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Examples:
        >>> from ultralytics.utils.downloads import get_google_drive_file_info
        >>> link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        >>> url, filename = get_google_drive_file_info(link)
    """
    import requests  # scoped as slow import

    file_id = link.split("/d/")[1].split("/view", 1)[0]
    drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    filename = None

    # Start session
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if "quota exceeded" in str(response.content.lower()):
            raise ConnectionError(
                emojis(
                    f"❌  Google Drive file download quota exceeded. "
                    f"Please try again later or download this file manually at {link}."
                )
            )
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                drive_url += f"&confirm={v}"  # v is token
        if cd := response.headers.get("content-disposition"):
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(
    url: str | Path,
    file: str | Path | None = None,
    dir: str | Path | None = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    retry: int = 3,
    min_bytes: float = 1e0,
    exist_ok: bool = False,
    progress: bool = True,
) -> Path | str:
    """
    Download files from a URL with options for retrying, unzipping, and deleting the downloaded file. Enhanced with
    robust partial download detection using Content-Length validation.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str | Path, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file.
        delete (bool, optional): Whether to delete the downloaded file after unzipping.
        curl (bool, optional): Whether to use curl command line tool for downloading.
        retry (int, optional): The number of times to retry the download in case of failure.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.
        progress (bool, optional): Whether to display a progress bar during the download.

    Returns:
        (Path | str): The path to the downloaded file or extracted directory.

    Examples:
        >>> from ultralytics.utils.downloads import safe_download
        >>> link = "https://ultralytics.com/assets/bus.jpg"
        >>> path = safe_download(link)
    """
    gdrive = url.startswith("https://drive.google.com/")  # check if the URL is a Google Drive link
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = Path(dir or ".") / (file or url2file(url))  # URL converted to filename
    if "://" not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        uri = (url if gdrive else clean_url(url)).replace(ASSETS_URL, "https://ultralytics.com/assets")  # clean
        desc = f"Downloading {uri} to '{f}'"
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        curl_installed = shutil.which("curl")
        for i in range(retry + 1):
            try:
                if (curl or i > 0) and curl_installed:  # curl download with retry, continue
                    s = "sS" * (not progress)  # silent
                    r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
                    assert r == 0, f"Curl return value {r}"
                    expected_size = None  # Can't get size with curl
                else:  # urllib download
                    with request.urlopen(url) as response:
                        expected_size = int(response.getheader("Content-Length", 0))
                        if i == 0 and expected_size > 1048576:
                            check_disk_space(expected_size, path=f.parent)
                        buffer_size = max(8192, min(1048576, expected_size // 1000)) if expected_size else 8192
                        with TQDM(
                            total=expected_size,
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                while True:
                                    data = response.read(buffer_size)
                                    if not data:
                                        break
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    file_size = f.stat().st_size
                    if file_size > min_bytes:
                        # Check if download is complete (only if we have expected_size)
                        if expected_size and file_size != expected_size:
                            LOGGER.warning(
                                f"Partial download: {file_size}/{expected_size} bytes ({file_size / expected_size * 100:.1f}%)"
                            )
                        else:
                            break  # success
                    f.unlink()  # remove partial downloads
            except MemoryError:
                raise  # Re-raise immediately - no point retrying if insufficient disk space
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Environment may be offline.")) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached. {e}")) from e
                LOGGER.warning(f"Download failure, retrying {i + 1}/{retry} {uri}... {e}")

    if unzip and f.exists() and f.suffix in {"", ".zip", ".tar", ".gz"}:
        from zipfile import is_zipfile

        unzip_dir = (dir or f.parent).resolve()  # unzip to dir if provided else unzip in place
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, exist_ok=exist_ok, progress=progress)  # unzip
        elif f.suffix in {".tar", ".gz"}:
            LOGGER.info(f"Unzipping {f} to {unzip_dir}...")
            subprocess.run(["tar", "xf" if f.suffix == ".tar" else "xfz", f, "--directory", unzip_dir], check=True)
        if delete:
            f.unlink()  # remove zip
        return unzip_dir
    return f


def get_github_assets(
    repo: str = "ultralytics/assets",
    version: str = "latest",
    retry: bool = False,
) -> tuple[str, list[str]]:
    """
    Retrieve the specified version's tag and assets from a GitHub repository.

    If the version is not specified, the function fetches the latest release assets.

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        version (str, optional): The release version to fetch assets from.
        retry (bool, optional): Flag to retry the request in case of a failure.

    Returns:
        tag (str): The release tag.
        assets (list[str]): A list of asset names.

    Examples:
        >>> tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
    """
    import requests  # scoped as slow import

    if version != "latest":
        version = f"tags/{version}"  # i.e. tags/v6.2
    url = f"https://api.github.com/repos/{repo}/releases/{version}"
    r = requests.get(url)  # github api
    if r.status_code != 200 and r.reason != "rate limit exceeded" and retry:  # failed and not 403 rate limit exceeded
        r = requests.get(url)  # try again
    if r.status_code != 200:
        LOGGER.warning(f"GitHub assets check failure for {url}: {r.status_code} {r.reason}")
        return "", []
    data = r.json()
    return data["tag_name"], [x["name"] for x in data["assets"]]  # tag, assets i.e. ['yolo11n.pt', 'yolov8s.pt', ...]


def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.3.0",
    **kwargs,
) -> str:
    """
    Attempt to download a file from GitHub release assets if it is not found locally.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        release (str, optional): The specific release version to be downloaded.
        **kwargs (Any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Examples:
        >>> file_path = attempt_download_asset("yolo11n.pt", repo="ultralytics/assets", release="latest")
    """
    from ultralytics.utils import SETTINGS  # scoped for circular import

    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        download_url = f"https://github.com/{repo}/releases/download"
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = url2file(name)  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # file already exists
            else:
                safe_download(url=url, file=file, min_bytes=1e5, **kwargs)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f"{download_url}/{release}/{name}", file=file, min_bytes=1e5, **kwargs)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # latest release
            if name in assets:
                safe_download(url=f"{download_url}/{tag}/{name}", file=file, min_bytes=1e5, **kwargs)

        return str(file)


def download(
    url: str | list[str] | Path,
    dir: Path = Path.cwd(),
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    threads: int = 1,
    retry: int = 3,
    exist_ok: bool = False,
) -> None:
    """
    Download files from specified URLs to a given directory.

    Supports concurrent downloads if multiple threads are specified.

    Args:
        url (str | list[str]): The URL or list of URLs of the files to be downloaded.
        dir (Path, optional): The directory where the files will be saved.
        unzip (bool, optional): Flag to unzip the files after downloading.
        delete (bool, optional): Flag to delete the zip files after extraction.
        curl (bool, optional): Flag to use curl for downloading.
        threads (int, optional): Number of threads to use for concurrent downloads.
        retry (int, optional): Number of retries in case of download failure.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.

    Examples:
        >>> download("https://ultralytics.com/assets/example.zip", dir="path/to/dir", unzip=True)
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    urls = [url] if isinstance(url, (str, Path)) else url
    if threads > 1:
        LOGGER.info(f"Downloading {len(urls)} file(s) with {threads} threads to {dir}...")
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0],
                    dir=x[1],
                    unzip=unzip,
                    delete=delete,
                    curl=curl,
                    retry=retry,
                    exist_ok=exist_ok,
                    progress=True,
                ),
                zip(urls, repeat(dir)),
            )
            pool.close()
            pool.join()
    else:
        for u in urls:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry, exist_ok=exist_ok)
