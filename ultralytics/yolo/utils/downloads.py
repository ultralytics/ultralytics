# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import os
import subprocess
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from zipfile import ZipFile

import requests
import torch

from ultralytics.yolo.utils import LOGGER


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def safe_download(url,
                  file=None,
                  dir=None,
                  unzip=True,
                  delete=False,
                  curl=False,
                  retry=3,
                  min_bytes=1E0,
                  progress=True):
    """
    Function for downloading files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url: str: The URL of the file to be downloaded.
        file: str, optional: The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir: str, optional: The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip: bool, optional: Whether to unzip the downloaded file. Default: True.
        delete: bool, optional: Whether to delete the downloaded file after unzipping. Default: False.
        curl: bool, optional: Whether to use curl command line tool for downloading. Default: False.
        retry: int, optional: The number of times to retry the download in case of failure. Default: 3.
        min_bytes: float, optional: The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        progress: bool, optional: Whether to display a progress bar during the download. Default: True.
    """
    if '://' not in str(url) and Path(url).is_file():  # exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    else:  # does not exist
        assert dir or file, 'dir or file required for download'
        f = dir / Path(url).name if dir else Path(file)
        LOGGER.info(f'Downloading {url} to {f}...')
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = 'sS' * (not progress)  # silent
                    r = os.system(f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -')
                else:  # torch download
                    r = torch.hub.download_url_to_file(url, f, progress=progress)
                assert r in {0, None}
            except Exception as e:
                if i >= retry:
                    raise ConnectionError(f'❌  Download failure for {url}') from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')
                continue

            if f.exists():
                if f.stat().st_size > min_bytes:
                    break  # success
                f.unlink()  # remove partial downloads

    if unzip and f.exists() and f.suffix in {'.zip', '.tar', '.gz'}:
        LOGGER.info(f'Unzipping {f}...')
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=f.parent)  # unzip
        elif f.suffix == '.tar':
            os.system(f'tar xf {f} --directory {f.parent}')  # unzip
        elif f.suffix == '.gz':
            os.system(f'tar xfz {f} --directory {f.parent}')  # unzip
        if delete:
            f.unlink()  # remove zip


def attempt_download_asset(file, repo='ultralytics/assets', release='v0.0.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.2', etc.
    from ultralytics.yolo.utils import SETTINGS

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov5m.pt', ...])
        # Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov8s.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.2
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    file = Path(str(file).strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(url=url, file=file, min_bytes=1E5)
            return file

        # GitHub assets
        assets = [f'yolov8{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        if name in assets:
            safe_download(url=f'https://github.com/{repo}/releases/download/{tag}/{name}', file=file, min_bytes=1E5)

        return str(file)


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0], dir=x[1], unzip=unzip, delete=delete, curl=curl, retry=retry, progress=threads <= 1),
                zip(url, repeat(dir)))
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry)
