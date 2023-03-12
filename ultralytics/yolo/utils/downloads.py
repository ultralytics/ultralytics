# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request
from zipfile import BadZipFile, ZipFile, is_zipfile

import requests
import torch
from tqdm import tqdm

from ultralytics.yolo.utils import LOGGER, checks, is_online

GITHUB_ASSET_NAMES = [f'yolov8{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '6', '-cls', '-seg')] + \
                     [f'yolov5{size}u.pt' for size in 'nsmlx'] + \
                     [f'yolov3{size}u.pt' for size in ('', '-spp', '-tiny')]
GITHUB_ASSET_STEMS = [Path(k).stem for k in GITHUB_ASSET_NAMES]


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    return False


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX')):
    """
    Unzip a *.zip file to path/, excluding files containing strings in exclude list
    Replaces: ZipFile(file).extractall(path=path)
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)
        return zipObj.namelist()[0]  # return unzip dir


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
        desc = f'Downloading {url} to {f}'
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = 'sS' * (not progress)  # silent
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o', f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:  # urllib download
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        from ultralytics.yolo.utils import TQDM_BAR_FORMAT
                        with request.urlopen(url) as response, tqdm(total=int(response.getheader('Content-Length', 0)),
                                                                    desc=desc,
                                                                    disable=not progress,
                                                                    unit='B',
                                                                    unit_scale=True,
                                                                    unit_divisor=1024,
                                                                    bar_format=TQDM_BAR_FORMAT) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(f'❌  Download failure for {url}. Environment is not online.') from e
                elif i >= retry:
                    raise ConnectionError(f'❌  Download failure for {url}. Retry limit reached.') from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')

    if unzip and f.exists() and f.suffix in ('.zip', '.tar', '.gz'):
        unzip_dir = dir or f.parent  # unzip to dir if provided else unzip in place
        LOGGER.info(f'Unzipping {f} to {unzip_dir}...')
        if f.suffix == '.zip':
            unzip_dir = unzip_file(file=f, path=unzip_dir)  # unzip
        elif f.suffix == '.tar':
            subprocess.run(['tar', 'xf', f, '--directory', unzip_dir], check=True)  # unzip
        elif f.suffix == '.gz':
            subprocess.run(['tar', 'xfz', f, '--directory', unzip_dir], check=True)  # unzip
        if delete:
            f.unlink()  # remove zip
        return unzip_dir


def attempt_download_asset(file, repo='ultralytics/assets', release='v0.0.0'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v6.2', etc.
    from ultralytics.yolo.utils import SETTINGS  # scoped for circular import

    def github_assets(repository, version='latest'):
        # Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov8s.pt', ...])
        if version != 'latest':
            version = f'tags/{version}'  # i.e. tags/v6.2
        response = requests.get(f'https://api.github.com/repos/{repository}/releases/{version}').json()  # github api
        return response['tag_name'], [x['name'] for x in response['assets']]  # tag, assets

    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {url} locally at {file}')  # file already exists
            else:
                safe_download(url=url, file=file, min_bytes=1E5)
            return file

        # GitHub assets
        assets = GITHUB_ASSET_NAMES
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output(['git', 'tag']).decode().split()[-1]
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
