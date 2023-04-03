# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from random import random

import requests
from tqdm import tqdm

from ultralytics.yolo.utils import (ENVIRONMENT, LOGGER, ONLINE, RANK, SETTINGS, TESTS_RUNNING, TQDM_BAR_FORMAT,
                                    TryExcept, __version__, colorstr, emojis, get_git_origin_url, is_colab, is_git_dir,
                                    is_pip_package)

PREFIX = colorstr('Ultralytics HUB: ')
HELP_MSG = 'If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance.'
HUB_API_ROOT = os.environ.get('ULTRALYTICS_HUB_API', 'https://api.ultralytics.com')


def check_dataset_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=2.0):
    """
    Check if there is sufficient disk space to download and store a dataset.

    Args:
        url (str, optional): The URL to the dataset file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.

    Returns:
        bool: True if there is sufficient disk space, False otherwise.
    """
    gib = 1 << 30  # bytes per GiB
    data = int(requests.head(url).headers['Content-Length']) / gib  # dataset size (GB)
    total, used, free = (x / gib for x in shutil.disk_usage('/'))  # bytes
    LOGGER.info(f'{PREFIX}{data:.3f} GB dataset, {free:.1f}/{total:.1f} GB free disk space')
    if data * sf < free:
        return True  # sufficient space
    LOGGER.warning(f'{PREFIX}WARNING: Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, '
                   f'training cancelled ❌. Please free {data * sf - free:.1f} GB additional disk space and try again.')
    return False  # insufficient space


def request_with_credentials(url: str) -> any:
    """
    Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        any: The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not is_colab():
        raise OSError('request_with_credentials() must run in a Colab environment')
    from google.colab import output  # noqa
    from IPython import display  # noqa
    display.display(
        display.Javascript("""
            window._hub_tmp = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("%s", {
                    method: 'POST',
                    credentials: 'include'
                })
                    .then((response) => resolve(response.json()))
                    .then((json) => {
                    clearTimeout(timeout);
                    }).catch((err) => {
                    clearTimeout(timeout);
                    reject(err);
                });
            });
            """ % url))
    return output.eval_js('_hub_tmp')


def split_key(key=''):
    """
    Verify and split a 'api_key[sep]model_id' string, sep is one of '.' or '_'

    Args:
        key (str): The model key to split. If not provided, the user will be prompted to enter it.

    Returns:
        Tuple[str, str]: A tuple containing the API key and model ID.
    """

    import getpass

    error_string = emojis(f'{PREFIX}Invalid API key ⚠️\n')  # error string
    if not key:
        key = getpass.getpass('Enter model key: ')
    sep = '_' if '_' in key else None  # separator
    assert sep, error_string
    api_key, model_id = key.split(sep)
    assert len(api_key) and len(model_id), error_string
    return api_key, model_id


def requests_with_progress(method, url, **kwargs):
    """
    Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        progress (bool, optional): Whether to display a progress bar. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        requests.Response: The response from the HTTP request.
    """
    progress = kwargs.pop('progress', False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get('content-length', 0))  # total size
    pbar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, bar_format=TQDM_BAR_FORMAT)
    for data in response.iter_content(chunk_size=1024):
        pbar.update(len(data))
    pbar.close()
    return response


def smart_request(method, url, retry=3, timeout=30, thread=True, code=-1, verbose=True, progress=False, **kwargs):
    """
    Makes an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up. Default is 3.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying. Default is 30.
        thread (bool, optional): Whether to execute the request in a separate daemon thread. Default is True.
        code (int, optional): An identifier for the request, used for logging purposes. Default is -1.
        verbose (bool, optional): A flag to determine whether to print out to console or not. Default is True.
        progress (bool, optional): Whether to show a progress bar during the request. Default is False.
        **kwargs: Keyword arguments to be passed to the requests function specified in method.

    Returns:
        requests.Response: The HTTP response object. If the request is executed in a separate thread, returns None.
    """
    retry_codes = (408, 500)  # retry only these codes

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        r = None  # response
        t0 = time.time()  # initial time for timer
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)  # i.e. get(url, data, json, files)
            if r.status_code == 200:
                break
            try:
                m = r.json().get('message', 'No JSON message.')
            except AttributeError:
                m = 'Unable to read JSON.'
            if i == 0:
                if r.status_code in retry_codes:
                    m += f' Retrying {retry}x for {timeout}s.' if retry else ''
                elif r.status_code == 429:  # rate limit
                    h = r.headers  # response headers
                    m = f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). " \
                        f"Please retry after {h['Retry-After']}s."
                if verbose:
                    LOGGER.warning(f'{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})')
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2 ** i)  # exponential standoff
        return r

    args = method, url
    kwargs['progress'] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)


class Traces:

    def __init__(self):
        """
        Initialize Traces for error tracking and reporting if tests are not currently running.
        Sets the rate limit, timer, and metadata attributes, and determines whether Traces are enabled.
        """
        self.rate_limit = 60.0  # rate limit (seconds)
        self.t = 0.0  # rate limit timer (seconds)
        self.metadata = {
            'sys_argv_name': Path(sys.argv[0]).name,
            'install': 'git' if is_git_dir() else 'pip' if is_pip_package() else 'other',
            'python': platform.python_version(),
            'release': __version__,
            'environment': ENVIRONMENT}
        self.enabled = \
            SETTINGS['sync'] and \
            RANK in (-1, 0) and \
            not TESTS_RUNNING and \
            ONLINE and \
            (is_pip_package() or get_git_origin_url() == 'https://github.com/ultralytics/ultralytics.git')
        self._reset_usage()

    def __call__(self, cfg, all_keys=False, traces_sample_rate=1.0):
        """
        Sync traces data if enabled in the global settings.

        Args:
            cfg (IterableSimpleNamespace): Configuration for the task and mode.
            all_keys (bool): Sync all items, not just non-default values.
            traces_sample_rate (float): Fraction of traces captured from 0.0 to 1.0.
        """

        # Increment usage
        self.usage['modes'][cfg.mode] = self.usage['modes'].get(cfg.mode, 0) + 1
        self.usage['tasks'][cfg.task] = self.usage['tasks'].get(cfg.task, 0) + 1

        t = time.time()  # current time
        if not self.enabled or random() > traces_sample_rate:
            # Traces disabled or not randomly selected, do nothing
            return
        elif (t - self.t) < self.rate_limit:
            # Time is under rate limiter, do nothing
            return
        else:
            # Time is over rate limiter, send trace now
            trace = {'uuid': SETTINGS['uuid'], 'usage': self.usage.copy(), 'metadata': self.metadata}

            # Send a request to the HUB API to sync analytics
            smart_request('post', f'{HUB_API_ROOT}/v1/usage/anonymous', json=trace, code=3, retry=0, verbose=False)

            # Reset usage and rate limit timer
            self._reset_usage()
            self.t = t

    def _reset_usage(self):
        """Reset the usage dictionary by initializing keys for each task and mode with a value of 0."""
        from ultralytics.yolo.cfg import MODES, TASKS
        self.usage = {'tasks': {k: 0 for k in TASKS}, 'modes': {k: 0 for k in MODES}}


# Run below code on hub/utils init -------------------------------------------------------------------------------------
traces = Traces()
