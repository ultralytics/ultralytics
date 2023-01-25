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

from ultralytics.yolo.utils import (DEFAULT_CFG_DICT, LOGGER, RANK, SETTINGS, TryExcept, colorstr, emojis,
                                    get_git_origin_url, is_colab, is_docker, is_git_dir, is_github_actions_ci,
                                    is_jupyter, is_kaggle, is_pip_package, is_pytest_running)

PREFIX = colorstr('Ultralytics: ')
HELP_MSG = 'If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance.'
HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")


def check_dataset_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=2.0):
    # Check that url fits on disk with safety factor sf, i.e. require 2GB free if url size is 1GB with sf=2.0
    gib = 1 << 30  # bytes per GiB
    data = int(requests.head(url).headers['Content-Length']) / gib  # dataset size (GB)
    total, used, free = (x / gib for x in shutil.disk_usage("/"))  # bytes
    LOGGER.info(f'{PREFIX}{data:.3f} GB dataset, {free:.1f}/{total:.1f} GB free disk space')
    if data * sf < free:
        return True  # sufficient space
    LOGGER.warning(f'{PREFIX}WARNING: Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, '
                   f'training cancelled ❌. Please free {data * sf - free:.1f} GB additional disk space and try again.')
    return False  # insufficient space


def request_with_credentials(url: str) -> any:
    """ Make an ajax request with cookies attached """
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
    return output.eval_js("_hub_tmp")


# Deprecated TODO: eliminate this function?
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
    sep = '_' if '_' in key else '.' if '.' in key else None  # separator
    assert sep, error_string
    api_key, model_id = key.split(sep)
    assert len(api_key) and len(model_id), error_string
    return api_key, model_id


def smart_request(*args, retry=3, timeout=30, thread=True, code=-1, method="post", verbose=True, **kwargs):
    """
    Makes an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        *args: Positional arguments to be passed to the requests function specified in method.
        retry (int, optional): Number of retries to attempt before giving up. Default is 3.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying. Default is 30.
        thread (bool, optional): Whether to execute the request in a separate daemon thread. Default is True.
        code (int, optional): An identifier for the request, used for logging purposes. Default is -1.
        method (str, optional): The HTTP method to use for the request. Choices are 'post' and 'get'. Default is 'post'.
        verbose (bool, optional): A flag to determine whether to print out to console or not. Default is True.
        **kwargs: Keyword arguments to be passed to the requests function specified in method.

    Returns:
        requests.Response: The HTTP response object. If the request is executed in a separate thread, returns None.
    """
    retry_codes = (408, 500)  # retry only these codes

    def func(*func_args, **func_kwargs):
        r = None  # response
        t0 = time.time()  # initial time for timer
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            if method == 'post':
                r = requests.post(*func_args, **func_kwargs)  # i.e. post(url, data, json, files)
            elif method == 'get':
                r = requests.get(*func_args, **func_kwargs)  # i.e. get(url, data, json, files)
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
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2 ** i)  # exponential standoff
        return r

    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)


class Traces:

    def __init__(self):
        """
        Initialize Traces for error tracking and reporting if tests are not currently running.
        """
        from ultralytics import __version__
        env = 'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Jupyter' if is_jupyter() else \
            'Docker' if is_docker() else platform.system()
        self.rate_limit = 3.0  # rate limit (seconds)
        self.t = time.time()  # rate limit timer (seconds)
        self.metadata = {
            "sys_argv_name": Path(sys.argv[0]).name,
            "install": 'git' if is_git_dir() else 'pip' if is_pip_package() else 'other',
            "python": platform.python_version(),
            "release": __version__,
            "environment": env}
        self.enabled = SETTINGS['sync'] and \
                       RANK in {-1, 0} and \
                       not is_pytest_running() and \
                       not is_github_actions_ci() and \
                       (is_pip_package() or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")

    @TryExcept(verbose=False)
    def __call__(self, cfg, all_keys=False, traces_sample_rate=1.0):
        """
       Sync traces data if enabled in the global settings

        Args:
            cfg (IterableSimpleNamespace): Configuration for the task and mode.
            all_keys (bool): Sync all items, not just non-default values.
            traces_sample_rate (float): Fraction of traces captured from 0.0 to 1.0
        """
        t = time.time()  # current time
        if self.enabled and random() < traces_sample_rate and (t - self.t) > self.rate_limit:
            self.t = t  # reset rate limit timer
            cfg = vars(cfg)  # convert type from IterableSimpleNamespace to dict
            if not all_keys:  # filter cfg
                include_keys = {'task', 'mode'}  # always include
                cfg = {k: v for k, v in cfg.items() if v != DEFAULT_CFG_DICT.get(k, None) or k in include_keys}
            trace = {'uuid': SETTINGS['uuid'], 'cfg': cfg, 'metadata': self.metadata}

            # Send a request to the HUB API to sync analytics
            smart_request(f'{HUB_API_ROOT}/v1/usage/anonymous',
                          json=trace,
                          headers=None,
                          code=3,
                          retry=0,
                          verbose=False)


# Run below code on hub/utils init -------------------------------------------------------------------------------------

traces = Traces()
