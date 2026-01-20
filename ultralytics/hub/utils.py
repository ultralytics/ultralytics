# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import threading
import time
from typing import Any

from ultralytics.utils import (
    IS_COLAB,
    LOGGER,
    TQDM,
    TryExcept,
    colorstr,
)

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

PREFIX = colorstr("Ultralytics HUB: ")
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> Any:
    """Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        (Any): The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not IS_COLAB:
        raise OSError("request_with_credentials() must run in a Colab environment")
    from google.colab import output
    from IPython import display

    display.display(
        display.Javascript(
            f"""
            window._hub_tmp = new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("{url}", {{
                    method: 'POST',
                    credentials: 'include'
                }})
                    .then((response) => resolve(response.json()))
                    .then((json) => {{
                    clearTimeout(timeout);
                    }}).catch((err) => {{
                    clearTimeout(timeout);
                    reject(err);
                }});
            }});
            """
        )
    )
    return output.eval_js("_hub_tmp")


def requests_with_progress(method: str, url: str, **kwargs):
    """Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (Any): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Notes:
        - If 'progress' is set to True, the progress bar will display the download progress for responses with a known
          content length.
        - If 'progress' is a number then progress bar will display assuming content length = progress.
    """
    import requests  # scoped as slow import

    progress = kwargs.pop("progress", False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)  # total size
    try:
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:  # avoid 'Connection broken: IncompleteRead' warnings
        response.close()
    return response


def smart_request(
    method: str,
    url: str,
    retry: int = 3,
    timeout: int = 30,
    thread: bool = True,
    code: int = -1,
    verbose: bool = True,
    progress: bool = False,
    **kwargs,
):
    """Make an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying.
        thread (bool, optional): Whether to execute the request in a separate daemon thread.
        code (int, optional): An identifier for the request, used for logging purposes.
        verbose (bool, optional): A flag to determine whether to print out to console or not.
        progress (bool, optional): Whether to show a progress bar during the request.
        **kwargs (Any): Keyword arguments to be passed to the requests function specified in method.

    Returns:
        (requests.Response | None): The HTTP response object. If the request is executed in a separate thread, returns
            None.
    """
    retry_codes = (408, 500)  # retry only these codes

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None  # response
        t0 = time.time()  # initial time for timer
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)  # i.e. get(url, data, json, files)
            if r.status_code < 300:  # return codes in the 2xx range are generally considered "good" or "successful"
                break
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:  # rate limit
                    h = r.headers  # response headers
                    m = (
                        f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). "
                        f"Please retry after {h['Retry-After']}s."
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)  # exponential standoff
        return r

    args = method, url
    kwargs["progress"] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)
