import shutil
import threading
import time

import requests

from ultralytics.yolo.utils import LOGGER, PREFIX, emojis

HELP_MSG = 'If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance.'


def check_dataset_disk_space(url='https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', sf=2.0):
    # Check that url fits on disk with safety factor sf, i.e. require 2GB free if url size is 1GB with sf=2.0
    gib = 1 / 1024 ** 3  # bytes to GiB
    data = int(requests.head(url).headers['Content-Length']) * gib  # dataset size (GB)
    total, used, free = (x * gib for x in shutil.disk_usage("/"))  # bytes
    print(f'{PREFIX}{data:.3f} GB dataset, {free:.1f}/{total:.1f} GB free disk space')
    if data * sf < free:
        return True  # sufficient space
    s = f'{PREFIX}WARNING: Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, ' \
        f'training cancelled ❌. Please free {data * sf - free:.1f} GB additional disk space and try again.'
    print(emojis(s))
    return False  # insufficient space


def request_with_credentials(url: str) -> any:
    """ Make a ajax request with cookies attached """
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


#! deprecated
def split_key(key=''):
    # Verify and split a 'api_key[sep]model_id' string, sep is one of '.' or '_'
    # key = 'ac0ab020186aeb50cc4c2a5272de17f58bbd2c0_RqFCDNBxgU4mOLmaBrcd'  # example
    # api_key='ac0ab020186aeb50cc4c2a5272de17f58bbd2c0', model_id='RqFCDNBxgU4mOLmaBrcd'  # example
    import getpass

    s = emojis(f'{PREFIX}Invalid API key ⚠️\n')  # error string
    if not key:
        key = getpass.getpass('Enter model key: ')
    sep = '_' if '_' in key else '.' if '.' in key else None  # separator
    assert sep, s
    api_key, model_id = key.split(sep)
    assert len(api_key) and len(model_id), s
    return api_key, model_id


def smart_request(*args, retry=3, timeout=30, thread=True, code='', method="post", **kwargs):
    # requests.post with exponential standoff retries up to timeout(seconds)
    retry_codes = (408, 500)  # retry only these codes
    methods = {'post': requests.post, 'get': requests.get}  # request methods

    def fcn(*args, **kwargs):
        t0 = time.time()
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = methods[method](*args, **kwargs)  # i.e. post(url, data, json, files)
            if r.status_code == 200:
                break
            try:
                m = r.json().get('message', 'No JSON message.')
            except Exception:
                m = 'Unable to read JSON.'
            if i == 0:
                if r.status_code in retry_codes:
                    m += f' Retrying {retry}x for {timeout}s.' if retry else ''
                elif r.status_code == 429:  # rate limit
                    h = r.headers  # response headers
                    m = f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). " \
                        f"Please retry after {h['Retry-After']}s."
                LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2 ** i)  # exponential standoff
        return r

    if thread:
        threading.Thread(target=fcn, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return fcn(*args, **kwargs)
