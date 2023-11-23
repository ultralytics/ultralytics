# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making cython reid evaluation module failed, exiting.")
        import sys

        sys.exit(1)
