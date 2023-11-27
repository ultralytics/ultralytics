#!/usr/bin/env python

import sys
from numpy import savez
from scipy.io import loadmat

assert len(sys.argv) > 1

files = sys.argv[1:]

for f in files:
    mat_vars = loadmat(f)
    mat_vars.pop('__version__')
    mat_vars.pop('__header__')
    mat_vars.pop('__globals__')

    fn = f.replace('.mat','.npz')
    savez(fn,**mat_vars)
