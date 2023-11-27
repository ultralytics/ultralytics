#!/usr/bin/env python

import sys
from numpy import load
from scipy.io import savemat

assert len(sys.argv) > 1

files = sys.argv[1:]

for f in files:
   data = load(f)
   fn = f.replace('.npz','')
   fn = fn.replace('.','_') #matlab cant handle dots
   savemat(fn,data)
