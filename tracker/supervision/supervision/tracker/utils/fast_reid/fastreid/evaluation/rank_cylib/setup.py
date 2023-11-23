from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'rank_cy',
        ['rank_cy.pyx'],
        include_dirs=[numpy_include()],
    ),
    Extension(
        'roc_cy',
        ['roc_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]

setup(
    name='Cython-based reid evaluation code',
    ext_modules=cythonize(ext_modules)
)
