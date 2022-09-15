"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib
import re

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()  # current path
long_description = (here / 'README.md').read_text(encoding='utf-8')  # Get the long description from the README file
with open(here / 'requirements.txt') as fp:  # read requirements.txt
    install_reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]


def get_version():
    file = here / 'ultralytics/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(), re.M).group(1)


setup(
    name='ultralytics',
    version=get_version(),
    description='Ultralytics YOLOv5 Python package, https://ultralytics.com',
    long_description=long_description, 
    long_description_content_type='text/markdown',
    url='https://github.com/ultralytics/ultralytics',
    author='Ultralytics',
    classifiers=['Intended Audience :: Developers',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Image Recognition',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 ], 
    keywords='machine-learning, deep-learning, ml, pytorch, YOLO, object-detection, YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv8', 
    package_dir={'': 'ultralytics'},
    packages=find_packages(where='ultralytics'),
    python_requires='>=3.7, <4',
    install_requires=install_reqs,  
    extras_require={'dev': ['check-manifest'],
                    'test': ['coverage'],
                    },
    package_data={'ultralytics': ['package_data.dat'],
                  },
    #data_files=[('my_data', ['data/data_file'])],  # Optional

    #entry_points={'console_scripts': ['ultralytics=ultralytics.console:main', ],},  # Optional

    project_urls={'Bug Reports': 'https://github.com/ultralytics/ultralytics/issues',
                  'Funding': 'https://www.ultralytics.com',
                  'Source': 'https://github.com/ultralytics/ultralytics/',
                  },
)