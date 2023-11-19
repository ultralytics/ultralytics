# Ultralytics YOLO 🚀, AGPL-3.0 license

import re
from pathlib import Path

from setuptools import setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')


def get_version():
    """
    Retrieve the version number from the 'ultralytics/__init__.py' file.

    Returns:
        (str): The version number extracted from the '__version__' attribute in the 'ultralytics/__init__.py' file.
    """
    file = PARENT / 'ultralytics/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        (List[str]): List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line.split('#')[0].strip())  # ignore inline comments

    return requirements


setup(
    name='ultralytics',  # name of pypi package
    version=get_version(),  # version of pypi package
    python_requires='>=3.8',
    license='AGPL-3.0',
    description=('Ultralytics YOLOv8 for SOTA object detection, multi-object tracking, instance segmentation, '
                 'pose estimation and image classification.'),
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ultralytics/ultralytics',
    project_urls={
        'Bug Reports': 'https://github.com/ultralytics/ultralytics/issues',
        'Funding': 'https://ultralytics.com',
        'Source': 'https://github.com/ultralytics/ultralytics'},
    author='Ultralytics',
    author_email='hello@ultralytics.com',
    packages=['ultralytics'] + [str(x) for x in Path('ultralytics').rglob('*/') if x.is_dir() and '__' not in str(x)],
    package_data={
        '': ['*.yaml'],
        'ultralytics.assets': ['*.jpg']},
    include_package_data=True,
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
    extras_require={
        'dev': [
            'ipython',
            'check-manifest',
            'pre-commit',
            'pytest',
            'pytest-cov',
            'coverage',
            'mkdocs-material',
            'mkdocstrings[python]',
            'mkdocs-redirects',  # for 301 redirects
            'mkdocs-ultralytics-plugin>=0.0.34',  # for meta descriptions and images, dates and authors
        ],
        'export': [
            'coremltools>=7.0',
            'openvino-dev>=2023.0',
            'tensorflow<=2.13.1',
            'tensorflowjs',  # automatically installs tensorflow
        ], },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows', ],
    keywords='machine-learning, deep-learning, vision, ML, DL, AI, YOLO, YOLOv3, YOLOv5, YOLOv8, HUB, Ultralytics',
    entry_points={'console_scripts': ['yolo = ultralytics.cfg:entrypoint', 'ultralytics = ultralytics.cfg:entrypoint']})
