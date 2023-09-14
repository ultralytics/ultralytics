# Ultralytics YOLO 🚀, AGPL-3.0 license

import re
from pathlib import Path

from setuptools import setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')


def get_version():
    file = PARENT / 'ultralytics/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]


def parse_requirements(file_path: Path):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (str | Path): Path to the requirements.txt file.

    Returns:
        List[str]: List of parsed requirements.
    """

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line.split('#')[0].strip())  # ignore inline comments

    return requirements


def find_packages(start_dir, exclude=()):
    """
    Custom implementation of setuptools.find_packages(). Finds all Python packages in a directory.

    Args:
        start_dir (str | Path, optional): The directory where the search will start. Defaults to the current directory.
        exclude (list | tuple, optional): List of package names to exclude. Defaults to None.

    Returns:
        List[str]: List of package names.
    """

    packages = []
    start_dir = Path(start_dir)
    root_package = start_dir.name

    if '__init__.py' in [child.name for child in start_dir.iterdir()]:
        packages.append(root_package)
    for package in start_dir.rglob('*'):
        if package.is_dir() and '__init__.py' in [child.name for child in package.iterdir()]:
            package_name = f"{root_package}.{package.relative_to(start_dir).as_posix().replace('/', '.')}"
            if package_name not in exclude:
                packages.append(package_name)

    return packages


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
    packages=find_packages(start_dir='ultralytics'),  # required
    include_package_data=True,
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
    extras_require={
        'dev': [
            'ipython',
            'check-manifest',
            'pytest',
            'pytest-cov',
            'coverage',
            'mkdocs-material',
            'mkdocstrings[python]',
            'mkdocs-redirects',  # for 301 redirects
            'mkdocs-ultralytics-plugin>=0.0.27',  # for meta descriptions and images, dates and authors
        ],
        'export': [
            'coremltools>=7.0.b1',
            'openvino-dev>=2023.0',
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
