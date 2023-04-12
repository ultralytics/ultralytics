from setuptools import setup

setup(
    name='RCA-YOLOv8',
    version='0.1',
    py_modules=['cli'],
    packages=['FOD_YOLOv8'],
    include_package_data=True,
    install_requires=[
        'Click',
        'ultralytics',
        'plotly',
        'optuna',
        'kaleido'
    ],
    entry_points='''
        [console_scripts]
        rcav8=cli:cli
    ''',
)