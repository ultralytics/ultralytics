# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
DagsHub callback
"""

import os
from glob import glob

from ultralytics.yolo.utils import TESTS_RUNNING

from .mlflow import mlflow as is_mlflow

try:
    import dagshub
    from dagshub.upload import Repo

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(dagshub, '__version__')  # verify package is not directory

except (ImportError, AssertionError):
    dagshub = None


def splitter(repo):
    # util function to split stdio
    splitted = repo.split('/')
    if len(splitted) != 2:
        raise ValueError(f'Invalid input, should be owner_name/repo_name, but got {repo} instead')
    return splitted[1], splitted[0]


def on_pretrain_routine_end(trainer):
    global repo

    # get dagshub remote information
    repo_name, repo_owner = os.getenv('DAGSHUB_REPO_NAME', None), os.getenv('DAGSHUB_REPO_OWNER', None)
    if not repo_name or not repo_owner:
        repo_name, repo_owner = splitter(input('Please insert your repository owner_name/repo_name:'))

    # setup dagshub repository and artifacts directory
    dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
    repo = Repo(owner=repo_owner, name=repo_name, branch=os.getenv('DAGSHUB_REPO_BRANCH', 'main'))

    # setup mlflow tracking uri
    if is_mlflow:
        token = dagshub.auth.get_token()
        os.environ['MLFLOW_TRACKING_USERNAME'] = token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        os.environ['MLFLOW_TRACKING_URI'] = f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow'

    # log dataset to repository
    if os.getenv('DAGSHUB_LOG_DATASET', '').lower() == 'true':
        dataset = trainer.data['path'].as_posix().split('/')[-1]
        repo.directory(f'data/{dataset}').add_dir(trainer.data['path'].as_posix(), commit_message=f'added {dataset}', force=True)


def on_model_save(trainer):
    # log artifacts to dagshub storage
    repo.directory('artifacts').add_dir(trainer.save_dir.as_posix(), glob_exclude='*.yaml', commit_message='added artifacts', force=True)
    for file in glob(os.path.join(trainer.save_dir.as_posix(), '*.yaml')):
        repo.upload(file,
                    directory_path='.',
                    commit_message=f"added {file.split('/')[-1]}",
                    versioning='git',
                    force=True)


def on_export_end(exporter):
    # log model exports
    artifacts.add_dir(exporter.file.parent.as_posix(),
                      glob_exclude='*.yaml',
                      commit_message='exported model',
                      force=True)


callbacks = {
    'on_model_save': on_model_save,
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_export_end': on_export_end} if dagshub else {}
