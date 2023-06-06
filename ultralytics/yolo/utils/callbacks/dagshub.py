# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
DagsHub callback
"""

import os
from glob import glob

from ultralytics.yolo.utils import TESTS_RUNNING, LOGGER

from .mlflow import mlflow as is_mlflow

try:
    import dagshub
    from dagshub.upload import Repo
    from dagshub.upload.wrapper import DataSet

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(dagshub, '__version__')  # verify package is not directory

    if not is_mlflow:
        LOGGER.warning('MLFlow is not initialized but DagsHub is initialized, not logging with MLFlow. \
                To log experiments with Dagshub\'s MLFlow Remote, run `pip install mlflow`')

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
    if 'dagshub' in os.getenv('MLFLOW_TRACKING_URI' , ''):
        repo_name, repo_owner = *os.getenv('MLFLOW_TRACKING_URI' , '')[:-7].split('/')[-2:]
    if not repo_name or not repo_owner:
        repo_name, repo_owner = splitter(input('Please insert your repository owner_name/repo_name:'))

    # setup dagshub authentication
    dagshub_auth = os.getenv("DAGSHUB_TOKEN")
    if dagshub_auth:
        dagshub.auth.add_app_token(dagshub_auth)

    # setup dagshub repository, mlflow environment variables and artifacts directory
    dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
    repo = Repo(owner=repo_owner, name=repo_name, branch=os.getenv('DAGSHUB_REPO_BRANCH', 'main'))

    # log artifacts with mlflow
    os.environ['MLFLOW_LOG_ARTIFACT'] = os.getenv('DAGSHUB_LOG_ARTIFACTS_WITH_MLFLOW', 'false').lower()

    # log dataset to repository
    if os.getenv('DAGSHUB_LOG_DATASET', 'false').lower() == 'true':
        dataset = trainer.data['path'].as_posix().split('/')[-1]
        repo.directory(f'data/{dataset}').add_dir(trainer.data['path'].as_posix(),
                                                  commit_message=f'added {dataset}',
                                                  force=True)


def on_val_end(validator):
    if os.getenv('DAGSHUB_LOG_SAMPLE', 'true').lower() == 'true':
        # upload a sample (1 batch) of the dataset
        repo.upload_files([DataSet.get_file(file, f"sample/{files.split('/')[-1]}") for file in next(validator.dataloader.iterator)['im_file']]
                          commit_message='added sample batch',
                          versioning='dvc',
                          force=True)


def on_model_save(trainer):
    # log artifacts to dagshub storage
    if os.getenv('DAGSHUB_LOG_ARTIFACTS_WITH_DVC', 'true').lower() == 'true':
        repo.directory('artifacts').add_dir(trainer.save_dir.as_posix(),
                                            glob_exclude='*.yaml',
                                            commit_message='added artifacts',
                                            force=True)
        for file in glob(os.path.join(trainer.save_dir.as_posix(), '*.yaml')):
            repo.upload(file,
                        directory_path='.',
                        commit_message=f"added {file.split('/')[-1]}",
                        versioning='git',
                        force=True)


def on_export_end(exporter):
    # log model exports
    repo.directory('exports').add_dir(exporter.file.parent.as_posix(),
                      glob_exclude='*.yaml',
                      commit_message='exported model',
                      force=True)


callbacks = {
    'on_model_save': on_model_save,
    'on_val_end': on_val_end,
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_export_end': on_export_end} if dagshub else {}
