# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
DagsHub callback
"""

import os
import re
from glob import glob
from pathlib import Path

from .mlflow import mlflow as is_mlflow
from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING

try:
    import dagshub
    from dagshub.upload import Repo

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(dagshub, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    dagshub = None

def splitter(repo):
    # util function to split stdio
    splitted = repo.split("/")
    if len(splitted) != 2: raise ValueError(f"Invalid input, should be owner_name/repo_name, but got {repo} instead")
    return splitted[1], splitted[0]

def on_pretrain_routine_end(trainer):
    # setup dagshub repository and mlflow tracking uri
    global repo, artifacts

    repo_name, repo_owner = os.getenv("DAGSHUB_REPO_NAME", None), os.getenv("DAGSHUB_REPO_OWNER", None)
    if not repo_name or not repo_owner: repo_name, repo_owner = splitter(input("Please insert your repository owner_name/repo_name:"))

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner)
    repo = Repo(owner=repo_owner,
                name=repo_name,
                branch=os.getenv("DAGSHUB_REPO_BRANCH", "main"))
    artifacts = repo.directory("artifacts")

    if is_mlflow:
        token = dagshub.auth.get_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

def on_model_save(trainer):
    # log artifacts to dagshub storage
    artifacts.add_dir(trainer.save_dir.as_posix(), glob_exclude="*.yaml", commit_message="added artifacts", force=True)
    for file in glob(os.path.join(trainer.save_dir.as_posix(), "*.yaml")):
        repo.upload(file, directory_path=".", commit_message=f"added {file.split('/')[-1]}", versioning="git", force=True)

def on_export_end(exporter):
    # log model exports
    artifacts.add_dir(exporter.file.parent.as_posix(), glob_exclude="*.yaml", commit_message="exported model", force=True)

callbacks = {
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_model_save": on_model_save,
    "on_export_end": on_export_end} if dagshub else {}
