---
description: Track data, artifacts, and training metrics with with the DagsHub integration. 
---

# Usage:

You can enable the integration by installing the dagshub package, using `pip install dagshub`.

If you would like to use experiment tracking with MLFlow, using dagshub remotes, also run `pip install mlflow`

## Feature Flags

You can set the following environment variables, based on your personal preference to enable or disable behaviors of the integration.

`DAGSHUB_LOG_DATASET=TRUE` logs the dataset with DDA. This is disabled by default.
`DAGSHUB_LOG_SAMPLE=TRUE` logs a sample of the dataset with DDA. This is enabled by default.
`LOG_ARTIFACTS_WITH_MLFLOW=TRUE` logs model artifacts with mlflow. This is disabled by default.
`LOG_ARTIFACTS_WITH_DVC=TRUE` logs model artifacts with DDA. This is enabled by default.

## Non-Interactive Running

If you would like to set the DagsHub integration to run in a noninteractive environment (i.e. no access to stdio), you can either pre-run `dagshub.init(*args)`, or set the following environment variables.

```
DAGSHUB_TOKEN=token
DAGSHUB_REPO_OWNER=username
DAGSHUB_REPO_NAME=repository
DAGSHUB_REPO_BRANCH=branch 
```

If the branch variable is not set, it default to 'main'.

# on_pretrain_routine_end
---
:::ultralytics.yolo.utils.callbacks.dagshub.on_pretrain_routine_end
<br><br>

# on_model_save
---
:::ultralytics.yolo.utils.callbacks.dagshub.on_model_save
<br><br>

# on_val_end
---
:::ultralytics.yolo.utils.callbacks.dagshub.on_val_end
<br><br>

# on_export_end
---
:::ultralytics.yolo.utils.callbacks.dagshub.on_export_end
<br><br>
