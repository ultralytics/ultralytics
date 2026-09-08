---
comments: true
description: Historical information about the retired Neptune experiment-tracking integration for Ultralytics YOLO.
keywords: Neptune, Ultralytics, experiment tracking, MLOps
---

# Experiment Tracking with Neptune

!!! warning "Retired integration"

    Neptune's hosted service shut down on March 6, 2026 at 12:00 UTC. Hosted data was permanently deleted and can no longer be exported.

Neptune was an experiment-tracking service previously supported by Ultralytics. Its hosted service and API are no longer available, and Ultralytics no longer includes the retired Neptune callback.

Choose an active [experiment-tracking integration](index.md) for new training runs.

## FAQ

### Why was the Neptune integration removed?

Neptune shut down its hosted service on March 6, 2026, so the callback no longer had a service to log to. Ultralytics removed it rather than ship a dead dependency.

### Which experiment trackers can I use instead?

Ultralytics ships callbacks for [Comet](comet.md), [ClearML](clearml.md), [MLflow](mlflow.md), [TensorBoard](tensorboard.md), [Weights & Biases](weights-biases.md), and [DVCLive](dvc.md). Enable one with `yolo settings`, for example `yolo settings comet=True`, and training runs are logged automatically.
