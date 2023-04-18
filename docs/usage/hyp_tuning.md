Hyperparameter tuning (or hyperparameter optimization) is the process of determining the right combination of hyperparameters that maximizes the model performance. It works by running multiple trials in a single training process.

Ultralytics YOLO hyperparameter tuning is based on Ray Tune. It is also integrated with wandb for tracking evolution progress
!!! tip "Installation"
    `pip install "ray[tune]"`

    `pip install wandb` (optional)


!!! example "Usage"
    ```
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    retust = model.tune(data="coco128.yaml")
    ```

Here are the arguments that `tune()` accepts

| Parameter      | Type             | Description                                                   | Default Value |
| -------------- | ---------------- | ------------------------------------------------------------- | ------------- |
| data           | str              | The dataset to run the tuner on.                             |               |
| space          | dict, optional   | The hyperparameter search space.                             |           |
| grace_period   | int, optional    | The grace period in epochs of the ASHA scheduler.             | 10            |
| gpu_per_trial  | int, optional    | The number of GPUs to allocate per trial.                     | None          |
| max_samples    | int, optional    | The maximum number of trials to run.                          | 10            |
| train_args     | dict, optional   | Additional arguments to pass to the `train()` method.        | {}            |

The Default search `space` if not provided by the users is a dictionary with following data:

| Parameter        | Value Range     |
|------------------|-----------------|
| lr0              | `tune.uniform(1e-5, 1e-1)` |
| lrf              | `tune.uniform(0.01, 1.0)` |
| momentum         | `tune.uniform(0.6, 0.98)` |
| weight_decay     | `tune.uniform(0.0, 0.001)` |
| warmup_epochs    | `tune.uniform(0.0, 5.0)` |
| warmup_momentum  | `tune.uniform(0.0, 0.95)` |
| box              | `tune.uniform(0.02, 0.2)` |
| cls              | `tune.uniform(0.2, 4.0)` |
| fl_gamma         | `tune.uniform(0.0, 2.0)` |
| hsv_h            | `tune.uniform(0.0, 0.1)` |
| hsv_s            | `tune.uniform(0.0, 0.9)` |
| hsv_v            | `tune.uniform(0.0, 0.9)` |
| degrees          | `tune.uniform(0.0, 45.0)` |
| translate        | `tune.uniform(0.0, 0.9)` |
| scale            | `tune.uniform(0.0, 0.9)` |
| shear            | `tune.uniform(0.0, 10.0)` |
| perspective      | `tune.uniform(0.0, 0.001)` |
| flipud           | `tune.uniform(0.0, 1.0)` |
| fliplr           | `tune.uniform(0.0, 1.0)` |
| mosaic           | `tune.uniform(0.0, 1.0)` |
| mixup            | `tune.uniform(0.0, 1.0)` |
| copy_paste       | `tune.uniform(0.0, 1.0)` |


!!! example "Example using custom search space"
    ```
    from ultralytics import YOLO
    from ray import tune

    model = YOLO("yolov8n.pt")
    retust = model.tune(data="coco128.yaml", sapce={"lr0": tune.uniform(1e-5, 1e-1)}, train_args={"epochs": 50})
    ```
