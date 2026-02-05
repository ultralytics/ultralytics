# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr


def run_ray_tune(
    model,
    space: dict | None = None,
    grace_period: int = 10,
    gpu_per_trial: int | None = None,
    max_samples: int = 10,
    **train_args,
):
    """Run hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. If not provided, uses default space.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial.
        max_samples (int, optional): The maximum number of trials to run.
        **train_args (Any): Additional arguments to pass to the `train()` method.

    Returns:
        (ray.tune.ResultGrid): A ResultGrid containing the results of the hyperparameter search.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")  # Load a YOLO26n model

        Start tuning hyperparameters for YOLO26n training on the COCO8 dataset
        >>> result_grid = model.tune(data="coco8.yaml", use_ray=True)
    """
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")
    try:
        checks.check_requirements("ray[tune]")

        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay
        "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
        "box": tune.uniform(0.02, 0.2),  # box loss gain
        "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "bgr": tune.uniform(0.0, 1.0),  # swap RGB↔BGR channels (probability)
        "mosaic": tune.uniform(0.0, 1.0),  # image mosaic (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
    }

    # Put the model in ray store
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        """Train the YOLO model with the specified hyperparameters and return results."""
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.reset_callbacks()
        config.update(train_args)

        # Set trial-specific name for W&B logging
        try:
            trial_id = tune.get_trial_id()  # Get current trial ID (e.g., "2c2fc_00000")
            trial_suffix = trial_id.split("_")[-1] if "_" in trial_id else trial_id
            config["name"] = f"{base_name}_{trial_suffix}"
        except Exception:
            # Not in Ray Tune context or error getting trial ID, use base name
            config["name"] = base_name

        results = model_to_train.train(**config)
        return results.results_dict

    # Get search space
    if not space and not train_args.get("resume"):
        space = default_space
        LOGGER.warning("Search space not provided, using default search space.")

    # Get dataset
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'Data not provided, using default "data={data}".')

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # Define the ASHA scheduler for hyperparameter search
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )

    # Define the callbacks for the hyperparameter search
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []

    # Create the Ray Tune hyperparameter search tuner
    tune_dir = get_save_dir(
        get_cfg(
            DEFAULT_CFG,
            {**train_args, **{"exist_ok": train_args.pop("resume", False)}},  # resume w/ same tune_dir
        ),
        name=train_args.pop("name", "tune"),  # runs/{task}/{tune_dir}
    )  # must be absolute dir
    tune_dir.mkdir(parents=True, exist_ok=True)
    if tune.Tuner.can_restore(tune_dir):
        LOGGER.info(f"{colorstr('Tuner: ')} Resuming tuning run {tune_dir}...")
        tuner = tune.Tuner.restore(str(tune_dir), trainable=trainable_with_resources, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=space,
            tune_config=tune.TuneConfig(
                scheduler=asha_scheduler,
                num_samples=max_samples,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir.parent, name=tune_dir.name),
        )

    # Run the hyperparameter search
    tuner.fit()

    # Get the results of the hyperparameter search
    results = tuner.get_results()

    # Shut down Ray to clean up workers
    ray.shutdown()

    return results
