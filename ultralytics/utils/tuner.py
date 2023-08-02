# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, NUM_THREADS


def run_ray_tune(model,
                 space: dict = None,
                 grace_period: int = 10,
                 gpu_per_trial: int = None,
                 max_samples: int = 10,
                 **train_args):
    """
    Runs hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. Defaults to None.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.
        max_samples (int, optional): The maximum number of trials to run. Defaults to 10.
        train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.

    Returns:
        (dict): A dictionary containing the results of the hyperparameter search.

    Raises:
        ModuleNotFoundError: If Ray Tune is not installed.
    """
    if train_args is None:
        train_args = {}

    try:
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.wandb import WandbLoggerCallback
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        raise ModuleNotFoundError('Tuning hyperparameters requires Ray Tune. Install with: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, '__version__')
    except (ImportError, AssertionError):
        wandb = False

    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        'lr0': tune.uniform(1e-5, 1e-1),
        'lrf': tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
        'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
        'box': tune.uniform(0.02, 0.2),  # box loss gain
        'cls': tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        'hsv_h': tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        'hsv_s': tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        'hsv_v': tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        'degrees': tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        'translate': tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        'scale': tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        'shear': tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        'perspective': tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        'flipud': tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        'fliplr': tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        'mosaic': tune.uniform(0.0, 1.0),  # image mixup (probability)
        'mixup': tune.uniform(0.0, 1.0),  # image mixup (probability)
        'copy_paste': tune.uniform(0.0, 1.0)}  # segment copy-paste (probability)

    def _tune(config):
        """
        Trains the YOLO model with the specified hyperparameters and additional arguments.

        Args:
            config (dict): A dictionary of hyperparameters to use for training.

        Returns:
            None.
        """
        model._reset_callbacks()
        config.update(train_args)
        model.train(**config)

    # Get search space
    if not space:
        space = default_space
        LOGGER.warning('WARNING ⚠️ search space not provided, using default search space.')

    # Get dataset
    data = train_args.get('data', TASK2DATA[model.task])
    space['data'] = data
    if 'data' not in train_args:
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {'cpu': NUM_THREADS, 'gpu': gpu_per_trial or 0})

    # Define the ASHA scheduler for hyperparameter search
    asha_scheduler = ASHAScheduler(time_attr='epoch',
                                   metric=TASK2METRIC[model.task],
                                   mode='max',
                                   max_t=train_args.get('epochs') or DEFAULT_CFG_DICT['epochs'] or 100,
                                   grace_period=grace_period,
                                   reduction_factor=3)

    # Define the callbacks for the hyperparameter search
    tuner_callbacks = [WandbLoggerCallback(project='YOLOv8-tune')] if wandb else []

    # Create the Ray Tune hyperparameter search tuner
    tuner = tune.Tuner(trainable_with_resources,
                       param_space=space,
                       tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
                       run_config=RunConfig(callbacks=tuner_callbacks, storage_path='./runs/tune'))

    # Run the hyperparameter search
    tuner.fit()

    # Return the results of the hyperparameter search
    return tuner.get_results()
