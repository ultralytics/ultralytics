from ultralytics.yolo.utils import LOGGER

try:
    from ray import tune
    from ray.air import RunConfig, session  # noqa
    from ray.air.integrations.wandb import WandbLoggerCallback  # noqa
    from ray.tune.schedulers import PopulationBasedTraining as PBT  # noqa

except ImportError:
    LOGGER.info("Tuning hyperparameters requires ray/tune. Install using `pip install 'ray[tune]'`")
    tune = None

default_space = {
    'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'RMSProp']),
    'lr0': tune.uniform(0.0001, 0.1),
    'weight_decay': tune.choice([0.005, 0.0005]),  # optimizer weight decay 5e-4
    'warmup_epochs': tune.uniform(2.0, 8.0),  # warmup epochs (fractions ok)
    'warmup_momentum': 0.8,  # warmup initial momentum
    'box': tune.uniform(5.5, 8.5),  # box loss gain
    'cls': tune.uniform(0.25, 0.65),  # cls loss gain (scale with pixels)
    'dfl': tune.uniform(1.0, 2.0),  # dfl loss gain
    'mixup': tune.uniform(0.0, 0.5),  # image mixup (probability)
    'copy_paste': tune.uniform(0.0, 0.5),  # segment copy-paste (probability)
}

task_metric_map = {
    'detect': 'metrics/mAP50-95(B)',
    'segment': 'metrics/mAP50-95(M)',
    'classify': 'top1_acc',
    'pose': None}
