from ultralytics.yolo.utils import LOGGER

try:
    from ray import tune
    from ray.air import RunConfig, session  # noqa
    from ray.air.integrations.wandb import WandbLoggerCallback  # noqa
    from ray.tune.schedulers import ASHAScheduler  # noqa
    from ray.tune.schedulers import AsyncHyperBandScheduler as AHB  # noqa

except ImportError:
    LOGGER.info("Tuning hyperparameters requires ray/tune. Install using `pip install 'ray[tune]'`")
    tune = None

default_space = {
    # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'RMSProp']),
    'lr0': tune.uniform(1e-5, 1e-1),
    'lrf': tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
    'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
    'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
    'warmup_epochs': tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
    'warmup_momentum': tune.uniform(0.0, 0.95),  # warmup initial momentum
    'box': tune.uniform(0.02, 0.2),  # box loss gain
    'cls': tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
    'fl_gamma': tune.uniform(0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
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

task_metric_map = {
    'detect': 'metrics/mAP50-95(B)',
    'segment': 'metrics/mAP50-95(M)',
    'classify': 'top1_acc',
    'pose': None}
