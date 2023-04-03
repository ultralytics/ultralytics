from ultralytics.yolo.utils import LOGGER


try:
    from ray import tune
except:
    LOGGER.info("Tuning hyperparameters requires ray/tune. Install using `pip install 'ray[tune]'`")
    tune = None

tune_space = {
"optimizer": ['SGD', 'Adam', 'AdamW', 'RMSProp'],
"lr0": tune.uniform(0.0001, 0.1),
"lrf": 0.01,  # final learning rate (lr0 * lrf)
"momentum": 0.937,  # SGD momentum/Adam beta1,
"weight_decay": [0.005, 0.0005],  # optimizer weight decay 5e-4
"warmup_epochs": 3.0,  # warmup epochs (fractions ok)
"warmup_momentum": 0.8,  # warmup initial momentum
"warmup_bias_lr": 0.1,  # warmup initial bias lr
"box": 7.5,  # box loss gain
"cls": 0.5 , # cls loss gain (scale with pixels)
"dfl": 1.5,  # dfl loss gain
"fl_gamma": 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
"label_smoothing": 0.0,  # label smoothing (fraction)
"nbs": 64,  # nominal batch size
"hsv_h":tune.uniform(0.01, 0.02),  # image HSV-Hue augmentation (fraction)
"hsv_s": tune.uniform(0.5, 0.8),  # image HSV-Saturation augmentation (fraction)
"hsv_v": 0.4 , # image HSV-Value augmentation (fraction)
"degrees": 0.0 , # image rotation (+/- deg)
"translate": 0.1,  # image translation (+/- fraction)
"scale": 0.5 , # image scale (+/- gain)
"shear": 0.0 , # image shear (+/- deg)
"perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
"flipud": 0.0,  # image flip up-down (probability)
"fliplr": 0.5 , # image flip left-right (probability)
"mosaic": 1.0,  # image mosaic (probability)
"mixup": 0.0 , # image mixup (probability)
"copy_paste": 0.0,  # segment copy-paste (probability)
}