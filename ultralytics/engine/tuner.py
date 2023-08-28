# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
This module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection,
instance segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Example:
    Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.tune(data='coco8.yaml', imgsz=640, epochs=30, iterations=300)
    ```
"""
import random
import time
from pathlib import Path

import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG, SETTINGS, callbacks


class Tuner:
    """
     Class responsible for hyperparameter tuning of YOLO models.

     The class evolves YOLO model hyperparameters over a given number of iterations
     by mutating them according to the search space and retraining the model to evaluate their performance.

     Attributes:
         space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
         save_dir (Path): Directory where evolution logs and results will be saved.
         evolve_csv (Path): Path to the CSV file where evolution logs are saved.

     Methods:
         _mutate(hyp: dict) -> dict:
             Mutates the given hyperparameters within the bounds specified in `self.space`.

         __call__():
             Executes the hyperparameter evolution across multiple iterations.

     Example:
         Tune hyperparameters for YOLOv8n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
         ```python
         from ultralytics import YOLO

         model = YOLO('yolov8n.pt')
         model.tune(data='coco8.yaml', imgsz=640, epochs=30, iterations=300)
         ```
     """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        self.args = args
        self.space = {
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            'lr0': (1e-5, 1e-1),
            'lrf': (0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (0.0, 0.001),  # optimizer weight decay 5e-4
            'warmup_epochs': (0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (0.0, 0.95),  # warmup initial momentum
            'box': (0.02, 0.2),  # box loss gain
            'cls': (0.2, 4.0),  # cls loss gain (scale with pixels)
            'hsv_h': (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (0.0, 45.0),  # image rotation (+/- deg)
            'translate': (0.0, 0.9),  # image translation (+/- fraction)
            'scale': (0.0, 0.9),  # image scale (+/- gain)
            'shear': (0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (0.0, 1.0),  # image mixup (probability)
            'mixup': (0.0, 1.0),  # image mixup (probability)
            'copy_paste': (0.0, 1.0)}  # segment copy-paste (probability)
        self.save_dir = Path(SETTINGS['runs_dir']) / self.args.task
        self.evolve_csv = self.save_dir / 'evolve.csv'
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def _mutate(self, hyp):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            hyp (dict): Dictionary containing current hyperparameters.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
            # Select parent(s)
            parent = 'single'  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt(self.evolve_csv, ndmin=2, delimiter=',', skiprows=1)
            fitness = x[:, -1]  # last column
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness)][:n]  # top n mutations
            w = fitness - fitness.min() + 1E-6  # weights (sum > 0)
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([self.space[k][0] for k in hyp.keys()])  # gains 0-1
            ng = len(self.space)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                hyp[k] = float(x[i + 7] * v[i])  # mutate

        # Constrain to limits
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

    def __call__(self, model=None, iterations=10):
        """
        Executes the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:
        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
           model (YOLO): A pre-initialized YOLO model to be used for training.
           iterations (int): The number of generations to run the evolution for.

        Note:
           The method utilizes the `self.evolve_csv` Path object to read and log hyperparameters and fitness scores.
           Ensure this path is set correctly in the Tuner instance.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for gen in range(iterations):
            # Load or initialize hyperparameters
            if self.evolve_csv.exists():
                with open(self.evolve_csv) as f:
                    hyp = yaml.safe_load(f)
            else:
                hyp = {}  # Initialize your default hyperparameters here

            # Mutate hyperparameters
            mutated_hyp = self._mutate(hyp)
            print(f'Running generation {gen + 1} with hyperparameters: {mutated_hyp}')

            # Initialize and train YOLOv8 model
            model = YOLO('yolov8n.pt')
            results = model.train(**{**vars(self.args), **mutated_hyp})

            # Save results and mutated_hyp to evolve_csv
            fitness_score = results.fitness  # Replace this with the metric you want to use for fitness
            log_row = [fitness_score] + [mutated_hyp[k] for k in self.space.keys()]
            with open(self.evolve_csv, 'a') as f:
                f.write(','.join(map(str, log_row)) + '\n')
