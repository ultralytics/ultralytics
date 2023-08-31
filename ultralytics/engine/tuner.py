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
    model.tune(data='coco8.yaml', imgsz=640, epochs=100, iterations=10)
    ```
"""
import random
import time
from copy import deepcopy

import numpy as np

from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, yaml_print, yaml_save


class Tuner:
    """
     Class responsible for hyperparameter tuning of YOLO models.

     The class evolves YOLO model hyperparameters over a given number of iterations
     by mutating them according to the search space and retraining the model to evaluate their performance.

     Attributes:
         space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
         tune_dir (Path): Directory where evolution logs and results will be saved.
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
         model.tune(data='coco8.yaml', imgsz=640, epochs=100, iterations=10, val=False, cache=True)
         ```
     """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        self.args = get_cfg(overrides=args)
        self.space = {  # key: (min, max, gain(optionaL))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            'lr0': (1e-5, 1e-1),
            'lrf': (0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.6, 0.98, 0.3),  # SGD momentum/Adam beta1
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
        self.tune_dir = get_save_dir(self.args, name='_tune')
        self.evolve_csv = self.tune_dir / 'evolve.csv'
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)
        LOGGER.info(f"Initialized Tuner instance with 'tune_dir={self.tune_dir}'.")

    def _mutate(self, parent='single', n=5, mutation=0.8, sigma=0.2):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
            # Select parent(s)
            x = np.loadtxt(self.evolve_csv, ndmin=2, delimiter=',', skiprows=1)
            fitness = x[:, 0]  # first column
            n = min(n, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness)][:n]  # top n mutations
            w = x[:, 0] - x[:, 0].min() + 1E-6  # weights (sum > 0)
            if parent == 'single' or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == 'weighted':
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            r = np.random  # method
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for k, v in self.space.items()])  # gains 0-1
            ng = len(self.space)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # Constrain to limits
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # lower limit
            hyp[k] = min(hyp[k], v[1])  # upper limit
            hyp[k] = round(hyp[k], 5)  # significant digits

        return hyp

    def __call__(self, model=None, iterations=10, prefix=colorstr('Tuner:')):
        """
        Executes the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:
        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
           model (Model): A pre-initialized YOLO model to be used for training.
           iterations (int): The number of generations to run the evolution for.

        Note:
           The method utilizes the `self.evolve_csv` Path object to read and log hyperparameters and fitness scores.
           Ensure this path is set correctly in the Tuner instance.
        """

        t0 = time.time()
        best_save_dir, best_metrics = None, None
        self.tune_dir.mkdir(parents=True, exist_ok=True)
        for i in range(iterations):
            # Mutate hyperparameters
            mutated_hyp = self._mutate()
            LOGGER.info(f'{prefix} Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}')

            try:
                # Train YOLO model with mutated hyperparameters
                train_args = {**vars(self.args), **mutated_hyp}
                results = (deepcopy(model) or YOLO(self.args.model)).train(**train_args)
                fitness = results.fitness
            except Exception as e:
                LOGGER.warning(f'WARNING ❌️ training failure for hyperparameter tuning iteration {i}\n{e}')
                fitness = 0.0

            # Save results and mutated_hyp to evolve_csv
            log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
            headers = '' if self.evolve_csv.exists() else (','.join(['fitness_score'] + list(self.space.keys())) + '\n')
            with open(self.evolve_csv, 'a') as f:
                f.write(headers + ','.join(map(str, log_row)) + '\n')

            # Print tuning results
            x = np.loadtxt(self.evolve_csv, ndmin=2, delimiter=',', skiprows=1)
            fitness = x[:, 0]  # first column
            best_idx = fitness.argmax()
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = results.save_dir
                best_metrics = {k: round(v, 5) for k, v in results.results_dict.items()}
            header = (f'{prefix} {i + 1} iterations complete ✅ ({time.time() - t0:.2f}s)\n'
                      f'{prefix} Results saved to {colorstr("bold", self.tune_dir)}\n'
                      f'{prefix} Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n'
                      f'{prefix} Best fitness metrics are {best_metrics}\n'
                      f'{prefix} Best fitness model is {best_save_dir}\n'
                      f'{prefix} Best fitness hyperparameters are printed below.\n')

            LOGGER.info('\n' + header)

            # Save turning results
            data = {k: float(x[0, i + 1]) for i, k in enumerate(self.space.keys())}
            header = header.replace(prefix, '#').replace('[1m/', '').replace('[0m', '') + '\n'
            yaml_save(self.tune_dir / 'best.yaml', data=data, header=header)
            yaml_print(self.tune_dir / 'best.yaml')
