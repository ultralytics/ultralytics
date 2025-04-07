# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection, instance
segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Examples:
    Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo11n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""

import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """
    A class for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations by mutating them according to the
    search space and retraining the model to evaluate their performance.

    Attributes:
        space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.
        args (dict): Configuration arguments for the tuning process.
        callbacks (list): Callback functions to be executed during tuning.
        prefix (str): Prefix string for logging messages.

    Methods:
        _mutate: Mutates the given hyperparameters within the specified bounds.
        __call__: Executes the hyperparameter evolution across multiple iterations.

    Examples:
        Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.tune(
        ...     data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False
        ... )

        Tune with custom search space.
        >>> model.tune(space={key1: val1, key2: val2})  # custom search space dictionary
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict): Configuration for hyperparameter evolution.
            _callbacks (list, optional): Callback functions to be executed during tuning.
        """
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            "shear": (0.0, 10.0),  # image shear (+/- deg)
            "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            "bgr": (0.0, 1.0),  # image channel bgr (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
        }
        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume  # resume w/ same tune_dir
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # reset to not affect training
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        Mutate hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        if self.tune_csv.exists():  # if CSV file exists: select best hyps and mutate
            # Select parent(s)
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # first column
            n = min(n, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness)][:n]  # top n mutations
            w = x[:, 0] - x[:, 0].min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # random selection
                x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

            # Mutate
            r = np.random  # method
            r.seed(int(time.time()))
            g = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # gains 0-1
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

    def __call__(self, model=None, iterations=10, cleanup=True):
        """
        Execute the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:

        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
            model (Model): A pre-initialized YOLO model to be used for training.
            iterations (int): The number of generations to run the evolution for.
            cleanup (bool): Whether to delete iteration weights to reduce storage space used during tuning.

        Note:
            The method utilizes the `self.tune_csv` Path object to read and log hyperparameters and fitness scores.
            Ensure this path is set correctly in the Tuner instance.
        """
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        start = 0
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            start = x.shape[0]
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
        for i in range(start, iterations):
            # Mutate hyperparameters
            mutated_hyp = self._mutate()
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            metrics = {}
            train_args = {**vars(self.args), **mutated_hyp}
            save_dir = get_save_dir(get_cfg(train_args))
            weights_dir = save_dir / "weights"
            try:
                # Train YOLO model with mutated hyperparameters (run in subprocess to avoid dataloader hang)
                launch = [__import__("sys").executable, "-m", "ultralytics.cfg.__init__"]  # workaround yolo not found
                cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
                return_code = subprocess.run(cmd, check=True).returncode
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
                metrics = torch.load(ckpt_file)["train_metrics"]
                assert return_code == 0, "training failed"

            except Exception as e:
                LOGGER.warning(f"WARNING ❌️ training failure for hyperparameter tuning iteration {i + 1}\n{e}")

            # Save results and mutated_hyp to CSV
            fitness = metrics.get("fitness", 0.0)
            log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
            headers = "" if self.tune_csv.exists() else (",".join(["fitness"] + list(self.space.keys())) + "\n")
            with open(self.tune_csv, "a", encoding="utf-8") as f:
                f.write(headers + ",".join(map(str, log_row)) + "\n")

            # Get best results
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # first column
            best_idx = fitness.argmax()
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = save_dir
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup:
                shutil.rmtree(weights_dir, ignore_errors=True)  # remove iteration weights/ dir to reduce storage space

            # Plot tune results
            plot_tune_results(self.tune_csv)

            # Save and print tune results
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}\n"
                f"{self.prefix}Best fitness hyperparameters are printed below.\n"
            )
            LOGGER.info("\n" + header)
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            yaml_save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            yaml_print(self.tune_dir / "best_hyperparameters.yaml")
