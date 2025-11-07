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

from __future__ import annotations

import gc
import random
import shutil
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """A class for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations by mutating them according to the
    search space and retraining the model to evaluate their performance. Supports both local CSV storage and distributed
    MongoDB Atlas coordination for multi-machine hyperparameter optimization.

    Attributes:
        space (dict[str, tuple]): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.
        args (dict): Configuration arguments for the tuning process.
        callbacks (list): Callback functions to be executed during tuning.
        prefix (str): Prefix string for logging messages.
        mongodb (MongoClient): Optional MongoDB client for distributed tuning.
        collection (Collection): MongoDB collection for storing tuning results.

    Methods:
        _mutate: Mutate hyperparameters based on bounds and scaling factors.
        __call__: Execute the hyperparameter evolution across multiple iterations.

    Examples:
        Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     plots=False,
        >>>     save=False,
        >>>     val=False
        >>> )

        Tune with distributed MongoDB Atlas coordination across multiple machines:
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     mongodb_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
        >>>     mongodb_db="ultralytics",
        >>>     mongodb_collection="tune_results"
        >>> )

        Tune with custom search space:
        >>> model.tune(space={"lr0": (1e-5, 1e-1), "momentum": (0.6, 0.98)})
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks: list | None = None):
        """Initialize the Tuner with configurations.

        Args:
            args (dict): Configuration for hyperparameter evolution.
            _callbacks (list | None, optional): Callback functions to be executed during tuning.
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
            "cls": (0.1, 4.0),  # cls loss gain (scale with pixels)
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
            "mosaic": (0.0, 1.0),  # image mosaic (probability)
            "mixup": (0.0, 1.0),  # image mixup (probability)
            "cutmix": (0.0, 1.0),  # image cutmix (probability)
            "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            "close_mosaic": (0.0, 10.0),  # close dataloader mosaic (epochs)
        }
        mongodb_uri = args.pop("mongodb_uri", None)
        mongodb_db = args.pop("mongodb_db", "ultralytics")
        mongodb_collection = args.pop("mongodb_collection", "tuner_results")

        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume  # resume w/ same tune_dir
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # reset to not affect training
        self.tune_csv = self.tune_dir / "tune_results.csv"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)

        # MongoDB Atlas support (optional)
        self.mongodb = None
        if mongodb_uri:
            self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _connect(self, uri: str = "mongodb+srv://username:password@cluster.mongodb.net/", max_retries: int = 3):
        """Create MongoDB client with exponential backoff retry on connection failures.

        Args:
            uri (str): MongoDB connection string with credentials and cluster information.
            max_retries (int): Maximum number of connection attempts before giving up.

        Returns:
            (MongoClient): Connected MongoDB client instance.
        """
        check_requirements("pymongo")

        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=40000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=30,
                    minPoolSize=3,
                    maxIdleTimeMS=60000,
                )
                client.admin.command("ping")  # Test connection
                LOGGER.info(f"{self.prefix}Connected to MongoDB Atlas (attempt {attempt + 1})")
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError):
                if attempt == max_retries - 1:
                    raise
                wait_time = 2**attempt
                LOGGER.warning(
                    f"{self.prefix}MongoDB connection failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def _init_mongodb(self, mongodb_uri="", mongodb_db="", mongodb_collection=""):
        """Initialize MongoDB connection for distributed tuning.

        Connects to MongoDB Atlas for distributed hyperparameter optimization across multiple machines. Each worker
        saves results to a shared collection and reads the latest best hyperparameters from all workers for evolution.

        Args:
            mongodb_uri (str): MongoDB connection string, e.g. 'mongodb+srv://username:password@cluster.mongodb.net/'.
            mongodb_db (str, optional): Database name.
            mongodb_collection (str, optional): Collection name.

        Notes:
            - Creates a fitness index for fast queries of top results
            - Falls back to CSV-only mode if connection fails
            - Uses connection pooling and retry logic for production reliability
        """
        self.mongodb = self._connect(mongodb_uri)
        self.collection = self.mongodb[mongodb_db][mongodb_collection]
        self.collection.create_index([("fitness", -1)], background=True)
        LOGGER.info(f"{self.prefix}Using MongoDB Atlas for distributed tuning")

    def _get_mongodb_results(self, n: int = 5) -> list:
        """Get top N results from MongoDB sorted by fitness.

        Args:
            n (int): Number of top results to retrieve.

        Returns:
            (list[dict]): List of result documents with fitness scores and hyperparameters.
        """
        try:
            return list(self.collection.find().sort("fitness", -1).limit(n))
        except Exception:
            return []

    def _save_to_mongodb(self, fitness: float, hyperparameters: dict[str, float], metrics: dict, iteration: int):
        """Save results to MongoDB with proper type conversion.

        Args:
            fitness (float): Fitness score achieved with these hyperparameters.
            hyperparameters (dict[str, float]): Dictionary of hyperparameter values.
            metrics (dict): Complete training metrics dictionary (mAP, precision, recall, losses, etc.).
            iteration (int): Current iteration number.
        """
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
                    "metrics": metrics,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_csv(self):
        """Sync MongoDB results to CSV for plotting compatibility.

        Downloads all results from MongoDB and writes them to the local CSV file in chronological order. This enables
        the existing plotting functions to work seamlessly with distributed MongoDB data.
        """
        try:
            # Get all results from MongoDB
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return

            # Write to CSV
            headers = ",".join(["fitness", *list(self.space.keys())]) + "\n"
            with open(self.tune_csv, "w", encoding="utf-8") as f:
                f.write(headers)
                for result in all_results:
                    fitness = result["fitness"]
                    hyp_values = [result["hyperparameters"][k] for k in self.space.keys()]
                    log_row = [round(fitness, 5), *hyp_values]
                    f.write(",".join(map(str, log_row)) + "\n")

        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to CSV sync failed: {e}")

    def _crossover(self, x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
        """BLX-α crossover from up to top-k parents (x[:,0]=fitness, rest=genes)."""
        k = min(k, len(x))
        # fitness weights (shifted to >0); fallback to uniform if degenerate
        weights = x[:, 0] - x[:, 0].min() + 1e-6
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        idxs = random.choices(range(len(x)), weights=weights, k=k)
        parents_mat = np.stack([x[i][1:] for i in idxs], 0)  # (k, ng) strip fitness
        lo, hi = parents_mat.min(0), parents_mat.max(0)
        span = hi - lo
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = 9,
        mutation: float = 0.5,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        """Mutate hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method (kept for API compatibility, unused in BLX mode).
            n (int): Number of top parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict[str, float]): A dictionary containing mutated hyperparameters.
        """
        x = None

        # Try MongoDB first if available
        if self.mongodb:
            if results := self._get_mongodb_results(n):
                # MongoDB already sorted by fitness DESC, so results[0] is best
                x = np.array([[r["fitness"]] + [r["hyperparameters"][k] for k in self.space.keys()] for r in results])
            elif self.collection.name in self.collection.database.list_collection_names():  # Tuner started elsewhere
                x = np.array([[0.0] + [getattr(self.args, k) for k in self.space.keys()]])

        # Fall back to CSV if MongoDB unavailable or empty
        if x is None and self.tune_csv.exists():
            csv_data = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            if len(csv_data) > 0:
                fitness = csv_data[:, 0]  # first column
                order = np.argsort(-fitness)
                x = csv_data[order][:n]  # top-n sorted by fitness DESC

        # Mutate if we have data, otherwise use defaults
        if x is not None:
            np.random.seed(int(time.time()))
            ng = len(self.space)

            # Crossover
            genes = self._crossover(x)

            # Mutation
            gains = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # gains 0-1
            factors = np.ones(ng)
            while np.all(factors == 1):  # mutate until a change occurs (prevent duplicates)
                mask = np.random.random(ng) < mutation
                step = np.random.randn(ng) * (sigma * gains)
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)
            hyp = {k: float(genes[i] * factors[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # Constrain to limits
        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)

        # Update types
        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])

        return hyp

    def __call__(self, model=None, iterations: int = 10, cleanup: bool = True):
        """Execute the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the specified number of iterations, performing the following steps:
        1. Sync MongoDB results to CSV (if using distributed mode)
        2. Mutate hyperparameters using the best previous results or defaults
        3. Train a YOLO model with the mutated hyperparameters
        4. Log fitness scores and hyperparameters to MongoDB and/or CSV
        5. Track the best performing configuration across all iterations

        Args:
            model (Model | None, optional): A pre-initialized YOLO model to be used for training.
            iterations (int): The number of generations to run the evolution for.
            cleanup (bool): Whether to delete iteration weights to reduce storage space during tuning.
        """
        t0 = time.time()
        best_save_dir, best_metrics = None, None
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)

        # Sync MongoDB to CSV at startup for proper resume logic
        if self.mongodb:
            self._sync_mongodb_to_csv()

        start = 0
        if self.tune_csv.exists():
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            start = x.shape[0]
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
        for i in range(start, iterations):
            # Linearly decay sigma from 0.2 → 0.1 over first 300 iterations
            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac

            # Mutate hyperparameters
            mutated_hyp = self._mutate(sigma=sigma_i)
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
                metrics = torch_load(ckpt_file)["train_metrics"]
                assert return_code == 0, "training failed"

                # Cleanup
                time.sleep(1)
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                LOGGER.error(f"training failure for hyperparameter tuning iteration {i + 1}\n{e}")

            # Save results - MongoDB takes precedence
            fitness = metrics.get("fitness", 0.0)
            if self.mongodb:
                self._save_to_mongodb(fitness, mutated_hyp, metrics, i + 1)
                self._sync_mongodb_to_csv()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    LOGGER.info(
                        f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                    )
                    break
            else:
                # Save to CSV only if no MongoDB
                log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]
                headers = "" if self.tune_csv.exists() else (",".join(["fitness", *list(self.space.keys())]) + "\n")
                with open(self.tune_csv, "a", encoding="utf-8") as f:
                    f.write(headers + ",".join(map(str, log_row)) + "\n")

            # Get best results
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)
            fitness = x[:, 0]  # first column
            best_idx = fitness.argmax()
            best_is_current = best_idx == i
            if best_is_current:
                best_save_dir = str(save_dir)
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}
                for ckpt in weights_dir.glob("*.pt"):
                    shutil.copy2(ckpt, self.tune_dir / "weights")
            elif cleanup and best_save_dir:
                shutil.rmtree(best_save_dir, ignore_errors=True)  # remove iteration dirs to reduce storage space

            # Plot tune results
            plot_tune_results(str(self.tune_csv))

            # Save and print tune results
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}"
            )
            LOGGER.info("\n" + header)
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}
            YAML.save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            YAML.print(self.tune_dir / "best_hyperparameters.yaml")
