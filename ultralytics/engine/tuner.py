# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection, instance
segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Examples:
    Tune hyperparameters for YOLO26n on COCO8 at imgsz=640 and epochs=10 for 300 tuning iterations.
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo26n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import CFG_INT_KEYS, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """A class for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations by mutating them according to the
    search space and retraining the model to evaluate their performance. Supports both local NDJSON storage and
    distributed MongoDB Atlas coordination for multi-machine hyperparameter optimization.

    Attributes:
        space (dict[str, tuple]): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_file (Path): Path to the NDJSON file where evolution logs are saved.
        args (SimpleNamespace): Configuration arguments for the tuning process.
        callbacks (dict): Callback functions to be executed during tuning.
        prefix (str): Prefix string for logging messages.
        mongodb (MongoClient): Optional MongoDB client for distributed tuning.
        collection (Collection): MongoDB collection for storing tuning results.

    Methods:
        _mutate: Mutate hyperparameters based on bounds and scaling factors.
        __call__: Execute the hyperparameter evolution across multiple iterations.

    Examples:
        Tune hyperparameters for YOLO26n on COCO8 at imgsz=640 and epochs=10 for 300 tuning iterations.
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
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
        >>> model.tune(space={"lr0": (1e-5, 1e-2), "momentum": (0.7, 0.98)})
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks: dict | None = None):
        """Initialize the Tuner with configurations.

        Args:
            args (dict): Configuration for hyperparameter evolution.
            _callbacks (dict | None, optional): Callback functions to be executed during tuning.
        """
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.1, 4.0),  # cls loss gain (scale with pixels)
            "cls_pw": (0.0, 1.0),  # cls power weight
            "dfl": (0.4, 12.0),  # dfl loss gain
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
        self.tune_file = self.tune_dir / "tune_results.ndjson"
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

    def _connect(self, uri: str = "", max_retries: int = 3):
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
            mongodb_uri (str): MongoDB connection string.
            mongodb_db (str, optional): Database name.
            mongodb_collection (str, optional): Collection name.

        Notes:
            - Creates a fitness index for fast queries of top results
            - Falls back to local NDJSON mode if connection fails
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

    @staticmethod
    def _json_default(x):
        """Convert tensor-like values for JSON serialization."""
        return x.item() if hasattr(x, "item") else str(x)

    def _result_record(
        self,
        iteration: int,
        fitness: float,
        hyperparameters: dict[str, float],
        datasets: dict[str, dict],
        save_dirs: dict[str, str] | None = None,
    ) -> dict:
        """Build one local tuning result record."""
        result = {
            "iteration": iteration,
            "fitness": round(fitness, 5),
            "hyperparameters": hyperparameters,
            "datasets": datasets,
        }
        if save_dirs:
            result["save_dirs"] = save_dirs
        return result

    def _save_to_mongodb(
        self,
        fitness: float,
        hyperparameters: dict[str, float],
        metrics: dict,
        datasets: dict[str, dict],
        iteration: int,
    ):
        """Save results to MongoDB with proper type conversion.

        Args:
            fitness (float): Fitness score achieved with these hyperparameters.
            hyperparameters (dict[str, float]): Dictionary of hyperparameter values.
            metrics (dict): Complete training metrics dictionary (mAP, precision, recall, losses, etc.).
            datasets (dict[str, dict]): Per-dataset metrics for the iteration.
            iteration (int): Current iteration number.
        """
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
                    "metrics": metrics,
                    "datasets": datasets,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_file(self):
        """Sync MongoDB results to the local NDJSON tuning log.

        Downloads all results from MongoDB and writes them to the local NDJSON file in chronological order. This keeps
        resume, mutation, and plotting on the same local source of truth when using distributed tuning.
        """
        try:
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return

            with open(self.tune_file, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(
                        json.dumps(
                            self._result_record(
                                result["iteration"],
                                result["fitness"] or 0.0,
                                result.get("hyperparameters", {}),
                                result.get("datasets", {}),
                                result.get("save_dirs"),
                            ),
                            default=self._json_default,
                        )
                        + "\n"
                    )

        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to NDJSON sync failed: {e}")

    def _load_local_results(self) -> list[dict]:
        """Load local tuning results from the NDJSON log."""
        if not self.tune_file.exists():
            return []
        with open(self.tune_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _local_results_to_array(self, results: list[dict], n: int | None = None) -> np.ndarray | None:
        """Convert local NDJSON records to a fitness-plus-hyperparameters numpy array."""
        if not results:
            return None
        x = np.array(
            [
                [r.get("fitness", 0.0)]
                + [r.get("hyperparameters", {}).get(k, getattr(self.args, k)) for k in self.space]
                for r in results
            ],
            dtype=float,
        )
        if n is None:
            return x
        order = np.argsort(-x[:, 0])
        return x[order][:n]

    def _save_local_result(self, result: dict):
        """Append one tuning result to the local NDJSON log."""
        with open(self.tune_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=self._json_default) + "\n")

    @staticmethod
    def _best_metrics(result: dict) -> dict | None:
        """Summarize best-result metrics for logging."""
        datasets = result.get("datasets", {})
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        if len(datasets) > 1:
            return {k: round(v.get("fitness") or 0.0, 5) for k, v in datasets.items()}
        return None

    @staticmethod
    def _dataset_names(data: list) -> list[str]:
        """Create stable unique dataset names for logging and per-run directories."""
        stems = [Path(str(d)).stem for d in data]
        totals, seen = Counter(stems), Counter()
        names = []
        for stem in stems:
            seen[stem] += 1
            names.append(f"{stem}-{seen[stem]}" if totals[stem] > 1 else stem)
        return names

    @staticmethod
    def _crossover(x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
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
        # given a small value when span is zero to avoid no mutation
        span = np.where(span == 0, np.random.uniform(0.01, 0.1, span.shape), span)
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = 9,
        mutation: float = 0.5,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        """Mutate hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
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
                x = np.array(
                    [
                        [r["fitness"]] + [r["hyperparameters"].get(k, self.args.get(k)) for k in self.space.keys()]
                        for r in results
                    ]
                )
            elif self.collection.name in self.collection.database.list_collection_names():  # Tuner started elsewhere
                x = np.array([[0.0] + [getattr(self.args, k) for k in self.space.keys()]])

        # Fall back to local NDJSON if MongoDB unavailable or empty
        if x is None:
            x = self._local_results_to_array(self._load_local_results(), n=n)

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
        if "epochs" in hyp:
            hyp["epochs"] = round(hyp["epochs"])

        return hyp

    def __call__(self, iterations: int = 10, cleanup: bool = True):
        """Execute the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the specified number of iterations, performing the following steps:
        1. Sync MongoDB results to local NDJSON (if using distributed mode)
        2. Mutate hyperparameters using the best previous results or defaults
        3. Train a YOLO model with the mutated hyperparameters
        4. Log fitness scores and hyperparameters to MongoDB and/or NDJSON
        5. Track the best performing configuration across all iterations

        Args:
            iterations (int): The number of generations to run the evolution for.
            cleanup (bool): Whether to delete iteration weights to reduce storage space during tuning.
        """
        t0 = time.time()
        self.tune_dir.mkdir(parents=True, exist_ok=True)
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        best_save_dirs = {}

        # Sync MongoDB to local NDJSON at startup for proper resume logic
        if self.mongodb:
            self._sync_mongodb_to_file()

        start = 0
        if self.tune_file.exists():
            start = len(self._load_local_results())
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
        for i in range(start, iterations):
            # Linearly decay sigma from 0.2 → 0.1 over first 300 iterations
            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac

            # Mutate hyperparameters
            mutated_hyp = self._mutate(sigma=sigma_i)
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            train_args = {**vars(self.args), **mutated_hyp}
            data = train_args.pop("data")
            if not isinstance(data, (list, tuple)):
                data = [data]
            dataset_names = self._dataset_names(data)
            save_dir = (
                [get_save_dir(get_cfg(train_args))]
                if len(data) == 1
                else [get_save_dir(get_cfg(train_args), name=name) for name in dataset_names]
            )
            weights_dir = [s / "weights" for s in save_dir]
            metrics = {}
            all_fitness = []
            dataset_metrics = {}
            for j, (d, dataset) in enumerate(zip(data, dataset_names)):
                metrics_i = {}
                try:
                    train_args["data"] = d
                    train_args["save_dir"] = str(save_dir[j])  # pass save_dir to subprocess to ensure same path is used
                    # Train YOLO model with mutated hyperparameters (run in subprocess to avoid dataloader hang)
                    launch = [
                        __import__("sys").executable,
                        "-m",
                        "ultralytics.cfg.__init__",
                    ]  # workaround yolo not found
                    cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
                    return_code = subprocess.run(cmd, check=True).returncode
                    ckpt_file = weights_dir[j] / ("best.pt" if (weights_dir[j] / "best.pt").exists() else "last.pt")
                    metrics_i = torch_load(ckpt_file)["train_metrics"]
                    metrics = metrics_i
                    assert return_code == 0, "training failed"

                    # Cleanup
                    time.sleep(1)
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    LOGGER.error(f"training failure for hyperparameter tuning iteration {i + 1}\n{e}")

                # Save results - MongoDB takes precedence
                dataset_metrics[dataset] = metrics_i or {"fitness": 0.0}
                all_fitness.append(dataset_metrics[dataset].get("fitness") or 0.0)
            fitness = sum(all_fitness) / len(all_fitness)
            result = self._result_record(
                i + 1,
                fitness,
                mutated_hyp,
                dataset_metrics,
                {dataset: str(s) for dataset, s in zip(dataset_names, save_dir)},
            )
            stop_after_iteration = False
            if self.mongodb:
                self._save_to_mongodb(fitness, mutated_hyp, metrics, dataset_metrics, i + 1)
                self._sync_mongodb_to_file()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    stop_after_iteration = True
            else:
                self._save_local_result(result)

            # Get best results
            results = self._load_local_results()
            x = self._local_results_to_array(results)
            fitness = x[:, 0]  # first column
            best_idx = fitness.argmax()
            best_result = results[best_idx]
            current_best_save_dirs = best_result.get("save_dirs", {})
            best_is_current = best_idx == i
            if best_is_current:
                if cleanup:
                    for s in best_save_dirs.values():
                        if s not in current_best_save_dirs.values():
                            shutil.rmtree(s, ignore_errors=True)
                for dataset, weight_dir in zip(dataset_names, weights_dir):
                    best_weights_dir = (
                        self.tune_dir / "weights" if len(data) == 1 else self.tune_dir / "weights" / dataset
                    )
                    best_weights_dir.mkdir(parents=True, exist_ok=True)
                    for ckpt in weight_dir.glob("*.pt"):
                        shutil.copy2(ckpt, best_weights_dir)
                best_save_dirs = current_best_save_dirs
            elif cleanup:
                for s in save_dir:
                    shutil.rmtree(s, ignore_errors=True)  # remove iteration dirs to reduce storage space
                best_save_dirs = current_best_save_dirs

            # Plot tune results
            plot_tune_results(str(self.tune_file))

            # Save and print tune results
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {self._best_metrics(best_result)}\n"
                f"{self.prefix}Best fitness model is "
                f"{self.tune_dir / 'weights' if len(best_result.get('datasets', {})) == 1 else 'not saved for multi-dataset tuning'}"
            )
            LOGGER.info("\n" + header)
            data = {k: int(v) if k in CFG_INT_KEYS else float(v) for k, v in zip(self.space.keys(), x[best_idx, 1:])}
            YAML.save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )
            YAML.print(self.tune_dir / "best_hyperparameters.yaml")
            if stop_after_iteration:
                LOGGER.info(
                    f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                )
                break
