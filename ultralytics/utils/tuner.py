# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr

RAY_SEARCH_ALG_REQUIREMENTS = {
    "random": None,
    "ax": "ax-platform",
    "bayesopt": "bayesian-optimization==1.4.3",
    "bohb": ["hpbandster", "ConfigSpace"],
    "hebo": "HEBO>=0.2.0",
    "hyperopt": "hyperopt",
    "nevergrad": "nevergrad",
    "optuna": "optuna",
    "zoopt": "zoopt",
}


def _sanitize_tune_value(value: dict):
    """Convert NumPy-backed Tune values into native Python types for YAML serialization.

    Args:
        value (dict): The value to convert. Can be a dict, list, tuple, NumPy scalar, or NumPy array.

    Returns:
        The converted value with NumPy types replaced by native Python types.
    """
    if isinstance(value, dict):
        return {k: _sanitize_tune_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_tune_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_tune_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _get_ray_search_alg_kind(search_alg):
    """Return the normalized Ray Tune search algorithm kind for known searcher objects.

    Args:
        search_alg (str | ray.tune.search.Searcher): The search algorithm to identify. Can be None, a string, or a Ray
            Tune searcher object.

    Returns:
        str | None: The normalized search algorithm name, or None if not recognized.
    """
    if search_alg is None:
        return None
    if isinstance(search_alg, str):
        normalized = search_alg.strip().lower()
        return normalized or None

    cls = search_alg.__class__
    module, name = cls.__module__, cls.__name__
    if name == "AxSearch" and module.startswith("ray.tune.search.ax"):
        return "ax"
    if name == "TuneBOHB" and module.startswith("ray.tune.search.bohb"):
        return "bohb"
    if name == "ZOOptSearch" and module.startswith("ray.tune.search.zoopt"):
        return "zoopt"
    return None


def _validate_ax_search_space(space):
    """Validate that a Tune search space can be consumed by Ax.

    Args:
        space (dict): The hyperparameter search space to validate.

    Returns:
        list: The converted Ax parameters.

    Raises:
        ImportError: If the required 'ax-platform' package is not installed.
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["ax"])

    from ray.tune.search.ax.ax_search import AxSearch

    return AxSearch.convert_search_space(space)


def _create_ax_search(space, task):
    """Create an Ax searcher with an initialized experiment.

    Args:
        space (dict): The hyperparameter search space.
        task (str): The task type (e.g., 'detect', 'segment', 'classify').

    Returns:
        AxSearch (ray.tune.search.Searcher): The configured Ax search algorithm.

    Raises:
        ImportError: If required Ax packages are not installed.
    """
    parameters = _validate_ax_search_space(space)

    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    from ray.tune.search.ax.ax_search import AxSearch

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters,
        objectives={TASK2METRIC[task]: ObjectiveProperties(minimize=False)},
    )
    return AxSearch(ax_client=ax_client)


def _convert_bohb_search_space(space):
    """Convert a Tune search space into BOHB-compatible ConfigSpace and fixed-only Tune param_space.

    Args:
        space (dict): The hyperparameter search space.

    Returns:
        (tuple): A tuple containing the ConfigSpace object and a dict of fixed parameters.

    Raises:
        ValueError: If the search space contains grid search parameters or unsupported samplers.
        ImportError: If required BOHB packages are not installed.
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["bohb"])

    import ConfigSpace
    from ray.tune.search.sample import Categorical, Float, Integer, LogUniform, Quantized, Uniform
    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(resolved_space)
    if grid_vars:
        raise ValueError("Grid search parameters cannot be automatically converted to a TuneBOHB search space.")

    cs = ConfigSpace.ConfigurationSpace()
    for path, domain in domain_vars:
        par = "/".join(str(p) for p in path)
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            raise ValueError("TuneBOHB does not support quantized search spaces with the current ConfigSpace version.")

        if isinstance(domain, Float) and isinstance(sampler, (Uniform, LogUniform)):
            cs.add(
                ConfigSpace.UniformFloatHyperparameter(
                    par, lower=domain.lower, upper=domain.upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Integer) and isinstance(sampler, (Uniform, LogUniform)):
            upper = domain.upper - 1  # Tune integer search spaces are exclusive on the upper bound
            cs.add(
                ConfigSpace.UniformIntegerHyperparameter(
                    par, lower=domain.lower, upper=upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Categorical) and isinstance(sampler, Uniform):
            cs.add(ConfigSpace.CategoricalHyperparameter(par, choices=domain.categories))
        else:
            raise ValueError(
                f"TuneBOHB does not support parameters of type {type(domain).__name__} "
                f"with sampler type {type(domain.sampler).__name__}."
            )

    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    return cs, fixed_param_space


def _create_bohb_search(space, task):
    """Create a BOHB searcher using a ConfigSpace definition compatible with current ConfigSpace versions.

    Args:
        space (dict): The hyperparameter search space.
        task (str): The task type (e.g., 'detect', 'segment', 'classify').

    Returns:
        (tuple): A tuple containing the TuneBOHB searcher and fixed parameter space dict.

    Raises:
        ImportError: If required BOHB packages are not installed.
    """
    cs, fixed_param_space = _convert_bohb_search_space(space)

    from ray.tune.search.bohb.bohb_search import TuneBOHB

    return TuneBOHB(space=cs, metric=TASK2METRIC[task], mode="max"), fixed_param_space


def _create_nevergrad_search(task):
    """Create a Nevergrad searcher with a default optimizer.

    Args:
        task (str): The task type (e.g., 'detect', 'segment', 'classify').

    Returns:
        (NevergradSearch): The configured Nevergrad search algorithm.

    Raises:
        ImportError: If the 'nevergrad' package is not installed.
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["nevergrad"])

    import nevergrad as ng
    from ray.tune.search.nevergrad import NevergradSearch

    return NevergradSearch(optimizer=ng.optimizers.OnePlusOne, metric=TASK2METRIC[task], mode="max")


def _convert_zoopt_search_space(space):
    """Convert a Tune search space into ZOOpt-compatible dimensions and fixed-only Tune param_space.

    Args:
        space (dict): The hyperparameter search space.

    Returns:
        (tuple): A tuple containing the ZOOpt dimension dict and fixed parameter space dict.

    Raises:
        ImportError: If the 'zoopt' package is not installed.
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["zoopt"])

    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.search.zoopt import ZOOptSearch
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, _, _ = parse_spec_vars(resolved_space)
    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    dim_dict = ZOOptSearch.convert_search_space(space)
    return dim_dict, fixed_param_space


def _create_zoopt_search(space, task, iterations):
    """Create a ZOOpt searcher with required budget and converted search space.

    Args:
        space (dict): The hyperparameter search space.
        task (str): The task type (e.g., 'detect', 'segment', 'classify').
        iterations (int): The maximum number of trials (budget) for ZOOpt.

    Returns:
        (tuple): A tuple containing the ZOOptSearch searcher and fixed parameter space dict.

    Raises:
        ImportError: If the 'zoopt' package is not installed.
    """
    dim_dict, fixed_param_space = _convert_zoopt_search_space(space)

    from ray.tune.search.zoopt import ZOOptSearch

    return ZOOptSearch(
        algo="asracos", budget=iterations, dim_dict=dim_dict, metric=TASK2METRIC[task], mode="max"
    ), fixed_param_space


def _resolve_ray_search_alg(search_alg, task, space, iterations):
    """Resolve search algorithms and normalize Tune param_space for known Ray Tune searchers.

    Args:
        search_alg (str | object | None): The search algorithm to use. Can be a string name, a pre-instantiated Ray Tune
            searcher object, or None for default behavior.
        task (str): The task type (e.g., 'detect', 'segment', 'classify').
        space (dict): The hyperparameter search space.
        iterations (int): The maximum number of trials to run.

    Returns:
        (tuple): A tuple containing (resolved_search_alg, tuner_param_space, resolved_search_alg_kind).
            - resolved_search_alg: The configured searcher or None.
            - tuner_param_space: The normalized parameter space for the tuner.
            - resolved_search_alg_kind: The normalized algorithm name or None.

    Raises:
        ValueError: If an unsupported search_alg string is provided.
        ModuleNotFoundError: If required dependencies for the chosen algorithm are not installed.
    """
    if search_alg is None:
        return None, space, None

    normalized = _get_ray_search_alg_kind(search_alg)
    if isinstance(search_alg, str):
        if not normalized:
            return None, space, None
        if normalized not in RAY_SEARCH_ALG_REQUIREMENTS:
            supported = ", ".join(sorted(RAY_SEARCH_ALG_REQUIREMENTS))
            raise ValueError(f"Unsupported Ray Tune search_alg '{search_alg}'. Supported values: {supported}.")
        if normalized == "random":
            return None, space, normalized

    try:
        if normalized == "ax":
            if isinstance(search_alg, str):
                return _create_ax_search(space, task), {}, normalized
            _validate_ax_search_space(space)
            return search_alg, {}, normalized
        if normalized == "bohb":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_bohb_search(space, task)
            else:
                _, tuner_param_space = _convert_bohb_search_space(space)
                resolved_search_alg = search_alg
            return resolved_search_alg, tuner_param_space, normalized
        if normalized == "nevergrad":
            return _create_nevergrad_search(task), space, normalized
        if normalized == "zoopt":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_zoopt_search(space, task, iterations)
            else:
                _, tuner_param_space = _convert_zoopt_search_space(space)
                resolved_search_alg = search_alg
            return resolved_search_alg, tuner_param_space, normalized
        if not isinstance(search_alg, str):
            return search_alg, space, None

        requirements = RAY_SEARCH_ALG_REQUIREMENTS[normalized]
        if requirements:
            checks.check_requirements(requirements)

        from ray.tune.search import create_searcher

        return create_searcher(normalized, metric=TASK2METRIC[task], mode="max"), space, normalized
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(
            f"Ray Tune search_alg '{search_alg}' requires additional dependencies. Original error: {e}"
        ) from e


def run_ray_tune(
    model,
    space: dict | None = None,
    grace_period: int = 10,
    gpu_per_trial: int | None = None,
    iterations: int = 10,
    search_alg=None,
    **train_args,
):
    """Run hyperparameter tuning using Ray Tune.

    Args:
        model (YOLO): Model to run the tuner on.
        space (dict, optional): The hyperparameter search space. If not provided, uses default space.
        grace_period (int, optional): The grace period in epochs of the ASHA scheduler.
        gpu_per_trial (int, optional): The number of GPUs to allocate per trial.
        iterations (int, optional): The maximum number of trials to run.
        search_alg (str | ray.tune.search.Searcher | ray.tune.search.SearchAlgorithm, optional): Search algorithm to
            use. Strings are resolved to supported Ray Tune searchers. Pre-instantiated objects are reused, and known
            searchers with special Tune param_space requirements are normalized automatically.
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
        from ray.tune import RunConfig
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
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
        "lr0": tune.uniform(1e-5, 1e-2),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": tune.uniform(0.7, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay
        "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
        "box": tune.uniform(1.0, 20.0),  # box loss gain
        "cls": tune.uniform(0.1, 4.0),  # cls loss gain (scale with pixels)
        "dfl": tune.uniform(0.4, 12.0),  # dfl loss gain
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.95),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "bgr": tune.uniform(0.0, 1.0),  # swap RGB↔BGR channels (probability)
        "mosaic": tune.uniform(0.0, 1.0),  # image mosaic (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
        "close_mosaic": tune.randint(0, 11),  # close dataloader mosaic (epochs)
    }

    # Put the model in ray store
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        """Train the YOLO model with the specified hyperparameters and return results."""
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.trainer = None
        model_to_train.reset_callbacks()
        config = _sanitize_tune_value(dict(config))
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

    resolved_search_alg, tuner_param_space, resolved_search_alg_kind = _resolve_ray_search_alg(
        search_alg, task, space, iterations
    )

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # Define the scheduler for hyperparameter search
    max_t = train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=max_t,
        grace_period=min(grace_period, max_t),
        reduction_factor=3,
    )
    if resolved_search_alg_kind == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="epoch",
            metric=TASK2METRIC[task],
            mode="max",
            max_t=max_t,
            reduction_factor=3,
        )

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
            param_space=tuner_param_space,
            tune_config=tune.TuneConfig(
                search_alg=resolved_search_alg,
                scheduler=scheduler,
                num_samples=iterations,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(storage_path=tune_dir.parent, name=tune_dir.name),
        )

    # Run the hyperparameter search
    tuner.fit()

    # Get the results of the hyperparameter search
    results = tuner.get_results()

    # Shut down Ray to clean up workers
    ray.shutdown()

    return results
