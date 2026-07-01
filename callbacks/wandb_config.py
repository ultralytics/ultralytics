"""Callback to log custom trainer attributes to WandB config and args.yaml.

Usage:
    from callbacks import wandb_config

    # CE experiments: pass params explicitly
    model.add_callback("on_pretrain_routine_start", wandb_config.log_config(loss_mode="ce", muon_w=0.1))

    # Text-aligned experiments: picks up attrs from TextClassificationTrainer
    model.add_callback("on_pretrain_routine_start", wandb_config.log_config())

    # Fork a parent run (e.g. to recover from a corrupted mid-training state)
    wandb_config.fork_and_attach("parent-run-id", fork_step=7, name="phase1-c3")
    model.train(...)  # Ultralytics' wb.init picks up WANDB_* env vars and attaches
"""

import os
from pathlib import Path

from ultralytics.utils import YAML

from . import paths

EXTRA_ATTRS = ("loss_mode", "muon_w", "use_clip_classifier", "teacher_variant", "teacher_temps", "grad_clip_norm")
_WANDB_INTERNAL_PREFIX = "_"


def log_config(**extra_kv):
    """Return on_pretrain_routine_start callback to log custom config.

    Args:
        **extra_kv: Key-value pairs to set on trainer and log. For TextClassificationTrainer, these attrs already exist;
            for ClassificationTrainer, they're set via extra_kv. Special keys exported as wandb env vars at module-load
        time so they take effect at ``wandb.init`` (post-init mutation does not persist server-side): ``wandb_group`` ->
            ``WANDB_RUN_GROUP``; ``tags`` (list) -> ``WANDB_TAGS`` (comma-joined); ``notes`` (str) -> ``WANDB_NOTES``.
    """
    # Set group/tags/notes via env so wandb.init picks them up; DDP subprocesses inherit via env.
    if "wandb_group" in extra_kv:
        os.environ["WANDB_RUN_GROUP"] = extra_kv["wandb_group"]
    if extra_kv.get("tags"):
        os.environ["WANDB_TAGS"] = ",".join(extra_kv["tags"])
    if extra_kv.get("notes"):
        os.environ["WANDB_NOTES"] = extra_kv["notes"]

    def callback(trainer):
        for k, v in extra_kv.items():
            if not hasattr(trainer, k):
                setattr(trainer, k, v)
        config = {k: getattr(trainer, k) for k in EXTRA_ATTRS if hasattr(trainer, k)}
        config.update(extra_kv)
        for k in ("wandb_group", "tags", "notes"):
            config.pop(k, None)
        # Update args.yaml
        args_path = Path(trainer.save_dir) / "args.yaml"
        if args_path.exists() and config:
            data = YAML.load(args_path)
            data.update(config)
            YAML.save(args_path, data)
        # Update WandB config (tags/notes already set via env at init)
        try:
            import wandb

            if wandb.run and config:
                wandb.run.config.update(config, allow_val_change=True)
        except ImportError:
            pass

    return callback


def fork_and_attach(
    parent_run_id: str,
    fork_step: int,
    name: str,
    entity: str = paths.WANDB_ENTITY,
    project: str = paths.WANDB_PROJECT,
    use_native_fork: bool = False,
) -> str:
    """Create a new wandb run that inherits parent's history up to ``fork_step``, then DDP-handoff via env vars.

    Two modes:
    - ``use_native_fork=True``: uses wandb's ``fork_from`` (private preview; requires support enablement).
    - ``use_native_fork=False`` (default): manual replay via ``wandb.Api`` — copies parent's per-step history
        rows up to ``fork_step`` into a fresh run, preserving the step axis. This is the portable fallback.

    After the forked run is created and finished, exports ``WANDB_RUN_ID`` + ``WANDB_RESUME`` + ``WANDB_PROJECT`` +
    ``WANDB_ENTITY`` so that DDP rank-0's ``wandb.init(...)`` attaches to the forked run instead of creating a new one.

    Args:
        parent_run_id (str): ID of the parent run to fork from.
        fork_step (int): Inclusive ``_step`` value in the parent run where the fork branches off.
        name (str): Display name for the forked run.
        entity (str, optional): WandB entity.
        project (str, optional): WandB project.
        use_native_fork (bool, optional): If True, use ``fork_from`` (requires account enablement).

    Returns:
        (str): ID of the newly created forked run.
    """
    import wandb

    if use_native_fork:
        run = wandb.init(
            entity=entity, project=project, name=name, fork_from=f"{parent_run_id}?_step={fork_step}"
        )
        forked_id = run.id
        run.finish()
    else:
        api = wandb.Api()
        parent = api.run(f"{entity}/{project}/{parent_run_id}")
        df = parent.history(pandas=True, samples=100000)
        df = df[df["_step"] <= fork_step].sort_values("_step").reset_index(drop=True)
        parent_config = {k: v for k, v in dict(parent.config).items() if not k.startswith(_WANDB_INTERNAL_PREFIX)}
        run = wandb.init(entity=entity, project=project, name=name, config=parent_config)
        for _, row in df.iterrows():
            step = int(row["_step"])
            metrics = {}
            for k, v in row.items():
                if k.startswith(_WANDB_INTERNAL_PREFIX):
                    continue
                if isinstance(v, float) and v != v:  # filter NaN
                    continue
                metrics[k] = v
            if metrics:
                run.log(metrics, step=step)
        forked_id = run.id
        run.finish()

    os.environ.update(WANDB_RUN_ID=forked_id, WANDB_RESUME="must", WANDB_PROJECT=project, WANDB_ENTITY=entity)
    return forked_id


def push_summary_to_parent(
    parent_run_id: str,
    summary: dict,
    entity: str = paths.WANDB_ENTITY,
    project: str = paths.WANDB_PROJECT,
) -> None:
    """Merge ``summary`` keys into a parent W&B run's summary via the public API.

    Used to attach phase-2 downstream metrics (e.g. multi-det macro mAP) back onto the phase-1 run that produced the
    backbone, so phase-1 sweep views directly rank recipes by downstream score. Merges by key; existing parent summary
    entries are preserved. Network or bad-id failures are logged but do not raise, so a multi-day phase-2 run does not
    abort at the final step on a typo.

    Args:
        parent_run_id (str): W&B run id of the parent (phase-1) run. Empty/None is a no-op.
        summary (dict): Keys to merge into the parent's summary.
        entity (str, optional): WandB entity.
        project (str, optional): WandB project.
    """
    if not parent_run_id or not summary:
        return
    import wandb

    try:
        wandb.Api().run(f"{entity}/{project}/{parent_run_id}").summary.update(summary)
    except Exception as e:
        print(f"[wandb] failed to push summary to parent {parent_run_id}: {e}")


def assert_parent_resolvable(
    parent_run_id: str,
    entity: str = paths.WANDB_ENTITY,
    project: str = paths.WANDB_PROJECT,
) -> None:
    """Fail fast if a non-empty parent run id does not resolve to a real W&B run.

    push_summary_to_parent swallows a bad id at the final step of a multi-day run, silently dropping the downstream
    link. This asserts at launch so a wrong id (e.g. a dir basename instead of the full timestamped run id) raises in
    seconds instead. Empty id is allowed and means no parent.

    Args:
        parent_run_id (str): W&B run id of the parent. Empty or None is allowed (no parent, no-op).
        entity (str, optional): WandB entity.
        project (str, optional): WandB project.
    """
    if not parent_run_id:
        return
    import wandb

    try:
        wandb.Api().run(f"{entity}/{project}/{parent_run_id}")
    except Exception as e:
        raise SystemExit(
            f"ERROR: phase1_wandb_id {parent_run_id!r} does not resolve to a W&B run in {entity}/{project} "
            f"({type(e).__name__}). Pass the full timestamped run id like phase1-foo_20260101_010101, or empty "
            f"for no parent. Otherwise push_summary_to_parent silently drops the downstream link at the final step."
        )


def resolve_run_id_by_name(
    name: str,
    entity: str = paths.WANDB_ENTITY,
    project: str = paths.WANDB_PROJECT,
) -> str:
    """Resolve a run display name to its full W&B run id, or empty string on any failure.

    Lets multi_det auto-link its downstream macro onto the phase-1 run when no explicit id was passed, resolving the
    phase-1 dir basename to its run id. Never raises: an absent, ambiguous, or unreachable name returns "" so the caller
    (push_summary_to_parent) no-ops instead of aborting a multi-day run. On duplicate names (crash-retry) the finished,
    longest-history, latest run wins.

    Args:
        name (str): Run display name (e.g. the phase-1 checkpoint's dir basename).
        entity (str, optional): WandB entity.
        project (str, optional): WandB project.

    Returns:
        (str): Full run id, or "" if it cannot be resolved.
    """
    import wandb

    try:  # empty match -> sorted([])[-1] raises IndexError -> caught -> ""
        return sorted(
            wandb.Api().runs(f"{entity}/{project}", filters={"displayName": name}),
            key=lambda r: (r.state == "finished", r.summary.get("_step", -1) if r.summary else -1, str(r.createdAt)),
        )[-1].id
    except Exception:
        return ""
