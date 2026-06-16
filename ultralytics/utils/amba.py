# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import TORCH_1_10, ModelEMA, unwrap_model


def normalize_imgsz(imgsz: int | list[int], default: int = 640) -> tuple[int, int]:
    """Return (height, width) from imgsz config."""
    if imgsz is None:
        imgsz = default
    if isinstance(imgsz, int):
        return imgsz, imgsz
    imgsz = list(imgsz)
    if len(imgsz) == 1:
        return imgsz[0], imgsz[0]
    return int(imgsz[0]), int(imgsz[1])


def set_amba_chipset(amba_chipset: str | None) -> None:
    """Set SpongeTorch target chipset when configured."""
    if amba_chipset is None:
        return
    import spongetorch

    spongetorch.set_target_chipset(amba_chipset)


def make_example_inputs(
    model: nn.Module,
    imgsz: int | list[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build random example inputs for spongetorch.prepare()."""
    h, w = normalize_imgsz(imgsz)
    yaml = getattr(unwrap_model(model), "yaml", {}) or {}
    channels = yaml.get("channels", 3)
    if device is None or dtype is None:
        param = next(model.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
    return torch.rand(1, channels, h, w, device=device, dtype=dtype)


def prepare_spongetorch_model(
    model: nn.Module,
    amba_config: str | Path,
    *,
    amba_chipset: str | None = None,
    imgsz: int | list[int] = 640,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    total_steps: int = 1,
) -> tuple[nn.Module, torch.optim.Optimizer | None, int | None]:
    """Run spongetorch.prepare() and return (model, optimizer, end_step)."""
    import spongetorch

    set_amba_chipset(amba_chipset)
    example_inputs = make_example_inputs(model, imgsz, device=device, dtype=dtype)
    # Parameter rebuilds under inference_mode become non-trainable inference tensors.
    prepare_ctx = torch.inference_mode(False) if TORCH_1_10 else contextlib.nullcontext()
    with prepare_ctx:
        model, optimizer = spongetorch.prepare(
            model,
            str(amba_config),
            base_optimizer=optimizer,
            total_steps=int(total_steps),
            inplace=True,
            example_inputs=example_inputs,
        )
    end_step = get_spongetorch_end_step(optimizer)
    return model, optimizer, end_step


def get_spongetorch_end_step(optimizer: torch.optim.Optimizer | None) -> int | None:
    """Return spongetorch scheduler end_step from an optimizer, if present."""
    callbacks = getattr(optimizer, "callbacks", None) if optimizer is not None else None
    if not callbacks:
        return None
    scheduler = getattr(callbacks[0], "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "end_step"):
        return None
    return int(scheduler.end_step)


def copy_amba_config(amba_config: str | Path, save_dir: Path) -> Path:
    """Copy spongetorch config into the run directory once."""
    dest = save_dir / "amba_config.prototxt"
    shutil.copy(amba_config, dest)
    LOGGER.info(f"Copied amba_config.prototxt to {dest}")
    return dest


def is_pruned_model(model: nn.Module) -> bool:
    """Return True if the module tree contains pruning masks."""
    from torch.nn.utils import prune as torch_prune

    return torch_prune.is_pruned(model)


def load_pt_for_amba_validation(
    weights: str | Path,
    *,
    device: torch.device,
    amba_config: str | Path | None,
    amba_chipset: str | None,
    imgsz: int | list[int],
) -> nn.Module:
    """Load a .pt checkpoint and apply spongetorch prepare when needed for validation."""
    from ultralytics.nn.tasks import load_checkpoint

    model, _ = load_checkpoint(weights, device=device, fuse=False)
    if amba_config is None:
        set_amba_chipset(amba_chipset)
        return model

    if is_pruned_model(model):
        set_amba_chipset(amba_chipset)
        LOGGER.info("SpongeTorch pruning is already present in the checkpoint; skipping validation-time prepare().")
        return model

    model, _, _ = prepare_spongetorch_model(
        model,
        amba_config,
        amba_chipset=amba_chipset,
        imgsz=imgsz,
        device=device,
        dtype=torch.float32,
        total_steps=1,
    )
    return model


def resolve_validation_model(
    model_ref: Any,
    args,
    *,
    device: torch.device,
) -> tuple[Any, bool]:
    """Resolve the validation model and AutoBackend ``fuse`` flag for SpongeTorch.

    Applies spongetorch ``prepare()`` (or chipset selection) when validating a ``.pt``
    checkpoint under an amba config, and disables Conv+BN fusion for amba/pruned models.

    Returns:
        (Any): Model reference to pass to ``AutoBackend`` (path or prepared module).
        (bool): Whether AutoBackend should fuse Conv+BN layers.
    """
    amba_mode = bool(getattr(args, "amba_config", None) or getattr(args, "amba_chipset", None))
    is_pruned = isinstance(model_ref, nn.Module) and is_pruned_model(model_ref)
    if amba_mode and isinstance(model_ref, (str, Path)) and str(model_ref).endswith(".pt"):
        if getattr(args, "amba_config", None):
            model_ref = load_pt_for_amba_validation(
                model_ref,
                device=device,
                amba_config=args.amba_config,
                amba_chipset=args.amba_chipset,
                imgsz=args.imgsz,
            )
        else:
            from ultralytics.nn.tasks import load_checkpoint

            model_ref, _ = load_checkpoint(model_ref, device=device, fuse=False)
            set_amba_chipset(args.amba_chipset)
        is_pruned = is_pruned_model(model_ref)
    return model_ref, not (amba_mode or is_pruned)


def prepare_export_model(
    source_model: nn.Module,
    amba_config: str | Path,
    *,
    amba_chipset: str | None = None,
    imgsz: int | list[int] = 640,
    task: str | None = None,
    ckpt_path: str | Path | None = None,
) -> nn.Module:
    """Rebuild/prepare a model for SpongeTorch export and restore sparse checkpoint weights."""
    # Cast a copy of the state dict to FP32 without mutating the caller's model dtype.
    source_state = {k: v.float() if v.is_floating_point() else v for k, v in source_model.state_dict().items()}
    has_spongetorch_state = any(k.endswith(("_orig", "_mask")) for k in source_state)
    if not has_spongetorch_state:
        raise ValueError(
            "Checkpoint has no SpongeTorch pruning state (_orig/_mask keys in state_dict). "
            "Use a compressed checkpoint from amba training before export."
        )

    model_yaml = deepcopy(source_model.yaml)
    model = type(source_model)(
        model_yaml,
        ch=model_yaml.get("channels", 3),
        nc=model_yaml.get("nc"),
        verbose=False,
    )

    model, _, _ = prepare_spongetorch_model(
        model,
        amba_config,
        amba_chipset=amba_chipset,
        imgsz=imgsz,
        total_steps=1,
    )
    try:
        model.load_state_dict(source_state)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load SpongeTorch checkpoint into prepared export model. "
            f"Verify amba_config='{amba_config}' matches training."
        ) from e

    model.args = getattr(source_model, "args", {})
    model.pt_path = getattr(source_model, "pt_path", ckpt_path)
    model.task = getattr(source_model, "task", task)
    model.names = getattr(source_model, "names", getattr(model, "names", None))
    return model


def get_export_model(
    model: nn.Module,
    args: dict[str, Any],
    *,
    task: str | None = None,
    ckpt_path: str | Path | None = None,
) -> nn.Module:
    """Return a model ready for Exporter; run SpongeTorch prepare when ``amba_config`` is set."""
    amba_config = args.get("amba_config")
    if amba_config is None:
        return model
    return prepare_export_model(
        model,
        amba_config,
        amba_chipset=args.get("amba_chipset"),
        imgsz=args.get("imgsz", 640),
        task=task,
        ckpt_path=ckpt_path,
    )


def get_spongetorch_step(optimizer: torch.optim.Optimizer | None, fallback: int) -> int:
    """Return current spongetorch scheduler step, or fallback when unavailable."""
    callbacks = getattr(optimizer, "callbacks", None) if optimizer is not None else None
    if not callbacks:
        return fallback
    scheduler = getattr(callbacks[0], "scheduler", None)
    if scheduler is None:
        return fallback
    for attr in ("step", "current_step", "global_step", "steps", "last_step", "_step"):
        value = getattr(scheduler, attr, None)
        if value is None or callable(value):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return fallback


def spongetorch_gate_open(end_step: int | None, current_step: int) -> bool:
    """Return True when spongetorch compression has finished (or gating is inactive)."""
    return end_step is None or current_step > int(end_step)


def make_training_ema(model: nn.Module) -> ModelEMA:
    """Create ModelEMA, safely deepcopying pruned SpongeTorch modules."""
    inner = unwrap_model(model)
    stashed = [
        (m, m.__dict__.pop("weight"))
        for m in inner.modules()
        if hasattr(m, "weight_orig") and "weight" in m.__dict__
    ]
    try:
        return ModelEMA(inner)
    finally:
        for m, w in stashed:
            m.__dict__["weight"] = w


@dataclass
class SpongeTorchTraining:
    """Optional SpongeTorch training state; inactive when ``amba_config`` is unset."""

    amba_config: str | Path | None = None
    amba_chipset: str | None = None
    imgsz: int | list[int] = 640
    end_step: int | None = None
    optimizer_steps: int = 0
    _save_gate_logged: bool = field(default=False, repr=False)
    _config_copied: bool = field(default=False, repr=False)

    @classmethod
    def from_args(cls, args) -> SpongeTorchTraining:
        """Build training state from trainer args."""
        return cls(
            amba_config=getattr(args, "amba_config", None),
            amba_chipset=getattr(args, "amba_chipset", None),
            imgsz=getattr(args, "imgsz", 640),
        )

    @property
    def active(self) -> bool:
        """Return True when SpongeTorch training is configured."""
        return self.amba_config is not None

    def prepare(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        save_dir: Path,
        nb: int,
        epochs: int,
        ckpt: dict | None = None,
        epoch: int | None = None,
        resume: bool = False,
        start_epoch: int = 0,
    ) -> tuple[nn.Module, torch.optim.Optimizer | None]:
        """Prepare model/optimizer for SpongeTorch training when configured."""
        if not self.active:
            return model, optimizer

        if not self._config_copied:
            copy_amba_config(self.amba_config, save_dir)
            self._config_copied = True

        if is_pruned_model(unwrap_model(model)):
            set_amba_chipset(self.amba_chipset)
            LOGGER.info("SpongeTorch pruning already present; skipping training-time prepare().")
            self.sync(optimizer)
            return model, optimizer

        total_steps = self._remaining_steps(nb, epochs, ckpt=ckpt, epoch=epoch, resume=resume, start_epoch=start_epoch)
        model, optimizer, self.end_step = prepare_spongetorch_model(
            model,
            self.amba_config,
            amba_chipset=self.amba_chipset,
            imgsz=self.imgsz,
            optimizer=optimizer,
            total_steps=total_steps,
        )
        self.sync(optimizer)
        return model, optimizer

    def sync(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Refresh step/end_step tracking after resume or when prepare is skipped."""
        if not self.active:
            return
        if self.end_step is None:
            self.end_step = get_spongetorch_end_step(optimizer)
        self.optimizer_steps = get_spongetorch_step(optimizer, self.optimizer_steps)

    def on_optimizer_step(self) -> None:
        """Track optimizer steps for SpongeTorch gate fallback."""
        self.optimizer_steps += 1

    def current_step(self, optimizer: torch.optim.Optimizer | None) -> int:
        """Return current SpongeTorch step."""
        return get_spongetorch_step(optimizer, self.optimizer_steps)

    def gate_open(self, optimizer: torch.optim.Optimizer | None) -> bool:
        """Return True when compression has finished or SpongeTorch is inactive."""
        if not self.active:
            return True
        return spongetorch_gate_open(self.end_step, self.current_step(optimizer))

    def can_save_checkpoint(self, optimizer: torch.optim.Optimizer | None, *, final_epoch: bool = False) -> bool:
        """Return True when checkpoints may be written."""
        if not self.active or self.end_step is None:
            return True

        current_step = self.current_step(optimizer)
        end_step = int(self.end_step)
        if current_step > end_step:
            LOGGER.info(f"spongetorch step {current_step} has crossed end_step {end_step}. Saving checkpoint...")
            return True
        if final_epoch:
            LOGGER.warning(
                f"spongetorch step {current_step} has not crossed end_step {end_step}, "
                "but this is the final epoch. Saving checkpoint anyway."
            )
            return True
        if not self._save_gate_logged:
            LOGGER.info(
                f"Skipping checkpoint save: spongetorch step {current_step} has not crossed end_step {end_step}."
            )
            self._save_gate_logged = True
        return False

    def log_delayed_best_save(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Log once when best.pt is delayed until compression finishes."""
        if not self.active or self.end_step is None or self._save_gate_logged:
            return
        current_step = self.current_step(optimizer)
        LOGGER.info(
            f"Delaying best.pt save: spongetorch step {current_step} has not crossed "
            f"end_step {int(self.end_step)}."
        )
        self._save_gate_logged = True

    @staticmethod
    def _remaining_steps(
        nb: int,
        epochs: int,
        *,
        ckpt: dict | None = None,
        epoch: int | None = None,
        resume: bool = False,
        start_epoch: int = 0,
    ) -> int:
        """Return remaining batch steps for spongetorch.prepare() total_steps."""
        if ckpt is not None and resume:
            start = ckpt.get("epoch", -1) + 1
        elif epoch is not None:
            start = epoch
        else:
            start = start_epoch
        return int(nb * max(epochs - start, 1))