# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for wandb project/entity precedence logic in Ray Tune integration."""

from unittest.mock import MagicMock

from ultralytics.utils.tuner import WANDB_ONLY_KEYS


class TestWandbPrecedence:
    """Test wandb project/entity setting precedence in run_ray_tune."""

    @staticmethod
    def _build_wandb_callback_kwargs(wandb_run=None, train_args=None):
        """Build WandbLoggerCallback kwargs using the same precedence logic as run_ray_tune.

        Priority: active wandb.run > train_args > default fallback.

        Args:
            wandb_run: Mock wandb.run object or None.
            train_args: Dict of training arguments (may contain 'project', 'entity').

        Returns:
            dict: kwargs that would be passed to WandbLoggerCallback.
        """
        if train_args is None:
            train_args = {}

        # Extract wandb-only keys (mimics run_ray_tune behavior)
        wandb_kwargs = {k: train_args.pop(k) for k in WANDB_ONLY_KEYS if k in train_args}

        wandb_project = (wandb_run.project if wandb_run else None) or train_args.get("project") or "YOLOv8-tune"
        wandb_entity = (wandb_run.entity if wandb_run else None) or wandb_kwargs.get("entity")

        cb_kwargs = {"project": wandb_project}
        if wandb_entity:
            cb_kwargs["entity"] = wandb_entity
        return cb_kwargs

    def test_default_fallback(self):
        """No wandb.run, no train_args project/entity => default 'YOLOv8-tune'."""
        result = self._build_wandb_callback_kwargs(wandb_run=None, train_args={})
        assert result == {"project": "YOLOv8-tune"}

    def test_train_args_project(self):
        """Train_args project used when no active wandb.run."""
        result = self._build_wandb_callback_kwargs(wandb_run=None, train_args={"project": "my-project"})
        assert result["project"] == "my-project"

    def test_train_args_entity(self):
        """Train_args entity used when no active wandb.run."""
        result = self._build_wandb_callback_kwargs(
            wandb_run=None, train_args={"project": "my-project", "entity": "my-team"}
        )
        assert result["project"] == "my-project"
        assert result["entity"] == "my-team"

    def test_wandb_run_overrides_train_args(self):
        """Active wandb.run settings take precedence over train_args."""
        mock_run = MagicMock()
        mock_run.project = "run-project"
        mock_run.entity = "run-entity"

        result = self._build_wandb_callback_kwargs(
            wandb_run=mock_run, train_args={"project": "arg-project", "entity": "arg-entity"}
        )
        assert result["project"] == "run-project"
        assert result["entity"] == "run-entity"

    def test_wandb_run_partial_override(self):
        """Wandb.run with empty entity falls back to train_args entity."""
        mock_run = MagicMock()
        mock_run.project = "run-project"
        mock_run.entity = ""  # empty string is falsy

        result = self._build_wandb_callback_kwargs(wandb_run=mock_run, train_args={"entity": "arg-entity"})
        assert result["project"] == "run-project"
        assert result["entity"] == "arg-entity"

    def test_entity_not_in_train_config(self):
        """Verify 'entity' is extracted from train_args and not forwarded to model.train()."""
        train_args = {"data": "coco8.yaml", "epochs": 5, "entity": "my-team"}
        _ = self._build_wandb_callback_kwargs(wandb_run=None, train_args=train_args)
        assert "entity" not in train_args, "'entity' should be removed from train_args"

    def test_wandb_only_keys_contains_entity(self):
        """Verify entity is in the WANDB_ONLY_KEYS set."""
        assert "entity" in WANDB_ONLY_KEYS
