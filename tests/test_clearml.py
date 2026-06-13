# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("project", "expected"),
    [
        (None, "Ultralytics"),
        ("", "Ultralytics"),
        ("runs/train", "runs/train"),
        ("/tmp/test/runs", "tmp/test/runs"),
        ("///tmp/test", "tmp/test"),
        ("/", "Ultralytics"),
        ("Ultralytics", "Ultralytics"),
        ("C:/tmp/test/runs", "C:/tmp/test/runs"),
        ("C:\\tmp\\test\\runs", "C:\\tmp\\test\\runs"),
        ("\\\\server\\share\\runs", "\\\\server\\share\\runs"),
        ("//server/share/runs", "//server/share/runs"),
    ],
)
def test_clearml_project_name(project, expected):
    """Ensure ClearML project names are normalized without changing relative project names."""
    from ultralytics.utils.callbacks.clearml import _clearml_project_name

    assert _clearml_project_name(project) == expected


@pytest.mark.parametrize(
    ("project", "expected"),
    [
        ("/tmp/test/runs", "tmp/test/runs"),
        ("runs/train", "runs/train"),
    ],
)
def test_clearml_project_name_passed_to_task_init(monkeypatch, project, expected):
    """Ensure ClearML receives the expected project name without changing other Task.init arguments."""
    from ultralytics.utils.callbacks import clearml

    args = SimpleNamespace(project=project, name="exp")
    connect_calls = []
    init_kwargs = {}

    class MockTask:
        @staticmethod
        def current_task():
            return None

        @staticmethod
        def init(**kwargs):
            init_kwargs.update(kwargs)
            return MockTask()

        def connect(self, *args, **kwargs):
            connect_calls.append((args, kwargs))

    trainer = SimpleNamespace(args=args)
    monkeypatch.setattr(clearml, "Task", MockTask, raising=False)

    clearml.on_pretrain_routine_start(trainer)

    assert init_kwargs["project_name"] == expected
    assert init_kwargs["task_name"] == "exp"
    assert init_kwargs["tags"] == ["Ultralytics"]
    assert init_kwargs["output_uri"] is True
    assert init_kwargs["reuse_last_task_id"] is False
    assert init_kwargs["auto_connect_frameworks"] == {"pytorch": False, "matplotlib": False}
    assert connect_calls == [((vars(args),), {"name": "General", "ignore_remote_overrides": True})]


def test_clearml_connect_warning_does_not_raise(monkeypatch):
    """Ensure ClearML callback handles connect failures without raising."""
    from ultralytics.utils.callbacks import clearml

    warnings = []

    class MockTask:
        @staticmethod
        def current_task():
            return None

        @staticmethod
        def init(**kwargs):
            return MockTask()

        def connect(self, *args, **kwargs):
            raise RuntimeError("connect failed")

    trainer = SimpleNamespace(args=SimpleNamespace(project="/tmp/test/runs", name="exp"))
    monkeypatch.setattr(clearml, "Task", MockTask, raising=False)
    monkeypatch.setattr(clearml.LOGGER, "warning", warnings.append)

    clearml.on_pretrain_routine_start(trainer)

    assert any("ClearML installed but not initialized correctly" in warning for warning in warnings)
