import os
from unittest import mock

import pytest
import torch

from ultralytics.utils.torch_utils import select_device


class MockDevice:
    """Mock class to replace torch.device, ensuring compatibility with isinstance() checks."""

    def __init__(self, arg):
        """Initializes the mock device with a device string."""
        # Record the input string for future assertions
        self.arg = str(arg)

    def __str__(self):
        """Returns the string representation of the device."""
        return self.arg

    def __repr__(self):
        """Returns the formal string representation of the device."""
        return f"device(type='{self.arg}')"


def test_npu_device_selection_logic():
    """Test NPU selection logic: verify environment variables and device strings."""
    # 1. Mock the torch_npu module and its internal methods
    mock_npu = mock.MagicMock()
    mock_npu.npu.is_available.return_value = True
    mock_npu.npu.device_count.return_value = 2
    mock_npu.npu.get_device_name.return_value = "Ascend910B"

    # 2. Inject the mock module into sys.modules to handle internal imports
    with mock.patch.dict("sys.modules", {"torch_npu": mock_npu, "torch_npu.contrib": mock.MagicMock()}):
        # 3. Critical Fix: Replace torch.device with the MockDevice class (not an instance)
        # to satisfy isinstance(device, torch.device) checks in the source code.
        with mock.patch("torch.device", MockDevice):
            # 4. Simultaneously mock the torch.npu attribute to prevent AttributeError in CI environments
            with mock.patch("torch.npu", mock_npu.npu, create=True):
                with mock.patch.dict(os.environ, {}, clear=False):
                    device_str = "npu:0,1"
                    result = select_device(device_str, verbose=False)

                    # Verify that environment variables are set according to logic
                    assert os.environ.get("ASCEND_VISIBLE_DEVICES") == "0,1"
                    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0,1"

                    # Verify the processed result from select_device.
                    # Logic: Input "npu:0,1" should return the device object for "npu:0".
                    assert str(result) == "npu:0"


def test_npu_fallback_logic():
    """Test fallback to CPU when torch_npu is unavailable."""
    # Mock scenario where the module is missing
    with mock.patch.dict("sys.modules", {"torch_npu": None, "torch_npu.contrib": None}):
        # Ensure the torch.npu attribute is also non-existent to simulate a clean environment
        if hasattr(torch, "npu"):
            with mock.patch.object(torch, "npu", None):
                result = select_device("npu:0", verbose=False)
        else:
            result = select_device("npu:0", verbose=False)

        assert str(result) == "cpu"


if __name__ == "__main__":
    # Execute tests using pytest with verbose output
    pytest.main([__file__, "-v"])
