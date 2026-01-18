import os
from unittest import mock

import pytest

from ultralytics.utils.torch_utils import select_device


def test_npu_device_selection_logic():
    """Test NPU selection logic: verify that environment variables are set correctly."""
    # Define a mock torch_npu module
    mock_npu = mock.MagicMock()
    mock_npu.npu.is_available.return_value = True
    mock_npu.npu.device_count.return_value = 2
    mock_npu.npu.get_device_name.return_value = "Ascend910B"

    # Inject the mock module into sys.modules so 'import torch_npu' retrieves it
    with mock.patch.dict("sys.modules", {"torch_npu": mock_npu, "torch_npu.contrib": mock.MagicMock()}):
        # Clear environment variable interference during the test
        with mock.patch.dict(os.environ, {}, clear=False):
            # Run the device selection logic
            device_str = "npu:0,1"
            result = select_device(device_str, verbose=False)

            # Verify the returned device and environment variables
            assert "npu" in str(result).lower()
            assert os.environ.get("ASCEND_VISIBLE_DEVICES") == "0,1"
            assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0,1"


def test_npu_fallback_logic():
    """Test fallback logic: should revert to CPU when torch_npu import fails."""
    # Mock the scenario where torch_npu is completely unavailable
    with mock.patch.dict("sys.modules", {"torch_npu": None, "torch_npu.contrib": None}):
        # Execute selection logic
        result = select_device("npu:0", verbose=False)

        # Verify that it falls back to CPU
        assert str(result) == "cpu"


if __name__ == "__main__":
    # Execute the tests using pytest
    pytest.main([__file__, "-v"])
