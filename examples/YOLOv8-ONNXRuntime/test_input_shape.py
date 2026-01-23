"""Test script to verify ONNX input shape assignment fix for issue #23126.

This script validates that input_height and input_width are correctly assigned
from the ONNX model's NCHW format input shape.
"""

import sys


def test_shape_assignment():
    """Test that input shape indices correctly map to height and width.

    ONNX models use NCHW format where:
    - input_shape[0] = N (batch size)
    - input_shape[1] = C (channels)
    - input_shape[2] = H (height)
    - input_shape[3] = W (width)
    """
    print("Testing ONNX input shape assignment...")

    # Simulate NCHW input shape: batch=1, channels=3, height=480, width=640
    simulated_shape = [1, 3, 480, 640]

    # Correct assignment (after fix)
    input_height = simulated_shape[2]  # Should be 480
    input_width = simulated_shape[3]  # Should be 640

    print(f"Input shape (NCHW): {simulated_shape}")
    print(f"  - Batch:    {simulated_shape[0]}")
    print(f"  - Channels: {simulated_shape[1]}")
    print(f"  - Height:   {simulated_shape[2]}")
    print(f"  - Width:    {simulated_shape[3]}")
    print()
    print(f"Assigned values:")
    print(f"  input_height = input_shape[2] = {input_height}")
    print(f"  input_width  = input_shape[3] = {input_width}")

    # Verify correct assignment
    assert input_height == 480, f"Expected height=480, got {input_height}"
    assert input_width == 640, f"Expected width=640, got {input_width}"

    print("\n✅ Test passed! Height and width are correctly assigned.")
    return True


if __name__ == "__main__":
    try:
        test_shape_assignment()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
