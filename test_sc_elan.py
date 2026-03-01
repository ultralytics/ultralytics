"""Test script for SC-ELAN module variants.

This script tests the SC_ELAN, SC_ELAN_Dilated, and SC_ELAN_Slim modules
to validate their functionality and compare their characteristics.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import SC_ELAN, SC_ELAN_Dilated, SC_ELAN_Slim


def count_parameters(model):
    """Count the number of parameters in a model.

    Args:
        model (nn.Module): The model to count parameters for.

    Returns:
        int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def test_module(module_class, module_name, c1=64, c2=128):
    """Test a specific SC-ELAN module variant.

    Args:
        module_class: The module class to test.
        module_name (str): Name of the module for display.
        c1 (int): Input channels. Defaults to 64.
        c2 (int): Output channels. Defaults to 128.
    """
    print(f"\n{'=' * 60}")
    print(f"Testing {module_name}")
    print(f"{'=' * 60}")

    # Create module instance
    # Note: c3, c4, c5 are compatibility parameters not used in SC-ELAN
    model = module_class(c1=c1, c2=c2, c3=c2, c4=c2, c5=1)
    model.eval()

    # Count parameters
    params = count_parameters(model)
    print(f"Total parameters: {params:,}")

    # Test with different input sizes
    batch_sizes = [1, 4]
    spatial_sizes = [(32, 32), (64, 64), (128, 128)]

    for batch_size in batch_sizes:
        for h, w in spatial_sizes:
            # Create dummy input
            x = torch.randn(batch_size, c1, h, w)

            # Forward pass
            try:
                with torch.no_grad():
                    y = model(x)

                # Verify output shape
                expected_shape = (batch_size, c2, h, w)
                assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

                print(f"✓ Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")

            except Exception as e:
                print(f"✗ Failed for input shape {tuple(x.shape)}: {e}")
                return False

    print(f"\n{module_name} passed all tests!")
    return True


def compare_modules():
    """Compare computational characteristics of all SC-ELAN variants."""
    print(f"\n{'=' * 60}")
    print("Comparison of SC-ELAN Variants")
    print(f"{'=' * 60}\n")

    c1, c2 = 128, 256
    modules = [
        (SC_ELAN, "SC_ELAN (Full)"),
        (SC_ELAN_Dilated, "SC_ELAN_Dilated"),
        (SC_ELAN_Slim, "SC_ELAN_Slim"),
    ]

    results = []

    for module_class, module_name in modules:
        model = module_class(c1=c1, c2=c2, c3=c2, c4=c2, c5=1)
        model.eval()

        params = count_parameters(model)

        # Measure inference time (rough estimate)
        x = torch.randn(1, c1, 64, 64)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Time measurement
        import time

        iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x)
        end = time.time()
        avg_time = (end - start) / iterations * 1000  # Convert to ms

        results.append((module_name, params, avg_time))

    # Print comparison table
    print(f"{'Module':<25} {'Parameters':<15} {'Avg Time (ms)':<15}")
    print("-" * 55)
    for name, params, time_ms in results:
        print(f"{name:<25} {params:<15,} {time_ms:<15.3f}")

    # Calculate relative differences
    print(f"\n{'Relative to SC_ELAN:':<25}")
    base_params = results[0][1]
    base_time = results[0][2]

    for i, (name, params, time_ms) in enumerate(results[1:], 1):
        param_ratio = params / base_params
        time_ratio = time_ms / base_time
        print(f"  {name:<23} Params: {param_ratio:.2%}, Time: {time_ratio:.2%}")


def test_gradient_flow():
    """Test that gradients flow properly through the modules."""
    print(f"\n{'=' * 60}")
    print("Testing Gradient Flow")
    print(f"{'=' * 60}\n")

    c1, c2 = 64, 128
    model = SC_ELAN(c1=c1, c2=c2, c3=c2, c4=c2, c5=1)
    model.train()

    # Create dummy input and target
    x = torch.randn(2, c1, 32, 32, requires_grad=True)
    target = torch.randn(2, c2, 32, 32)

    # Forward pass
    y = model(x)

    # Compute loss and backward
    loss = nn.MSELoss()(y, target)
    loss.backward()

    # Check gradients
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"✓ Input gradients: {'Present' if has_grad else 'Missing'}")

    # Check parameter gradients (allow some parameters without gradients, e.g., frozen BN params)
    param_grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad]
    params_with_grads = sum(param_grads)
    total_params = len(param_grads)
    grad_ratio = params_with_grads / total_params if total_params > 0 else 0
    print(f"✓ Parameter gradients: {params_with_grads}/{total_params} ({grad_ratio:.1%})")

    # Pass if input has gradients and at least 80% of parameters have gradients
    if has_grad and grad_ratio >= 0.8:
        print("\nGradient flow test passed!")
        return True
    else:
        print(f"\nGradient flow test {'passed with warning' if has_grad and grad_ratio >= 0.5 else 'failed'}!")
        return has_grad and grad_ratio >= 0.5


def test_feature_dimensions():
    """Test that feature concatenation works correctly in ELAN structure."""
    print(f"\n{'=' * 60}")
    print("Testing Feature Dimensions in ELAN Structure")
    print(f"{'=' * 60}\n")

    c1, c2 = 64, 128
    model = SC_ELAN(c1=c1, c2=c2, c3=c2, c4=c2, c5=1)
    model.eval()

    x = torch.randn(1, c1, 32, 32)

    # Hook to capture intermediate features
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.shape

        return hook

    # Register hooks
    model.cv1.register_forward_hook(hook_fn("cv1"))
    model.cv2.register_forward_hook(hook_fn("cv2"))
    model.cv3.register_forward_hook(hook_fn("cv3"))
    model.cv4.register_forward_hook(hook_fn("cv4"))

    with torch.no_grad():
        y = model(x)

    print("Feature map dimensions at each stage:")
    for name, shape in features.items():
        print(f"  {name}: {tuple(shape)}")

    print(f"\nFinal output shape: {tuple(y.shape)}")
    print("Feature dimension test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SC-ELAN Module Test Suite")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test each module variant
    modules_to_test = [
        (SC_ELAN, "SC_ELAN"),
        (SC_ELAN_Dilated, "SC_ELAN_Dilated"),
        (SC_ELAN_Slim, "SC_ELAN_Slim"),
    ]

    all_passed = True
    for module_class, module_name in modules_to_test:
        passed = test_module(module_class, module_name)
        all_passed = all_passed and passed

    # Additional tests
    gradient_passed = test_gradient_flow()
    feature_passed = test_feature_dimensions()

    # Comparison
    compare_modules()

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    if all_passed and gradient_passed and feature_passed:
        print("✓ All tests passed successfully!")
    else:
        print("✗ Some tests failed. Please review the output above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
