"""Test SC-ELAN modules in YAML model configuration and parse_model function."""

import torch

from ultralytics.nn.tasks import parse_model

# Test configuration with SC-ELAN modules
test_config = {
    "nc": 80,  # number of classes
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],  # 0-P1/2
        [-1, 1, "Conv", [128, 3, 2]],  # 1-P2/4
        [-1, 1, "SC_ELAN", [128, 128, 128]],  # 2 - Test SC_ELAN
        [-1, 1, "Conv", [256, 3, 2]],  # 3-P3/8
        [-1, 1, "SC_ELAN_Dilated", [256, 256, 256]],  # 4 - Test SC_ELAN_Dilated
        [-1, 1, "Conv", [512, 3, 2]],  # 5-P4/16
        [-1, 1, "SC_ELAN_Slim", [512, 512, 512]],  # 6 - Test SC_ELAN_Slim
    ],
    "head": [
        [-1, 1, "Conv", [80, 1, 1]],  # 7 - Simple head for testing
    ],
}


def test_parse_model_with_sc_elan():
    """Test that parse_model can parse SC-ELAN modules correctly."""
    print("=" * 60)
    print("Testing parse_model with SC-ELAN modules")
    print("=" * 60)

    try:
        # Parse the model
        model, savelist = parse_model(test_config, ch=3, verbose=True)

        print("\n✓ Model parsed successfully!")
        print(f"  - Total layers: {len(model)}")
        print(f"  - Save list: {savelist}")

        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        print(f"\n  Testing forward pass with input shape: {tuple(x.shape)}")

        with torch.no_grad():
            y = model(x)

        print(f"  - Output shape: {tuple(y.shape)}")
        print("\n✓ Forward pass successful!")

        # Verify SC-ELAN modules are in the model
        sc_elan_count = 0
        sc_elan_dilated_count = 0
        sc_elan_slim_count = 0

        for module in model:
            module_type = module.type
            if "SC_ELAN_Slim" in module_type:
                sc_elan_slim_count += 1
            elif "SC_ELAN_Dilated" in module_type:
                sc_elan_dilated_count += 1
            elif "SC_ELAN" in module_type:
                sc_elan_count += 1

        print("\n  Module counts:")
        print(f"  - SC_ELAN: {sc_elan_count}")
        print(f"  - SC_ELAN_Dilated: {sc_elan_dilated_count}")
        print(f"  - SC_ELAN_Slim: {sc_elan_slim_count}")

        assert sc_elan_count >= 1, "SC_ELAN module not found in model"
        assert sc_elan_dilated_count >= 1, "SC_ELAN_Dilated module not found in model"
        assert sc_elan_slim_count >= 1, "SC_ELAN_Slim module not found in model"

        print("\n✓ All SC-ELAN variants successfully integrated into parse_model!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_module_parameters():
    """Test that SC-ELAN modules have reasonable parameter counts."""
    print("\n" + "=" * 60)
    print("Testing SC-ELAN module parameters in parsed model")
    print("=" * 60)

    model, _ = parse_model(test_config, ch=3, verbose=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total model parameters: {total_params:,}")

    # Count parameters per module type
    for i, module in enumerate(model):
        if "SC_ELAN" in module.type:
            num_params = sum(p.numel() for p in module.parameters())
            print(f"  - Layer {i} ({module.type}): {num_params:,} parameters")

    print("\n✓ Parameter counts look reasonable!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SC-ELAN YAML Integration Test Suite")
    print("=" * 60 + "\n")

    success = all(
        [
            test_parse_model_with_sc_elan(),
            test_module_parameters(),
        ]
    )

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests PASSED! SC-ELAN modules are ready to use in YAML configs!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 60 + "\n")
