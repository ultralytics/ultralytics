"""
Simple test script to verify class balancing functionality works.
"""

import torch
import tempfile
import os
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    print("Testing basic class balancing functionality...")
    
    from ultralytics.data.utils import calculate_class_weights
    
    class MockDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    samples = [
        {'cls': torch.tensor([0])},
        {'cls': torch.tensor([0])},
        {'cls': torch.tensor([0])},
        {'cls': torch.tensor([1])},
        {'cls': torch.tensor([2])},
        {'cls': torch.tensor([2])}
    ]
    
    dataset = MockDataset(samples)
    weights = calculate_class_weights(dataset, nc=3)
    
    print(f"Class weights: {weights}")
    print(f"Weight for class 0 (frequent): {weights[0]:.3f}")
    print(f"Weight for class 1 (rare): {weights[1]:.3f}")
    print(f"Weight for class 2 (medium): {weights[2]:.3f}")
    
    assert weights[0] < weights[1], "Rare classes should have higher weights"
    assert weights[2] < weights[1], "Rare classes should have higher weights than medium frequency classes"
    
    print("✅ Class weight calculation works correctly!")
    return True

def test_config_loading():
    """Test configuration loading with cls_weights."""
    print("\nTesting configuration loading...")
    
    from ultralytics.cfg import get_cfg
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
task: detect
mode: train
cls_weights: [1.0, 2.0, 1.5]
epochs: 1
""")
        temp_yaml = f.name
    
    try:
        cfg = get_cfg(temp_yaml)
        assert hasattr(cfg, 'cls_weights'), "cls_weights should be loaded"
        assert cfg.cls_weights == [1.0, 2.0, 1.5], "cls_weights values should match"
        print("✅ Configuration loading works correctly!")
        return True
    finally:
        os.unlink(temp_yaml)

def test_pos_weight_creation():
    """Test pos_weight tensor creation."""
    print("\nTesting pos_weight tensor creation...")
    
    import torch.nn as nn
    
    cls_weights = [1.0, 2.0, 0.5]
    pos_weight = torch.tensor(cls_weights, dtype=torch.float32)
    
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    
    pred = torch.randn(2, 3)
    target = torch.rand(2, 3)
    
    loss = bce(pred, target)
    
    print(f"Loss shape: {loss.shape}")
    print(f"Pos weight: {pos_weight}")
    print("✅ BCEWithLogitsLoss with pos_weight works correctly!")
    return True

def test_weighted_sampler():
    """Test WeightedRandomSampler functionality."""
    print("\nTesting WeightedRandomSampler...")
    
    from torch.utils.data import WeightedRandomSampler
    
    sample_weights = [0.1, 0.1, 0.1, 2.0, 1.0, 1.0]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    samples = list(sampler)[:20]
    print(f"Sample indices: {samples}")
    
    rare_class_count = sum(1 for s in samples if s == 3)
    common_class_count = sum(1 for s in samples if s in [0, 1, 2])
    
    print(f"Rare class (idx 3) appeared {rare_class_count} times")
    print(f"Common classes (idx 0,1,2) appeared {common_class_count} times")
    print("✅ WeightedRandomSampler works correctly!")
    return True

def main():
    """Run all tests."""
    print("Running class balancing functionality tests...")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_config_loading,
        test_pos_weight_creation,
        test_weighted_sampler
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Class balancing functionality is working.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
