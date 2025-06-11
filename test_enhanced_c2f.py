# fixed_test_enhanced_c2f.py
# Fixed version of the test script

import torch
import time
from ultralytics.nn.modules.block import EnhancedC2f, C2f, EnhancedC2fConfig

def test_block_compatibility():
    """Test if EnhancedC2f is compatible with original C2f."""
    print("Testing block compatibility...")
    
    # Test input
    x = torch.randn(1, 128, 64, 64)
    
    # Original C2f
    original = C2f(c1=128, c2=128, n=2)
    
    # Enhanced C2f variants with FIXED parameters
    lightweight = EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.lightweight())
    balanced = EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.balanced())
    full = EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.full())
    
    # Test forward pass
    with torch.no_grad():
        out_orig = original(x)
        out_light = lightweight(x)
        out_balanced = balanced(x)
        out_full = full(x)
    
    print(f"Original output shape: {out_orig.shape}")
    print(f"Lightweight output shape: {out_light.shape}")
    print(f"Balanced output shape: {out_balanced.shape}")
    print(f"Full output shape: {out_full.shape}")
    
    # Check shapes match
    assert out_orig.shape == out_light.shape == out_balanced.shape == out_full.shape
    print("✅ All output shapes match!")
    
    return True

def benchmark_performance():
    """Benchmark performance of different variants."""
    print("\nBenchmarking performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4, 128, 64, 64).to(device)  # Batch of 4
    
    models = {
        'Original C2f': C2f(c1=128, c2=128, n=2).to(device),
        'Enhanced Lightweight': EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.lightweight()).to(device),
        'Enhanced Balanced': EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.balanced()).to(device),
        'Enhanced Full': EnhancedC2f(c1=128, c2=128, n=2, **EnhancedC2fConfig.full()).to(device),
    }
    
    # Warmup
    for model in models.values():
        with torch.no_grad():
            _ = model(x)
    
    # Benchmark
    results = {}
    for name, model in models.items():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):  # 100 iterations
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # ms
        results[name] = avg_time
        
        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20} | {avg_time:6.2f}ms | {params:8,} params")
    
    # Calculate relative overhead
    baseline = results['Original C2f']
    for name, time_ms in results.items():
        if name != 'Original C2f':
            overhead = ((time_ms - baseline) / baseline) * 100
            print(f"{name:20} | {overhead:+5.1f}% overhead")
    
    return results

def test_simple_loading():
    """Simple test without YAML files."""
    print("\nTesting simple model creation...")
    
    try:
        # Test creating enhanced C2f directly
        model = EnhancedC2f(c1=64, c2=64, n=1, **EnhancedC2fConfig.lightweight())
        
        # Test forward pass
        x = torch.randn(1, 64, 32, 32)
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✅ Simple model creation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simple model creation: {e}")
        return False

def debug_channel_flow():
    """Debug the channel flow through the enhanced model."""
    print("\nDebugging channel flow...")
    
    try:
        # Create model with debug info
        c1, c2, n = 128, 128, 2
        model = EnhancedC2f(c1=c1, c2=c2, n=n, **EnhancedC2fConfig.balanced())
        
        x = torch.randn(1, c1, 32, 32)
        print(f"Input: {x.shape}")
        
        # Step through the model manually
        with torch.no_grad():
            # Initial convolution
            y = model.cv1(x)
            print(f"After cv1: {y.shape}")
            
            # Split
            y_list = list(y.chunk(2, 1))
            print(f"After split: {[yi.shape for yi in y_list]}")
            
            # Process through bottlenecks
            for i, m in enumerate(model.m):
                feat = m(y_list[-1])
                print(f"After bottleneck {i}: {feat.shape}")
                y_list.append(feat)
            
            print(f"Total features: {len(y_list)}")
            print(f"Feature shapes: {[yi.shape for yi in y_list]}")
            
            # Test fusion
            if model.use_adaptive_fusion:
                fused = model.feature_fusion(y_list)
                print(f"After adaptive fusion: {fused.shape}")
                final = model.cv2(fused)
            else:
                concat = torch.cat(y_list, 1)
                print(f"After concatenation: {concat.shape}")
                final = model.cv2(concat)
            
            print(f"Final output: {final.shape}")
        
        print("✅ Channel flow debugging successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in channel flow: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test():
    """Quick test to verify enhanced C2f works."""
    print("Quick Enhanced C2f Test")
    print("-" * 30)
    
    try:
        # Test input
        x = torch.randn(1, 128, 32, 32)
        
        # Create enhanced model with lightweight config
        model = EnhancedC2f(
            c1=128, c2=128, n=2,
            **EnhancedC2fConfig.lightweight()
        )
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
        
        if output.shape[0] == x.shape[0] and output.shape[1] == 128:
            print("✅ Quick test PASSED!")
            return True
        else:
            print("❌ Quick test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("FIXED ENHANCED C2F TESTING SUITE")
    print("=" * 60)
    
    tests = [
        ("Debug Channel Flow", debug_channel_flow),
        ("Block Compatibility", test_block_compatibility),
        ("Performance Benchmark", benchmark_performance),
        ("Simple Loading", test_simple_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} | {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return results

if __name__ == "__main__":
    # Run individual test for debugging
    quick_test()
    
    # Or run all tests
    run_all_tests()