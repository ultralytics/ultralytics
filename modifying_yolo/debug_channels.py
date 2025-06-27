# debug_channels.py
import torch
from ultralytics.nn.modules.block import EnhancedC2f

def debug_channel_calculation():
    """Debug channel calculations in Enhanced C2f."""
    
    test_configs = [
        {'c1': 128, 'c2': 128, 'n': 1, 'e': 0.5},
        {'c1': 256, 'c2': 256, 'n': 2, 'e': 0.5},
        {'c1': 512, 'c2': 512, 'n': 3, 'e': 0.5},
    ]
    
    for config in test_configs:
        print(f"\nTesting config: {config}")
        
        c1, c2, n, e = config['c1'], config['c2'], config['n'], config['e']
        hidden_c = max(1, int(c2 * e))
        total_features = 2 + n
        fusion_channels = total_features * hidden_c
        
        print(f"  Hidden channels (c): {hidden_c}")
        print(f"  Total features: {total_features}")
        print(f"  Fusion channels: {fusion_channels}")
        
        try:
            model = EnhancedC2f(**config)
            x = torch.randn(1, c1, 32, 32)
            output = model(x)
            print(f"  ✅ Success: {x.shape} -> {output.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    debug_channel_calculation()