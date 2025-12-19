#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for T018, T019, T020: StereoCenterNetHead instantiation and model forward pass.

T018: Verify parse_model() correctly instantiates StereoCenterNetHead from YAML
T019: Test model instantiation from refactored config
T020: Verify model forward pass works with StereoCenterNetHead from YAML
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics.nn.tasks import yaml_model_load, parse_model


class TestStereo3DDetT018T019T020:
    """Test suite for T018, T019, T020."""

    def test_t018_parse_model_instantiates_stereo_center_net_head(self):
        """Test T018: Verify parse_model() correctly instantiates StereoCenterNetHead from YAML."""
        config_path = "ultralytics/cfg/models/stereo3ddet_full.yaml"
        config_dict = yaml_model_load(config_path)
        
        # Find StereoCenterNetHead in head section
        head_section = config_dict.get("head", [])
        stereo_head_layer = None
        for layer in head_section:
            if isinstance(layer, list) and len(layer) >= 3 and layer[2] == "StereoCenterNetHead":
                stereo_head_layer = layer
                break
        
        assert stereo_head_layer is not None, "StereoCenterNetHead not found in head section"
        
        # Test parse_model with minimal config
        test_config = {
            "nc": 3,
            "backbone": [[-1, 1, "Conv", [256, 3, 1]]],  # Dummy backbone
            "head": [stereo_head_layer],
        }
        
        try:
            model, save = parse_model(test_config, ch=256, verbose=False)
            assert model is not None and len(model) > 0
            
            last_layer = model[-1]
            from ultralytics.models.yolo.stereo3ddet.stereo_yolo_v11 import StereoCenterNetHead
            assert isinstance(last_layer, StereoCenterNetHead), f"Expected StereoCenterNetHead, got {type(last_layer)}"
            assert hasattr(last_layer, "branches") and len(last_layer.branches) == 10
            
            print("✓ T018: parse_model() correctly instantiates StereoCenterNetHead from YAML")
            print(f"  - Head type: {type(last_layer).__name__}")
            print(f"  - Branches: {len(last_layer.branches)}")
        except KeyError as e:
            if "StereoCenterNetHead" in str(e):
                pytest.fail(f"parse_model() could not resolve StereoCenterNetHead: {e}")
            raise
        except Exception as e:
            pytest.fail(f"parse_model() failed: {e}")

    def test_t019_model_instantiation_from_refactored_config(self):
        """Test T019: Test model instantiation from refactored config."""
        config_path = "ultralytics/cfg/models/stereo3ddet_full.yaml"
        config_dict = yaml_model_load(config_path)
        
        assert "backbone" in config_dict and "head" in config_dict and "nc" in config_dict
        assert config_dict["nc"] == 3
        assert config_dict.get("input_channels") == 6
        
        from ultralytics.models.yolo.stereo3ddet.model import Stereo3DDetModel
        
        try:
            model = Stereo3DDetModel(cfg=config_dict, ch=6, nc=3, verbose=False)
            assert model is not None and model.model is not None
            assert model.task == "stereo3ddet"
            
            last_layer = model.model[-1]
            from ultralytics.models.yolo.stereo3ddet.stereo_yolo_v11 import StereoCenterNetHead
            assert isinstance(last_layer, StereoCenterNetHead)
            
            print("✓ T019: Model instantiated successfully from refactored config")
            print(f"  - Task: {model.task}")
            print(f"  - Last layer: {type(last_layer).__name__}")
        except Exception as e:
            pytest.fail(f"Model instantiation failed: {e}")

    def test_t020_model_forward_pass_with_stereo_center_net_head(self):
        """Test T020: Verify model forward pass works with StereoCenterNetHead from YAML."""
        config_path = "ultralytics/cfg/models/stereo3ddet_full.yaml"
        from ultralytics.models.yolo.stereo3ddet.model import Stereo3DDetModel
        
        config_dict = yaml_model_load(config_path)
        model = Stereo3DDetModel(cfg=config_dict, ch=6, nc=3, verbose=False)
        model.model.eval()
        
        # Create 6-channel stereo input
        dummy_input = torch.randn(1, 6, 384, 384 * 4)
        
        with torch.no_grad():
            try:
                outputs = model.model(dummy_input)
                assert outputs is not None
                
                if isinstance(outputs, dict):
                    expected_branches = [
                        "heatmap", "offset", "bbox_size", "lr_distance", "right_width",
                        "dimensions", "orientation", "vertices", "vertex_offset", "vertex_dist"
                    ]
                    for branch_name in expected_branches:
                        assert branch_name in outputs, f"Missing branch: {branch_name}"
                        assert isinstance(outputs[branch_name], torch.Tensor)
                    
                    print("✓ T020: Model forward pass successful with StereoCenterNetHead")
                    print(f"  - Output branches: {len(outputs)}")
                    print(f"  - Heatmap shape: {outputs['heatmap'].shape}")
                else:
                    print("✓ T020: Model forward pass successful")
                    print(f"  - Output type: {type(outputs)}")
            except Exception as e:
                pytest.fail(f"Forward pass failed: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("T018, T019, T020: StereoCenterNetHead Tests")
    print("=" * 70)
    print()
    
    test_suite = TestStereo3DDetT018T019T020()
    
    try:
        print("Test T018: parse_model() instantiation...")
        test_suite.test_t018_parse_model_instantiates_stereo_center_net_head()
        print()
        
        print("Test T019: Model instantiation...")
        test_suite.test_t019_model_instantiation_from_refactored_config()
        print()
        
        print("Test T020: Forward pass...")
        test_suite.test_t020_model_forward_pass_with_stereo_center_net_head()
        print()
        
        print("=" * 70)
        print("✓ All tests passed! T018, T019, T020 are complete.")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

