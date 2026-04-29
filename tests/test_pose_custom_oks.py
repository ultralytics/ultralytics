import pytest
import numpy as np
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import DEFAULT_CFG

class DummyModel:
    """A minimal mock model to satisfy DetectionValidator.init_metrics."""
    names = {0: 'person'}

def test_pose_validator_custom_oks():
    """Test custom OKS sigmas loading and validation from YAML."""
    validator = PoseValidator(args=DEFAULT_CFG)
    model = DummyModel()
    
    # Case 1: Correct Configuration (17 keypoints)
    validator.data = {
        'kpt_shape': [17, 3],
        'kpt_oks_sigmas': [0.1] * 17
    }
    
    validator.init_metrics(model)
    assert np.allclose(validator.sigma, np.array([0.1] * 17))

    # Case 2: Length error (10 instead of 17)
    validator.data = {
        'kpt_shape': [17, 3],
        'kpt_oks_sigmas': [0.1] * 10
    }
    
    with pytest.raises(ValueError, match="Length of 'kpt_oks_sigmas'"):
        validator.init_metrics(model)

    # Case 3: sigma values <= must provoke errors
    validator.data = {
        'kpt_shape': [17, 3],
        'kpt_oks_sigmas': [0.0] * 17
    }
    with pytest.raises(ValueError, match="strictly positive"):
        validator.init_metrics(model)