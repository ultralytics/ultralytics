# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for class filtering and reindexing functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D


class TestClassMappingUtilities:
    """Test suite for class mapping utility functions (T091)."""

    def test_get_paper_class_mapping(self):
        """Test get_paper_class_mapping returns correct mapping dictionaries."""
        from ultralytics.models.yolo.stereo3ddet.utils import (
            ORIGINAL_TO_PAPER,
            PAPER_TO_ORIGINAL,
            get_paper_class_mapping,
        )

        original_to_paper, paper_to_original = get_paper_class_mapping()

        # Verify mapping structure
        assert isinstance(original_to_paper, dict)
        assert isinstance(paper_to_original, dict)

        # Verify expected mappings
        assert original_to_paper[0] == 0  # Car -> Car
        assert original_to_paper[3] == 1  # Pedestrian -> Pedestrian
        assert original_to_paper[5] == 2  # Cyclist -> Cyclist

        # Verify reverse mapping
        assert paper_to_original[0] == 0  # Car -> Car
        assert paper_to_original[1] == 3  # Pedestrian -> Pedestrian
        assert paper_to_original[2] == 5  # Cyclist -> Cyclist

        # Verify constants match
        assert original_to_paper == ORIGINAL_TO_PAPER
        assert paper_to_original == PAPER_TO_ORIGINAL

    def test_filter_and_remap_class_id_valid(self):
        """Test filter_and_remap_class_id with valid class IDs."""
        from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        # Test valid mappings
        assert filter_and_remap_class_id(0) == 0  # Car -> Car
        assert filter_and_remap_class_id(3) == 1  # Pedestrian -> Pedestrian
        assert filter_and_remap_class_id(5) == 2  # Cyclist -> Cyclist

    def test_filter_and_remap_class_id_invalid(self):
        """Test filter_and_remap_class_id with invalid class IDs (should return None)."""
        from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        # Test invalid class IDs (should return None)
        assert filter_and_remap_class_id(1) is None  # Van - filtered out
        assert filter_and_remap_class_id(2) is None  # Truck - filtered out
        assert filter_and_remap_class_id(4) is None  # Person_sitting - filtered out
        assert filter_and_remap_class_id(6) is None  # Tram - filtered out
        assert filter_and_remap_class_id(7) is None  # Misc - filtered out
        assert filter_and_remap_class_id(99) is None  # Invalid ID


@pytest.fixture
def mock_dataset_structure(tmp_path):
    """Create mock dataset directory structure."""
    # Create required directories
    (tmp_path / "images" / "train" / "left").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train" / "right").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "calib" / "train").mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    import cv2
    import numpy as np
    dummy_img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "train" / "left" / "000000.png"), dummy_img)
    cv2.imwrite(str(tmp_path / "images" / "train" / "right" / "000000.png"), dummy_img)
    
    # Create dummy calibration file
    calib_file = tmp_path / "calib" / "train" / "000000.txt"
    calib_file.write_text("fx: 721.5377\nfy: 721.5377\ncx: 609.5593\ncy: 172.8540\nbaseline: 0.54\n")
    
    return tmp_path

class TestStereo3DDetAdapterDatasetClassRemapping:
    """Test suite for Stereo3DDetAdapterDataset class remapping (T093)."""

    def test_adapter_dataset_remaps_classes(self, mock_dataset_structure):
        """Test that Stereo3DDetAdapterDataset remaps class IDs correctly."""
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset

        # Create label file with mixed classes
        label_file = mock_dataset_structure / "labels" / "train" / "000000.txt"
        label_content = """0 0.5 0.5 0.1 0.1 0.5 0.1 1.5 1.7 3.9 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
3 0.7 0.7 0.1 0.1 0.7 0.1 1.7 0.5 0.8 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
5 0.9 0.9 0.1 0.1 0.9 0.1 1.8 0.6 1.8 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
1 0.3 0.3 0.1 0.1 0.3 0.1 1.5 1.7 3.9 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"""
        label_file.write_text(label_content)

        # Create adapter dataset with filtered class names (3 classes)
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        adapter = Stereo3DDetAdapterDataset(root=str(mock_dataset_structure), split="train", imgsz=384, names=names)

        # Get item
        sample = adapter[0]
        
        # Verify sample structure
        assert "img" in sample, "Sample should contain 'img' key"
        assert "labels" in sample, "Sample should contain 'labels' key"
        
        labels = sample["labels"]

        # Verify class IDs are remapped in labels
        # Should have 3 labels (Car=0, Pedestrian=1, Cyclist=2), Van (1) filtered out
        assert len(labels) == 3, f"Expected 3 labels after filtering, got {len(labels)}"
        
        # Verify all class IDs are in {0, 1, 2}
        class_ids = [label["class_id"] for label in labels]
        assert set(class_ids) == {0, 1, 2}, f"Expected class IDs {{0, 1, 2}}, got {set(class_ids)}"
        
        # Verify specific mappings
        for label in labels:
            cid = label["class_id"]
            assert cid in {0, 1, 2}, f"Class ID {cid} not in paper classes"
            # Verify original_class_id is stored if available
            if "original_class_id" in label:
                orig_id = label["original_class_id"]
                if orig_id == 0:
                    assert cid == 0, "Car should map to 0"
                elif orig_id == 3:
                    assert cid == 1, "Pedestrian should map to 1"
                elif orig_id == 5:
                    assert cid == 2, "Cyclist should map to 2"


class TestDecodeStereo3dOutputsWithFilteredClasses:
    """Test suite for decode_stereo3d_outputs with filtered class names (T094)."""

    def test_decode_uses_paper_class_names(self):
        """Test that decode_stereo3d_outputs uses paper class names (3 classes)."""
        from ultralytics.models.yolo.stereo3ddet.val import decode_stereo3d_outputs

        batch_size = 1
        num_classes = 3  # Paper uses 3 classes
        h, w = 96, 320

        # Create mock outputs
        outputs = {
            "heatmap": torch.zeros(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w) * 10 + 20,
            "lr_distance": torch.randn(batch_size, 1, h, w) * 5 + 10,
            "right_width": torch.randn(batch_size, 1, h, w) * 5 + 10,
            "dimensions": torch.randn(batch_size, 3, h, w) * 0.5,
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w),
        }

        # Set high confidence for class 0 (Car)
        outputs["heatmap"][0, 0, 10, 20] = 0.9

        calib = {
            "fx": 721.5377,
            "fy": 721.5377,
            "cx": 609.5593,
            "cy": 172.8540,
            "baseline": 0.54,
        }

        boxes3d = decode_stereo3d_outputs(outputs, conf_threshold=0.5, top_k=100, calib=calib)

        # Verify boxes use paper class names
        assert len(boxes3d) > 0
        for box in boxes3d:
            assert box.class_id in {0, 1, 2}, f"Class ID {box.class_id} not in paper classes"
            assert box.class_label in {"Car", "Pedestrian", "Cyclist"}, f"Class label {box.class_label} not in paper classes"


class TestLabelsToBox3dListWithFiltering:
    """Test suite for _labels_to_box3d_list with class filtering and remapping (T095)."""

    def test_labels_to_box3d_list_filters_and_remaps(self):
        """Test that _labels_to_box3d_list filters and remaps class IDs."""
        from ultralytics.models.yolo.stereo3ddet.val import _labels_to_box3d_list
        from ultralytics.data.stereo.calib import CalibrationParameters

        # Create labels with mixed classes (using original KITTI class IDs)
        labels = [
            {
                "class_id": 0,  # Car -> should become 0 (already correct)
                "left_box": {"center_x": 0.5, "center_y": 0.5, "width": 0.1, "height": 0.1},
                "dimensions": {"height": 1.5, "width": 1.7, "length": 3.9},
                "alpha": 0.0,
            },
            {
                "class_id": 3,  # Pedestrian -> should become 1
                "left_box": {"center_x": 0.7, "center_y": 0.7, "width": 0.1, "height": 0.1},
                "dimensions": {"height": 1.7, "width": 0.5, "length": 0.8},
                "alpha": 0.0,
            },
            {
                "class_id": 1,  # Van -> should be filtered out
                "left_box": {"center_x": 0.3, "center_y": 0.3, "width": 0.1, "height": 0.1},
                "dimensions": {"height": 1.5, "width": 1.7, "length": 3.9},
                "alpha": 0.0,
            },
        ]

        calib = CalibrationParameters(
            fx=721.5377,
            fy=721.5377,
            cx=609.5593,
            cy=172.8540,
            baseline=0.54,
            image_width=1242,
            image_height=375,
        )

        boxes3d = _labels_to_box3d_list(labels, calib)

        # Should only have 2 boxes (Car and Pedestrian, Van filtered out)
        assert len(boxes3d) == 2, f"Expected 2 boxes, got {len(boxes3d)}"

        # Verify class IDs are remapped
        class_ids = [box.class_id for box in boxes3d]
        assert set(class_ids) == {0, 1}, f"Expected class IDs {0, 1}, got {set(class_ids)}"

        # Verify class labels
        for box in boxes3d:
            if box.class_id == 0:
                assert box.class_label == "Car"
            elif box.class_id == 1:
                assert box.class_label == "Pedestrian"

class TestLabelClassValidation:
    def test_validation_passes_when_names_match_labels(self, mock_dataset_structure):
        """Test that validation passes when names parameter matches actual label classes (T118)."""
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset

        # Create label file with classes 0, 3, 5 (Car, Pedestrian, Cyclist)
        label_file = mock_dataset_structure / "labels" / "train" / "000000.txt"
        label_content = """0 0.5 0.5 0.1 0.1 0.5 0.1 1.5 1.7 3.9 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
3 0.7 0.7 0.1 0.1 0.7 0.1 1.7 0.5 0.8 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
5 0.9 0.9 0.1 0.1 0.9 0.1 1.8 0.6 1.8 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"""
        label_file.write_text(label_content)

        # Create adapter with matching names (all 3 classes)
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        
        # Should not raise an error
        adapter = Stereo3DDetAdapterDataset(root=str(mock_dataset_structure), split="train", imgsz=384, names=names)
        assert adapter is not None
        assert len(adapter) > 0

    def test_validation_edge_cases(self, mock_dataset_structure):
        """Test edge cases for label class validation (T119)."""
        from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset

        # Test case 1: Empty labels (no objects in label file)
        label_file = mock_dataset_structure / "labels" / "train" / "000000.txt"
        label_file.write_text("")  # Empty label file
        
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        # Should not raise error for empty labels (no classes to validate against)
        adapter = Stereo3DDetAdapterDataset(root=str(mock_dataset_structure), split="train", imgsz=384, names=names)
        assert adapter is not None

        # Test case 2: Missing label file (should be handled gracefully by base dataset)
        label_file.unlink()  # Remove label file
        
        # Should not raise error for missing label file (base dataset handles it)
        adapter = Stereo3DDetAdapterDataset(root=str(mock_dataset_structure), split="train", imgsz=384, names=names)
        assert adapter is not None

        # Test case 3: All classes filtered out (only non-paper classes in labels)
        label_file = mock_dataset_structure / "labels" / "train" / "000000.txt"
        label_content = """1 0.5 0.5 0.1 0.1 0.5 0.1 1.5 1.7 3.9 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2
2 0.7 0.7 0.1 0.1 0.7 0.1 1.7 0.5 0.8 0.0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"""
        label_file.write_text(label_content)  # Only Van (1) and Truck (2), no paper classes
        
        # Should not raise error - all classes will be filtered out, but validation should pass
        # because there are no paper classes to validate against
        adapter = Stereo3DDetAdapterDataset(root=str(mock_dataset_structure), split="train", imgsz=384, names=names)
        assert adapter is not None

