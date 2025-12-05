"""Unit tests for dataset max_samples parameter functionality."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ultralytics.data.kitti_stereo import KITTIStereoDataset
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset


@pytest.fixture
def mock_dataset_structure(tmp_path):
    """Create a mock KITTI dataset structure with 20 samples."""
    root = Path(tmp_path) / "kitti_dataset"
    
    # Create directory structure
    for split in ["train", "val"]:
        (root / "images" / split / "left").mkdir(parents=True, exist_ok=True)
        (root / "images" / split / "right").mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "calib" / split).mkdir(parents=True, exist_ok=True)
    
    # Create 20 sample images and labels
    for i in range(20):
        image_id = f"{i:06d}"
        
        # Create dummy images (100x100 RGB)
        left_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        cv2.imwrite(str(root / "images" / "train" / "left" / f"{image_id}.png"), left_img)
        cv2.imwrite(str(root / "images" / "train" / "right" / f"{image_id}.png"), right_img)
        
        # Create dummy label file (YOLO 3D format: class_id, left_box, right_box, dims, alpha, vertices)
        # Use original KITTI class IDs (0=Car, 3=Pedestrian, 5=Cyclist) to ensure all 3 classes are represented
        # These will be filtered and remapped to paper IDs (0, 1, 2) by the dataset
        original_class_ids = [0, 3, 5]  # Car, Pedestrian, Cyclist in original KITTI format
        class_id = original_class_ids[i % 3]  # Cycle through 0 (Car), 3 (Pedestrian), 5 (Cyclist)
        label_file = root / "labels" / "train" / f"{image_id}.txt"
        with open(label_file, "w") as f:
            f.write(f"{class_id} 100 100 50 50 100 50 1.5 1.5 3.0 0.0 10 10 20 10 30 20 40 20\n")
        
        # Create dummy calibration file
        calib_file = root / "calib" / "train" / f"{image_id}.txt"
        with open(calib_file, "w") as f:
            f.write("fx: 721.5377\n")
            f.write("fy: 721.5377\n")
            f.write("cx: 609.5593\n")
            f.write("cy: 172.8540\n")
            f.write("baseline: 0.54\n")
    
    return root


class TestKITTIStereoDatasetMaxSamples:
    """Test max_samples parameter in KITTIStereoDataset."""
    
    def test_kitti_dataset_respects_max_samples(self, mock_dataset_structure):
        """Test that KITTIStereoDataset limits to max_samples when specified (T182)."""
        dataset = KITTIStereoDataset(
            root=mock_dataset_structure,
            split="train",
            max_samples=10
        )
        
        assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
        assert len(dataset.image_ids) == 10, f"Expected 10 image_ids, got {len(dataset.image_ids)}"
        
        # Verify we can access all 10 samples
        for i in range(10):
            sample = dataset[i]
            assert "left_img" in sample
            assert "right_img" in sample
            assert "labels" in sample
            assert "calib" in sample
    
    def test_max_samples_none_no_limit(self, mock_dataset_structure):
        """Test that max_samples=None loads all samples (backward compatibility) (T184)."""
        dataset = KITTIStereoDataset(
            root=mock_dataset_structure,
            split="train",
            max_samples=None
        )
        
        assert len(dataset) == 20, f"Expected all 20 samples, got {len(dataset)}"
        assert len(dataset.image_ids) == 20, f"Expected 20 image_ids, got {len(dataset.image_ids)}"
    
    def test_max_samples_validation(self, mock_dataset_structure):
        """Test that max_samples <= 0 raises ValueError (T185)."""
        with pytest.raises(ValueError, match="max_samples must be > 0"):
            KITTIStereoDataset(
                root=mock_dataset_structure,
                split="train",
                max_samples=0
            )
        
        with pytest.raises(ValueError, match="max_samples must be > 0"):
            KITTIStereoDataset(
                root=mock_dataset_structure,
                split="train",
                max_samples=-1
            )
    
    def test_max_samples_greater_than_total(self, mock_dataset_structure):
        """Test that max_samples > len(dataset) works correctly (loads all available) (T186)."""
        dataset = KITTIStereoDataset(
            root=mock_dataset_structure,
            split="train",
            max_samples=100  # More than available 20 samples
        )
        
        assert len(dataset) == 20, f"Expected all 20 samples, got {len(dataset)}"
        assert len(dataset.image_ids) == 20, f"Expected 20 image_ids, got {len(dataset.image_ids)}"


class TestStereo3DDetAdapterDatasetMaxSamples:
    """Test max_samples parameter in Stereo3DDetAdapterDataset."""
    
    def test_adapter_dataset_respects_max_samples(self, mock_dataset_structure):
        """Test that Stereo3DDetAdapterDataset limits to max_samples when specified (T183)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        dataset = Stereo3DDetAdapterDataset(
            root=mock_dataset_structure,
            split="train",
            imgsz=384,
            names=names,
            max_samples=5
        )
        
        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
        assert len(dataset.base.image_ids) == 5, f"Expected 5 image_ids in base dataset, got {len(dataset.base.image_ids)}"
        
        # Verify we can access all 5 samples
        for i in range(5):
            sample = dataset[i]
            assert "img" in sample
            assert "labels" in sample  # Dataset returns 'labels', not 'targets'
            assert sample["img"].shape[0] == 6  # 6-channel input
    
    def test_adapter_max_samples_none_no_limit(self, mock_dataset_structure):
        """Test that max_samples=None loads all samples in adapter dataset (T184)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        dataset = Stereo3DDetAdapterDataset(
            root=mock_dataset_structure,
            split="train",
            imgsz=384,
            names=names,
            max_samples=None
        )
        
        assert len(dataset) == 20, f"Expected all 20 samples, got {len(dataset)}"
        assert len(dataset.base.image_ids) == 20, f"Expected 20 image_ids, got {len(dataset.base.image_ids)}"


class TestDataLoaderIntegration:
    """Test DataLoader integration with limited dataset."""
    
    def test_dataloader_with_limited_dataset(self, mock_dataset_structure):
        """Test that DataLoader works correctly with limited dataset (T187)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        dataset = Stereo3DDetAdapterDataset(
            root=mock_dataset_structure,
            split="train",
            imgsz=384,
            names=names,
            max_samples=8
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Verify we can iterate through all batches
        total_samples = 0
        for batch in dataloader:
            assert "img" in batch
            assert batch["img"].shape[0] <= 2  # Batch size
            total_samples += batch["img"].shape[0]
        
        assert total_samples == 8, f"Expected 8 samples total, got {total_samples}"
    
    def test_dataloader_with_max_samples_none(self, mock_dataset_structure):
        """Test that DataLoader works with max_samples=None (all samples)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        dataset = Stereo3DDetAdapterDataset(
            root=mock_dataset_structure,
            split="train",
            imgsz=384,
            names=names,
            max_samples=None
        )
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        total_samples = 0
        for batch in dataloader:
            total_samples += batch["img"].shape[0]
        
        assert total_samples == 20, f"Expected 20 samples total, got {total_samples}"

