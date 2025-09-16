import os
from typing import Any
from ultralytics.data import YOLODataset
from motion_utils import color_diff
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.patches import imread



class MotionDataset(YOLODataset):
    """A dataset class for motion-based data loading and processing."""

    def __init__(self, *args, data, task='pose', **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)
        self.windows_size = 3
        # step1: build the dataset structure
        # step2: generate the motion graphs for each image.
        # check the format in seriesA_seriesB_frameID.jpg
        for img_file in self.im_files:
            if not self._check_format(img_file):
                raise ValueError(f"Image file name {img_file} does not follow the expected format 'seriesA_seriesB_frameID.jpg'.")
        
        # organize the images into sequences based on seriesA and seriesB
        self.sequences = {}
        for img_file in self.im_files:
            base_name = os.path.basename(img_file)
            seriesA, seriesB, frameID_with_ext = base_name.split('_')
            frameID = os.path.splitext(frameID_with_ext)[0]
            series_key = f"{seriesA}_{seriesB}"
            if series_key not in self.sequences:
                self.sequences[series_key] = []
            self.sequences[series_key].append((int(frameID), img_file))
        
        # sort each sequence by frameID
        for series_key in self.sequences:
            self.sequences[series_key].sort(key=lambda x: x[0])
            self.sequences[series_key] = [img_file for _, img_file in self.sequences[series_key]]
        
        
        # generate motion graphs for each sequence
        self.motion_graphs = {}
        for series_key, img_files in self.sequences.items():
            if not img_files:
                continue

            # Create a padded list of image file paths using replication padding
            padded_files = [img_files[0]] * (self.windows_size - 1) + img_files

            # Generate windows and compute color differences
            for i in range(len(img_files)):
                window_paths = padded_files[i : i + self.windows_size]
                
                # Load images for the current window
                window_frames = [imread(p) for p in window_paths]
                valid_frames = [frame for frame in window_frames if frame is not None]

                # Compute the color difference for the window
                diffs = list(color_diff(valid_frames))
                # The primary motion graph for a frame is the sum of diffs in its window
                # We'll take the first diff as representative, but you could also sum them
                motion_graph = diffs[0] if diffs else None
                
                if motion_graph is not None:
                    # Associate the generated motion graph with the original image file path
                    original_img_path = img_files[i]
                    self.motion_graphs[original_img_path] = motion_graph

    
    def _check_format(self, img_file: str) -> bool:
        """Check if the image file name follows the expected format."""
        base_name = os.path.basename(img_file)
        parts = base_name.split('_')
        if len(parts) < 3:
            return False
        seriesA, seriesB, frameID_with_ext = parts[0], parts[1], parts[2]
        frameID = os.path.splitext(frameID_with_ext)[0]
        return seriesA.isalnum() and seriesB.isalnum() and frameID.isdigit()

if __name__ == "__main__":
    data = check_det_dataset("ultralytics/cfg/datasets/motion.yaml")
    dataset = MotionDataset(data=data, img_path="/root/autodl-tmp/Dataset_YOLO/images/train")
    print(f"Number of samples in the dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample data: {sample}")