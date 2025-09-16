from collections.abc import Generator
import numpy as np
import torch
from ultralytics.utils.patches import imread, imshow, imwrite
import cv2


def color_diff(frames: list[np.ndarray]) -> Generator[np.ndarray, None, None]:
    """Compute color differences between consecutive frames.

    Args:
        frames (list of np.ndarray): List of frames in HWC format.

    Returns:
        generator of np.ndarray: Generator yielding color difference images.
    """
    assert len(frames) >= 2, "At least two frames are required to compute color differences."
    for i in range(1, len(frames)):
        yield cv2.absdiff(frames[i], frames[i - 1])

def generate_color_diff_window_img(frames: list[np.ndarray], window_size: int) -> Generator[np.ndarray, None, None]:
    """Compute color differences over a sliding window of frames.

    Args:
        frames (list of np.ndarray): List of frames in HWC format.
        window_size (int): Size of the sliding window.

    Returns:
        generator of np.ndarray: Generator yielding color difference images in shape HWC. C = window_size.
    """
    assert len(frames) >= window_size, "Number of frames must be at least equal to the window size."
    # Padding by replicating the first frame
    padded_frames = [frames[0]] * (window_size - 1) + frames
    diff_frames = list(color_diff(padded_frames))

    # Ensure the each diff frame is one channel
    for df_f in diff_frames:
        if df_f.ndim == 3 and df_f.shape[2] == 3:
            df_f = cv2.cvtColor(df_f, cv2.COLOR_BGR2GRAY)
            df_f = np.expand_dims(df_f, axis=-1)

    results = []
    for i in range(window_size-1, len(diff_frames)):
        window = diff_frames[i:i + window_size]
        stacked = np.stack(window, axis=-1)
        yield stacked
    
        
if __name__ == "__main__":
    # test graph_diff function
    graph_1_path = "/root/autodl-tmp/Dataset_YOLO/images/val/game10_Clip10_0000.jpg"
    graph_2_path = "/root/autodl-tmp/Dataset_YOLO/images/val/game10_Clip10_0001.jpg"
    graph_3_path = "/root/autodl-tmp/Dataset_YOLO/images/val/game10_Clip10_0002.jpg"
    graph_4_path = "/root/autodl-tmp/Dataset_YOLO/images/val/game10_Clip10_0003.jpg"

    # convert to one channel
    g1 = imread(graph_1_path)
    g2 = imread(graph_2_path)
    g3 = imread(graph_3_path)
    g4 = imread(graph_4_path)

    # Create a list of images and filter out any that failed to load
    images = [g1, g2, g3, g4]
    valid_images = [img for img in images if img is not None]

    # Check if there are any valid images before proceeding
    if valid_images:
        # test color_diff
        diffs = color_diff(valid_images)
        for i, diff in enumerate(diffs):
            imwrite(f'motion_color_diff_{i}.jpg', diff)

    # test generate_color_diff_window_img
    window_size = 3
    if valid_images and len(valid_images) >= window_size:
        windowed_diffs = generate_color_diff_window_img(valid_images, window_size)
        for i, diff in enumerate(windowed_diffs):
            imwrite(f'motion_color_diff_window_{i}.jpg', diff)