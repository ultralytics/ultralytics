from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """
    A class to store video information, including width, height, fps and
        total number of frames.

    Attributes:
        width (int): width of the video in pixels
        height (int): height of the video in pixels
        fps (int): frames per second of the video
        total_frames (int, optional): total number of frames in the video,
            default is None

    Examples:
        ```python
        >>> import supervision as sv

        >>> video_info = sv.VideoInfo.from_video_path(video_path='video.mp4')

        >>> video_info
        VideoInfo(width=3840, height=2160, fps=25, total_frames=538)

        >>> video_info.resolution_wh
        (3840, 2160)
        ```
    """

    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution_wh(self) -> Tuple[int, int]:
        return self.width, self.height


class VideoSink:
    """
    Context manager that saves video frames to a file using OpenCV.

    Attributes:
        target_path (str): The path to the output file where the video will be saved.
        video_info (VideoInfo): Information about the video resolution, fps,
            and total frame count.
        codec (str): FOURCC code for video format

    Example:
        ```python
        >>> import supervision as sv

        >>> video_info = sv.VideoInfo.from_video_path('source.mp4')
        >>> frames_generator = get_video_frames_generator('source.mp4')

        >>> with sv.VideoSink(target_path='target.mp4', video_info=video_info) as sink:
        ...     for frame in frames_generator:
        ...         sink.write_frame(frame=frame)
        ```
    """

    def __init__(self, target_path: str, video_info: VideoInfo, codec: str = "mp4v"):
        self.target_path = target_path
        self.video_info = video_info
        self.__codec = codec
        self.__writer = None

    def __enter__(self):
        try:
            self.__fourcc = cv2.VideoWriter_fourcc(*self.__codec)
        except TypeError as e:
            print(str(e) + ". Defaulting to mp4v...")
            self.__fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.__writer = cv2.VideoWriter(
            self.target_path,
            self.__fourcc,
            self.video_info.fps,
            self.video_info.resolution_wh,
        )
        return self

    def write_frame(self, frame: np.ndarray):
        self.__writer.write(frame)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__writer.release()


def _validate_and_setup_video(source_path: str, start: int, end: Optional[int]):
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is not None and end > total_frames:
        raise Exception("Requested frames are outbound")
    start = max(start, 0)
    end = min(end, total_frames) if end is not None else total_frames
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    return video, start, end


def get_video_frames_generator(
    source_path: str, stride: int = 1, start: int = 0, end: Optional[int] = None
) -> Generator[np.ndarray, None, None]:
    """
    Get a generator that yields the frames of the video.

    Args:
        source_path (str): The path of the video file.
        stride (int): Indicates the interval at which frames are returned,
            skipping stride - 1 frames between each.
        start (int): Indicates the starting position from which
            video should generate frames
        end (Optional[int]): Indicates the ending position at which video
            should stop generating frames. If None, video will be read to the end.

    Returns:
        (Generator[np.ndarray, None, None]): A generator that yields the
            frames of the video.

    Examples:
        ```python
        >>> import supervision as sv

        >>> for frame in sv.get_video_frames_generator(source_path='source_video.mp4'):
        ...     ...
        ```
    """
    video, start, end = _validate_and_setup_video(source_path, start, end)
    frame_position = start
    while True:
        success, frame = video.read()
        if not success or frame_position >= end:
            break
        yield frame
        for _ in range(stride - 1):
            success = video.grab()
            if not success:
                break
        frame_position += stride
    video.release()


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """
    Process a video file by applying a callback function on each frame
        and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in
            a numpy ndarray representation of a video frame and an
            int index of the frame and returns a processed numpy ndarray
            representation of the frame.

    Examples:
        ```python
        >>> import supervision as sv

        >>> def callback(scene: np.ndarray, index: int) -> np.ndarray:
        ...     ...

        >>> process_video(
        ...     source_path='...',
        ...     target_path='...',
        ...     callback=callback
        ... )
        ```
    """
    source_video_info = VideoInfo.from_video_path(video_path=source_path)
    with VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            get_video_frames_generator(source_path=source_path)
        ):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)


class FPSMonitor:
    """
    A class for monitoring frames per second (FPS) to benchmark latency.
    """

    def __init__(self, sample_size: int = 30):
        """
        Args:
            sample_size (int): The maximum number of observations for latency
                benchmarking.

        Examples:
            ```python
            >>> import supervision as sv

            >>> frames_generator = sv.get_video_frames_generator('source.mp4')
            >>> fps_monitor = sv.FPSMonitor()

            >>> for frame in frames_generator:
            ...     # your processing code here
            ...     fps_monitor.tick()
            ...     fps = fps_monitor()
            ```
        """
        self.all_timestamps = deque(maxlen=sample_size)

    def __call__(self) -> float:
        """
        Computes and returns the average FPS based on the stored time stamps.

        Returns:
            float: The average FPS. Returns 0.0 if no time stamps are stored.
        """

        if not self.all_timestamps:
            return 0.0
        taken_time = self.all_timestamps[-1] - self.all_timestamps[0]
        return (len(self.all_timestamps)) / taken_time if taken_time != 0 else 0.0

    def tick(self) -> None:
        """
        Adds a new time stamp to the deque for FPS calculation.
        """
        self.all_timestamps.append(time.monotonic())

    def reset(self) -> None:
        """
        Clears all the time stamps from the deque.
        """
        self.all_timestamps.clear()
