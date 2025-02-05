# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread


@dataclass
class SourceTypes:
    """
    Class to represent various types of input sources for predictions.

    This class uses dataclass to define boolean flags for different types of input sources that can be used for
    making predictions with YOLO models.

    Attributes:
        stream (bool): Flag indicating if the input source is a video stream.
        screenshot (bool): Flag indicating if the input source is a screenshot.
        from_img (bool): Flag indicating if the input source is an image file.

    Examples:
        >>> source_types = SourceTypes(stream=True, screenshot=False, from_img=False)
        >>> print(source_types.stream)
        True
        >>> print(source_types.from_img)
        False
    """

    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """
    Stream Loader for various types of video streams.

    Supports RTSP, RTMP, HTTP, and TCP streams. This class handles the loading and processing of multiple video
    streams simultaneously, making it suitable for real-time video analysis tasks.

    Attributes:
        sources (List[str]): The source input paths or URLs for the video streams.
        vid_stride (int): Video frame-rate stride.
        buffer (bool): Whether to buffer input streams.
        running (bool): Flag to indicate if the streaming thread is running.
        mode (str): Set to 'stream' indicating real-time capture.
        imgs (List[List[np.ndarray]]): List of image frames for each stream.
        fps (List[float]): List of FPS for each stream.
        frames (List[int]): List of total frames for each stream.
        threads (List[Thread]): List of threads for each stream.
        shape (List[Tuple[int, int, int]]): List of shapes for each stream.
        caps (List[cv2.VideoCapture]): List of cv2.VideoCapture objects for each stream.
        bs (int): Batch size for processing.

    Methods:
        update: Read stream frames in daemon thread.
        close: Close stream loader and release resources.
        __iter__: Returns an iterator object for the class.
        __next__: Returns source paths, transformed, and original images for processing.
        __len__: Return the length of the sources object.

    Examples:
        >>> stream_loader = LoadStreams("rtsp://example.com/stream1.mp4")
        >>> for sources, imgs, _ in stream_loader:
        ...     # Process the images
        ...     pass
        >>> stream_loader.close()

    Notes:
        - The class uses threading to efficiently load frames from multiple streams simultaneously.
        - It automatically handles YouTube links, converting them to the best available stream URL.
        - The class implements a buffer system to manage frame storage and retrieval.
    """

    def __init__(self, sources="file.streams", vid_stride=1, buffer=False):
        """Initialize stream loader for multiple video sources, supporting various stream types."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f"{i + 1}/{n}: {s}... "
            if urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Jsn8D3aC840' or 'https://youtu.be/Jsn8D3aC840'
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{st}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f"{st}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

    def update(self, i, cap, stream):
        """Read stream frames in daemon thread and update image buffer."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Terminates stream loader, stops threads, and releases video capture resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f"WARNING ⚠️ Could not release VideoCapture object: {e}")
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns the next batch of frames from multiple video streams for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord("q"):  # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"WARNING ⚠️ Waiting for stream {i}")

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self):
        """Return the number of video streams in the LoadStreams object."""
        return self.bs  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    """
    Ultralytics screenshot dataloader for capturing and processing screen images.

    This class manages the loading of screenshot images for processing with YOLO. It is suitable for use with
    `yolo predict source=screen`.

    Attributes:
        source (str): The source input indicating which screen to capture.
        screen (int): The screen number to capture.
        left (int): The left coordinate for screen capture area.
        top (int): The top coordinate for screen capture area.
        width (int): The width of the screen capture area.
        height (int): The height of the screen capture area.
        mode (str): Set to 'stream' indicating real-time capture.
        frame (int): Counter for captured frames.
        sct (mss.mss): Screen capture object from `mss` library.
        bs (int): Batch size, set to 1.
        fps (int): Frames per second, set to 30.
        monitor (Dict[str, int]): Monitor configuration details.

    Methods:
        __iter__: Returns an iterator object.
        __next__: Captures the next screenshot and returns it.

    Examples:
        >>> loader = LoadScreenshots("0 100 100 640 480")  # screen 0, top-left (100,100), 640x480
        >>> for source, im, im0s, vid_cap, s in loader:
        ...     print(f"Captured frame: {im.shape}")
    """

    def __init__(self, source):
        """Initialize screenshot capture with specified screen and region parameters."""
        check_requirements("mss")
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.mode = "stream"
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1
        self.fps = 30

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        """Yields the next screenshot image from the specified screen or region for processing."""
        return self

    def __next__(self):
        """Captures and returns the next screenshot as a numpy array using the mss library."""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA to BGR
        s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        self.frame += 1
        return [str(self.screen)], [im0], [s]  # screen, img, string


class LoadImagesAndVideos:
    """
    A class for loading and processing images and videos for YOLO object detection.

    This class manages the loading and pre-processing of image and video data from various sources, including
    single image files, video files, and lists of image and video paths.

    Attributes:
        files (List[str]): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (List[bool]): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate.
        bs (int): Batch size.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during __iter__().
        ni (int): Number of images.

    Methods:
        __init__: Initialize the LoadImagesAndVideos object.
        __iter__: Returns an iterator object for VideoStream or ImageFolder.
        __next__: Returns the next batch of images or video frames along with their paths and metadata.
        _new_video: Creates a new video capture object for the given path.
        __len__: Returns the number of batches in the object.

    Examples:
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # Process batch of images or video frames
        ...     pass

    Notes:
        - Supports various image formats including HEIC.
        - Handles both local files and directories.
        - Can read from a text file containing paths to images and videos.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize dataloader for images and videos, supporting various input formats."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images or videos
        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS:
                images.append(f)
            elif suffix in VID_FORMATS:
                videos.append(f)
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"  # default to video if no images
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """Iterates through image/video files, yielding source paths, images, and metadata."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if imgs:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                success = False
                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                # Handle image files (including HEIC)
                self.mode = "image"
                if path.split(".")[-1].lower() == "heic":
                    # Load HEIC image using Pillow with pillow-heif
                    check_requirements("pillow-heif")

                    from pillow_heif import register_heif_opener

                    register_heif_opener()  # Register HEIF opener with Pillow
                    with Image.open(path) as img:
                        im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # convert image to BGR nparray
                else:
                    im0 = imread(path)  # BGR
                if im0 is None:
                    LOGGER.warning(f"WARNING ⚠️ Image Read Error {path}")
                else:
                    paths.append(path)
                    imgs.append(im0)
                    info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info

    def _new_video(self, path):
        """Creates a new video capture object for the given path and initializes video-related attributes."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of files (images and videos) in the dataset."""
        return math.ceil(self.nf / self.bs)  # number of batches


class LoadPilAndNumpy:
    """
    Load images from PIL and Numpy arrays for batch processing.

    This class manages loading and pre-processing of image data from both PIL and Numpy formats. It performs basic
    validation and format conversion to ensure that the images are in the required format for downstream processing.

    Attributes:
        paths (List[str]): List of image paths or autogenerated filenames.
        im0 (List[np.ndarray]): List of images stored as Numpy arrays.
        mode (str): Type of data being processed, set to 'image'.
        bs (int): Batch size, equivalent to the length of `im0`.

    Methods:
        _single_check: Validate and format a single image to a Numpy array.

    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> pil_img = Image.new("RGB", (100, 100))
        >>> np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> loader = LoadPilAndNumpy([pil_img, np_img])
        >>> paths, images, _ = next(iter(loader))
        >>> print(f"Loaded {len(images)} images")
        Loaded 2 images
    """

    def __init__(self, im0):
        """Initializes a loader for PIL and Numpy images, converting inputs to a standardized format."""
        if not isinstance(im0, list):
            im0 = [im0]
        # use `image{i}.jpg` when Image.filename returns an empty path.
        self.paths = [getattr(im, "filename", "") or f"image{i}.jpg" for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.mode = "image"
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array, ensuring RGB order and contiguous memory."""
        assert isinstance(im, (Image.Image, np.ndarray)), f"Expected PIL/np.ndarray image type, but got {type(im)}"
        if isinstance(im, Image.Image):
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute, representing the number of loaded images."""
        return len(self.im0)

    def __next__(self):
        """Returns the next batch of images, paths, and metadata for processing."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __iter__(self):
        """Iterates through PIL/numpy images, yielding paths, raw images, and metadata for processing."""
        self.count = 0
        return self


class LoadTensor:
    """
    A class for loading and processing tensor data for object detection tasks.

    This class handles the loading and pre-processing of image data from PyTorch tensors, preparing them for
    further processing in object detection pipelines.

    Attributes:
        im0 (torch.Tensor): The input tensor containing the image(s) with shape (B, C, H, W).
        bs (int): Batch size, inferred from the shape of `im0`.
        mode (str): Current processing mode, set to 'image'.
        paths (List[str]): List of image paths or auto-generated filenames.

    Methods:
        _single_check: Validates and formats an input tensor.

    Examples:
        >>> import torch
        >>> tensor = torch.rand(1, 3, 640, 640)
        >>> loader = LoadTensor(tensor)
        >>> paths, images, info = next(iter(loader))
        >>> print(f"Processed {len(images)} images")
    """

    def __init__(self, im0) -> None:
        """Initialize LoadTensor object for processing torch.Tensor image data."""
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = "image"
        self.paths = [getattr(im, "filename", f"image{i}.jpg") for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """Validates and formats a single image tensor, ensuring correct shape and normalization."""
        s = (
            f"WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) "
            f"divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible."
        )
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0 + torch.finfo(im.dtype).eps:  # torch.float32 eps is 1.2e-07
            LOGGER.warning(
                f"WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. "
                f"Dividing input by 255."
            )
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """Yields an iterator object for iterating through tensor image data."""
        self.count = 0
        return self

    def __next__(self):
        """Yields the next batch of tensor images and metadata for processing."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, [""] * self.bs

    def __len__(self):
        """Returns the batch size of the tensor input."""
        return self.bs


def autocast_list(source):
    """Merges a list of sources into a list of numpy arrays or PIL images for Ultralytics prediction."""
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(
                f"type {type(im).__name__} is not a supported Ultralytics prediction source type. \n"
                f"See https://docs.ultralytics.com/modes/predict for supported source types."
            )

    return files


def get_best_youtube_url(url, method="pytube"):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        method (str): The method to use for extracting video info. Options are "pytube", "pafy", and "yt-dlp".
            Defaults to "pytube".

    Returns:
        (str | None): The URL of the best quality MP4 video stream, or None if no suitable stream is found.

    Examples:
        >>> url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        >>> best_url = get_best_youtube_url(url)
        >>> print(best_url)
        https://rr4---sn-q4flrnek.googlevideo.com/videoplayback?expire=...

    Notes:
        - Requires additional libraries based on the chosen method: pytubefix, pafy, or yt-dlp.
        - The function prioritizes streams with at least 1080p resolution when available.
        - For the "yt-dlp" method, it looks for formats with video codec, no audio, and *.mp4 extension.
    """
    if method == "pytube":
        # Switched from pytube to pytubefix to resolve https://github.com/pytube/pytube/issues/1954
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube

        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)  # sort streams by resolution
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:  # check if resolution is at least 1080p
                return stream.url

    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy  # noqa

        return pafy.new(url).getbestvideo(preftype="mp4").url

    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in reversed(info_dict.get("formats", [])):  # reversed because best is usually last
            # Find a format with video codec, no audio, *.mp4 extension at least 1920x1080 size
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")


# Define constants
LOADERS = (LoadStreams, LoadPilAndNumpy, LoadImagesAndVideos, LoadScreenshots)
