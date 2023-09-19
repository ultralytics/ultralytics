# Ultralytics YOLO 🚀, AGPL-3.0 license

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

from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import LOGGER, is_colab, is_kaggle, ops
from ultralytics.utils.checks import check_requirements


@dataclass
class SourceTypes:
    webcam: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False


class LoadStreams:
    """YOLOv8 streamloader, i.e. `yolo predict source='rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`."""

    def __init__(self, sources='file.streams', imgsz=640, vid_stride=1, buffer=False):
        """Initialize instance variables and check for consistent input stream shapes."""
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = 'stream'
        self.imgsz = imgsz
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.sources = [ops.clean_str(x) for x in sources]  # clean source names for later
        self.imgs, self.fps, self.frames, self.threads, self.shape = [[]] * n, [0] * n, [0] * n, [None] * n, [[]] * n
        self.caps = [None] * n  # video capture objects
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
                # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/LNwODJXcvt4'
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            if s == 0 and (is_colab() or is_kaggle()):
                raise NotImplementedError("'source=0' webcam not supported in Colab and Kaggle notebooks. "
                                          "Try running 'source=0' in a local environment.")
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f'{st}Failed to open {s}')
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                'inf')  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                raise ConnectionError(f'{st}Failed to read images from {s}')
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # newline

        # Check for common shapes
        self.bs = self.__len__()

    def update(self, i, cap, stream):
        """Read stream `i` frames in daemon thread."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through YOLO image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns source paths, transformed and original images for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):

            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f'WARNING ⚠️ Waiting for stream {i}')

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, None, ''

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadScreenshots:
    """YOLOv8 screenshot dataloader, i.e. `yolo predict source=screen`."""

    def __init__(self, source, imgsz=640):
        """source = [screen_number left top width height] (pixels)."""
        check_requirements('mss')
        import mss  # noqa

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)
        self.imgsz = imgsz
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()
        self.bs = 1

        # Parse monitor shape
        monitor = self.sct.monitors[self.screen]
        self.top = monitor['top'] if top is None else (monitor['top'] + top)
        self.left = monitor['left'] if left is None else (monitor['left'] + left)
        self.width = width or monitor['width']
        self.height = height or monitor['height']
        self.monitor = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}

    def __iter__(self):
        """Returns an iterator of the object."""
        return self

    def __next__(self):
        """mss screen capture: get raw pixels from the screen as np array."""
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]  # BGRA to BGR
        s = f'screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: '

        self.frame += 1
        return [str(self.screen)], [im0], None, s  # screen, img, vid_cap, string


class LoadImages:
    """YOLOv8 image/video dataloader, i.e. `yolo predict source=image.jpg/vid.mp4`."""

    def __init__(self, path, imgsz=640, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if '*' in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, '*.*'))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.imgsz = imgsz
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = 1
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f'No images or videos found in {p}. '
                                    f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}')

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Return next image, path and metadata from dataset."""
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            success, im0 = self.cap.retrieve()
            while not success:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                success, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f'Image Not Found {path}')
            s = f'image {self.count}/{self.nf} {path}: '

        return [path], [im0], self.cap, s

    def _new_video(self, path):
        """Create a new video capture object."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of files in the object."""
        return self.nf  # number of files


class LoadPilAndNumpy:

    def __init__(self, im0, imgsz=640):
        """Initialize PIL and Numpy Dataloader."""
        if not isinstance(im0, list):
            im0 = [im0]
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]
        self.im0 = [self._single_check(im) for im in im0]
        self.imgsz = imgsz
        self.mode = 'image'
        # Generate fake paths
        self.bs = len(self.im0)

    @staticmethod
    def _single_check(im):
        """Validate and format an image to numpy array."""
        assert isinstance(im, (Image.Image, np.ndarray)), f'Expected PIL/np.ndarray image type, but got {type(im)}'
        if isinstance(im, Image.Image):
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = np.asarray(im)[:, :, ::-1]
            im = np.ascontiguousarray(im)  # contiguous
        return im

    def __len__(self):
        """Returns the length of the 'im0' attribute."""
        return len(self.im0)

    def __next__(self):
        """Returns batch paths, images, processed images, None, ''."""
        if self.count == 1:  # loop only once as it's batch inference
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __iter__(self):
        """Enables iteration for class LoadPilAndNumpy."""
        self.count = 0
        return self


class LoadTensor:

    def __init__(self, im0) -> None:
        self.im0 = self._single_check(im0)
        self.bs = self.im0.shape[0]
        self.mode = 'image'
        self.paths = [getattr(im, 'filename', f'image{i}.jpg') for i, im in enumerate(im0)]

    @staticmethod
    def _single_check(im, stride=32):
        """Validate and format an image to torch.Tensor."""
        s = f'WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) ' \
            f'divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.'
        if len(im.shape) != 4:
            if len(im.shape) != 3:
                raise ValueError(s)
            LOGGER.warning(s)
            im = im.unsqueeze(0)
        if im.shape[2] % stride or im.shape[3] % stride:
            raise ValueError(s)
        if im.max() > 1.0:
            LOGGER.warning(f'WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. '
                           f'Dividing input by 255.')
            im = im.float() / 255.0

        return im

    def __iter__(self):
        """Returns an iterator object."""
        self.count = 0
        return self

    def __next__(self):
        """Return next item in the iterator."""
        if self.count == 1:
            raise StopIteration
        self.count += 1
        return self.paths, self.im0, None, ''

    def __len__(self):
        """Returns the batch size."""
        return self.bs


def autocast_list(source):
    """
    Merges a list of source of different types into a list of numpy arrays or PIL images
    """
    files = []
    for im in source:
        if isinstance(im, (str, Path)):  # filename or uri
            files.append(Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im))
        elif isinstance(im, (Image.Image, np.ndarray)):  # PIL or np Image
            files.append(im)
        else:
            raise TypeError(f'type {type(im).__name__} is not a supported Ultralytics prediction source type. \n'
                            f'See https://docs.ultralytics.com/modes/predict for supported source types.')

    return files


LOADERS = LoadStreams, LoadPilAndNumpy, LoadImages, LoadScreenshots  # tuple


def get_best_youtube_url(url, use_pafy=False):
    """
    Retrieves the URL of the best quality MP4 video stream from a given YouTube video.

    This function uses the pafy or yt_dlp library to extract the video info from YouTube. It then finds the highest
    quality MP4 format that has video codec but no audio codec, and returns the URL of this video stream.

    Args:
        url (str): The URL of the YouTube video.
        use_pafy (bool): Use the pafy package, default=True, otherwise use yt_dlp package.

    Returns:
        (str): The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    if use_pafy:
        check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        import pafy  # noqa
        return pafy.new(url).getbestvideo(preftype='mp4').url
    else:
        check_requirements('yt-dlp')
        import yt_dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # extract info
        for f in reversed(info_dict.get('formats', [])):  # reversed because best is usually last
            # Find a format with video codec, no audio, *.mp4 extension at least 1920x1080 size
            good_size = (f.get('width') or 0) >= 1920 or (f.get('height') or 0) >= 1080
            if good_size and f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4':
                return f.get('url')
