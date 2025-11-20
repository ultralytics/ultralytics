# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import contextlib
import os
import queue
import re
import time
from threading import Condition, get_ident, Lock, Thread

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image

from sam3.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

IS_MAIN_PROCESS = os.getenv("IS_MAIN_PROCESS", "1") == "1"
RANK = int(os.getenv("RANK", "0"))

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def load_resource_as_video_frames(
    resource_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="cv2",
):
    """
    Load video frames from either a video or an image (as a single-frame video).
    Alternatively, if input is a list of PIL images, convert its format
    """
    if isinstance(resource_path, list):
        img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        assert all(isinstance(img_pil, Image.Image) for img_pil in resource_path)
        assert len(resource_path) is not None
        orig_height, orig_width = resource_path[0].size
        orig_height, orig_width = (
            orig_width,
            orig_height,
        )  # For some reason, this method returns these swapped
        images = []
        for img_pil in resource_path:
            img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
            assert img_np.dtype == np.uint8, "np.uint8 is expected for JPEG images"
            img_np = img_np / 255.0
            img = torch.from_numpy(img_np).permute(2, 0, 1)
            # float16 precision should be sufficient for image tensor storage
            img = img.to(dtype=torch.float16)
            # normalize by mean and std
            img -= img_mean
            img /= img_std
            images.append(img)
        images = torch.stack(images)
        if not offload_video_to_cpu:
            images = images.cuda()
        return images, orig_height, orig_width

    is_image = (
        isinstance(resource_path, str)
        and os.path.splitext(resource_path)[-1].lower() in IMAGE_EXTS
    )
    if is_image:
        return load_image_as_single_frame_video(
            image_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    else:
        return load_video_frames(
            video_path=resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )


def load_image_as_single_frame_video(
    image_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    video_loader_type="cv2",
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    assert isinstance(video_path, str)
    if video_path.startswith("<load-dummy-video"):
        # Check for pattern <load-dummy-video-N> where N is an integer
        match = re.match(r"<load-dummy-video-(\d+)>", video_path)
        num_frames = int(match.group(1)) if match else 60
        return load_dummy_video(image_size, offload_video_to_cpu, num_frames=num_frames)
    elif os.path.isdir(video_path):
        return load_video_frames_from_image_folder(
            image_folder=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )
    elif os.path.splitext(video_path)[-1].lower() in VIDEO_EXTS:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
    else:
        raise NotImplementedError("Only video files and image folders are supported")


def load_video_frames_from_image_folder(
    image_folder,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
):
    """
    Load the video frames from a directory of image files ("<frame_index>.<img_ext>" format)
    """
    frame_names = [
        p
        for p in os.listdir(image_folder)
        if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
    ]
    try:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.<img_ext>"
        logger.warning(
            f'frame names are not in "<frame_index>.<img_ext>" format: {frame_names[:5]=}, '
            f"falling back to lexicographic sort."
        )
        frame_names.sort()
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {image_folder}")
    img_paths = [os.path.join(image_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncImageFrameLoader(
            img_paths, image_size, offload_video_to_cpu, img_mean, img_std
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # float16 precision should be sufficient for image tensor storage
    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
    video_height, video_width = None, None
    for n, img_path in enumerate(
        tqdm(img_paths, desc=f"frame loading (image folder) [rank={RANK}]")
    ):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean,
    img_std,
    async_loading_frames,
    gpu_acceleration=False,
    gpu_device=None,
    video_loader_type="cv2",
):
    """Load the video frames from a video file."""
    if video_loader_type == "cv2":
        return load_video_frames_from_video_file_using_cv2(
            video_path=video_path,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu,
        )
    elif video_loader_type == "torchcodec":
        logger.info("Using torchcodec to load video file")
        lazy_images = AsyncVideoFileLoaderWithTorchCodec(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            gpu_acceleration=gpu_acceleration,
            gpu_device=gpu_device,
        )
        # The `AsyncVideoFileLoaderWithTorchCodec` class always loads the videos asynchronously,
        # so we just wait for its loading thread to finish if async_loading_frames=False.
        if not async_loading_frames:
            async_thread = lazy_images.thread
            if async_thread is not None:
                async_thread.join()
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    else:
        raise RuntimeError("video_loader_type must be either 'cv2' or 'torchcodec'")


def load_video_frames_from_video_file_using_cv2(
    video_path: str,
    image_size: int,
    img_mean: tuple = (0.5, 0.5, 0.5),
    img_std: tuple = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False,
) -> torch.Tensor:
    """
    Load video from path, convert to normalized tensor with specified preprocessing

    Args:
        video_path: Path to video file
        image_size: Target size for square frames (height and width)
        img_mean: Normalization mean (RGB)
        img_std: Normalization standard deviation (RGB)

    Returns:
        torch.Tensor: Preprocessed video tensor in shape (T, C, H, W) with float16 dtype
    """
    import cv2  # delay OpenCV import to avoid unnecessary dependency

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = num_frames if num_frames > 0 else None

    frames = []
    pbar = tqdm(desc=f"frame loading (OpenCV) [rank={RANK}]", total=num_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC
        )
        frames.append(frame_resized)
        pbar.update(1)
    cap.release()
    pbar.close()

    # Convert to tensor
    frames_np = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, C)
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

    img_mean = torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std, dtype=torch.float16).view(1, 3, 1, 1)
    if not offload_video_to_cpu:
        video_tensor = video_tensor.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    video_tensor -= img_mean
    video_tensor /= img_std
    return video_tensor, original_height, original_width


def load_dummy_video(image_size, offload_video_to_cpu, num_frames=60):
    """
    Load a dummy video with random frames for testing and compilation warmup purposes.
    """
    video_height, video_width = 480, 640  # dummy original video sizes
    images = torch.randn(num_frames, 3, image_size, image_size, dtype=torch.float16)
    if not offload_video_to_cpu:
        images = images.cuda()
    return images, video_height, video_width


def _load_img_as_tensor(img_path, image_size):
    """Load and resize an image and convert it into a PyTorch tensor."""
    img = Image.open(img_path).convert("RGB")
    orig_width, orig_height = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    img = TF.to_tensor(img)
    return img, orig_height, orig_width


class AsyncImageFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(
                    range(len(self.images)),
                    desc=f"frame loading (image folder) [rank={RANK}]",
                ):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # float16 precision should be sufficient for image tensor storage
        img = img.to(dtype=torch.float16)
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.cuda()
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


class TorchCodecDecoder:
    """
    A wrapper to support GPU device and num_threads in TorchCodec decoder,
    which are not supported by `torchcodec.decoders.SimpleVideoDecoder` yet.
    """

    def __init__(self, source, dimension_order="NCHW", device="cpu", num_threads=1):
        from torchcodec import _core as core

        self._source = source  # hold a reference to the source to prevent it from GC
        if isinstance(source, str):
            self._decoder = core.create_from_file(source, "exact")
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source, "exact")
        else:
            raise TypeError(f"Unknown source type: {type(source)}.")
        assert dimension_order in ("NCHW", "NHWC")

        device_string = str(device)
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(
            self._decoder,
            dimension_order=dimension_order,
            device=device_string,
            num_threads=(1 if "cuda" in device_string else num_threads),
        )
        video_metadata = core.get_container_metadata(self._decoder)
        best_stream_index = video_metadata.best_video_stream_index
        assert best_stream_index is not None
        self.metadata = video_metadata.streams[best_stream_index]
        assert self.metadata.num_frames_from_content is not None
        self._num_frames = self.metadata.num_frames_from_content

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key: int):
        from torchcodec import _core as core

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )
        frame_data, *_ = core.get_frame_at_index(
            self._decoder,
            frame_index=key,
        )
        return frame_data


class FIFOLock:
    """A lock that ensures FIFO ordering of lock acquisitions."""

    def __init__(self):
        self._lock = Lock()
        self._waiters = queue.Queue()
        self._condition = Condition()

    def acquire(self):
        ident = get_ident()
        with self._condition:
            self._waiters.put(ident)
            while self._waiters.queue[0] != ident or not self._lock.acquire(
                blocking=False
            ):
                self._condition.wait()
                # got the lock and it's our turn

    def release(self):
        with self._condition:
            self._lock.release()
            self._waiters.get()
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()

    def __exit__(self, t, v, tb):
        self.release()


class AsyncVideoFileLoaderWithTorchCodec:
    """
    Loading frames from video files asynchronously without blocking session start.

    Unlike `AsyncVideoFileLoader`, this class uses PyTorch's offical TorchCodec library
    for video decoding, which is more efficient and supports more video formats.
    """

    def __init__(
        self,
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        gpu_acceleration=True,
        gpu_device=None,
        use_rand_seek_in_loading=False,
    ):
        # Check and possibly infer the output device (and also get its GPU id when applicable)
        assert gpu_device is None or gpu_device.type == "cuda"
        gpu_id = (
            gpu_device.index
            if gpu_device is not None and gpu_device.index is not None
            else torch.cuda.current_device()
        )
        if offload_video_to_cpu:
            out_device = torch.device("cpu")
        else:
            out_device = torch.device("cuda") if gpu_device is None else gpu_device
        self.out_device = out_device
        self.gpu_acceleration = gpu_acceleration
        self.gpu_id = gpu_id
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        if not isinstance(img_mean, torch.Tensor):
            img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_mean = img_mean
        if not isinstance(img_std, torch.Tensor):
            img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self.img_std = img_std

        if gpu_acceleration:
            self.img_mean = self.img_mean.to(f"cuda:{self.gpu_id}")
            self.img_std = self.img_std.to(f"cuda:{self.gpu_id}")
            decoder_option = {"device": f"cuda:{self.gpu_id}"}
        else:
            self.img_mean = self.img_mean.cpu()
            self.img_std = self.img_std.cpu()
            decoder_option = {"num_threads": 1}  # use a single thread to save memory

        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.async_reader = TorchCodecDecoder(video_path, **decoder_option)

        # `num_frames_from_content` is the true number of frames in the video content
        # from the scan operation (rather than from the metadata, which could be wrong)
        self.num_frames = self.async_reader.metadata.num_frames_from_content
        self.video_height = self.async_reader.metadata.height
        self.video_width = self.async_reader.metadata.width

        # items in `self._images` will be loaded asynchronously
        self.images_loaded = [False] * self.num_frames
        self.images = torch.zeros(
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
            dtype=torch.float16,
            device=self.out_device,
        )
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        self.use_rand_seek_in_loading = use_rand_seek_in_loading
        self.rand_seek_idx_queue = queue.Queue()
        # use a lock to avoid race condition between concurrent access to torchcodec
        # libs (which are not thread-safe); the lock is replaced with a nullcontext
        # when the video is fully loaded
        self.torchcodec_access_lock = FIFOLock()
        self._start_video_loading()

    def _load_one_frame(self, idx):
        frame_resized = self._transform_frame(self.async_reader[idx])
        return frame_resized

    @torch.inference_mode()
    def _start_video_loading(self):
        desc = f"frame loading (TorchCodec w/ {'GPU' if self.gpu_acceleration else 'CPU'}) [rank={RANK}]"
        pbar = tqdm(desc=desc, total=self.num_frames)
        self.num_loaded_frames = 0
        # load the first frame synchronously to cache it before the session is opened
        idx = self.num_loaded_frames
        self.images[idx] = self._load_one_frame(idx)
        self.images_loaded[idx] = True
        self.num_loaded_frames += 1
        pbar.update(n=1)
        self.all_frames_loaded = self.num_loaded_frames == self.num_frames

        # load the frames asynchronously without blocking the session start
        def _load_frames():
            finished = self.all_frames_loaded
            chunk_size = 16
            while not finished:
                # asynchronously load `chunk_size` frames each time we acquire the lock
                with self.torchcodec_access_lock, torch.inference_mode():
                    for _ in range(chunk_size):
                        try:
                            idx = self.num_loaded_frames
                            self.images[idx] = self._load_one_frame(idx)
                            self.images_loaded[idx] = True
                            self.num_loaded_frames += 1
                            pbar.update(n=1)
                            if self.num_loaded_frames >= self.num_frames:
                                finished = True
                                break
                        except Exception as e:
                            self.exception = e
                            raise

                    # also read the frame that is being randomly seeked to
                    while True:
                        try:
                            idx = self.rand_seek_idx_queue.get_nowait()
                            if not self.images_loaded[idx]:
                                self.images[idx] = self._load_one_frame(idx)
                                self.images_loaded[idx] = True
                        except queue.Empty:
                            break
                        except Exception as e:
                            self.exception = e
                            raise

            # finished -- check whether we have loaded the total number of frames
            if self.num_loaded_frames != self.num_frames:
                raise RuntimeError(
                    f"There are {self.num_frames} frames in the video, but only "
                    f"{self.num_loaded_frames} frames can be loaded successfully."
                )
            else:
                self.all_frames_loaded = True
                pbar.close()
                with self.torchcodec_access_lock:
                    import gc

                    # all frames have been loaded, so we can release the readers and free their memory
                    # also remove pbar and thread (which shouldn't be a part of session saving)
                    reader = self.async_reader
                    if reader is not None:
                        reader._source = None
                    self.async_reader = None
                    self.pbar = None
                    self.thread = None
                    self.rand_seek_idx_queue = None
                    gc.collect()
                # remove the lock (replace it with nullcontext) when the video is fully loaded
                self.torchcodec_access_lock = contextlib.nullcontext()

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def _transform_frame(self, frame):
        frame = frame.clone()  # make a copy to avoid modifying the original frame bytes
        frame = frame.float()  # convert to float32 before interpolation
        frame_resized = F.interpolate(
            frame[None, :],
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )[0]
        # float16 precision should be sufficient for image tensor storage
        frame_resized = frame_resized.half()  # uint8 -> float16
        frame_resized /= 255
        frame_resized -= self.img_mean
        frame_resized /= self.img_std
        if self.offload_video_to_cpu:
            frame_resized = frame_resized.cpu()
        elif frame_resized.device != self.out_device:
            frame_resized = frame_resized.to(device=self.out_device, non_blocking=True)
        return frame_resized

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        max_tries = 1200
        for _ in range(max_tries):
            # use a lock to avoid race condition between concurrent access to torchcodec
            # libs (which are not thread-safe); the lock is replaced with a nullcontext
            # when the video is fully loaded
            with self.torchcodec_access_lock:
                if self.images_loaded[index]:
                    return self.images[index]

                if self.use_rand_seek_in_loading:
                    # async loading hasn't reached this frame yet, so we load this frame individually
                    # (it will be loaded by in _load_frames thread and added to self.images[index])
                    self.rand_seek_idx_queue.put(index)

            time.sleep(0.1)

        raise RuntimeError(f"Failed to load frame {index} after {max_tries} tries")

    def __len__(self):
        return len(self.images)

    def __getstate__(self):
        """
        Remove a few attributes during pickling, so that this async video loader can be
        saved and loaded as a part of the model session.
        """
        # wait for async video loading to finish before pickling
        async_thread = self.thread
        if async_thread is not None:
            async_thread.join()
        # release a few objects that cannot be pickled
        reader = self.async_reader
        if reader is not None:
            reader._source = None
        self.async_reader = None
        self.pbar = None
        self.thread = None
        self.rand_seek_idx_queue = None
        self.torchcodec_access_lock = contextlib.nullcontext()
        return self.__dict__.copy()
