# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
from PIL import Image, ImageOps
import threading

import queue
from torch.utils.data import DataLoader

from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        # handle grayscale mixed in RGB images
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image


"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
This is a single-function package that transforms arbitrary generator into a background-thead generator that 
prefetches several batches of data in a parallel background thead.

This is useful if you have a computationally heavy process (CPU or GPU) that 
iteratively processes minibatches from the generator while the generator 
consumes some other resource (disk IO / loading from database / more CPU if you have unused cores). 

By default these two processes will constantly wait for one another to finish. If you make generator work in 
prefetch mode (see examples below), they will work in parallel, potentially saving you your GPU time.
We personally use the prefetch generator when iterating minibatches of data for deep learning with PyTorch etc.

Quick usage example (ipython notebook) - https://github.com/justheuristic/prefetch_generator/blob/master/example.ipynb
This package contains this object
 - BackgroundGenerator(any_other_generator[,max_prefetch = something])
"""


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(
            local_rank
        )  # create a new cuda stream in each process
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def _shutdown_background_thread(self):
        if not self.iter.is_alive():
            # avoid re-entrance or ill-conditioned thread state
            return

        # Set exit event to True for background threading stopping
        self.iter.exit_event.set()

        # Exhaust all remaining elements, so that the queue becomes empty,
        # and the thread should quit
        for _ in self.iter:
            pass

        # Waiting for background thread to quit
        self.iter.join()

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if isinstance(self.batch[k], torch.Tensor):
                    self.batch[k] = self.batch[k].to(
                        device=self.local_rank, non_blocking=True
                    )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(
            self.stream
        )  # wait tensor to put on GPU
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        # If the dataloader is to be freed, shutdown its BackgroundGenerator
        self._shutdown_background_thread()
