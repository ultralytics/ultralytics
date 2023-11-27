# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import atexit
import bisect
from collections import deque

import cv2
import torch
import torch.multiprocessing as mp

from supervision.tracker.utils.fast_reid.fastreid.engine import DefaultPredictor

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


class FeatureExtractionDemo(object):
    def __init__(self, cfg, parallel=False):
        """
        Args:
            cfg (CfgNode):
            parallel (bool) whether to run the model in different processes from visualization.:
                Useful since the visualization logic can be slow.
        """
        self.cfg = cfg
        self.parallel = parallel

        if parallel:
            self.num_gpus = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, self.num_gpus)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, original_image):
        """

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (np.ndarray): normalized feature of the model.
        """
        # the model expects RGB inputs
        original_image = original_image[:, :, ::-1]
        # Apply pre-processing to image.
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for
        # network input
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        predictions = self.predictor(image)
        return predictions

    def run_on_loader(self, data_loader):
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            batch_data = deque()

            for cnt, batch in enumerate(data_loader):
                batch_data.append(batch)
                self.predictor.put(batch["images"])

                if cnt >= buffer_size:
                    batch = batch_data.popleft()
                    predictions = self.predictor.get()
                    yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()

            while len(batch_data):
                batch = batch_data.popleft()
                predictions = self.predictor.get()
                yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()
        else:
            for batch in data_loader:
                predictions = self.predictor(batch["images"])
                yield predictions, batch["targets"].cpu().numpy(), batch["camids"].cpu().numpy()


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because when the amount of data is large.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """

        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # Make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
