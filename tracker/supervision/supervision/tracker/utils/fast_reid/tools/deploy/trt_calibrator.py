# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com

Create custom calibrator, use to calibrate int8 TensorRT model.
Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.
"""

# based on:
# https://github.com/qq995431104/Pytorch2TensorRT/blob/master/myCalibrator.py

import os
import sys

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import torchvision.transforms as T

sys.path.append('../..')

from supervision.tracker.utils.fast_reid.fastreid.data.build import _root
from supervision.tracker.utils.fast_reid.fastreid.data.data_utils import read_image
from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
import logging

from supervision.tracker.utils.fast_reid.fastreid.data.transforms import ToTensor


logger = logging.getLogger('trt_export.calibrator')


class FeatEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, args):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = 'reid_feat.cache'

        self.batch_size = args.batch_size
        self.channel = args.channel
        self.height = args.height
        self.width = args.width
        self.transform = T.Compose([
            T.Resize((self.height, self.width), interpolation=3),  # [h,w]
            ToTensor(),
        ])

        dataset = DATASET_REGISTRY.get(args.calib_data)(root=_root)
        self._data_items = dataset.train + dataset.query + dataset.gallery
        np.random.shuffle(self._data_items)
        self.imgs = [item[0] for item in self._data_items]

        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size

        self.data_size = self.batch_size * self.channel * self.height * self.width * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.channel, self.height, self.width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = read_image(f)
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size // self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            logger.info("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            batch_imgs = batch_imgs.ravel()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
