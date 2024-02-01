# -*- coding: utf-8 -*-
"""
@Author: CaptainHu
@Date: 2022年 10月 27日 星期二 10:32:12 CST
@Description: 张量调试自用
"""
from typing import Optional, Union, Dict, List, TypeVar
import math
from copy import deepcopy
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.getLogger("PIL").setLevel(logging.WARNING)
Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)

ImgData = TypeVar(
    "ImgData", np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]
)


def _normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range * 255


def _get_idx_order(channel_flag: str):
    order = (channel_flag.find("C"), channel_flag.find("H"), channel_flag.find("W"))
    return order


def _convert_shape(data: Tensor, channel: str):
    if len(data.shape) > 4 or len(data.shape) < 2:
        raise ValueError("dim num of input tensor must >1 and <5")
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if len(data.shape) == 2:
        if len(channel) == 2:
            channel = "C" + channel
        data = np.expand_dims(data, axis=0)

    if len(data.shape) == 4:
        order = [x + 1 for x in _get_idx_order(channel)]
        order.insert(0, 0)
        order = tuple(order)
        data = np.transpose(data, order)
        return [data[i] for i in range(data.shape[0])]

    data = np.transpose(data, _get_idx_order(channel))
    return data


class _ImgSeq(object):
    def __init__(self, imgs, channel_order):
        self._img_list = []
        self._o_imgs_channel_order = self._analy_seq_img_channel(imgs, channel_order)
        self._standardize_tensors_channel(imgs)

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, idx):
        return self._img_list[idx]

    def _standardize_tensors_channel(self, tensor_list):
        for t, c in zip(tensor_list, self._o_imgs_channel_order):
            chw_tensor = _convert_shape(t, c)
            if isinstance(chw_tensor, list):
                self._img_list.extend(chw_tensor)
            else:
                self._img_list.append(chw_tensor)

    def _analy_seq_img_channel(
        self, img_ori: ImgData, channel_order: Union[str, Dict[Union[str, int], str]]
    ):
        # 防止有时候显示opencv图片忘了写参数
        if isinstance(channel_order, str):
            default_flag = channel_order
        else:
            default_flag = "CHW"
        default_channel = []
        for idx, data in enumerate(img_ori):
            if len(data.shape) == 3 and np.argmin(data.shape) == 2:
                default_channel.append("HWC")
            else:
                default_channel.append(default_flag)
        if isinstance(channel_order, str):
            return default_channel

        for k, v in channel_order.items():
            if k.find(",") != -1:
                for i in k.split(","):
                    default_channel[int(i)] = v
            elif k.find(":") != -1:
                start_i = 0
                end_i = len(img_ori)
                if ":" == k[-1]:
                    start_i = int(k[:-1])
                elif ":" == k[0]:
                    end_i = int(k[1:])
                else:
                    idx = [int(x) for x in k.split(":")]
                    start_i = idx[0]
                    end_i = idx[-1]
                for i in range(start_i, end_i):
                    default_channel[int(i)] = v
            else:
                default_channel[int(k)] = v
        return default_channel


def normalize_tensor(img: np.ndarray):
    if img.min() < 0 or img.max() > 255:
        img = _normalization(img)
        print("img have norm")
        if img.shape[0] == 3:
            img = img.astype(np.uint8)
    else:
        if not (img % 1).any():
            img = img.astype(np.uint8)
            print("img type to uint8")
        else:
            img = img.astype(np.float)
    return img


def show_img(
    img_ori: ImgData,
    channel_order: Union[str, Dict[Union[str, int], str]] = "CHW",
    text: Optional[str] = None,
    cvreader: bool = True,
    delay: float = 0,
):
    r"""使用matplotlib阻塞显示张量,列表传入的图片或者是4D的张量会被拆分到不同的子图中显示

    Args:
        img_ori (ImgData):
            2~4D 的np.ndarray或者torch.Tensor,或是列表形式
        channel_order (Union[str, Dict[Union[str, int], str]], optional):
            默认除开3D且最短轴的张量是"HWC"外,其余所有张量都是"CHW",
            选项是"CHW",三个字母任意顺序组合,对于4D张量也只用设置这3者顺序即可,默认认为第一轴是batch.
            若图片传入的是一个列表,且列表中各个张量通道顺序不一致,则可使用字典表示,例如:
                >>> channel_order={
                >>>     "1,3":"WCH",
                >>>     "2": "CWH",
                >>>     "4:6":"HCW",
                >>>     "6:":"CWH",
                >>> }
            表示img_ori[0]是CHW(默认),img_ori[1,3]是WCH,img_ori[2]是CWH,img_ori[4:6]是HCW,依次类推...

        text (Optional[str], optional):
            用于显示一些提示信息. Defaults to None.
        cvreader (bool, optional):
            图片是否使用的是opencv读的,若是默认会将通道反转. Defaults to True.
        delay (float, optional):
            显示时的延时,为0为一直阻塞. Defaults to 0.

    """
    imgs = deepcopy(img_ori)
    if not isinstance(imgs, (list, tuple)):
        if hasattr(imgs, "show"):
            imgs.show()
            return 0
        else:
            imgs = [imgs]

    imgss = _ImgSeq(imgs, channel_order)
    img_num = len(imgss)
    row_n = math.ceil(math.sqrt(img_num))
    col_n = max(math.ceil(img_num / row_n), 1)
    fig, axs = plt.subplots(
        row_n, col_n, figsize=(15 * row_n, 15 * col_n), layout="constrained"
    )
    plt.rcParams["figure.constrained_layout.use"] = True
    for idx, img in enumerate(imgss):
        img_t = normalize_tensor(img)
        img_grid = _split_channel2grid(img_t)
        if cvreader:
            img_grid = img_grid[:, :, ::-1]
        if isinstance(axs, np.ndarray):
            if 2 == len(axs.shape):
                axs[idx % row_n][idx // row_n].imshow(img_grid)
                axs[idx % row_n][idx // row_n].set_title(str(idx))
            else:
                axs[idx % row_n].imshow(img_grid)
                axs[idx % row_n].set_title(str(idx))
        else:
            axs.imshow(img_grid)
            axs.set_title(str(idx))
    if text:
        plt.text(0, 0, text, fontsize=15)
    if delay <= 0:
        plt.show()
    else:
        plt.draw()
        plt.pause(delay)
        plt.close()


def _split_channel2grid(data: np.ndarray):
    """data should be 3D tensor and CHW"""
    if data.shape[0] in (1, 3):
        return data.transpose((1, 2, 0))
    else:
        nrow = int(np.sqrt(data.shape[0]))
        n_t = data.shape[0] % nrow
        n_t = nrow - n_t if n_t else 0
        ncol = data.shape[0] // nrow + min(n_t, 1)

        if n_t:
            fill_t = np.zeros((n_t, *data.shape[1:]))
            data = np.concatenate((data, fill_t), axis=0)

        data = np.reshape(data, (data.shape[0] // nrow, data.shape[1] * nrow, -1))
        data = data.transpose((1, 0, 2))  # HCW
        data = data.reshape((data.shape[0], 1, -1))
        data = data.transpose((0, 2, 1))  # HWC
        return data


def show_pos_mask(mask_gt: torch.Tensor, size=640):
    """调试 mask_gt使用

    Args:
        mask_gt (torch.Tensor): _description_
        size (int, optional): _description_. Defaults to 640.

    Returns:
        _type_: _description_
    """
    s1 = size // 8
    s2 = size // 16
    s3 = size // 32

    b, gt_num, _ = mask_gt.shape

    p1, p2, p3 = mask_gt.split([s1 * s1, s2 * s2, s3 * s3], dim=-1)
    p1 = p1.view(b, gt_num, s1, s1)
    p2 = p2.view(b, gt_num, s2, s2)
    p3 = p3.view(b, gt_num, s3, s3)
    show_img([p1, p2, p3])


def show_pd_scores(ps_scores: torch.Tensor, size=640):
    """调试 ps_scores使用

    Args:
        ps_scores (torch.Tensor): _description_
        size (int, optional): _description_. Defaults to 640.

    Returns:
        _type_: _description_
    """
    s1 = size // 8
    s2 = size // 16
    s3 = size // 32

    b, anchor_num, nc = ps_scores.shape
    p1, p2, p3 = ps_scores.permute(0, 2, 1).split([s1 * s1, s2 * s2, s3 * s3], dim=-1)
    p1 = p1.view(b, nc, s1, s1)
    p2 = p2.view(b, nc, s2, s2)
    p3 = p3.view(b, nc, s3, s3)
    show_img([p1, p2, p3])


def show_bbox_scores(bbox_scores: torch.Tensor, size=640):
    s1 = size // 8
    s2 = size // 16
    s3 = size // 32

    b, gt_max_num, anchor_num = bbox_scores.shape
    p1, p2, p3 = bbox_scores.split([s1 * s1, s2 * s2, s3 * s3], dim=-1)
    p1 = p1.view(b, gt_max_num, s1, s1)
    p2 = p2.view(b, gt_max_num, s2, s2)
    p3 = p3.view(b, gt_max_num, s3, s3)
    show_img([p1, p2, p3])


def show_flatten_tensor(t: torch.Tensor, flatten_dim: int, stage_num=3, size=640):
    if len(t.shape)>4:
        return
    s = [0] * stage_num
    for i in range(stage_num):
        s[i] = size // (8 * (2**i))

    dim_order=list(range(len(t.shape)))
    dim_order.pop(flatten_dim)
    dim_order.append(flatten_dim)

    dim_num=list(t.shape)
    dim_num.pop(flatten_dim)

    split_args=[i*i for i in s]
    if flatten_dim!=len(t.shape)-1:
        tt=t.permute(dim_order)
    else:
        tt=t
    r=tt.split(split_args,dim=-1)
    r=[m.view(*(dim_num+[st,st])) for m,st in zip(r,s)]
    show_img(r)



