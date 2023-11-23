import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.modeling import build_model
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer


# from torch.backends import cudnn



# cudnn.benchmark = True


def setup_cfg(config_file, opts, device='cuda'):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = device

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if device == 'cuda':
            self.device = 'cuda'
        elif device == 'mps':
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path], device=self.device)

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))
        # features = np.zeros((0, 768))

        for patches in batch_patches:

            # Run model
            patches_ = torch.clone(patches)
            pred = self.model(patches)
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features

