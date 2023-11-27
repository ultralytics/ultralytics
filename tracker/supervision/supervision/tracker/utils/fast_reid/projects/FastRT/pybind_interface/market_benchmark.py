import random
import numpy as np
import cv2
import fs
import argparse
import io
import sys
import torch
import time
import os
import torchvision.transforms as T

sys.path.append('../../..')
sys.path.append('../')
from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch import build_model
from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.utils.logger import setup_logger
from supervision.tracker.utils.fast_reid.fastreid.data import build_reid_train_loader, build_reid_test_loader
from supervision.tracker.utils.fast_reid.fastreid.evaluation.rank import eval_market1501

from build.pybind_interface.ReID import ReID


FEATURE_DIM = 2048
GPU_ID = 0

def map(wrapper):
	model = wrapper
	cfg = get_cfg()
	test_loader, num_query = build_reid_test_loader(cfg, "Market1501", T.Compose([]))

	feats = []
	pids = []
	camids = []

	for batch in test_loader:
		for image_path in batch["img_paths"]:
			t = torch.Tensor(np.array([model.infer(cv2.imread(image_path))]))
			t.to(torch.device(GPU_ID))
			feats.append(t)
		pids.extend(batch["targets"].numpy())
		camids.extend(batch["camids"].numpy())
		
	feats = torch.cat(feats, dim=0)
	q_feat = feats[:num_query]
	g_feat = feats[num_query:]
	q_pids = np.asarray(pids[:num_query])
	g_pids = np.asarray(pids[num_query:])
	q_camids = np.asarray(camids[:num_query])
	g_camids = np.asarray(camids[num_query:])

	
	distmat = 1 - torch.mm(q_feat, g_feat.t())
	distmat = distmat.numpy()
	all_cmc, all_AP, all_INP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, 5)
	mAP = np.mean(all_AP)
	print("mAP {}, rank-1 {}".format(mAP, all_cmc[0]))


if __name__ == '__main__':
	infer = ReID(GPU_ID)
	infer.build("../build/sbs_R50-ibn.engine")
	map(infer)
