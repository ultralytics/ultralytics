# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils
import vision_transformer as vits
from eval_knn import extract_features
from PIL import Image, ImageFile
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms


class CopydaysDataset():

    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (['original', 'strong'] + ['jpegqual/%d' % i for i in [3, 5, 8, 10, 15, 20, 30, 50, 75]] +
                            ['crops/%d' % i for i in [10, 15, 20, 30, 40, 50, 60, 70, 80]])
        self.nblocks = len(self.block_names)

        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + '/' + self.block_names[i]
        fnames = [dirname + '/' + fname for fname in sorted(os.listdir(dirname)) if fname.endswith('.jpg')]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + '/' + subdir_name
        return [fname for fname in sorted(os.listdir(dirname)) if fname.endswith('.jpg')]

    def eval_result(self, ids, distances):
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]  # block size
            sum_AP = 0
            if block_name != 'strong':
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames('original')
                strongs = self.get_block_filenames('strong')

                # check if prefixes match
                positives_per_query = [[j for j, bname in enumerate(originals) if bname[:4] == qname[:4]]
                                       for qname in strongs]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            print('eval on {} mAP={:.3f}'.format(block_name, sum_AP / (j1 - j0)))
            j0 = j1


# from the Holidays evaluation package
def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


class ImgListDataset(torch.utils.data.Dataset):

    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp']:
        return True
    return False


@torch.no_grad()
def extract_features(image_list, model, args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.imsize, args.imsize), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
    tempdataset = ImgListDataset(image_list, transform=transform)
    data_loader = torch.utils.data.DataLoader(tempdataset,
                                              batch_size=args.batch_size_per_gpu,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              sampler=torch.utils.data.DistributedSampler(tempdataset, shuffle=False))
    features = None
    for samples, index in utils.MetricLogger(delimiter='  ').log_every(data_loader, 10):
        samples, index = samples.cuda(non_blocking=True), index.cuda(non_blocking=True)
        feats = model.get_intermediate_layers(samples, n=1)[0].clone()

        cls_output_token = feats[:, 0, :]  #  [CLS] token
        # GeM with exponent 4 for output patch tokens
        b, h, w, d = len(samples), int(samples.shape[-2] / model.patch_embed.patch_size), int(
            samples.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
        feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(dist.get_world_size(),
                                feats.size(0),
                                feats.size(1),
                                dtype=feats.dtype,
                                device=feats.device)
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features  # features is still None for every rank which is not 0 (main)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Copy detection on Copydays')
    parser.add_argument('--data_path',
                        default='/path/to/copydays/',
                        type=str,
                        help='See https://lear.inrialpes.fr/~jegou/data.php#copydays')
    parser.add_argument('--whitening_path',
                        default='/path/to/whitening_data/',
                        type=str,
                        help="""Path to directory with images used for computing the whitening operator.
        In our paper, we use 20k random images from YFCC100M.""")
    parser.add_argument(
        '--distractors_path',
        default='/path/to/distractors/',
        type=str,
        help='Path to directory with distractors images. In our paper, we use 10k random images from YFCC100M.')
    parser.add_argument('--imsize', default=320, type=int, help='Image size (square image)')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pretrained weights to evaluate.')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--checkpoint_key',
                        default='teacher',
                        type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dist_url',
                        default='env://',
                        type=str,
                        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--local_rank', default=0, type=int, help='Please ignore and do not set this argument.')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print('git:\n  {}\n'.format(utils.get_sha()))
    print('\n'.join('{}: {}'.format(k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    if 'vit' in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f'Model {args.arch} {args.patch_size}x{args.patch_size} built.')
    else:
        print(f'Architecture {args.arch} non supported')
        sys.exit(1)
    if args.use_cuda:
        model.cuda()
    model.eval()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    dataset = CopydaysDataset(args.data_path)

    # ============ Extract features ... ============
    # extract features for queries
    queries = []
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, args))
    if utils.get_rank() == 0:
        queries = torch.cat(queries)
        print(f'Extraction of queries features done. Shape: {queries.shape}')

    # extract features for database
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, args))

    # extract features for distractors
    if os.path.isdir(args.distractors_path):
        print('Using distractors...')
        list_distractors = [
            os.path.join(args.distractors_path, s) for s in os.listdir(args.distractors_path) if is_image_file(s)]
        database.append(extract_features(list_distractors, model, args))
    if utils.get_rank() == 0:
        database = torch.cat(database)
        print(f'Extraction of database and distractors features done. Shape: {database.shape}')

    # ============ Whitening ... ============
    if os.path.isdir(args.whitening_path):
        print(f'Extracting features on images from {args.whitening_path} for learning the whitening operator.')
        list_whit = [os.path.join(args.whitening_path, s) for s in os.listdir(args.whitening_path) if is_image_file(s)]
        features_for_whitening = extract_features(list_whit, model, args)
        if utils.get_rank() == 0:
            # center
            mean_feature = torch.mean(features_for_whitening, dim=0)
            database -= mean_feature
            queries -= mean_feature
            pca = utils.PCA(dim=database.shape[-1], whit=0.5)
            # compute covariance
            cov = torch.mm(features_for_whitening.T, features_for_whitening) / features_for_whitening.shape[0]
            pca.train_pca(cov.cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)

    # ============ Copy detection ... ============
    if utils.get_rank() == 0:
        # l2 normalize the features
        database = nn.functional.normalize(database, dim=1, p=2)
        queries = nn.functional.normalize(queries, dim=1, p=2)

        # similarity
        similarity = torch.mm(queries, database.T)
        distances, indices = similarity.topk(20, largest=True, sorted=True)

        # evaluate
        retrieved = dataset.eval_result(indices, distances)
    dist.barrier()
