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
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import utils
import vision_transformer as vits
from PIL import Image
from torchvision import transforms as pth_transforms
from tqdm import tqdm

FOURCC = {
    'mp4': cv2.VideoWriter_fourcc(*'MP4V'),
    'avi': cv2.VideoWriter_fourcc(*'XVID'), }
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class VideoGenerator:

    def __init__(self, args):
        self.args = args
        # self.model = None
        # Don't need to load model if you only want a video
        if not self.args.video_only:
            self.model = self.__load_model()

    def run(self):
        if self.args.input_path is None:
            print(f'Provided input path {self.args.input_path} is non valid.')
            sys.exit(1)
        else:
            if self.args.video_only:
                self._generate_video_from_images(self.args.input_path, self.args.output_path)
            else:
                # If input path exists
                if os.path.exists(self.args.input_path):
                    # If input is a video file
                    if os.path.isfile(self.args.input_path):
                        frames_folder = os.path.join(self.args.output_path, 'frames')
                        attention_folder = os.path.join(self.args.output_path, 'attention')

                        os.makedirs(frames_folder, exist_ok=True)
                        os.makedirs(attention_folder, exist_ok=True)

                        self._extract_frames_from_video(self.args.input_path, frames_folder)

                        self._inference(
                            frames_folder,
                            attention_folder,
                        )

                        self._generate_video_from_images(attention_folder, self.args.output_path)

                    # If input is a folder of already extracted frames
                    if os.path.isdir(self.args.input_path):
                        attention_folder = os.path.join(self.args.output_path, 'attention')

                        os.makedirs(attention_folder, exist_ok=True)

                        self._inference(self.args.input_path, attention_folder)

                        self._generate_video_from_images(attention_folder, self.args.output_path)

                # If input path doesn't exists
                else:
                    print(f"Provided input path {self.args.input_path} doesn't exists.")
                    sys.exit(1)

    def _extract_frames_from_video(self, inp: str, out: str):
        vidcap = cv2.VideoCapture(inp)
        self.args.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(f'Video: {inp} ({self.args.fps} fps)')
        print(f'Extracting frames to {out}')

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(out, f'frame-{count:04}.jpg'),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def _generate_video_from_images(self, inp: str, out: str):
        img_array = []
        attention_images_list = sorted(glob.glob(os.path.join(inp, 'attn-*.jpg')))

        # Get size of the first image
        with open(attention_images_list[0], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            size = (img.width, img.height)
            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        print(f'Generating video {size} to {out}')

        for filename in tqdm(attention_images_list[1:]):
            with open(filename, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        out = cv2.VideoWriter(
            os.path.join(out, 'video.' + self.args.video_format),
            FOURCC[self.args.video_format],
            self.args.fps,
            size,
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print('Done')

    def _inference(self, inp: str, out: str):
        print(f'Generating attention images to {out}')

        for img_path in tqdm(sorted(glob.glob(os.path.join(inp, '*.jpg')))):
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')

            if self.args.resize is not None:
                transform = pth_transforms.Compose([
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(self.args.resize),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
            else:
                transform = pth_transforms.Compose([
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

            img = transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % self.args.patch_size,
                img.shape[2] - img.shape[2] % self.args.patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // self.args.patch_size
            h_featmap = img.shape[-1] // self.args.patch_size

            attentions = self.model.get_last_selfattention(img.to(DEVICE))

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (nn.functional.interpolate(
                th_attn.unsqueeze(0),
                scale_factor=self.args.patch_size,
                mode='nearest',
            )[0].cpu().numpy())

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=self.args.patch_size,
                mode='nearest',
            )[0].cpu().numpy())

            # save attentions heatmaps
            fname = os.path.join(out, 'attn-' + os.path.basename(img_path))
            plt.imsave(
                fname=fname,
                arr=sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0])),
                cmap='inferno',
                format='jpg',
            )

    def __load_model(self):
        # build model
        model = vits.__dict__[self.args.arch](patch_size=self.args.patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(DEVICE)

        if os.path.isfile(self.args.pretrained_weights):
            state_dict = torch.load(self.args.pretrained_weights, map_location='cpu')
            if (self.args.checkpoint_key is not None and self.args.checkpoint_key in state_dict):
                print(f'Take key {self.args.checkpoint_key} in provided checkpoint dict')
                state_dict = state_dict[self.args.checkpoint_key]
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(self.args.pretrained_weights, msg))
        else:
            print('Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.')
            url = None
            if self.args.arch == 'vit_small' and self.args.patch_size == 16:
                url = 'dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
            elif self.args.arch == 'vit_small' and self.args.patch_size == 8:
                url = 'dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth'  # model used for visualizations in our paper
            elif self.args.arch == 'vit_base' and self.args.patch_size == 16:
                url = 'dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth'
            elif self.args.arch == 'vit_base' and self.args.patch_size == 8:
                url = 'dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth'
            if url is not None:
                print('Since no pretrained weights have been provided, we load the reference pretrained DINO weights.')
                state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/dino/' + url)
                model.load_state_dict(state_dict, strict=True)
            else:
                print('There is no reference weights available for this model => We use random weights.')
        return model


def parse_args():
    parser = argparse.ArgumentParser('Generation self-attention video')
    parser.add_argument(
        '--arch',
        default='vit_small',
        type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help='Architecture (support only ViT atm).',
    )
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the self.model.')
    parser.add_argument(
        '--pretrained_weights',
        default='',
        type=str,
        help='Path to pretrained weights to load.',
    )
    parser.add_argument(
        '--checkpoint_key',
        default='teacher',
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        '--input_path',
        required=True,
        type=str,
        help="""Path to a video file if you want to extract frames
            or to a folder of images already extracted by yourself.
            or to a folder of attention images.""",
    )
    parser.add_argument(
        '--output_path',
        default='./',
        type=str,
        help="""Path to store a folder of frames and / or a folder of attention images.
            and / or a final video. Default to current directory.""",
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        '--resize',
        default=None,
        type=int,
        nargs='+',
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or W H): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        '--video_only',
        action='store_true',
        help="""Use this flag if you only want to generate a video and not all attention images.
            If used, --input_path must be set to the folder of attention images. Ex: ./attention/""",
    )
    parser.add_argument(
        '--fps',
        default=30.0,
        type=float,
        help='FPS of input / output video. Automatically set if you extract frames from a video.',
    )
    parser.add_argument(
        '--video_format',
        default='mp4',
        type=str,
        choices=['mp4', 'avi'],
        help='Format of generated video (mp4 or avi).',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    vg = VideoGenerator(args)
    vg.run()
