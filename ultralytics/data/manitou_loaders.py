import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import FORMATS_HELP_MSG, IMG_FORMATS, VID_FORMATS
from ultralytics.utils import IS_COLAB, IS_KAGGLE, LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import imread
from ultralytics.utils.radar_data import ManitouRadarPC


class LoadManitouImagesAndRadar:
    """Load images and radar data from a directory for Manitou dataset."""
    
    def __init__(self, data_cfg, radar_accumulation=1, batch=1, pre_transform=None, use_radar=True):
        """
        Args:
            data_cfg (dict): Configuration dictionary containing dataset paths and calibration parameters.
            radar_accumulation (int): Number of radar frames to accumulate. Default is 1.
            batch (int): Batch size for loading data.
            pre_transform (list of callable, optional): List of pre-transform functions to apply to the images.
            use_radar (bool): Whether to use (load) radar data. Default is True.
        """
        self._check_data_cfg(data_cfg)
        self.cam1 = data_cfg['camera1']
        self.cam2 = data_cfg['camera2']
        self.cam3 = data_cfg['camera3']
        self.cam4 = data_cfg['camera4']
        self.radar1 = data_cfg['radar1']
        self.radar2 = data_cfg['radar2']
        self.radar3 = data_cfg['radar3']
        self.radar4 = data_cfg['radar4']
        self.calib_params = data_cfg['calib_params']
        self.filter_cfg = data_cfg['filter_cfg']
        self.radar_accumulation = radar_accumulation
        self.pre_transform = pre_transform if pre_transform is not None else []
        self.use_radar = use_radar

        # Check if the paths is a directory or a file
        self.cam1 = self._check_path(self.cam1)
        self.cam2 = self._check_path(self.cam2)
        self.cam3 = self._check_path(self.cam3)
        self.cam4 = self._check_path(self.cam4)
        if self.use_radar:
            self.radar1 = self._check_path(self.radar1)
            self.radar2 = self._check_path(self.radar2)
            self.radar3 = self._check_path(self.radar3)
            self.radar4 = self._check_path(self.radar4)

        if self.use_radar:
            assert len(self.cam1) == len(self.cam2) == len(self.cam3) == len(self.cam4) \
                    == len(self.radar1) == len(self.radar2) == len(self.radar3) == len(self.radar4), \
                "The number of images in each camera and radar should be the same."
        else:
            assert len(self.cam1) == len(self.cam2) == len(self.cam3) == len(self.cam4), \
                "The number of images in each camera should be the same."
            
        self.nf = len(self.cam1)  # number of frames
        self.bs = batch
        
    def _check_data_cfg(self, data_cfg):
        """Check if the data configuration is valid. 
        A valid configuration should contain following keys:
            - 'camera1', 'camera2', 'camera3', 'camera4': Paths to the camera images.
            - 'radar1', 'radar2', 'radar3', 'radar4': Paths to the radar data.
            - 'filter_cfg' (dict): Configuration for filtering the data.
            - 'calib_params' (dict): Calibration parameters for the cameras and radars.
        """
        required_keys = ['camera1', 'camera2', 'camera3', 'camera4', 
                         'radar1', 'radar2', 'radar3', 'radar4', 
                         'filter_cfg',
                         'calib_params']
        for key in required_keys:
            if key not in data_cfg:
                LOGGER.warning(f"Required keys are: {', '.join(required_keys)}")
                raise ValueError(f"Missing required key '{key}' in data configuration.")
  
    def _check_path(self, path):
        """If the path is a directory, return the path to the images."""
        if isinstance(path, str) and Path(path).suffix.lower() in IMG_FORMATS:
            return [path]
        elif isinstance(path, list):
            return sorted(path)
        
        if Path(path).is_dir():
            path = sorted(glob.glob(os.path.join(path, '*')))
            if not path:
                raise FileNotFoundError(f"No images found in directory: {path}")
            return path
        else:
            raise FileNotFoundError(f"Path should be a image path or a directory containing images or a list of image paths, but got: {path}")
        
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        paths = []
        batch = []
        info = []
        while len(batch) < self.bs:
            if self.count >= self.nf:
                if batch:
                    return paths, batch, info
                else:
                    raise StopIteration
            cam1_pth = self.cam1[self.count]
            cam2_pth = self.cam2[self.count]
            cam3_pth = self.cam3[self.count]
            cam4_pth = self.cam4[self.count]
            if self.use_radar:
                radar1_pth = self.radar1[self.count]
                radar2_pth = self.radar2[self.count]
                radar3_pth = self.radar3[self.count]
                radar4_pth = self.radar4[self.count]
            frame_name = os.path.basename(cam1_pth).split('.')[0]
            
            # Load images and radar data
            cam1_img = imread(cam1_pth)
            cam2_img = imread(cam2_pth)
            cam3_img = imread(cam3_pth)
            cam4_img = imread(cam4_pth)
            
            h, w = cam1_img.shape[:2]
            orig_images = {
                'cam1': cam1_img,
                'cam2': cam2_img,
                'cam3': cam3_img,
                'cam4': cam4_img
            }
            
            # Apply pre-transforms to the images if any
            for transform in self.pre_transform:
                cam1_img, cam2_img, cam3_img, cam4_img = \
                    transform(images=(cam1_img, cam2_img, cam3_img, cam4_img))
            
            if self.use_radar:
                radar1_pc = self._load_radar(radar1_pth, self.radar_accumulation)
                radar2_pc = self._load_radar(radar2_pth, self.radar_accumulation)
                radar3_pc = self._load_radar(radar3_pth, self.radar_accumulation)
                radar4_pc = self._load_radar(radar4_pth, self.radar_accumulation)
                
                radar_pc = ManitouRadarPC(radar1=radar1_pc, 
                                        radar2=radar2_pc, 
                                        radar3=radar3_pc, 
                                        radar4=radar4_pc,
                                        calib_params= self.calib_params,
                                        filter_cfg=self.filter_cfg)
            
            # Create a dictionary for the batch
            paths.append({
                'cam1': cam1_pth,
                'cam2': cam2_pth,
                'cam3': cam3_pth,
                'cam4': cam4_pth,
                'radar1': radar1_pth if self.use_radar else None,
                'radar2': radar2_pth if self.use_radar else None,
                'radar3': radar3_pth if self.use_radar else None,
                'radar4': radar4_pth if self.use_radar else None
            })
            batch.append({
                'cam1': cam1_img,
                'cam2': cam2_img,
                'cam3': cam3_img,
                'cam4': cam4_img,
                'radar': radar_pc if self.use_radar else None,
                'orig_images': orig_images,
                'frame_name': frame_name,
            })
            info.append(f'image {self.count + 1}/{self.nf} frame: {frame_name}: ')
            self.count += 1
            
        return paths, batch, info
    
    def __len__(self):
        return self.nf
    
    def _load_radar(self, path, accumulation):
        radar_pc = np.loadtxt(path, delimiter=' ', dtype=np.float32)
        # # load only previous frames if accumulation > 1
        # if accumulation > 1:
        #     for _ in range(accumulation - 1):
        #         frame_name = os.path.basename(path).split('.')[0]
        #         frame_name = int(frame_name)
        #         if frame_name - 1 < 0:
        #             break
        #         prev_name = f"{frame_name - 1:06d}.{path.split('.')[-1]}"
        #         path = str(Path(path).parent / f"{prev_name}")
        #         radar_pc = np.concatenate((radar_pc, np.loadtxt(path, delimiter=' ', dtype=np.float32)), axis=0)
        
        # load from previous and future frames if accumulation > 1 (half of accumulation frames before and after)
        if accumulation > 1:
            half_accum = accumulation // 2
            frame_name = os.path.basename(path).split('.')[0]
            frame_name = int(frame_name)
            for i in range(-half_accum, half_accum + 1):
                if i == 0:
                    continue  
                # if frame_name + i < 0 or frame_name + i >= self.nf:
                #     continue
                try:
                    new_name = f"{frame_name + i:06d}.{path.split('.')[-1]}"
                    new_path = str(Path(path).parent / f"{new_name}")
                    radar_pc = np.concatenate((radar_pc, np.loadtxt(new_path, delimiter=' ', dtype=np.float32)), axis=0)
                    # print(f"Loading radar data from {new_path} for accumulation. (current frame: {frame_name}, accumulation: {accumulation})")
                except FileNotFoundError:
                    continue  # Skip if the file does not exist

        return radar_pc
            


if __name__ == "__main__":
    # Example usage
    import json
    from scipy.spatial.transform import Rotation as R
    from ultralytics.data.manitou_api import ManitouAPI
    
    # Get the extrinsic parameters of the camera
    def get_cam_extrinsics(path, invert=False):
        with open(path, 'r') as f:
            data = json.load(f)
        T = np.eye(4)
        T[:3, :3] = np.array(data['R']['data']).reshape(3, 3)
        T[:3, 3] = np.array(data['T']['data']).reshape(3)
        if invert:
            T = np.linalg.inv(T)  # camera to ego
        return T

    # Get the intrinsic parameters of the camera and extrinsic parameters of the radar
    def get_radar_params(path, to_meter):
        with open(path, 'r') as f:
            data = json.load(f)
        matrix_K = np.array(data['camera_matrix']).reshape(3, 3)
        dist = np.array(data['distortion']).reshape(4, )
        rvec = np.array(data['cam_radar_rvecs']).reshape(3, )
        tvec = np.array(data['cam_radar_tvecs']).reshape(3, )
        if to_meter:
            tvec = tvec / 100  # convert to meter
        Tcam_radar = np.eye(4)
        Tcam_radar[:3, :3] = R.from_rotvec(rvec).as_matrix()
        Tcam_radar[:3, 3] = tvec
        return matrix_K, dist, Tcam_radar    
    
    radar1_path = '/home/shu/dataset/manitou/calibration/calibration_params_radar1.json'
    radar2_path = '/home/shu/dataset/manitou/calibration/calibration_params_radar2.json'
    radar3_path = '/home/shu/dataset/manitou/calibration/calibration_params_radar3.json'
    radar4_path = '/home/shu/dataset/manitou/calibration/calibration_params_radar4.json'
    cam1_path = '/home/shu/dataset/manitou/calibration/cam1.json'
    cam2_path = '/home/shu/dataset/manitou/calibration/cam2.json'
    cam3_path = '/home/shu/dataset/manitou/calibration/cam3.json'
    cam4_path = '/home/shu/dataset/manitou/calibration/cam4.json'
    
    K1, dist1, Tcam1_radar1 = get_radar_params(radar1_path, to_meter=True)
    K2, dist2, Tcam2_radar2 = get_radar_params(radar2_path, to_meter=True)
    K3, dist3, Tcam3_radar3 = get_radar_params(radar3_path, to_meter=True)
    K4, dist4, Tcam4_radar4 = get_radar_params(radar4_path, to_meter=True)
    
    Tego_cam1 = get_cam_extrinsics(cam1_path, invert=True)
    Tego_cam2 = get_cam_extrinsics(cam2_path, invert=True)
    Tego_cam3 = get_cam_extrinsics(cam3_path, invert=True)
    Tego_cam4 = get_cam_extrinsics(cam4_path, invert=True)
    
    Tego_radar1 = Tego_cam1 @ Tcam1_radar1
    Tego_radar2 = Tego_cam2 @ Tcam2_radar2
    Tego_radar3 = Tego_cam3 @ Tcam3_radar3
    Tego_radar4 = Tego_cam4 @ Tcam4_radar4
    
    calib_params = {
        'camera1_K': K1,
        'camera1_D': dist1,
        'camera2_K': K2,
        'camera2_D': dist2,
        'camera3_K': K3,
        'camera3_D': dist3,
        'camera4_K': K4,
        'camera4_D': dist4,
        'eCamera1': Tego_cam1,
        'eCamera2': Tego_cam2,
        'eCamera3': Tego_cam3,
        'eCamera4': Tego_cam4,
        'eRadar1': Tego_radar1,
        'eRadar2': Tego_radar2,
        'eRadar3': Tego_radar3,
        'eRadar4': Tego_radar4
    }
    
    data_root = '/home/shu/Documents/PROTECH/ultralytics/datasets/manitou'
    radar_dir = 'radars'
    cam_dir = 'key_frames'
    
    annotations_path = os.path.join(data_root, 'annotations_multi_view_mini', 'manitou_coco_val_remap.json')
    manitou = ManitouAPI(annotations_path)
    manitou.info()
    
    # get bag ids
    video_ids = manitou.get_vid_ids(bagIds=[2])
    
    cam1_ids = []
    cam2_ids = []
    cam3_ids = []
    cam4_ids = []

    for video_id in video_ids:
        imgs = manitou.get_img_ids_from_vid(video_id)
        if 'camera1' in manitou.vids[video_id]['name']:
            cam1_ids = imgs
        elif 'camera2' in manitou.vids[video_id]['name']:
            cam2_ids = imgs
        elif 'camera3' in manitou.vids[video_id]['name']:
            cam3_ids = imgs
        elif 'camera4' in manitou.vids[video_id]['name']:
            cam4_ids = imgs
    
    assert len(cam1_ids) == len(cam2_ids) == len(cam3_ids) == len(cam4_ids), "Camera IDs must be of the same length."
    
    cam1_paths = []
    cam2_paths = []
    cam3_paths = []
    cam4_paths = []
    radar1_paths = []
    radar2_paths = []
    radar3_paths = []
    radar4_paths = []
    for c1, c2, c3, c4 in zip(cam1_ids, cam2_ids, cam3_ids, cam4_ids):
        cam1_name = manitou.imgs[c1]['file_name']
        cam2_name = manitou.imgs[c2]['file_name']
        cam3_name = manitou.imgs[c3]['file_name']
        cam4_name = manitou.imgs[c4]['file_name']
        cam1_paths.append(os.path.join(data_root, cam_dir, cam1_name))
        cam2_paths.append(os.path.join(data_root, cam_dir, cam2_name))
        cam3_paths.append(os.path.join(data_root, cam_dir, cam3_name))
        cam4_paths.append(os.path.join(data_root, cam_dir, cam4_name))
        radar1_name = manitou.radars[c1]['file_name']
        radar2_name = manitou.radars[c2]['file_name']
        radar3_name = manitou.radars[c3]['file_name']
        radar4_name = manitou.radars[c4]['file_name']
        radar1_paths.append(os.path.join(data_root, radar_dir, radar1_name))
        radar2_paths.append(os.path.join(data_root, radar_dir, radar2_name))
        radar3_paths.append(os.path.join(data_root, radar_dir, radar3_name))
        radar4_paths.append(os.path.join(data_root, radar_dir, radar4_name))

    data_cfg = {
        'camera1': cam1_paths,
        'camera2': cam2_paths,
        'camera3': cam3_paths,
        'camera4': cam4_paths,
        'radar1': radar1_paths,
        'radar2': radar2_paths,
        'radar3': radar3_paths,
        'radar4': radar4_paths,
        'calib_params': calib_params,
        'filter_cfg': {}
    }
    loader = LoadManitouImagesAndRadar(data_cfg, radar_accumulation=2, batch=1)
    print(f"Number of frames: {len(loader)}")
    for paths, batch, info in loader:
        print(info)
        for i, b in enumerate(batch):
            print(f"Batch {i+1}:")
            print(f"Camera 1: {b['cam1'].shape}, Camera 2: {b['cam2'].shape}, "
                  f"Camera 3: {b['cam3'].shape}, Camera 4: {b['cam4'].shape}")
            print(f"Radar: {b['radar'].global_radar.shape}")