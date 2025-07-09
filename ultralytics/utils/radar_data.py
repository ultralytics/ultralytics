"""
Radar Point Cloud Fields:
[0] x
[1] y
[2] z
[3] radial_distance
[4] radial_velocity
[5] azimuth_angle
[6] elevation_angle
[7] radar_cross_section
[8] signal_noise_ratio
[9] radial_distance_variance
[10] radial_velocity_variance
[11] azimuth_angle_variance
[12] elevation_angle_variance
[13] radial_distance_velocity_covariance
[14] velocity_resolution_processing_probability
[15] azimuth_angle_probability
[16] elevation_angle_probability
[17] measurement_status
[18] idx_azimuth_ambiguity_peer.
"""

from types import SimpleNamespace

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN


def apply_clustering(radars, eps=0.5, min_samples=1):
    """
    Apply clustering to radar points.

    Args:
        radars: (N, 19) array of radar points.

    Returns:
        means: List of mean points for each cluster. (x,y,z,radial_distance).
    """

    def apply_dbscan(xyz, eps, min_samples=1, **kwargs):
        db = DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit(xyz)
        labels = db.labels_
        return labels

    points = radars[:, :3].copy()
    points[:, 2] = points[:, 2] / 5  # scale z-axis
    labels = apply_dbscan(points, eps=eps, min_samples=min_samples, n_jobs=3, metric="l2")
    means = []
    for label in np.unique(labels):
        if label != -1:
            means.append(np.mean(radars[:, :4][labels == label], axis=0))
    return np.array(means, dtype=np.float32)


def equidistance_projection(points, K, D):
    """
    Project points to the camera image using equidistance projection.

    Args:
        points: (N, 3) array of points in the camera frame.
        K: (3, 3) camera intrinsic matrix.
        D: (4,) distortion coefficients.

    Returns:
        (N, 2) array of projected points in the image plane.
    """
    if points.shape[0] == 0:
        return np.zeros((0, 2))  # Return empty array if no points

    Xc = points[:, 0]  # x-coordinates in camera frame
    Yc = points[:, 1]  # y-coordinates in camera frame
    Zc = points[:, 2]  # z-coordinates in camera frame

    r = np.hypot(Xc, Yc)  # sqrt(x^2 + y^2)
    theta = np.arctan2(r, Zc)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4

    # equidiatance model
    k1, k2, k3, k4 = D.flatten()[:4]
    theta_d = theta + k1 * theta * theta2 + k2 * theta * theta4 + k3 * theta * theta6 + k4 * theta * theta8
    scale = np.where(r > 1e-8, theta_d / r, 0.0)
    xd = Xc * scale
    yd = Yc * scale

    # to pixel coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * xd + cx
    v = fy * yd + cy

    return np.vstack((u, v)).T


class FilterParam(dict):
    def __init__(self, args):
        """
        Params:
            args:
                -range: tuple (x_min, x_max, y_min, y_max, z_min, z_max), range of radar data to filter
                -rcs: float, RCS threshold
                -vel: float, velocity threshold
                -elevation_angle: tuple (min_angle, max_angle), elevation angle range.
        """
        if isinstance(args, SimpleNamespace):
            args_dict = vars(args)
        elif isinstance(args, dict):
            args_dict = args
        else:
            raise TypeError(f"FilterParam expects a SimpleNamespace or dict, got {type(args)}")
        # add keys to the dictionary
        self["range"] = args_dict.get("range", None)
        if self["range"] is None:
            self["range"] = [0, 45, -45, 45, -1, 6]
        self["rcs"] = args_dict.get("rcs", -15)
        self["snr"] = args_dict.get("snr", 10)
        self["vel"] = args_dict.get("vel", None)
        self["dist_var"] = args_dict.get("dist_var", 0.4)
        self["azimuth_var"] = args_dict.get("azimuth_var", 0.4)
        self["elevation_var"] = args_dict.get("elevation_var", 0.4)
        self["elevation_angle"] = args_dict.get("elevation_angle", [-1, 12])
        self["measurement_status"] = args_dict.get("measurement_status", 10)


class ManitouRadarPC:
    """Radar Point Cloud class for processing and managing radar point cloud data."""

    def __init__(self, radar1, radar2, radar3, radar4, calib_params, filter_cfg):
        self._raw_radar1 = radar1
        self._raw_radar2 = radar2
        self._raw_radar3 = radar3
        self._raw_radar4 = radar4
        # Calibration parameters (NOTE: the translations are in centimeters)
        self._camera1_K = calib_params["camera1_K"]
        self._camera2_K = calib_params["camera2_K"]
        self._camera3_K = calib_params["camera3_K"]
        self._camera4_K = calib_params["camera4_K"]
        self._camera1_D = calib_params["camera1_D"]
        self._camera2_D = calib_params["camera2_D"]
        self._camera3_D = calib_params["camera3_D"]
        self._camera4_D = calib_params["camera4_D"]
        self._Tec1 = calib_params["eCamera1"]
        self._Tec2 = calib_params["eCamera2"]
        self._Tec3 = calib_params["eCamera3"]
        self._Tec4 = calib_params["eCamera4"]
        self._Ter1 = calib_params["eRadar1"]
        self._Ter2 = calib_params["eRadar2"]
        self._Ter3 = calib_params["eRadar3"]
        self._Ter4 = calib_params["eRadar4"]

        self._filtered_param = FilterParam(filter_cfg)

        self._filtered_radar1 = self._filter_radar_data(self._raw_radar1)
        self._filtered_radar2 = self._filter_radar_data(self._raw_radar2)
        self._filtered_radar3 = self._filter_radar_data(self._raw_radar3)
        self._filtered_radar4 = self._filter_radar_data(self._raw_radar4)

        # Convert to centimeters after filtering
        self._global_radar = self._to_global_coordinates()
        self._global_colors = self.generate_colors_distance(self._global_radar[:, 3], 0, 30)

    def _filter_radar_data(self, points):
        """Filter radar data based on the range (x_min, x_max, y_min, y_max, z_min, z_max) and other parameters."""
        filtered_points = points
        if self._filtered_param["measurement_status"] is not None:
            status_max = self._filtered_param["measurement_status"]
            mask = filtered_points[:, 17] <= status_max
            filtered_points = filtered_points[mask]
        if self._filtered_param["rcs"] is not None:
            rcs_min = self._filtered_param["rcs"]
            mask = filtered_points[:, 7] >= rcs_min
            filtered_points = filtered_points[mask]
        if self._filtered_param["snr"] is not None:
            snr_min = self._filtered_param["snr"]
            mask = filtered_points[:, 8] >= snr_min
            filtered_points = filtered_points[mask]
        if self._filtered_param["dist_var"] is not None:
            dist_var_max = self._filtered_param["dist_var"]
            mask = filtered_points[:, 9] <= dist_var_max
            filtered_points = filtered_points[mask]
        if self._filtered_param["azimuth_var"] is not None:
            azimuth_var_max = self._filtered_param["azimuth_var"]
            mask = filtered_points[:, 11] <= azimuth_var_max
            filtered_points = filtered_points[mask]
        if self._filtered_param["elevation_var"] is not None:
            elevation_var_max = self._filtered_param["elevation_var"]
            mask = filtered_points[:, 12] <= elevation_var_max
            filtered_points = filtered_points[mask]
        if self._filtered_param["vel"] is not None:
            vel_min = self._filtered_param["vel"]
            mask = np.abs(filtered_points[:, 4]) >= vel_min
            filtered_points = filtered_points[mask]
        if self._filtered_param["elevation_angle"] is not None:
            ele_min, ele_max = self._filtered_param["elevation_angle"]
            # to radian
            ele_min = np.radians(ele_min)
            ele_max = np.radians(ele_max)
            mask = (ele_min <= filtered_points[:, 6]) & (filtered_points[:, 6] <= ele_max)
            filtered_points = filtered_points[mask]
        if self._filtered_param["range"] is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self._filtered_param["range"]
            mask = (
                (x_min <= filtered_points[:, 0])
                & (filtered_points[:, 0] <= x_max)
                & (y_min <= filtered_points[:, 1])
                & (filtered_points[:, 1] <= y_max)
                & (z_min <= filtered_points[:, 2])
                & (filtered_points[:, 2] <= z_max)
            )
            filtered_points = filtered_points[mask]

        return filtered_points

    def _transform_pc(self, points, T):
        """Transform point cloud data using the transformation matrix T."""
        if points.shape[0] == 0:
            return points
        transformed_points = np.dot(T, np.vstack([points[:, :3].T, np.ones(points.shape[0])])).T[:, :3]
        points[:, :3] = transformed_points[:, :3]  # Update x, y, z coordinates
        if points.shape[1] > 3:
            # re-compute the radial distance
            points[:, 3] = np.linalg.norm(transformed_points[:, :3], axis=1)
        return points

    def _to_global_coordinates(self):
        """Concatenate all radar point clouds into a single point cloud in global coordinates."""
        # Transform each radar point cloud to global coordinates
        radar1_transformed = self._transform_pc(self._filtered_radar1.copy(), self._Ter1)
        radar2_transformed = self._transform_pc(self._filtered_radar2.copy(), self._Ter2)
        radar3_transformed = self._transform_pc(self._filtered_radar3.copy(), self._Ter3)
        radar4_transformed = self._transform_pc(self._filtered_radar4.copy(), self._Ter4)

        global_pcs = np.vstack((radar1_transformed, radar2_transformed, radar3_transformed, radar4_transformed))
        # add a global index to each point
        global_pcs = np.hstack((global_pcs, np.arange(global_pcs.shape[0]).reshape(-1, 1)))
        return global_pcs

    def _project_points(self, points, T, K, D, HFOV=195):
        if points.shape[0] == 0:
            return np.zeros((0, 2)), np.array([], dtype=bool)
        # Transform points to camera frame
        points_cam = self._transform_pc(points[:, :3].copy(), T)
        # mask_z = points_cam[:, 2] > 0  #  points in front of the camera
        angles = np.arctan2(points_cam[:, 0], points_cam[:, 2])
        hfov_rad = np.deg2rad(HFOV)
        mask_hfov = np.abs(angles) <= (hfov_rad / 2)  # points within the horizontal FOV
        mask = mask_hfov
        points_cam = points_cam[mask]
        # Project points to image plane
        pix_cam = equidistance_projection(points_cam[:, :3], K, D)

        return pix_cam, mask

    def project_to_camera(self, cam_idx):
        assert cam_idx in [1, 2, 3, 4], "Camera index must be 1, 2, 3, or 4."
        if cam_idx == 1:
            K = self._camera1_K
            D = self._camera1_D
            Tgc = self._Tec1
        elif cam_idx == 2:
            K = self._camera2_K
            D = self._camera2_D
            Tgc = self._Tec2
        elif cam_idx == 3:
            K = self._camera3_K
            D = self._camera3_D
            Tgc = self._Tec3
        elif cam_idx == 4:
            K = self._camera4_K
            D = self._camera4_D
            Tgc = self._Tec4

        # Project global radar points to camera image plane
        Tcg = np.linalg.inv(Tgc)  # global to camera transformation
        projected_points, mask = self._project_points(self._global_radar, Tcg, K, D)

        return projected_points, mask

    def get_overlay_image(self, cam_idx, img):
        """
        Overlay radar points on the camera image.

        Args:
            cam_idx: Camera index (1, 2, 3, or 4).
            img (str or np.ndarray): Path to the camera image or the image itself.

        Returns:
            Image with radar points overlaid.
        """
        assert cam_idx in [1, 2, 3, 4], "Camera index must be 1, 2, 3, or 4."
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"Image not found at {img}")
        elif isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Image must be a 3-channel RGB image.")
        else:
            raise TypeError("Image must be a file path or a 3-channel RGB numpy array.")

        projected_points, mask = self.project_to_camera(cam_idx)
        colors = self._global_colors[mask]
        for p, c in zip(projected_points, colors):
            cv2.circle(img, tuple(p.astype(np.int32)), 4, [int(i) for i in c], -1)
        return img

    def get_radar_bev(self, rangeX=(-30, 30), rangeY=(-30, 30)):
        canvas_width = int((rangeY[1] - rangeY[0]) * 20)
        canvas_height = int((rangeX[1] - rangeX[0]) * 20)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 0  # Black background

        X_min = rangeX[0]
        X_max = rangeX[1]
        Y_min = rangeY[0]
        Y_max = rangeY[1]

        canvas_height, canvas_width = canvas.shape[:2]

        # Draw radar points
        for p, c in zip(self._global_radar, self._global_colors):
            x = canvas_width - int((p[1] - Y_min) / (Y_max - Y_min) * canvas_width)
            y = canvas_height - int((p[0] - X_min) / (X_max - X_min) * canvas_height)
            cv2.circle(canvas, (x, y), 1, [int(i) for i in c], -1)

        return canvas

    def generate_colors_distance(self, distances, min_dist, max_dist):
        """Generate colors based on distances using a colormap."""
        norm = mcolors.Normalize(vmin=min_dist, vmax=max_dist)
        cmap = plt.get_cmap("jet")
        # warm
        # cmap = plt.get_cmap('autumn')
        clip_dist = np.clip(distances, min_dist, max_dist)
        colors = cmap(norm(clip_dist))[:, :3] * 255
        colors = colors.astype(np.uint8)  # [:,::-1]
        return colors

    @property
    def global_radar(self):
        return self._global_radar.copy()

    def update_camera_intrinsics(self, cam_idx, new_K):
        """
        Update camera intrinsic parameters.

        Args:
            cam_idx: Camera index (1, 2, 3, or 4).
            new_K: New camera intrinsic matrix (3, 3).
        """
        assert cam_idx in [1, 2, 3, 4], "Camera index must be 1, 2, 3, or 4."
        assert new_K.shape == (3, 3), "New camera intrinsic matrix must be of shape (3, 3)."
        if cam_idx == 1:
            self._camera1_K = new_K
        elif cam_idx == 2:
            self._camera2_K = new_K
        elif cam_idx == 3:
            self._camera3_K = new_K
        elif cam_idx == 4:
            self._camera4_K = new_K


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path

    from ultralytics.data.manitou_api import ManitouAPI

    # Get the extrinsic parameters of the camera
    def get_cam_extrinsics(path, invert=False):
        with open(path) as f:
            data = json.load(f)
        T = np.eye(4)
        T[:3, :3] = np.array(data["R"]["data"]).reshape(3, 3)
        T[:3, 3] = np.array(data["T"]["data"]).reshape(3)
        if invert:
            T = np.linalg.inv(T)  # camera to ego
        return T

    # Get the intrinsic parameters of the camera and extrinsic parameters of the radar
    def get_radar_params(path, to_meter):
        with open(path) as f:
            data = json.load(f)
        matrix_K = np.array(data["camera_matrix"]).reshape(3, 3)
        dist = np.array(data["distortion"]).reshape(
            4,
        )
        rvec = np.array(data["cam_radar_rvecs"]).reshape(
            3,
        )
        tvec = np.array(data["cam_radar_tvecs"]).reshape(
            3,
        )
        if to_meter:
            tvec = tvec / 100  # convert to meter
        Tcam_radar = np.eye(4)
        Tcam_radar[:3, :3] = R.from_rotvec(rvec).as_matrix()
        Tcam_radar[:3, 3] = tvec
        return matrix_K, dist, Tcam_radar

    radar1_path = "/home/shu/dataset/manitou/calibration/calibration_params_radar1.json"
    radar2_path = "/home/shu/dataset/manitou/calibration/calibration_params_radar2.json"
    radar3_path = "/home/shu/dataset/manitou/calibration/calibration_params_radar3.json"
    radar4_path = "/home/shu/dataset/manitou/calibration/calibration_params_radar4.json"
    cam1_path = "/home/shu/dataset/manitou/calibration/cam1.json"
    cam2_path = "/home/shu/dataset/manitou/calibration/cam2.json"
    cam3_path = "/home/shu/dataset/manitou/calibration/cam3.json"
    cam4_path = "/home/shu/dataset/manitou/calibration/cam4.json"

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
        "camera1_K": K1,
        "camera1_D": dist1,
        "camera2_K": K2,
        "camera2_D": dist2,
        "camera3_K": K3,
        "camera3_D": dist3,
        "camera4_K": K4,
        "camera4_D": dist4,
        "eCamera1": Tego_cam1,
        "eCamera2": Tego_cam2,
        "eCamera3": Tego_cam3,
        "eCamera4": Tego_cam4,
        "eRadar1": Tego_radar1,
        "eRadar2": Tego_radar2,
        "eRadar3": Tego_radar3,
        "eRadar4": Tego_radar4,
    }

    data_root = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou"
    radar_dir = "radars"
    cam_dir = "key_frames"

    annotations_path = os.path.join(data_root, "annotations_multi_view_mini", "manitou_coco_val_remap.json")
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
        if "camera1" in manitou.vids[video_id]["name"]:
            cam1_ids = imgs
        elif "camera2" in manitou.vids[video_id]["name"]:
            cam2_ids = imgs
        elif "camera3" in manitou.vids[video_id]["name"]:
            cam3_ids = imgs
        elif "camera4" in manitou.vids[video_id]["name"]:
            cam4_ids = imgs

    assert len(cam1_ids) == len(cam2_ids) == len(cam3_ids) == len(cam4_ids), "Camera IDs must be of the same length."

    # Manitou
    import math

    from ultralytics.data.augmentV1 import ManitouResizeCrop

    pre_crop_cfg = {"is_crop": False, "scale": 1, "target_size": (1552, 1936), "original_size": (1552, 1936)}
    h = 1552 // 32 * 32
    w = math.ceil(1936 / 32) * 32
    pre_crop_cfg["is_crop"] = True
    pre_crop_cfg["scale"] = w / 1936
    pre_crop_cfg["target_size"] = (h, w)
    imgsz = (h, w)
    print(f"Image size {(1552, 1936)} is not divisible by stride {32}, resizing and cropping to {(h, w)}")
    pre_crop = ManitouResizeCrop(
        pre_crop_cfg["scale"], pre_crop_cfg["target_size"], pre_crop_cfg["original_size"], p=1.0
    )

    # update camera intrinsics for MaintouResizeCrop
    h, w = pre_crop_cfg["original_size"]
    crop_h, crop_w = pre_crop_cfg["target_size"]
    new_h, new_w = int(h * pre_crop_cfg["scale"]), int(w * pre_crop_cfg["scale"])
    y_off = new_h - crop_h
    for cam_idx in range(1, 5):
        calib_params[f"camera{cam_idx}_K"] = pre_crop.update_camera_intrinsics(calib_params[f"camera{cam_idx}_K"])

    # # update camera intrinsics for left-right flipped images
    # for cam_idx in range(1, 5):
    #     calib_params[f"camera{cam_idx}_K"] = np.array([[-1, 0, w-1],
    #                                                   [0, 1, 0],
    #                                                   [0, 0, 1]]) @ calib_params[f"camera{cam_idx}_K"]

    for idx in range(0, 60):
        cam1_name = manitou.imgs[cam1_ids[idx]]["file_name"]
        cam1_path = os.path.join(data_root, cam_dir, cam1_name)
        cam2_name = manitou.imgs[cam2_ids[idx]]["file_name"]
        cam2_path = os.path.join(data_root, cam_dir, cam2_name)
        cam3_name = manitou.imgs[cam3_ids[idx]]["file_name"]
        cam3_path = os.path.join(data_root, cam_dir, cam3_name)
        cam4_name = manitou.imgs[cam4_ids[idx]]["file_name"]
        cam4_path = os.path.join(data_root, cam_dir, cam4_name)
        radar1_name = manitou.radars[cam1_ids[idx]]["file_name"]
        radar1_path = os.path.join(data_root, radar_dir, radar1_name)
        radar2_name = manitou.radars[cam2_ids[idx]]["file_name"]
        radar2_path = os.path.join(data_root, radar_dir, radar2_name)
        radar3_name = manitou.radars[cam3_ids[idx]]["file_name"]
        radar3_path = os.path.join(data_root, radar_dir, radar3_name)
        radar4_name = manitou.radars[cam4_ids[idx]]["file_name"]
        radar4_path = os.path.join(data_root, radar_dir, radar4_name)
        # idx = 45
        radar1 = np.loadtxt(radar1_path, delimiter=" ", dtype=np.float64)
        radar2 = np.loadtxt(radar2_path, delimiter=" ", dtype=np.float64)
        radar3 = np.loadtxt(radar3_path, delimiter=" ", dtype=np.float64)
        radar4 = np.loadtxt(radar4_path, delimiter=" ", dtype=np.float64)

        accumulation = 1
        for _ in range(accumulation - 1):
            radar1_frame_name = radar1_name.split("/")[-1].split(".")[0]
            if int(radar1_frame_name) - 1 < 0:
                break
            radar1_frame_name = str(Path(radar1_path).parent / f"{int(radar1_frame_name) - 1:06d}.txt")
            radar1 = np.concatenate((np.loadtxt(radar1_frame_name, delimiter=" ", dtype=np.float64), radar1), axis=0)
        for _ in range(accumulation - 1):
            radar2_frame_name = radar2_name.split("/")[-1].split(".")[0]
            if int(radar2_frame_name) - 1 < 0:
                break
            radar2_frame_name = str(Path(radar2_path).parent / f"{int(radar2_frame_name) - 1:06d}.txt")
            radar2 = np.concatenate((np.loadtxt(radar2_frame_name, delimiter=" ", dtype=np.float64), radar2), axis=0)
        for _ in range(accumulation - 1):
            radar3_frame_name = radar3_name.split("/")[-1].split(".")[0]
            if int(radar3_frame_name) - 1 < 0:
                break
            radar2_frame_name = str(Path(radar3_path).parent / f"{int(radar3_frame_name) - 1:06d}.txt")
            radar3 = np.concatenate((np.loadtxt(radar2_frame_name, delimiter=" ", dtype=np.float64), radar3), axis=0)
        for _ in range(accumulation - 1):
            radar4_frame_name = radar4_name.split("/")[-1].split(".")[0]
            if int(radar4_frame_name) - 1 < 0:
                break
            radar4_frame_name = str(Path(radar4_path).parent / f"{int(radar4_frame_name) - 1:06d}.txt")
            radar4 = np.concatenate((np.loadtxt(radar4_frame_name, delimiter=" ", dtype=np.float64), radar4), axis=0)

        name = cam1_name.split("/")[-1].split(".")[0]  # Get the name from the camera path

        filter_cfg = {
            # 'range': [0, 20, -20, 20, 0, 5],  # x_min, x_max, y_min, y_max, z_min, z_max
            # 'rcs': -15.0,  # RCS threshold
            # 'vel': None,  # Velocity threshold
            # 'elevation_angle': [3, 12]  # Elevation angle range
        }

        radar_pc = ManitouRadarPC(radar1, radar2, radar3, radar4, calib_params, filter_cfg)

        print(f"Global radar point cloud shape: {radar_pc.global_radar.shape}")
        # clean the saved directory
        os.makedirs("/home/shu/Documents/test_radar_data/cam1", exist_ok=True)
        os.makedirs("/home/shu/Documents/test_radar_data/cam2", exist_ok=True)
        os.makedirs("/home/shu/Documents/test_radar_data/cam3", exist_ok=True)
        os.makedirs("/home/shu/Documents/test_radar_data/cam4", exist_ok=True)
        os.makedirs("/home/shu/Documents/test_radar_data/radar_bev", exist_ok=True)

        # get radar points on camera 1 image
        img1 = cv2.imread(cam1_path)
        img1 = pre_crop(image=img1)
        # left-right flip the image
        # img1 = cv2.flip(img1, 1)  # flip horizontally
        img_with_radar = radar_pc.get_overlay_image(1, img1)
        cv2.imwrite(f"/home/shu/Documents/test_radar_data/cam1/{name}_radar_overlay.jpg", img_with_radar)

        img2 = cv2.imread(cam2_path)
        img2 = pre_crop(image=img2)
        # left-right flip the image
        # img2 = cv2.flip(img2, 1)
        img_with_radar = radar_pc.get_overlay_image(2, img2)
        cv2.imwrite(f"/home/shu/Documents/test_radar_data/cam2/{name}_radar_overlay.jpg", img_with_radar)

        img3 = cv2.imread(cam3_path)
        img3 = pre_crop(image=img3)
        # left-right flip the image
        # img3 = cv2.flip(img3, 1)  # flip horizontally
        img_with_radar = radar_pc.get_overlay_image(3, img3)
        cv2.imwrite(f"/home/shu/Documents/test_radar_data/cam3/{name}_radar_overlay.jpg", img_with_radar)

        img4 = cv2.imread(cam4_path)
        img4 = pre_crop(image=img4)
        # left-right flip the image
        # img4 = cv2.flip(img4, 1)
        img_with_radar = radar_pc.get_overlay_image(4, img4)
        cv2.imwrite(f"/home/shu/Documents/test_radar_data/cam4/{name}_radar_overlay.jpg", img_with_radar)

        # get the radar BEV
        radar_bev = radar_pc.get_radar_bev(rangeX=(-20, 20), rangeY=(-20, 20))
        cv2.imwrite(f"/home/shu/Documents/test_radar_data/radar_bev/{name}_radar_bev.jpg", radar_bev)
