import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO, _isArrayLike
from scipy.spatial.transform import Rotation as R

from ultralytics.utils import LOGGER, yaml_load


class ManitouAPI(COCO):
    """
    Inherit official COCO class in order to parse the annotations of Manitou dataset to support video tasks.
    Part of the code is adapted from: mmtracking/mmtrack/datasets/api_wrappers/coco_video_api.py.

    Methods:
        createIndex: Create index for the dataset.
        info: Print information about the dataset.
        get_img_ids_from_vid: Get image ids from given video id.
        get_ins_ids_from_vid: Get instance ids from given video id.
        get_img_ids_from_ins_id: Get image ids from given instance id.
        load_vids: Get video information of given video ids.
        get_ann_ids: Get annotation ids that satisfy given filter conditions.
        get_img_ids: Get image ids that satisfy given filter conditions.
        get_radar_ids: Get radar ids that satisfy given filter conditions.
        get_vid_ids: Get video ids that satisfy given filter conditions.
        get_bag_ids: Get bag ids that satisfy given filter conditions.
        load_radars: Load radars with the specified ids.
        load_bags: Load bags with the specified ids.
        load_anns: Load annotations with the specified ids.
        load_imgs: Load images with the specified ids.
    """

    def __init__(self, annotation_file):
        super().__init__(annotation_file)

    def createIndex(self):
        print("Creating index...")

        anns, cats, imgs, radars, vids, bags = {}, {}, {}, {}, {}, {}
        (imgToAnns, catToImgs, vidToImgs, vidToInstances, instancesToImgs, bagToVids) = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann
                if "instance_id" in ann:
                    instancesToImgs[ann["instance_id"]].append(ann["image_id"])
                    if "video_id" in ann and ann["instance_id"] not in vidToInstances[ann["video_id"]]:
                        vidToInstances[ann["video_id"]].append(ann["instance_id"])

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                vidToImgs[img["video_id"]].append(img["id"])
                imgs[img["id"]] = img

        if "radars" in self.dataset:
            for radar in self.dataset["radars"]:
                radars[radar["id"]] = radar

        if "videos" in self.dataset:
            for vid in self.dataset["videos"]:
                vids[vid["id"]] = vid

        if "bags" in self.dataset:
            for bag in self.dataset["bags"]:
                bags[bag["id"]] = bag

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        if "bags" in self.dataset and "videos" in self.dataset:
            for vid in self.dataset["videos"]:
                bagToVids[vid["bag_id"]].append(vid["id"])

        print("Index created.")

        self.anns = anns
        self.cats = cats
        self.imgs = imgs
        self.radars = radars
        self.vids = vids
        self.bags = bags
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.vidToImgs = vidToImgs
        self.vidToInstances = vidToInstances
        self.instancesToImgs = instancesToImgs
        self.bagToVids = bagToVids

    def info(self):
        """
        Print information about the annotation file.

        Returns:
            None.
        """
        if "info" in self.dataset:
            for key, value in self.dataset["info"].items():
                print(f"{key}: {value}")
        else:
            print("No info field in annotation file.")
            print(f"Number of bags: {len(self.bags)}")
            print(f"Number of videos: {len(self.vids)}")
            print(f"Number of images: {len(self.imgs)}")
            print(f"Number of radars: {len(self.radars)}")
            print(f"Number of categories: {len(self.cats)}")
            print(f"Number of annotations: {len(self.anns)}")

    def get_img_ids_from_vid(self, vidId):
        """
        Get image ids from given video id.

        Args:
            vidId (int): The given video id.

        Returns:
            list[int]: Image ids of given video id.
        """
        img_ids = self.vidToImgs[vidId]
        ids = list(np.zeros([len(img_ids)], dtype=np.int64))
        for img_id in img_ids:
            ids[self.imgs[img_id]["frame_id"]] = self.imgs[img_id]["id"]
        return ids

    def get_ins_ids_from_vid(self, vidId):
        """
        Get instance ids from given video id.

        Args:
            vidId (int): The given video id.

        Returns:
            list[int]: Instance ids of given video id.
        """
        return self.vidToInstances[vidId]

    def get_img_ids_from_ins_id(self, inside):
        """
        Get image ids from given instance id.

        Args:
            inside (int): The given instance id.

        Returns:
            list[int]: Image ids of given instance id.
        """
        return self.instancesToImgs[inside]

    def load_vids(self, ids=[]):
        """
        Get video information of given video ids.

        Default return all videos information.

        Args:
            ids (list[int]): The given video ids. Defaults to [].

        Returns:
            list[dict]: List of video information.
        """
        if _isArrayLike(ids):
            return [self.videos[id] for id in ids]
        elif type(ids) == int:
            return [self.videos[ids]]

    def get_ann_ids(self, imgIds=[], catIds=[], areaRng=[], vidIds=[], bagIds=[], isCrowd=None, isEgo=None):
        return self.getAnnIds(imgIds, catIds, areaRng, vidIds, bagIds, isCrowd, isEgo)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], vidIds=[], bagIds=[], isCrowd=None, isEgo=None):
        """
        Get annotation ids that satisfy given filter conditions.
        Params:
            imgIds (int array)    : Get annotations for given images
            catIds (int array)    : Get annotations for given categories
            areaRng (float array) : Get annotations for given area range (e.g. [0, inf])
            vidIds (int array)    : Get annotations for given videos
            bagIds (int array)    : Get annotations for given bags
            isCrowd (bool)       : Get annotations for given isCrowd (True or False)
            isEgo (bool)         : Get annotations for given isEgo (True or False).

        Returns:
            ids (int array) : Integer array of annotation ids.
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        bagIds = bagIds if _isArrayLike(bagIds) else [bagIds]

        if len(imgIds) == len(catIds) == len(areaRng) == len(vidIds) == len(bagIds) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(imgIds) + len(vidIds) + len(bagIds) > 0:
                # bagIds to vidIds
                bag2vids = [self.bagToVids[bagId] for bagId in bagIds]
                bag_vidIds = list(itertools.chain.from_iterable(bag2vids))

                # vidIds to imgIds
                _vidIds = []
                _vidIds.extend(vidIds + bag_vidIds)
                vid_imgIds = [self.vidToImgs[vidId] for vidId in _vidIds]
                vid_imgIds = list(itertools.chain.from_iterable(vid_imgIds))

                # imgIds to annIds
                _imgIds = []
                _imgIds.extend(imgIds + vid_imgIds)
                img_annIds = [self.imgToAnns[imgId] for imgId in _imgIds]
                anns = list(itertools.chain.from_iterable(img_annIds))
            else:
                anns = self.dataset["annotations"]

            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann["category_id"] in catIds]
            anns = (
                anns
                if len(areaRng) == 0
                else [ann for ann in anns if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]]
            )

        if isCrowd is not None:
            ids1 = [ann["id"] for ann in anns if ann["iscrowd"] == isCrowd]
        else:
            ids1 = [ann["id"] for ann in anns]
        if isEgo is not None:
            ids2 = [ann["id"] for ann in anns if ann["is_ego"] == isEgo]
        else:
            ids2 = [ann["id"] for ann in anns]
        ids = list(set(ids1).intersection(ids2))
        return ids

    def get_img_ids(self, imgIds=[], catIds=[], vidIds=[], bagIds=[]):
        return self.getImgIds(imgIds, catIds, vidIds, bagIds)

    def getImgIds(self, imgIds=[], catIds=[], vidIds=[], bagIds=[]):
        """
        Get image ids that satisfy given filter conditions.

        Params:
            imgIds (int array) : Get images for given image ids
            catIds (int array) : Get images for given category ids
            vidIds (int array) : Get images for given video ids
            bagIds (int array) : Get images for given bag ids
        Returns:
            ids (int array) : Integer array of image ids.
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        bagIds = bagIds if _isArrayLike(bagIds) else [bagIds]

        if len(imgIds) == len(catIds) == len(vidIds) == len(bagIds) == 0:
            ids = self.imgs.keys()
        else:
            bag2vids = [self.bagToVids[bagId] for bagId in bagIds]
            bag_vidIds = list(itertools.chain.from_iterable(bag2vids))
            _vidIds = []
            _vidIds.extend(vidIds + bag_vidIds)
            vid2imgIds = [self.vidToImgs[vidId] for vidId in _vidIds]
            vid_imgIds = list(itertools.chain.from_iterable(vid2imgIds))
            _imgIds = []
            _imgIds.extend(imgIds + vid_imgIds)

            ids = set(_imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids = ids.intersection(set(self.catToImgs[catId]))
        return list(ids)

    def get_radar_ids(self, radarIds=[], catIds=[], vidIds=[], bagIds=[]):
        return self.getRadarIds(radarIds, catIds, vidIds, bagIds)

    def getRadarIds(self, radarIds=[], catIds=[], vidIds=[], bagIds=[]):
        return self.getImgIds(radarIds, catIds, vidIds, bagIds)

    def get_vid_ids(self, vidIds=[], bagIds=[]):
        """
        Get video ids that satisfy given filter conditions.

        Params:
            vidIds (int array) : Get videos for given video ids
            bagIds (int array) : Get videos for given bag ids
        Returns:
            ids (int array) : Integer array of video ids.
        """
        vidIds = vidIds if _isArrayLike(vidIds) else [vidIds]
        bagIds = bagIds if _isArrayLike(bagIds) else [bagIds]

        if len(vidIds) == len(bagIds) == 0:
            ids = self.vids.keys()
        else:
            _vidIds = []
            _vidIds.extend(vidIds)
            bag2vids = [self.bagToVids[bagId] for bagId in bagIds]
            bag_vidIds = list(itertools.chain.from_iterable(bag2vids))
            _vidIds.extend(bag_vidIds)
            ids = set(_vidIds)
        return list(ids)

    def get_bag_ids(self, bagIds=[]):
        return self.getBagIds(bagIds)

    def getBagIds(self, bagIds=[]):
        """
        Get bag ids that satisfy given filter conditions.

        Params:
            bagIds (int array) : Get bags for given bag ids
        Returns:
            ids (int array) : Integer array of bag ids.
        """
        bagIds = bagIds if _isArrayLike(bagIds) else [bagIds]

        if len(bagIds) == 0:
            ids = self.bags.keys()
        else:
            ids = set(bagIds)
        return list(ids)

    def load_radars(self, ids=[]):
        return self.loadRadars(ids)

    def loadRadars(self, ids=[]):
        """
        Load radars with the specified ids.

        Params:
            ids (int array) : Integer array of radar ids
        Returns:
            radars (object array) : Array of radar objects.
        """
        if _isArrayLike(ids):
            return [self.radars[id] for id in ids]
        elif type(ids) == int:
            return [self.radars[ids]]

    def load_bags(self, ids=[]):
        return self.loadBags(ids)

    def loadBags(self, ids=[]):
        """
        Load bags with the specified ids.

        Params:
            ids (int array) : Integer array of bag ids
        Returns:
            bags (object array) : Array of bag objects.
        """
        if _isArrayLike(ids):
            return [self.bags[id] for id in ids]
        elif type(ids) == int:
            return [self.bags[ids]]

    def load_anns(self, ids=[]):
        return self.loadAnns(ids)

    def load_imgs(self, ids=[]):
        return self.loadImgs(ids)

    def get_cat_ids(self, catIds=[]):
        return self.getCatIds(catIds)


def _update_dataset_meta(data):
    """
    Get metadata from the json annotation file.

    Args:
        data (dict): The data dictionary containing the path to the annotation file.

    Returns:
        tuple(list[dict], list): A list of annotation and a list of valid data indices.
    """
    ann_path = data["train"]
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"❌ Annotation file {ann_path} does not exist.")
    manitou = ManitouAPI(ann_path)
    data["cat_ids"] = manitou.get_cat_ids(data["names"])
    data["cat2label"] = {cat_id: cat_id - 1 for cat_id in data["cat_ids"]}  # from 1-based to 0-based
    data["names"] = {l: manitou.cats[c]["name"] for c, l in data["cat2label"].items()}
    data["nc"] = len(data["cat_ids"])

    return data


def get_manitou_calibrations(data_cfg):
    # Get the extrinsic parameters of the camera
    def get_cam_extrinsics(path, invert=False):
        try:
            with open(path) as f:
                data = json.load(f)
            T = np.eye(4)
            T[:3, :3] = np.array(data["R"]["data"]).reshape(3, 3)
            T[:3, 3] = np.array(data["T"]["data"]).reshape(3)
            if invert:
                T = np.linalg.inv(T)  # camera to ego
            return T
        except Exception:
            print(f"❌ Error reading calibration file: {path}")

    # Get the intrinsic parameters of the camera and extrinsic parameters of the radar
    def get_radar_params(path, to_meter=True):
        try:
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
        except Exception:
            print(f"❌ Error reading calibration file: {path}")

    if isinstance(data_cfg, str):
        calib_path = Path(data_cfg).resolve()
    elif isinstance(data_cfg, dict):
        data_root = Path(data_cfg["path"]).resolve()
        calib_path = data_root / data_cfg["calib_prefix"]
    else:
        raise ValueError("❌ Invalid calibration path configuration. It should be a string or a dictionary.")

    if not calib_path.exists():
        raise FileNotFoundError(f"❌ Calibration file path: {calib_path} does not exist.")

    # hardcode camera and radar calibration files, maybe change in the future
    radar1_path = calib_path / "calibration_params_radar1.json"
    radar2_path = calib_path / "calibration_params_radar2.json"
    radar3_path = calib_path / "calibration_params_radar3.json"
    radar4_path = calib_path / "calibration_params_radar4.json"
    cam1_path = calib_path / "cam1.json"
    cam2_path = calib_path / "cam2.json"
    cam3_path = calib_path / "cam3.json"
    cam4_path = calib_path / "cam4.json"

    k1, dist1, Tcam1_radar1 = get_radar_params(radar1_path)
    k2, dist2, Tcam2_radar2 = get_radar_params(radar2_path)
    k3, dist3, Tcam3_radar3 = get_radar_params(radar3_path)
    k4, dist4, Tcam4_radar4 = get_radar_params(radar4_path)

    Tego_cam1 = get_cam_extrinsics(cam1_path, invert=True)
    Tego_cam2 = get_cam_extrinsics(cam2_path, invert=True)
    Tego_cam3 = get_cam_extrinsics(cam3_path, invert=True)
    Tego_cam4 = get_cam_extrinsics(cam4_path, invert=True)

    Tego_radar1 = Tego_cam1 @ Tcam1_radar1
    Tego_radar2 = Tego_cam2 @ Tcam2_radar2
    Tego_radar3 = Tego_cam3 @ Tcam3_radar3
    Tego_radar4 = Tego_cam4 @ Tcam4_radar4

    calib_params = {
        "camera1_K": k1,
        "camera1_D": dist1,
        "camera2_K": k2,
        "camera2_D": dist2,
        "camera3_K": k3,
        "camera3_D": dist3,
        "camera4_K": k4,
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

    return calib_params


def get_manitou_dataset(cfg_path):
    """
    Get the annotation path from the yaml file.

    Returns:
        (tuple): Tuple containing training and validation datasets.
    """
    try:
        data = yaml_load(cfg_path)
    except Exception as e:
        LOGGER.error(f"❌ Error loading data configuration file: {e}")
        raise

    data["channels"] = data.get("channels", 3)
    data["calib_prefix"] = data.get("calib_prefix", "")
    data["img_prefix"] = data.get("img_prefix", "")
    data["radar_prefix"] = data.get("radar_prefix", "")
    data["accumulation"] = data.get("accumulation", 1)
    data["names"] = data.get("names", None)
    if data["names"] is None:
        raise ValueError("❌ Class names are not provided in the data configuration file.")

    if isinstance(data["names"], dict):
        data["names"] = list(data["names"].values())  # convert dict (id: name) to list (name)

    path = Path(data["path"]).resolve()
    data["path"] = path
    data["train"] = str(path / data["train"])
    data["val"] = str(path / data["val"])

    data = _update_dataset_meta(data)

    return data


if __name__ == "__main__":
    import os

    # Example usage
    data_root = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou"
    radar_dir = "radars"
    cam_dir = "key_frames"

    annotations_path = os.path.join(data_root, "annotations", "manitou_coco_train.json")
    manitou = ManitouAPI(annotations_path)
    manitou.info()

    # get bag ids
    bag_ids = manitou.get_bag_ids(bagIds=[1])
    print(f"Bag ids: {bag_ids}")

    # get video ids for each bag
    for bag_id in bag_ids:
        video_ids = manitou.get_vid_ids(bagIds=[bag_id])
        print(f"Bag id: {bag_id}, Bag name: {manitou.bags[bag_id]['name']} has {len(video_ids)} videos")
        for video_id in video_ids:
            print(f"\tVideo id: {video_id}, Video name: {manitou.vids[video_id]['name']}")

            # get image ids for each video
            image_ids = manitou.get_img_ids(vidIds=[video_id])
            print(f"\t\tVideo id: {video_id}, Video name: {manitou.vids[video_id]['name']} has {len(image_ids)} images")

            # get radar ids for each video
            radar_ids = manitou.get_radar_ids(vidIds=[video_id])
            print(f"\t\tVideo id: {video_id}, Video name: {manitou.vids[video_id]['name']} has {len(radar_ids)} radars")

    # get ann ids for one image
    image_ids = manitou.get_img_ids(bagIds=[1])
    image_id = image_ids[100]
    print(f"Image id: {image_id}, Image name: {manitou.imgs[image_id]['file_name']}")
    ann_ids = manitou.get_ann_ids(imgIds=[image_id])
    print(f"Image id: {image_id} has {len(ann_ids)} annotations")
    anns = manitou.load_anns(ann_ids)

    for ann in anns:
        print(
            f"\tAnnotation id: {ann['id']}, Category id: {ann['category_id']}, Area: {ann['area']}, Is Ego: {ann['is_ego']}"
        )
        print(f"\t\tBounding box: {ann['bbox']}")

    # get radar name
    video_ids = manitou.get_vid_ids(bagIds=[1])
    image_ids = manitou.get_img_ids(vidIds=[video_ids[0]])
    print(f"Video id: {video_ids[0]}, Video name: {manitou.vids[video_ids[0]]['name']}")
    for image_id in image_ids:
        img_name = manitou.imgs[image_id]["file_name"]
        img_time_stamp = manitou.imgs[image_id]["time_stamp"]
        radar_name = manitou.radars[image_id]["file_name"]
        radar_time_stamp = manitou.radars[image_id]["time_stamp"]

        print(
            f"\tImage name: {img_name}, Radar name: {radar_name}, \n\t\tTimeStamp: {img_time_stamp} <-> {radar_time_stamp}"
        )
