# TODO: clone the trackeval repo
# TODO: rebuild trackeval to support numpy > 2.0
# TODO: add the MotManitou2DBox to trackeval

import os
from functools import wraps
from collections import defaultdict
import tempfile
import trackeval
from trackeval.utils import TrackEvalException
from pathlib import Path
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLOManitou
from ultralytics.data import ManitouAPI, get_manitou_dataset
from ultralytics.utils import LOGGER, TQDM


class MotManitou2DBox(trackeval.datasets.MotChallenge2DBox):
    def __init__(self, class_name_to_class_id, config=None):
        super(trackeval.datasets.MotChallenge2DBox).__init__()
        # Fill non-given config values with defaults
        self.config = trackeval.utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        self.gt_set = gt_set
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = list(class_name_to_class_id.keys())
        self.class_list = [cls if cls in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. Only pedestrian class is valid.')
        self.class_name_to_class_id = class_name_to_class_id
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))
                        
    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        else:
            file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, is_zipped=self.data_is_zipped, zip_file=None)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str( t+ 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t+1)
            if time_key in read_data.keys():
                try:
                    time_data = np.asarray(read_data[time_key], dtype=np.float32)
                except ValueError:
                    if is_gt:
                        raise TrackEvalException(
                            'Cannot convert gt data for sequence %s to float. Is data corrupted?' % seq)
                    else:
                        raise TrackEvalException(
                            'Cannot convert tracking data from tracker %s, sequence %s to float. Is data corrupted?' % (
                                tracker, seq))
                try:
                    raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                    raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                except IndexError:
                    if is_gt:
                        err = 'Cannot load gt data from sequence %s, because there is not enough ' \
                              'columns in the data.' % seq
                        raise TrackEvalException(err)
                    else:
                        err = 'Cannot load tracker data from tracker %s, sequence %s, because there is not enough ' \
                              'columns in the data.' % (tracker, seq)
                        raise TrackEvalException(err)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    else:
                        raise TrackEvalException(
                            'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                                seq, t))
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.atleast_1d(time_data[:, 6].astype(int))}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 6])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @trackeval._timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        distractor_class_names = []
        distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked']

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            # tracker_confidences = np.ones(tracker_dets.shape[0])   # TODO: delete when bug is fixed
            similarity_scores = raw_data['similarity_scores'][t]

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_tracker = np.array([], int)
            if self.do_preproc and self.benchmark != 'MOT15' and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:

                # Check all classes are valid:
                invalid_classes = np.setdiff1d(np.unique(gt_classes), self.valid_class_numbers)
                if len(invalid_classes) > 0:
                    print(' '.join([str(x) for x in invalid_classes]))
                    raise(TrackEvalException('Attempting to evaluate using invalid gt classes. '
                                             'This warning only triggers if preprocessing is performed, '
                                             'e.g. not for MOT15 or where prepropressing is explicitly disabled. '
                                             'Please either check your gt data, or disable preprocessing. '
                                             'The following invalid classes were found in timestep ' + str(t) + ': ' +
                                             ' '.join([str(x) for x in invalid_classes])))
                    
                # only keep current class
                tracker_go_keep_mask = np.equal(tracker_classes, cls_id)
                tracker_ids = tracker_ids[tracker_go_keep_mask]
                tracker_dets = tracker_dets[tracker_go_keep_mask, :]
                tracker_confidences = tracker_confidences[tracker_go_keep_mask]
                similarity_scores = similarity_scores[:, tracker_go_keep_mask]

                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                to_remove_tracker = match_cols[is_distractor_class]

            # Apply preprocessing to remove all unwanted tracker dets.
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            if self.do_preproc and self.benchmark != 'MOT15':
                gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                  (np.equal(gt_classes, cls_id))
            else:
                # There are no classes for MOT15
                gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

def get_tmpdir() -> str:
    """return the same tmpdir for all processes."""
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
    tmpdir = tempfile.mkdtemp()
    tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
    dir_tensor[:len(tmpdir)] = tmpdir
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    return tmpdir


class EvalManitouMOT:
    default_prefix = 'ManitouMOT-metric'
    allowed_metrics = ['HOTA', 'CLEAR', 'Identity']
    TRACKER = 'user-tracker'
    allowed_benchmarks = ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'DanceTrack']
    
    def __init__(self, data_cfg, tracker_cfg, model, imgsz, conf_thr, max_det, device=None, format_only=False, verbose=False, num_extra_frames=0):
        """Initialize the evaluation class.
        
        Args:
            data_cfg (str): Path to the data configuration file.
            tracker_cfg (str): Path to the tracker configuration file.
            model (Model): The pre-trained model to use for tracking. (need to be initialized before)
            imgsz (tuple): Image size of the dataset
            conf_thr (float): Confidence threshold for the tracker.
            max_det (int): Maximum number of detections per image.
            device (str): Device to use for inference. 
            format_only (bool): Whether to only format the data without evaluation.
            verbose (bool): Whether to print progress information.
            num_extra_frames (int): Number of extra frames to use for tracking. It decides how many additional frames
                are interpolated for two adjacent keyframes. 
        """
        self.data = self.get_data(data_cfg)
        self.model = model
        self.verbose = verbose
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.track_args = {
            "imgsz": imgsz,
            "conf": conf_thr,
            "max_det": max_det,
            "tracker": tracker_cfg,
            "device": device,
            "persist": True,
            "verbose": verbose,
        }
        self.num_extra_frames = num_extra_frames
        self.vid_data = self.prepare_data()     
        
        self.format_only = format_only
        self.class_name_to_class_id = {name: label for label, name in self.data["names"].items()}
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        self.metrics = self.allowed_metrics
 
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir.name = get_tmpdir()
        self.seq_info = defaultdict(
            lambda: dict(seq_length=-1, gt_tracks=[], pred_tracks=[]))
        self.gt_dir = self._get_gt_dir()
        self.pred_dir = self._get_pred_dir()
        LOGGER.info(f"Temporary directory: {self.tmp_dir.name} is created.")
        self.seqmap = os.path.join(self.pred_dir, 'videoseq.txt')
        with open(self.seqmap, 'w') as f:
            f.write('name\n') 
        
    def get_data(self, data_cfg):
        data_cfg = Path(data_cfg)
        if data_cfg.is_file():
            data = get_manitou_dataset(data_cfg)
        else:
            raise FileNotFoundError(f"Data configuration file {data_cfg} not found.")
        
        return data
    
    def prepare_data(self):
        ann_path = self.data["val"]
        cat_ids = self.data["cat_ids"]
        manitou = ManitouAPI(ann_path)

        vid_ids = manitou.get_vid_ids()
        vid_data = defaultdict(dict)
        for vid in vid_ids:
            img_list = []
            label_list = []
            video_name = manitou.vids[vid]["name"]
            img_ids = manitou.get_img_ids_from_vid(vid)
            
            vid_data[video_name]["vid_id"] = vid
            vid_data[video_name]["video_length"] = len(img_ids)
            for img_id in img_ids:
                raw_img_info = manitou.load_imgs([img_id])[0]
                raw_img_info["img_id"] = img_id
                
                ann_ids = manitou.get_ann_ids(imgIds=[img_id], catIds=cat_ids)
                raw_ann_info = manitou.load_anns(ann_ids)
                parsed_label_info = self.parse_label_info({"raw_img_info": raw_img_info, "raw_ann_info": raw_ann_info})
                label_list.append(parsed_label_info)
                img_list.append(parsed_label_info["im_file"])
            vid_data[video_name]["img_list"] = img_list
            vid_data[video_name]["label_list"] = label_list
            
        return vid_data
            
    def parse_label_info(self, raw_label_info):
        """
        Parse label information to target format.
        """
        img_info = raw_label_info["raw_img_info"]
        ann_info = raw_label_info["raw_ann_info"]
        
        img_path = str(Path(self.data["path"]) / self.data["prefix"] / img_info["file_name"])
        img_shape = (img_info["height"], img_info["width"])  # (h, w)
        img_timestamp = img_info["time_stamp"]
        img_frame_id = img_info["frame_id"]
        img_frame_name = img_info["frame_name"]
        is_start = img_info["is_start"]
        is_end = img_info["is_end"]
        
        boxes = []
        classes = []
        ins_ids = []
        mot_confs = []
        visibilities = []
        
        for i, ann in enumerate(ann_info):
            if ann.get("is_ego", False):
                continue
            
            if ann.get('ignore', False):
                continue
            
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            
            # if ann['area'] <= 400 or w < 10 or h < 20:
            #     continue
            
            if ann['category_id'] not in self.data["cat_ids"]:
                continue
            
            # The coco box format is [top_left_x, top_left_y, width, height]  
            # to xyxy          
            boxes.append([x1, y1, x1 + w, y1 + h])
            
            classes.append(self.data["cat2label"][ann['category_id']])
            ins_ids.append(ann['instance_id'])
            mot_confs.append(ann['mot_conf'])
            visibilities.append(ann['visibility'])
        
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            classes = np.array(classes, dtype=np.float32).reshape(-1, 1)
            ins_ids = np.array(ins_ids, dtype=np.float32).reshape(-1, 1)
            mot_confs = np.array(mot_confs, dtype=np.float32).reshape(-1, 1)
            visibilities = np.array(visibilities, dtype=np.float32).reshape(-1, 1)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0, 1), dtype=np.float32)
            ins_ids = np.zeros((0, 1), dtype=np.float32)
            mot_confs = np.zeros((0, 1), dtype=np.float32)
            visibilities = np.zeros((0, 1), dtype=np.float32)
            
        return {
            "im_file": img_path,
            "shape": img_shape,
            "img_timestamp": img_timestamp,
            "img_frame_id": img_frame_id,
            "im_frame_name": img_frame_name,
            "is_start": is_start,
            "is_end": is_end,
            "cls": classes,
            "bboxes": boxes,
            "ins_ids": ins_ids,
            "mot_confs": mot_confs,
            "visibilities": visibilities,
            "bbox_format": "xyxy",
        }
        
    def _augment_frames(self, img_list, label_list):
        if self.num_extra_frames <= 0:
            return img_list, label_list
        
        img_dir = Path(img_list[0]).parent
        suffix = Path(img_list[0]).suffix
        width = len(Path(img_list[0]).stem)
        
        # original key frame index
        base_idx = [int(Path(p).stem) for p in img_list]
        
        new_imgs = []
        new_labels = []
        n_extra = self.num_extra_frames
        
        # Interpolate for two adjacent keyframes
        for i in range(len(base_idx) - 1):
            new_imgs.append(img_list[i])
            new_labels.append(label_list[i])
            
            start, end = base_idx[i], base_idx[i + 1]
            assert start < end, f"Image frames in a wrong order: {start} -> {end}"
            assert n_extra < end - start, f"Number of extra frames {n_extra} should be less than the gap between two keyframes {end - start}"
            for j in range(1, n_extra + 1):
                # uniformly interpolate the frame index
                frac = j / (n_extra + 1)
                idx = int(round(start + (end - start) * frac))
                fname = f"{idx:0{width}d}{suffix}"
                new_imgs.append(str(img_dir / fname))
                new_labels.append(None)  # no label for the interpolated frames
                
        # Add the last key frame
        new_imgs.append(img_list[-1])
        new_labels.append(label_list[-1])
        
        return new_imgs, new_labels
           
    def run(self):
        for video_name, video_data in self.vid_data.items():
            LOGGER.info(f"Processing video: {video_name}")
            img_list = video_data["img_list"]
            label_list = video_data["label_list"]
            
            if self.seq_info[video_name]["seq_length"] == -1:
                self.seq_info[video_name]["seq_length"] = len(self.vid_data[video_name]["img_list"])  # ONLY len(Key-frames)
            
            proc_imgs, proc_labels = self._augment_frames(img_list, label_list)
            if not self.verbose:
                iterator = TQDM(zip(proc_imgs, proc_labels), total=len(proc_imgs), desc=video_name)
            else:
                iterator = zip(proc_imgs, proc_labels)
            
            for img_path, label_info in iterator:
                res = self.model.track(source=img_path, **self.track_args)[0]
                if label_info is not None:
                    seq_info = self.prepare_results(res, label_info)
                    self.seq_info[video_name]["pred_tracks"].extend(seq_info["pred_tracks"])
                    self.seq_info[video_name]["gt_tracks"].extend(seq_info["gt_tracks"])
            
            # Save the results of the video
            self._save_one_video_gts_preds(video_name)
            
            for i in range(len(self.model.predictor.trackers)):
                self.model.predictor.trackers[i].reset()  # Reset the tracker for the next video
        
        # Evaluate         
        self.evaluate()
                
    def prepare_results(self, pred, label):
        seq_info = {}
        frame_id = label["img_frame_id"]
        # load gts
        gt_tracks = [
            np.array([
                frame_id + 1, 
                label["ins_ids"][i][0],
                label["bboxes"][i][0],
                label["bboxes"][i][1],
                label["bboxes"][i][2] - label["bboxes"][i][0],
                label["bboxes"][i][3] - label["bboxes"][i][1],
                label["mot_confs"][i][0],
                label["cls"][i][0],
                label["visibilities"][i][0],
            ]) for i in range(len(label["ins_ids"]))
        ]
        seq_info["gt_tracks"] = gt_tracks
        
        # load preds
        if pred.boxes.id is None:  # to compatible with the implementation of trackers in ultralytics
            pred_tracks = []
        else:    
            boxes = pred.boxes.xyxy.cpu().numpy()
            track_ids = pred.boxes.id.cpu().tolist()
            cls_ids = pred.boxes.cls.cpu().tolist()
            confs = pred.boxes.conf.cpu().tolist()
            pred_tracks = [
                np.array([
                    frame_id + 1, 
                    track_ids[i],
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2] - boxes[i][0],
                    boxes[i][3] - boxes[i][1],
                    confs[i],
                    cls_ids[i],
                ]) for i in range(len(track_ids))
            ]
        seq_info["pred_tracks"] = pred_tracks
        
        return seq_info
    
    def _save_one_video_gts_preds(self, video_name):
        """Save the gt and prediction results."""
        info = self.seq_info[video_name]
        # save predictions
        pred_file = os.path.join(self.pred_dir, video_name + '.txt')

        pred_tracks = np.array(info['pred_tracks'])

        with open(pred_file, 'wt') as f:
            for tracks in pred_tracks:
                line = '%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,-1,-1\n' % (
                    tracks[0], tracks[1], tracks[2], tracks[3], tracks[4],
                    tracks[5], tracks[6], tracks[7])
                f.writelines(line)

        info['pred_tracks'] = []
        # save gts
        if info['gt_tracks']:
            gt_file = os.path.join(self.gt_dir, video_name + '.txt')
            with open(gt_file, 'wt') as f:
                for tracks in info['gt_tracks']:
                    line = '%d,%d,%d,%d,%d,%d,%d,%d,%.5f\n' % (
                        tracks[0], tracks[1], tracks[2], tracks[3], tracks[4],
                        tracks[5], tracks[6], tracks[7], tracks[8])
                    f.writelines(line)
            info['gt_tracks'].clear()
        # save seq info
        with open(self.seqmap, 'a') as f:
            f.write(video_name + '\n')
            f.close()
        
    def _get_gt_dir(self):
        """Get directory to save the gt files."""
        output_dir = os.path.join(self.tmp_dir.name, 'gt')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_pred_dir(self):
        """Get directory to save the prediction results."""
        pred_dir = os.path.join(self.tmp_dir.name, self.TRACKER)
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir
    
    def __del__(self):
        # To avoid tmpdir being cleaned up too early, because in multiple
        # consecutive ValLoops, the value of `self.tmp_dir.name` is unchanged,
        # and calling `tmp_dir.cleanup()` in compute_metrics will cause errors.
        self.tmp_dir.cleanup()
        LOGGER.info(f"Temporary directory: {self.tmp_dir.name} is deleted.")
            
    def compute_metrics(self):
        eval_results = defaultdict(dict)
        
        if self.format_only:
            return eval_results
        
        eval_config = trackeval.Evaluator.get_default_eval_config()
        # need to split out the tracker name
        # caused by the implementation of TrackEval
        pred_dir_tmp = self.pred_dir.rsplit(os.path.sep, 1)[0]
        dataset_config = self.get_dataset_cfg(self.gt_dir, pred_dir_tmp)
        
        evaluator = trackeval.Evaluator(eval_config)
        dataset = MotManitou2DBox(self.class_name_to_class_id, dataset_config)
        dataset = [dataset]
        
        metrics = [
            getattr(trackeval.metrics,
                    metric)(dict(METRICS=[metric], THRESHOLD=0.5))
            for metric in self.metrics
        ]
        output_results, _ = evaluator.evaluate(dataset, metrics)
        for cls in self.class_name_to_class_id.keys():
            output_res = output_results['MotManitou2DBox'][
                self.TRACKER]['COMBINED_SEQ'][cls]

            if 'HOTA' in self.metrics:
                eval_results[cls]['HOTA'] = np.average(output_res['HOTA']['HOTA'])
                eval_results[cls]['AssA'] = np.average(output_res['HOTA']['AssA'])
                eval_results[cls]['DetA'] = np.average(output_res['HOTA']['DetA'])

            if 'CLEAR' in self.metrics:
                eval_results[cls]['MOTA'] = np.average(output_res['CLEAR']['MOTA'])
                eval_results[cls]['MOTP'] = np.average(output_res['CLEAR']['MOTP'])
                eval_results[cls]['IDSW'] = np.average(output_res['CLEAR']['IDSW'])
                eval_results[cls]['TP'] = np.average(output_res['CLEAR']['CLR_TP'])
                eval_results[cls]['FP'] = np.average(output_res['CLEAR']['CLR_FP'])
                eval_results[cls]['FN'] = np.average(output_res['CLEAR']['CLR_FN'])
                eval_results[cls]['Frag'] = np.average(output_res['CLEAR']['Frag'])
                eval_results[cls]['MT'] = np.average(output_res['CLEAR']['MT'])
                eval_results[cls]['ML'] = np.average(output_res['CLEAR']['ML'])

            if 'Identity' in self.metrics:
                eval_results[cls]['IDF1'] = np.average(output_res['Identity']['IDF1'])
                eval_results[cls]['IDTP'] = np.average(output_res['Identity']['IDTP'])
                eval_results[cls]['IDFN'] = np.average(output_res['Identity']['IDFN'])
                eval_results[cls]['IDFP'] = np.average(output_res['Identity']['IDFP'])
                eval_results[cls]['IDP'] = np.average(output_res['Identity']['IDP'])
                eval_results[cls]['IDR'] = np.average(output_res['Identity']['IDR'])

        # clean all txt file
        for txt_name in Path(self.tmp_dir.name).rglob('*.txt'):  # TODO: delete
            os.remove(txt_name)
            
        self._print_eval_results(eval_results)
        return eval_results
    
    def _print_eval_results(self, results):
        for cls, eval_result in results.items():
            tab = self._format_eval_results(eval_result)
            print(f"Evaluation for {cls}:\n{tab}")
            
    def _format_eval_results(self, eval_result):
        # 按 key 排序（可根据需要改成自定义顺序）
        items = sorted(eval_result.items(), key=lambda x: x[0])
        metrics = [k for k, _ in items]
        values = [f"{v:.3f}" for _, v in items]

        # 每列宽度 = metric 名称和 value 长度的最大值
        col_widths = [
            max(len(m), len(val))
            for m, val in zip(metrics, values)
        ]

        # 构造分隔行：+----+------+---+
        sep_pieces = [ '-' * (w + 2) for w in col_widths ]
        sep_line = '+' + '+'.join(sep_pieces) + '+'

        # 第一行：| metric1 | metric2 | ... |
        header_cells = [
            ' ' + m.center(w) + ' '
            for m, w in zip(metrics, col_widths)
        ]
        header_line = '|' + '|'.join(header_cells) + '|'

        # 第二行：| value1  | value2  | ... |
        value_cells = [
            ' ' + val.rjust(w) + ' '
            for val, w in zip(values, col_widths)
        ]
        value_line = '|' + '|'.join(value_cells) + '|'

        # 组合
        return "\n".join([sep_line, header_line, sep_line, value_line, sep_line])     
    
    def evaluate(self):
        """Evaluate the model performance of the whole dataset after processing all batches."""   
        _metrics = self.compute_metrics()
        _metrics = {
            '/'.join([self.default_prefix, k]): v
            for k, v in _metrics.items()
        }
        return _metrics
    
    def get_dataset_cfg(self, gt_folder: str, tracker_folder: str):
        """Get default configs for trackeval.datasets.MotChallenge2DBox.

        Args:
            gt_folder (str): the name of the GT folder
            tracker_folder (str): the name of the tracker folder

        Returns:
            Dataset Configs for MotChallenge2DBox.
        """
        dataset_config = dict(
            # Location of GT data
            GT_FOLDER=gt_folder,
            # Trackers location
            TRACKERS_FOLDER=tracker_folder,
            # Where to save eval results
            # (if None, same as TRACKERS_FOLDER)
            OUTPUT_FOLDER=None,
            # Use self.TRACKER as the default tracker
            TRACKERS_TO_EVAL=[self.TRACKER],
            # Option values: ['pedestrian']
            CLASSES_TO_EVAL=list(self.class_name_to_class_id.keys()),
            # Option Values: 'MOT15', 'MOT16', 'MOT17', 'MOT20', 'DanceTrack'
            BENCHMARK="MOT17",
            # Option Values: 'train', 'test'
            SPLIT_TO_EVAL='train',
            # Whether tracker input files are zipped
            INPUT_AS_ZIP=False,
            # Whether to print current config
            PRINT_CONFIG=True,
            # Whether to perform preprocessing
            # (never done for MOT15)
            DO_PREPROC= True,
            # Tracker files are in
            # TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            TRACKER_SUB_FOLDER='',
            # Output files are saved in
            # OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            OUTPUT_SUB_FOLDER='',
            # Names of trackers to display
            # (if None: TRACKERS_TO_EVAL)
            TRACKER_DISPLAY_NAMES=None,
            # Where seqmaps are found
            # (if None: GT_FOLDER/seqmaps)
            SEQMAP_FOLDER=None,
            # Directly specify seqmap file
            # (if none use seqmap_folder/benchmark-split_to_eval)
            SEQMAP_FILE=self.seqmap,
            # If not None, specify sequences to eval
            # and their number of timesteps
            SEQ_INFO={
                seq: info['seq_length']
                for seq, info in self.seq_info.items()
            },
            # '{gt_folder}/{seq}.txt'
            GT_LOC_FORMAT='{gt_folder}/{seq}.txt',
            # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, the middle 'benchmark-split' folder is skipped for both.
            SKIP_SPLIT_FOL=True,
        )

        return dataset_config


if __name__ == "__main__":
    data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou_mini.yaml" 
    imgsz = (1552, 1936)  # (height, width)
    checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train/weights/best.pt'
    verbose = False
    num_extra_frames = 0
    
    model = YOLOManitou(checkpoint)

    evaluator = EvalManitouMOT(data_cfg, 
                               tracker_cfg="botsort.yaml", 
                               model=model, 
                               imgsz=imgsz, 
                               conf_thr=0.25, 
                               max_det=100, 
                               verbose=verbose, 
                               num_extra_frames=num_extra_frames)
    evaluator.run()