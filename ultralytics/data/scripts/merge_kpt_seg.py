import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from ultralytics.utils.ops import segments2boxes


def match_pose_to_segment(seg_line, pose_lines):
    seg_parts = [x.split() for x in seg_line.strip().splitlines() if len(x)]
    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in seg_parts]  # (cls, xy1...)
    seg_bbox = segments2boxes(segments)[0]

    best_match = None
    min_bbox_diff = float("inf")

    lb = [x.split() for x in pose_lines if len(x)]

    for i, pose_bbox in enumerate([np.array(x[1:5], dtype=np.float32) for x in lb]):
        bbox_diff = sum(abs(seg_bbox[i] - pose_bbox[i]) for i in range(4))
        if bbox_diff < min_bbox_diff:
            min_bbox_diff = bbox_diff
            best_match = pose_lines[i]

    return best_match


def merge_annotations(seg_path, pose_path, output_base_path):
    for subdir, _, _ in os.walk(seg_path):
        relative_path = os.path.relpath(subdir, seg_path)
        pose_subdir = os.path.join(pose_path, relative_path)
        output_subdir = os.path.join(output_base_path, relative_path)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        seg_files = glob.glob(os.path.join(subdir, "*.txt"))

        if not seg_files:
            continue

        for seg_file in tqdm(seg_files, desc=f"Processing {subdir} labels", unit="file"):
            pose_file = os.path.join(pose_subdir, os.path.basename(seg_file))
            output_file = os.path.join(output_subdir, os.path.basename(seg_file))

            if os.path.exists(pose_file):
                with open(seg_file, "r") as seg, open(pose_file, "r") as pose, open(output_file, "w") as out:
                    seg_lines = seg.readlines()
                    pose_lines = pose.readlines()

                    for seg_line in seg_lines:
                        seg_class_index = seg_line.strip().split()[0]
                        if seg_class_index == "0":  # Process only if class index is 0
                            best_match = match_pose_to_segment(seg_line, pose_lines)
                            if best_match:
                                pose_parts = best_match.strip().split()
                                seg_parts = seg_line.strip().split()
                                merged_line = (
                                    pose_parts[0]
                                    + " "
                                    + " ".join(pose_parts[5:])
                                    + " "
                                    + " ".join(seg_parts[1:])
                                    + "\n"
                                )
                                out.write(merged_line)
                        else:
                            # Write segmentation line without pose points
                            out.write(seg_line)


def main(keypoint_dataset, segmentation_dataset, output_dataset):
    merge_annotations(segmentation_dataset, keypoint_dataset, output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge keypoint and segmentation datasets.")
    parser.add_argument("-kpt", "--keypoint_dataset", required=True, help="Path to the keypoint dataset")
    parser.add_argument("-seg", "--segmentation_dataset", required=True, help="Path to the segmentation dataset")
    parser.add_argument("-o", "--output_dataset", required=True, help="Path to the output dataset")

    args = parser.parse_args()

    main(args.keypoint_dataset, args.segmentation_dataset, args.output_dataset)

    print("Merging datasets completed.")
