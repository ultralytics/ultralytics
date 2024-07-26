# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import cv2

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator

FILE = Path(__file__).resolve()  # get path of file


class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self, **kwargs):
        """Initialize the AiGYM class with kwargs arguments."""
        self.args = get_cfg(extract_cfg_data(FILE))
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        self.angle = None
        self.count = None
        self.stage = None
        self.count = []
        self.angle = []
        self.stage = []
        self.annotator = None
        self.env_check = check_imshow(warn=True)

    def start_counting(self, im0, results):
        """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data.
        """
        if not len(results[0]):
            return im0

        if len(results[0]) > len(self.count):
            new_human = len(results[0]) - len(self.count)
            self.count += [0] * new_human
            self.angle += [0] * new_human
            self.stage += ["-"] * new_human

        keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=self.args.line_thickness)

        for ind, k in enumerate(reversed(keypoints)):
            # Estimate angle and draw specific points based on pose type
            if self.args.pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.args.kpts_to_check[0])].cpu(),
                    k[int(self.args.kpts_to_check[1])].cpu(),
                    k[int(self.args.kpts_to_check[2])].cpu(),
                )
                im0 = self.annotator.draw_specific_points(k, self.args.kpts_to_check, shape=(640, 640), radius=10)

                # Check and update pose stages and counts based on angle
                if self.args.pose_type in {"abworkout", "pullup"}:
                    if self.angle[ind] > self.args.pose_up_angle:
                        self.stage[ind] = "down"
                    if self.angle[ind] < self.args.pose_down_angle and self.stage[ind] == "down":
                        self.stage[ind] = "up"
                        self.count[ind] += 1

                elif self.args.pose_type in {"pushup", "squat"}:
                    if self.angle[ind] > self.args.pose_up_angle:
                        self.stage[ind] = "up"
                    if self.angle[ind] < self.args.pose_down_angle and self.stage[ind] == "up":
                        self.stage[ind] = "down"
                        self.count[ind] += 1

                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.args.kpts_to_check[1])],
                )

            # Draw keypoints
            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        # Display the image if environment supports it and view_img is True
        if self.env_check and self.args.view_img:
            cv2.imshow(self.args.window_name, im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    kpts_to_check = [0, 1, 2]  # example keypoints
    aigym = AIGym(kpts_to_check=kpts_to_check)
