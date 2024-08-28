# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator


class AIGym:
    """A class for managing gym exercises by analyzing people's poses in a real-time video stream."""

    def __init__(self, **kwargs):
        """
        Initializes the AIGym instance with specified parameters for analyzing gym activities in a video stream.

        Args:
            kwargs (dict): Dictionary of arguments that allow customization of the AIGym instance. These can include various settings such as keypoints to choose, angles, and other relevant configurations.
        """
        DEFAULT_CFG_DICT.update(kwargs)
        self.model = YOLO(DEFAULT_CFG_DICT["model"])

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.stage = None
        self.count = []
        self.angle = []
        self.stage = []
        self.annotator = None
        self.env_check = check_imshow(warn=True)  # Check if environment supports imshow
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

    def start_counting(self, im0):
        """
        A function to count gym steps.

        Args:
            im0 (ndarray): The current frame from the video stream.

        Returns:
            im0 (ndarray): The processed image frame.
            results (list): Pose estimation data.
        """
        results = self.model.track(
            source=im0,
            persist=True,
            tracker=DEFAULT_CFG_DICT["tracker"],
            classes=DEFAULT_CFG_DICT["classes"],
            iou=DEFAULT_CFG_DICT["iou"],
            conf=DEFAULT_CFG_DICT["conf"],
        )
        if not len(results[0]):
            return im0

        if len(results[0]) > len(self.count):
            new_human = len(results[0]) - len(self.count)
            self.count += [0] * new_human
            self.angle += [0] * new_human
            self.stage += ["-"] * new_human

        keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=DEFAULT_CFG_DICT["line_width"])

        for ind, k in enumerate(reversed(keypoints)):
            # Estimate angle and draw specific points based on pose type
            if DEFAULT_CFG_DICT["pose_type"] in {"pushup", "pullup", "abworkout", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(DEFAULT_CFG_DICT["kpts_to_check"][0])].cpu(),
                    k[int(DEFAULT_CFG_DICT["kpts_to_check"][1])].cpu(),
                    k[int(DEFAULT_CFG_DICT["kpts_to_check"][2])].cpu(),
                )
                im0 = self.annotator.draw_specific_points(
                    k, DEFAULT_CFG_DICT["kpts_to_check"], shape=(640, 640), radius=DEFAULT_CFG_DICT["line_width"] * 4
                )

                # Check and update pose stages and counts based on angle
                if DEFAULT_CFG_DICT["pose_type"] in {"abworkout", "pullup"}:
                    if self.angle[ind] > DEFAULT_CFG_DICT["pose_up_angle"]:
                        self.stage[ind] = "down"
                    if self.angle[ind] < DEFAULT_CFG_DICT["pose_down_angle"] and self.stage[ind] == "down":
                        self.stage[ind] = "up"
                        self.count[ind] += 1

                elif DEFAULT_CFG_DICT["pose_type"] in {"pushup", "squat"}:
                    if self.angle[ind] > DEFAULT_CFG_DICT["pose_up_angle"]:
                        self.stage[ind] = "up"
                    if self.angle[ind] < DEFAULT_CFG_DICT["pose_down_angle"] and self.stage[ind] == "up":
                        self.stage[ind] = "down"
                        self.count[ind] += 1

                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(DEFAULT_CFG_DICT["kpts_to_check"][1])],
                )

        # Display the image if the environment supports it and view_img is set to True
        if self.env_check and DEFAULT_CFG_DICT["show"]:
            cv2.imshow("Ultralytics Solutions", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return im0


if __name__ == "__main__":
    kpts_to_check = {"kpts_to_check": {0, 1, 2}}  # example keypoints
    aigym = AIGym(data=kpts_to_check)
