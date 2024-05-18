# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.solutions import Solutions
from ultralytics.utils.plotting import Annotator, colors


class AIGym(Solutions):
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def start_counting(self, im0, results, frame_count):
        """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data
            frame_count (int): store current frame count
        """
        self.im0 = im0

        if frame_count == 1:
            self.workouts_counts = [0] * len(results[0])
            self.stage = [0] * len(results[0])
            self.angle = ["-" for _ in results[0]]

        keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=self.tf)

        for ind, k in enumerate(reversed(keypoints)):
            if self.pose_type in {"pushup", "pullup", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.kpts_to_check[0])].cpu(),
                    k[int(self.kpts_to_check[1])].cpu(),
                    k[int(self.kpts_to_check[2])].cpu(),
                )
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)

            if self.pose_type == "abworkout":
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.kpts_to_check[0])].cpu(),
                    k[int(self.kpts_to_check[1])].cpu(),
                    k[int(self.kpts_to_check[2])].cpu(),
                )
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)
                if self.angle[ind] > self.pose_up_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < self.pose_down_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.workouts_counts[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.workouts_counts[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                    color=self.count_bg_color,
                    txt_color=self.count_txt_color,
                )

            if self.pose_type == "pushup" or self.pose_type == "squat":
                if self.angle[ind] > self.pose_up_angle:
                    self.stage[ind] = "up"
                if self.angle[ind] < self.pose_down_angle and self.stage[ind] == "up":
                    self.stage[ind] = "down"
                    self.workouts_counts[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.workouts_counts[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                    color=self.count_bg_color,
                    txt_color=self.count_txt_color,
                )
            if self.pose_type == "pullup":
                if self.angle[ind] > self.pose_up_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < self.pose_down_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.workouts_counts[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.workouts_counts[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                    color=self.count_bg_color,
                    txt_color=self.count_txt_color,
                )

            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        self.display_frames(self.im0, self.window_name)

        return self.im0
