# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from . import (
    Annotator,
    bg_color_rgb,
    colors,
    display_frames,
    env_check,
    ps_type,
    psdown_angle,
    psup_angle,
    tf,
    txt_color_rgb,
    workout_kpts,
)


class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self):
        """Initializes the AIGym class with default parameters."""

        self.im0 = None  # variable for im0 storage
        self.count = None  # storage of reps count
        self.angle = None  # storage of kpts angle
        self.stage = None  # storage of current stage
        self.annotator = None  # annotator object initialized

        self.window_name = "Ultralytics YOLOv8 Workouts Monitor"  # visual window name
        print("Workouts monitoring app initialized...")

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
            self.count = [0] * len(results[0])
            self.angle = [0] * len(results[0])
            self.stage = ["-" for _ in results[0]]

        keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=tf)

        for ind, k in enumerate(reversed(keypoints)):
            if ps_type in {"pushup", "pullup", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(workout_kpts[0])].cpu(),
                    k[int(workout_kpts[1])].cpu(),
                    k[int(workout_kpts[2])].cpu(),
                )
                self.im0 = self.annotator.draw_specific_points(k, workout_kpts, shape=(640, 640), radius=10)

            if ps_type == "abworkout":
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(workout_kpts[0])].cpu(),
                    k[int(workout_kpts[1])].cpu(),
                    k[int(workout_kpts[2])].cpu(),
                )
                self.im0 = self.annotator.draw_specific_points(k, workout_kpts, shape=(640, 640), radius=10)
                if self.angle[ind] > psup_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < psdown_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(workout_kpts[1])],
                    color=bg_color_rgb,
                    txt_color=txt_color_rgb,
                )

            if ps_type == "pushup" or ps_type == "squat":
                if self.angle[ind] > psup_angle:
                    self.stage[ind] = "up"
                if self.angle[ind] < psdown_angle and self.stage[ind] == "up":
                    self.stage[ind] = "down"
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(workout_kpts[1])],
                    color=bg_color_rgb,
                    txt_color=txt_color_rgb,
                )
            if ps_type == "pullup":
                if self.angle[ind] > psup_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < psdown_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(workout_kpts[1])],
                    color=bg_color_rgb,
                    txt_color=txt_color_rgb,
                )

            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        display_frames(self.im0, self.window_name)

        return self.im0
