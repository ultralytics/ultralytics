# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator


class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(
        self,
        kpts_to_check,
        line_thickness=2,
        view_img=False,
        pose_up_angle=145.0,
        pose_down_angle=90.0,
        pose_type="pullup",
    ):
        """
        Initializes the AIGym class with the specified parameters.

        Args:
            kpts_to_check (list): Indices of keypoints to check.
            line_thickness (int, optional): Thickness of the lines drawn. Defaults to 2.
            view_img (bool, optional): Flag to display the image. Defaults to False.
            pose_up_angle (float, optional): Angle threshold for the 'up' pose. Defaults to 145.0.
            pose_down_angle (float, optional): Angle threshold for the 'down' pose. Defaults to 90.0.
            pose_type (str, optional): Type of pose to detect ('pullup', 'pushup', 'abworkout'). Defaults to "pullup".
        """

        # Image and line thickness
        self.im0 = None
        self.tf = line_thickness

        # Keypoints and count information
        self.keypoints = None
        self.poseup_angle = pose_up_angle
        self.posedown_angle = pose_down_angle
        self.threshold = 0.001

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.stage = None
        self.pose_type = pose_type
        self.kpts_to_check = kpts_to_check

        # Visual Information
        self.view_img = view_img
        self.annotator = None

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

    def start_counting(self, im0, results, frame_count):
        """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data.
            frame_count (int): Current frame count.
        """

        self.im0 = im0

        # Initialize count, angle, and stage lists on the first frame
        if frame_count == 1:
            self.count = [0] * len(results[0])
            self.angle = [0] * len(results[0])
            self.stage = ["-" for _ in results[0]]

        self.keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=self.tf)

        for ind, k in enumerate(reversed(self.keypoints)):
            # Estimate angle and draw specific points based on pose type
            if self.pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.kpts_to_check[0])].cpu(),
                    k[int(self.kpts_to_check[1])].cpu(),
                    k[int(self.kpts_to_check[2])].cpu(),
                )
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)

                # Check and update pose stages and counts based on angle
                if self.pose_type in {"abworkout", "pullup"}:
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "down"
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "down":
                        self.stage[ind] = "up"
                        self.count[ind] += 1

                elif self.pose_type == "pushup" or self.pose_type == "squat":
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "up"
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "up":
                        self.stage[ind] = "down"
                        self.count[ind] += 1

                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                )

            # Draw keypoints
            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        # Display the image if environment supports it and view_img is True
        if self.env_check and self.view_img:
            cv2.imshow("Ultralytics YOLOv8 AI GYM", self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    kpts_to_check = [0, 1, 2]  # example keypoints
    aigym = AIGym(kpts_to_check)
