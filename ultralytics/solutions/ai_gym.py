# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator


class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self):
        """Initializes the AIGym with default values for Visual and Image parameters."""

        # Image and line thickness
        self.im0 = None
        self.tf = None

        # Keypoints and count information
        self.keypoints = None
        self.poseup_angle = None
        self.posedown_angle = None
        self.threshold = 0.001

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.stage = None
        self.pose_type = "pushup"
        self.kpts_to_check = None

        # Visual Information
        self.view_img = False
        self.annotator = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        kpts_to_check,
        line_thickness=2,
        view_img=False,
        pose_up_angle=145.0,
        pose_down_angle=90.0,
        pose_type="pullup",
    ):
        """
        Configures the AIGym line_thickness, save image and view image parameters
        Args:
            kpts_to_check (list): 3 keypoints for counting
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): display the im0
            pose_up_angle (float): Angle to set pose position up
            pose_down_angle (float): Angle to set pose position down
            pose_type: "pushup", "pullup" or "abworkout"
        """
        self.kpts_to_check = kpts_to_check
        self.tf = line_thickness
        self.view_img = view_img
        self.poseup_angle = pose_up_angle
        self.posedown_angle = pose_down_angle
        self.pose_type = pose_type

    def start_counting(self, im0, results, frame_count):
        """
        Function used to count the gym steps
        Args:
            im0 (ndarray): Current frame from the video stream.
            results: Pose estimation data
            frame_count: store current frame count
        """
        self.im0 = im0
        if frame_count == 1:
            self.count = [0] * len(results[0])
            self.angle = [0] * len(results[0])
            self.stage = ["-" for _ in results[0]]
        self.keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=2)

        for ind, k in enumerate(reversed(self.keypoints)):
            if self.pose_type == "pushup" or self.pose_type == "pullup":
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
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < self.posedown_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                    line_thickness=self.tf,
                )

            if self.pose_type == "pushup":
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
                    line_thickness=self.tf,
                )
            if self.pose_type == "pullup":
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = "down"
                if self.angle[ind] < self.posedown_angle and self.stage[ind] == "down":
                    self.stage[ind] = "up"
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                    line_thickness=self.tf,
                )

            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        if self.env_check and self.view_img:
            cv2.imshow("Ultralytics YOLOv8 AI GYM", self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        return self.im0


if __name__ == "__main__":
    AIGym()
