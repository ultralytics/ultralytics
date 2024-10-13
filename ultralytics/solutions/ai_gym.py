# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator


class AIGym(BaseSolution):
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self, **kwargs):
        """Initialization function for AiGYM class, a child class of BaseSolution class, can be used for workouts
        monitoring.
        """
        # Check if the model name ends with '-pose'
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"

        super().__init__(**kwargs)
        self.count = []  # List for counts, necessary where there are multiple objects in frame
        self.angle = []  # List for angle, necessary where there are multiple objects in frame
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage
        self.lw = self.CFG["line_width"]  # Store line_width for usage

    def monitor(self, im0):
        """
        Monitor the workouts using Ultralytics YOLOv8 Pose Model: https://docs.ultralytics.com/tasks/pose/.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        # Extract tracks
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])[0]

        if tracks.boxes.id is not None:
            # Extract and check keypoints
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # Initialize annotator
            self.annotator = Annotator(im0, line_width=self.lw)

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.lw * 3)

                # Determine stage and count logic based on angle thresholds
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # Display angle, count, and stage text
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                )

        self.display_output(im0)  # Display output image, if environment support display
        return im0  # return an image for writing or further usage
