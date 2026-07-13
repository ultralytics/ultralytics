# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """A class to manage gym steps of people in a real-time video stream based on their poses.

    This class extends BaseSolution to monitor workouts using YOLO pose estimation models. It tracks and counts
    repetitions of exercises based on predefined angle thresholds for up and down positions.

    Attributes:
        states (dict[int, dict[str, float | int | str]]): Per-track angle, rep count, and stage for workout monitoring.
        up_angle (float): Angle threshold for considering the 'up' position of an exercise.
        down_angle (float): Angle threshold for considering the 'down' position of an exercise.
        kpts (list[int]): Indices of keypoints used for angle calculation.

    Methods:
        process: Process a frame to detect poses, calculate angles, and count repetitions.

    Examples:
        >>> gym = AIGym(model="yolo26n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize AIGym for workout monitoring using pose estimation and predefined angles.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class constructor including:
                - model (str): Model name or path, defaults to "yolo26n-pose.pt".
        """
        kwargs["model"] = kwargs.get("model", "yolo26n-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})  # Dict for count, angle and stage

        # Extract details from CFG single time for usage later
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage

    def process(self, im0) -> SolutionResults:
        """Monitor workouts using Ultralytics YOLO Pose Model.

        This function processes an input image to track and analyze human poses for workout monitoring. It uses the YOLO
        Pose model to detect keypoints, estimate angles, and count repetitions based on predefined angle thresholds.

        Args:
            im0 (np.ndarray): Input image for processing.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'workout_count' (list[int] of completed reps, one
                entry per currently tracked individual), 'workout_stage' (list[str] of current stages), 'workout_angle'
                (list[float] of current angles), and 'total_tracks' (int, total number of tracked individuals). All
                per-track lists are aligned with the currently visible tracks so that ``len(results.workout_count)
                == results.total_tracks``.

        Examples:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data.cpu().numpy()  # one host transfer, avoids per-keypoint GPU sync

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]  # get state details
                # Get keypoints and estimate the angle
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # Determine stage and count logic based on angle thresholds
                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

                # Display angle, count, and stage text
                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],  # angle text for display
                        count_text=state["count"],  # count text for workouts
                        stage_text=state["stage"],  # stage position text
                        center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output image, if environment support display

        # Return SolutionResults aligned with currently tracked individuals only.
        # self.states is keyed by track_id and may contain entries for people who have
        # already left the frame; iterating self.track_ids keeps the per-track lists in
        # sync with total_tracks so len(workout_count) == total_tracks at all times.
        return SolutionResults(
            plot_im=plot_im,
            workout_count=[self.states[tid]["count"] for tid in self.track_ids],
            workout_stage=[self.states[tid]["stage"] for tid in self.track_ids],
            workout_angle=[self.states[tid]["angle"] for tid in self.track_ids],
            total_tracks=len(self.track_ids),
        )
