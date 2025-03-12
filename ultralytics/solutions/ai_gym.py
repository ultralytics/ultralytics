# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """
    A class to manage gym steps of people in a real-time video stream based on their poses.

    This class extends BaseSolution to monitor workouts using YOLO pose estimation models. It tracks and counts
    repetitions of exercises based on predefined angle thresholds for up and down positions.

    Attributes:
        count (List[int]): Repetition counts for each detected person.
        angle (List[float]): Current angle of the tracked body part for each person.
        stage (List[str]): Current exercise stage ('up', 'down', or '-') for each person.
        initial_stage (str | None): Initial stage of the exercise.
        up_angle (float): Angle threshold for considering the 'up' position of an exercise.
        down_angle (float): Angle threshold for considering the 'down' position of an exercise.
        kpts (List[int]): Indices of keypoints used for angle calculation.

    Methods:
        process: Processes a frame to detect poses, calculate angles, and count repetitions.

    Examples:
        >>> gym = AIGym(model="yolo11n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs):
        """
        Initialize AIGym for workout monitoring using pose estimation and predefined angles.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class constructor.
                model (str): Model name or path, defaults to "yolo11n-pose.pt".
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-pose.pt")
        super().__init__(**kwargs)
        self.count = []  # List for counts, necessary where there are multiple objects in frame
        self.angle = []  # List for angle, necessary where there are multiple objects in frame
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage

    def process(self, im0):
        """
        Monitor workouts using Ultralytics YOLO Pose Model.

        This function processes an input image to track and analyze human poses for workout monitoring. It uses
        the YOLO Pose model to detect keypoints, estimate angles, and count repetitions based on predefined
        angle thresholds.

        Args:
            im0 (np.ndarray): Input image for processing.

        Returns:
            (SolutionResults): Contains processed image `plot_im`,
                'workout_count' (list of completed reps),
                'workout_stage' (list of current stages),
                'workout_angle' (list of angles), and
                'total_tracks' (total number of tracked individuals).

        Examples:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)
        tracks = self.tracks[0]

        if tracks.boxes.id is not None:
            if len(tracks) > len(self.count):  # Add new entries for newly detected people
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = annotator.estimate_pose_angle(*kpts)
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # Determine stage and count logic based on angle thresholds
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # Display angle, count, and stage text
                annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                )
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output image, if environment support display

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            workout_count=self.count,
            workout_stage=self.stage,
            workout_angle=self.angle,
            total_tracks=len(self.track_ids),
        )
