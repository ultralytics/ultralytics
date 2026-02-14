import cv2

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_imshow


class RegionSelector:
    """Interactive region selector for video frames.

    Allows users to draw a polygon region on the first frame of a video using mouse clicks.

    Attributes:
        video_path (str): Path to the video file.
        points (list): List of polygon points being selected.
        image (np.ndarray): Current image with drawings.
        original_image (np.ndarray): Original first frame.
    """

    def __init__(self, video_path):
        """Initialize RegionSelector with video path.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.points = []
        self.image = None
        self.original_image = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection.

        Args:
            event: OpenCV mouse event type.
            x (int): X coordinate of mouse.
            y (int): Y coordinate of mouse.
            flags: OpenCV event flags.
            param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            LOGGER.info(f"Point {len(self.points)} added: ({x}, {y})")
            self.draw_polygon()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed_point = self.points.pop()
                LOGGER.info(f"Point removed: {removed_point}")
                self.draw_polygon()

    def draw_polygon(self):
        """Draw the current polygon on the image."""
        self.image = self.original_image.copy()

        # Draw existing points
        for i, point in enumerate(self.points):
            cv2.circle(self.image, point, 5, (0, 255, 0), -1)
            cv2.putText(
                self.image, str(i + 1), (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

        # Draw lines connecting points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.image, self.points[i], self.points[i + 1], (0, 255, 0), 2)

        # Draw closing line if we have at least 3 points
        if len(self.points) >= 3:
            cv2.line(self.image, self.points[-1], self.points[0], (0, 255, 0), 2)

        # Display instructions
        text_y = 30
        cv2.putText(
            self.image,
            "LEFT CLICK: Add point | RIGHT CLICK: Remove last point",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.image,
            "ENTER: Confirm | ESC: Cancel",
            (10, text_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.image, f"Points: {len(self.points)}", (10, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        cv2.imshow("Region Selector", self.image)

    def select_region(self):
        """Main function to select region from first frame.

        Returns:
            list: List of selected polygon points as tuples, or None if cancelled.
        """
        # Check if display is available for imshow
        if not check_imshow(warn=True):
            raise RuntimeError(
                "Region selection requires a display. Run in a GUI environment or provide region coordinates directly."
            )

        # Open video and read first frame
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            LOGGER.warning("Cannot open video file")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            LOGGER.warning("Cannot read first frame")
            return None

        self.original_image = frame.copy()
        self.image = frame.copy()

        # Create window and set mouse callback
        cv2.namedWindow("Region Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Region Selector", 1280, 720)  # Set larger window size
        cv2.setMouseCallback("Region Selector", self.mouse_callback)

        # Display initial frame
        self.draw_polygon()

        # Wait for user input
        while True:
            key = cv2.waitKey(0)

            if key == 13:  # ENTER key
                if len(self.points) >= 3:
                    region = self.points.copy()
                    LOGGER.info(f"Region confirmed! Points: {region}")
                    cv2.destroyAllWindows()
                    return region
                else:
                    LOGGER.warning("Please select at least 3 points for a polygon!")

            elif key == 27:  # ESC key
                LOGGER.info("Region selection cancelled!")
                cv2.destroyAllWindows()
                return None


def select_region(video_path):
    """Convenience function to interactively select a region from video's first frame.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: List of selected polygon points as tuples, or None if cancelled.

    Examples:
        >>> from ultralytics.solutions import select_region
        >>> region = select_region("video.mp4")
        >>> if region:
        ...     print(f"Selected region: {region}")
    """
    selector = RegionSelector(video_path)
    return selector.select_region()
