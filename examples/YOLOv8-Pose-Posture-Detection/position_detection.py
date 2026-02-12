"""
Example: Human Posture Detection (Standing / Sitting / Unknown).
---------------------------------------------------------------
This example uses YOLOv8-Pose to classify human posture from webcam input
based on geometric relationships between body keypoints.

"""

import math

import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER


class PoseDetector:
    def __init__(self, model_path="yolov8n-pose.pt"):
        # Initializing the YOLOv8 model for pose estimation
        self.model = YOLO(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_angle(self, p1, p2, p3):
        # Calculate the angle formed by three points p1, p2, p3 with p2 as the vertex
        ang = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
        ang = ang + 360 if ang < 0 else ang

        # I normalize the angle between 0 and 180
        if ang > 180:
            ang = 360 - ang
        return ang

    def distance_between_two_points(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def detect_position(self, keypoints):
        """Determine the person's location based on keypoints.

        Keypoints YOLO Pose (COCO format):
        0: nose, 1-2: eyes, 3-4: ears
        5: left shoulder, 6: right shoulder
        7: left elbow, 8: right elbow
        9: left wrist, 10: right wrist
        11: left hip, 12: right hip
        13: left knee, 14: right knee
        15: left ankle, 16: right ankle
        """
        # Extract key points
        keypoints[0]
        l_shoulder = keypoints[5]  # left shoulder
        r_shoulder = keypoints[6]  # right shoulder
        l_hip = keypoints[11]  # left hip
        r_hip = keypoints[12]  # right hip
        l_knee = keypoints[13]  # left knee
        r_knee = keypoints[14]  # right knee

        # Check that the main points are detected (confidence > 0.5)
        valid_points = all(
            [
                l_shoulder[2] > 0.5,
                r_shoulder[2] > 0.5,
                l_hip[2] > 0.5,
                r_hip[2] > 0.5,
            ]
        )

        if not valid_points:
            return "Unknown", (0, 0, 255)

        # Calculate centers
        center_shoulders = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)

        center_hips = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

        # Bust height
        bust_height = abs(center_shoulders[1] - center_hips[1])

        # Calculate leg/torso ratio (used to detect the SITTING position in a frontal manner)
        leg_bust_ratio = None
        if bust_height > 10:  # Avoid division by zero
            legs_length = []

            # Length left leg
            if l_hip[2] > 0.5 and l_knee[2] > 0.5:
                left_leg = self.distance_between_two_points(l_hip[0], l_hip[1], l_knee[0], l_knee[1])
                legs_length.append(left_leg)

            # Length right leg
            if r_hip[2] > 0.5 and r_knee[2] > 0.5:
                right_leg = self.distance_between_two_points(r_hip[0], r_hip[1], r_knee[0], r_knee[1])
                legs_length.append(right_leg)

            # If at least one leg has been detected
            if legs_length:
                average_legs_height = sum(legs_length) / len(legs_length)
                leg_bust_ratio = average_legs_height / bust_height

        # Calculate knee angles with hip and shoulder with HIP vertex
        left_knee_corner = None
        right_knee_corner = None

        if l_knee[2] > 0.5:
            left_knee_corner = self.calculate_angle(l_shoulder[:2], l_hip[:2], l_knee[:2])

        if r_knee[2] > 0.5:
            right_knee_corner = self.calculate_angle(r_shoulder[:2], r_hip[:2], r_knee[:2])

        # Sitting: combination of knee angle AND leg/bust ratio
        sitting_by_angle = False
        sitting_by_ratio = False

        # Criterion 1: Knee angle
        if left_knee_corner is not None or right_knee_corner is not None:
            angles = [a for a in [left_knee_corner, right_knee_corner] if a is not None]
            medium_angle = sum(angles) / len(angles)
            if 50 < medium_angle < 150:  # Bent knees
                sitting_by_angle = True

        # Criterion 2: Leg/Bust ratio
        if leg_bust_ratio is not None:
            if leg_bust_ratio < 0.6:
                sitting_by_ratio = True

        # The person is classified as sitting when at least one of these geometric conditions is met:
        # 1) Knee–hip–shoulder angle indicates bent legs (50° < θ < 150°) [Side view]
        # 2) Leg/bust ratio indicates compressed legs (ratio < 0.6) [Front view]

        if sitting_by_ratio or sitting_by_angle:
            return "Sitting", (0, 255, 255)

        # STANDING: torso upright, knees straight
        if bust_height > 80:
            if left_knee_corner and 150 < left_knee_corner < 210:
                return "Standing", (0, 255, 0)
            if right_knee_corner and 150 < right_knee_corner < 210:
                return "Standing", (0, 255, 0)
            # If it doesn't detect the ankles but the torso is vertical
            return "Standing", (0, 255, 0)

        return "Unknown", (0, 0, 255)

    def process_frame(self, frame):
        """Process a single frame and detect positions."""
        # Run inference
        results = self.model(frame, verbose=False)

        # Process every person detected
        for result in results:
            if result.keypoints is not None:
                for person_keypoints in result.keypoints.data:
                    # person_keypoints shape: (17, 3) - 17 keypoints with x, y, confidence
                    keypoints = person_keypoints.cpu().numpy()

                    # Determine position
                    position, color = self.detect_position(keypoints)

                    # Draw skeleton
                    self.draw_skeleton(frame, keypoints, color)

                    # Add label
                    if keypoints[0][2] > 0.5:  # If the nose is visible
                        x, y = int(keypoints[0][0]), int(keypoints[0][1])
                        cv2.putText(frame, position, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def draw_skeleton(self, frame, keypoints, color):
        """Draw the skeleton of the person."""
        # Skeleton connections (COCO format)
        connections = [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        # Draw lines for skeleton
        for start, end in connections:
            if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:  # If confidence > 0.5
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 4, color, -1)

    def run_webcam(self):
        """Run the detector on the webcam."""
        cap = cv2.VideoCapture(0)

        LOGGER.info("Press 'q' to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame = self.process_frame(frame)

            # Show result
            cv2.imshow("YOLO Pose - Position Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PoseDetector("yolov8n-pose.pt")

    detector.run_webcam()
