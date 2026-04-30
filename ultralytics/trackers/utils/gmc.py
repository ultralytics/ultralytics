# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of frames for computational efficiency.

    Attributes:
        method (str | None): The tracking method to use. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', None.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.ndarray | None): Previous frame for tracking.
        prevKeyPoints (tuple | np.ndarray | None): Keypoints from the previous frame.
        prevDescriptors (np.ndarray | None): Descriptors from the previous frame.
        initializedFirstFrame (bool): Flag indicating if the first frame has been processed.

    Methods:
        apply: Apply the chosen method to a raw frame and optionally use provided detections.
        apply_ecc: Apply the ECC algorithm to a raw frame.
        apply_features: Apply feature-based methods like ORB or SIFT to a raw frame.
        apply_sparseoptflow: Apply the Sparse Optical Flow method to a raw frame.
        reset_params: Reset the internal parameters of the GMC object.

    Examples:
        Create a GMC object and apply it to a frame
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> warp = gmc.apply(frame)
        >>> print(warp.shape)
        (2, 3)
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        """Initialize a Generalized Motion Compensation (GMC) object with tracking method and downscale factor.

        Args:
            method (str): The tracking method to use. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
            downscale (int): Downscale factor for processing frames.
        """
        super().__init__()

        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == "ecc":
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == "sparseOptFlow":
            self.feature_params = dict(
                maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
            )

        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"Unknown GMC method: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        """Estimate a 2×3 motion compensation warp for a frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (list, optional): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.rand(480, 640, 3)
            >>> transformation_matrix = gmc.apply(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        if self.method in {"orb", "sift"}:
            return self.apply_features(raw_frame, detections)
        elif self.method == "ecc":
            return self.apply_ecc(raw_frame)
        elif self.method == "sparseOptFlow":
            return self.apply_sparseoptflow(raw_frame)
        else:
            return np.eye(2, 3)

    def apply_ecc(self, raw_frame: np.ndarray) -> np.ndarray:
        """Apply the ECC (Enhanced Correlation Coefficient) algorithm to a raw frame for motion compensation.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="ecc")
            >>> processed_frame = gmc.apply_ecc(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(processed_frame)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image for computational efficiency
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Handle first frame initialization
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        # Run the ECC algorithm to find transformation matrix
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            LOGGER.warning(f"findTransformECC failed; using identity warp. {e}")

        return H

    def apply_features(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        """Apply feature-based methods like ORB or SIFT to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (list, optional): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC(method="orb")
            >>> raw_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> transformation_matrix = gmc.apply_features(raw_frame)
            >>> print(transformation_matrix.shape)
            (2, 3)
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)

        # Downscale image for computational efficiency
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Create mask for keypoint detection, excluding border regions
        mask = np.zeros_like(frame)
        mask[int(0.02 * height) : int(0.98 * height), int(0.02 * width) : int(0.98 * width)] = 255

        # Exclude detection regions from mask to avoid tracking detected objects
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        # Find keypoints and compute descriptors
        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame initialization
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        # Match descriptors between previous and current frame
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filter matches based on spatial distance constraints
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Apply Lowe's ratio test and spatial distance filtering
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (
                    prevKeyPointLocation[0] - currKeyPointLocation[0],
                    prevKeyPointLocation[1] - currKeyPointLocation[1],
                )

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and (
                    np.abs(spatialDistance[1]) < maxSpatialDistance[1]
                ):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        # Filter outliers using statistical analysis
        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)
        inliers = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        # Extract good matches and corresponding points
        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Estimate transformation matrix using RANSAC
        if prevPoints.shape[0] > 4:
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Scale translation components back to original resolution
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("not enough matching points")

        # Store current frame data for next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def apply_sparseoptflow(self, raw_frame: np.ndarray) -> np.ndarray:
        """Apply Sparse Optical Flow method to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Transformation matrix with shape (2, 3).

        Examples:
            >>> gmc = GMC()
            >>> result = gmc.apply_sparseoptflow(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(result)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)

        # Downscale image for computational efficiency
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Find good features to track
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame initialization
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        # Calculate optical flow using Lucas-Kanade method
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # Extract successfully tracked points
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Estimate transformation matrix using RANSAC
        if (prevPoints.shape[0] > 4) and (prevPoints.shape[0] == currPoints.shape[0]):
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Scale translation components back to original resolution
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("not enough matching points")

        # Store current frame data for next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def reset_params(self) -> None:
        """Reset the internal parameters including previous frame, keypoints, and descriptors."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
