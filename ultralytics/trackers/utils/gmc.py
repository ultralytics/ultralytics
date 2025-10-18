# Ultralytics YOLO 🚀, AGPL-3.0 license

import copy

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class GMC:
    """
    Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of  frames for computational efficiency.

    Attributes:
        method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.array): Stores the previous frame for tracking.
        prevKeyPoints (list): Stores the keypoints from the previous frame.
        prevDescriptors (np.array): Stores the descriptors from the previous frame.
        initializedFirstFrame (bool): Flag to indicate if the first frame has been processed.

    Methods:
        __init__(self, method='sparseOptFlow', downscale=2): Initializes a GMC object with the specified method
                                                              and downscale factor.
        apply(self, raw_frame, detections=None): Applies the chosen method to a raw frame and optionally uses
                                                 provided detections.
        applyEcc(self, raw_frame, detections=None): Applies the ECC algorithm to a raw frame.
        applyFeatures(self, raw_frame, detections=None): Applies feature-based methods like ORB or SIFT to a raw frame.
        applySparseOptFlow(self, raw_frame, detections=None): Applies the Sparse Optical Flow method to a raw frame.
    """

    def __init__(self, method="sparseOptFlow", downscale=2):
        """Initialize a video tracker with specified parameters."""
        super().__init__()

        self.method = method
        self.downscale = max(1, int(downscale))

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

        elif self.method in ["none", "None", None]:
            self.method = None
        else:
            raise ValueError(f"Error: Unknown GMC method:{method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame, detections=None):
        """Apply object detection on a raw frame using specified method."""
        if self.method in ["orb", "sift"]:
            return self.applyFeatures(raw_frame, detections)
        elif self.method == "ecc":
            return self.applyEcc(raw_frame, detections)
        elif self.method == "sparseOptFlow":
            return self.applySparseOptFlow(raw_frame, detections)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):
        """Initialize."""
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            LOGGER.warning(f"WARNING: find transform failed. Set warp as identity {e}")

        return H

    def applyFeatures(self, raw_frame, detections=None):
        """Initialize."""
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height) : int(0.98 * height), int(0.02 * width) : int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1] : tlbr[3], tlbr[0] : tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # Compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

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

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliers = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

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

        # Draw the keypoint matches on the output image
        # if False:
        #     import matplotlib.pyplot as plt
        #     matches_img = np.hstack((self.prevFrame, frame))
        #     matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
        #     W = np.size(self.prevFrame, 1)
        #     for m in goodMatches:
        #         prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
        #         curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
        #         curr_pt[0] += W
        #         color = np.random.randint(0, 255, 3)
        #         color = (int(color[0]), int(color[1]), int(color[2]))
        #
        #         matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
        #         matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
        #         matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)
        #
        #     plt.figure()
        #     plt.imshow(matches_img)
        #     plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("WARNING: not enough matching points")

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame, detections=None):
        """Initialize."""
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # Leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            LOGGER.warning("WARNING: not enough matching points")

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H
