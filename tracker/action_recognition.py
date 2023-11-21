from itertools import combinations

import cv2
import networkx as nx
import numpy as np


class ActionRecognizer:
    def __init__(self, config):
        self.gathering_enabled = config["gathering"]["enabled"]
        self.distance_threshold = config["gathering"]["distance_threshold"]
        self.area_threshold = config["gathering"]["area_threshold"]

    def recognize_frame_gathering(self, detections):
        """
        Recognizes actions in a frame using the gathering method.
        Args:
            detections (sv.Detections): iterable object with the detections in the frame.
        """
        pairs = []
        # Iterate over all pairs of detections
        for i, j in combinations(range(len(detections)), 2):
            det1 = detections[int(i)]
            det2 = detections[int(j)]

            # If both detections are people proceed to computation
            # TODO: maybe constrain on speed?
            if (det1.class_id.item() == 0) and (det2.class_id.item() == 0):
                distance = self.compute_ned(det1, det2)

                # If distance is below threshold and the ratio between the areas is within the threshold add pair
                if distance <= self.distance_threshold and \
                        (self.area_threshold <= (det1.area.item()/det2.area.item()) <= (1/self.area_threshold)):
                    pairs.append([i, j])

        # Find independent chains in the graph
        crowds = self.get_independent_chains(pairs)

        # TODO store it on top of detections object so that it can be used in annotate_frame

        # For each crowd, compute the bounding box and store it in the results dictionary
        if len(crowds) > 0:
            results = {}
            for k, crowd in enumerate(crowds):
                crowd_box = self.compute_crowd_box(detections, crowd)
                results[k] = crowd_box
            return results
        else:
            return None

    @staticmethod
    def compute_ned(det1, det2):
        """
        Computes the normalized euclidean distance between two detections using the center of mass of the bounding
        boxes, and the mean area of the bounding boxes as normalization factor.
        Args:
            det1 (sv.Detection): first detection.
            det2 (sv.Detection): second detection.
        Returns:
            distance (float): normalized euclidean distance between the two detections.
        """
        x_cm_1 = det1.xyxy[0][0] + (det1.xyxy[0][2] - det1.xyxy[0][0]) / 2
        y_cm_1 = det1.xyxy[0][1] + (det1.xyxy[0][3] - det1.xyxy[0][1]) / 2
        x_cm_2 = det2.xyxy[0][0] + (det2.xyxy[0][2] - det2.xyxy[0][0]) / 2
        y_cm_2 = det2.xyxy[0][1] + (det2.xyxy[0][3] - det2.xyxy[0][1]) / 2

        distance = np.sqrt((x_cm_1 - x_cm_2)**2 + (y_cm_1 - y_cm_2)**2)/np.sqrt((det1.area.item()+det2.area.item())/2)
        return distance

    @staticmethod
    def get_independent_chains(pairs):
        """
        Finds the independent chains in a graph, where each node is a detection and each edge is a pair of detections
        linked by the distance threshold.
        Args:
            pairs (list): list of pairs of detections.
        Returns:
            valid_chains (list): list of lists of detections, where each list is an independent chain.
        """
        # Initialize graph
        g = nx.Graph()
        g.add_edges_from(pairs)
        # Find connected components
        independent_chains = list(nx.connected_components(g))
        # Filter out chains having less than 3 elements
        valid_chains = [chain for chain in independent_chains if len(chain) > 2]
        return valid_chains

    @staticmethod
    def compute_crowd_box(detections, crowd):
        """
        Computes the bounding box of a crowd of people.
        Args:
            detections (sv.Detections): iterable object with the detections in the frame.
            crowd (list): list of detections that form a crowd.
        Returns:
            crowd_box (list): list containing the coordinates of the bounding box of the crowd.
        """
        # Get the coordinates of the bounding boxes of the detections in the crowd
        crowd_boxes = [detections[i].xyxy[0] for i in crowd]
        # Compute the bounding box of the crowd
        crowd_box = [min([box[0] for box in crowd_boxes]),
                     min([box[1] for box in crowd_boxes]),
                     max([box[2] for box in crowd_boxes]),
                     max([box[3] for box in crowd_boxes])]
        #TODO maybe add more space to the bounding box, not narrow it down to the crowd
        return np.array(crowd_box)

    def annotate_crowd(self, frame, crowd_results):
        if crowd_results is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for idx, bbox in crowd_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 0),
                    thickness=2,
                )

                text = f"crowd #{idx}"

                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=0.5,
                    thickness=1,
                )[0]

                # Text must be top right corner of the bounding box of the crowd but outside the frame
                text_x = x2 - 10 - text_width
                text_y = y2 + 10 + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * 10 - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * 10 + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=(0, 0, 0),
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )


        return frame
