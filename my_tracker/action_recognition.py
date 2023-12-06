from itertools import combinations

import cv2
import networkx as nx
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1
text_color = (255, 255, 255)
background_color = (100, 100, 100)
text_padding = 7


class ActionRecognizer:
    def __init__(self, config):
        # Gathering parameters
        self.g_enabled = config["gather"]["enabled"]
        self.g_distance_threshold = config["gather"]["distance_threshold"]
        self.g_area_threshold = config["gather"]["area_threshold"]
        # Standing still parameters
        self.ss_enabled = config["stand_still"]["enabled"]
        self.ss_speed_threshold = config["stand_still"]["speed_threshold"]
        # Fast approach parameters
        self.fa_enabled = config["fast_approach"]["enabled"]
        self.fa_distance_threshold = config["fast_approach"]["distance_threshold"]
        self.fa_speed_threshold = config["fast_approach"]["speed_threshold"]
        # Suddenly running parameters
        self.sr_enabled = config["suddenly_run"]["enabled"]
        self.sr_acceleration_threshold = config["suddenly_run"]["acceleration_threshold"]

        self.osb_enabled = config["overstep_boundary"]["enabled"]
        self.osb_line = config["overstep_boundary"]["line"]  # Coords: [x1, y1, x2, y2]
        self.osb_direction = config["overstep_boundary"]["direction"]  # "up" or "down"
        self.osb_distance_threshold = config["overstep_boundary"]["distance_threshold"]

    def recognize_frame(self, tracks, frame):
        """
        Recognizes actions in a frame.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
            frame (np.array): frame to be annotated.
        """
        ar_results = {}

        if self.g_enabled:
            ar_results["gather"] = self.recognize_gather(tracks)
        if self.ss_enabled:
            ar_results["stand_still"] = self.recognize_stand_still(tracks)
        if self.fa_enabled:
            ar_results["fast_approach"] = self.recognize_fast_approach(tracks, frame)
        if self.sr_enabled:
            ar_results["suddenly_run"] = self.recognize_suddenly_run(tracks)
        if self.osb_enabled:
            ar_results["overstepboundry"] = self.recognize_overstep_boundary(tracks, frame)

        # TODO: merge individual actions so that we only loop and annotate once
        return ar_results

    def annotate(self, frame, ar_results):
        """
        Annotates the frame with the results of the action recognition.
        Args:
            frame (np.array): frame to be annotated.
            ar_results (dict): dictionary containing the results of the action recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        if self.g_enabled:
            frame = self.annotate_gather(frame, ar_results["gather"])

        # TODO: merge individual actions so that we only loop and annotate once
        if self.ss_enabled:
            frame = self.annotate_stand_still(frame, ar_results["stand_still"])

        if self.fa_enabled:
            frame = self.annotate_fast_approach(frame, ar_results["fast_approach"])

        if self.sr_enabled:
            frame = self.annotate_suddenly_run(frame, ar_results["suddenly_run"])

        if self.osb_enabled:
            frame = self.annotate_overstep_boundary(frame, ar_results["overstepboundry"])

        return frame

    def recognize_gather(self, tracks):
        """
        Recognizes gatherings in a frame by computing the normalized euclidean distance between all pairs of detections
        and conditioning pairs to have similar areas. Then, it finds the independent chains in the graph and computes
        the bounding box of each crowd.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
        Returns:
            results (dict): dictionary containing the results of the action recognition.
        """
        pairs = []
        # Iterate over all pairs of detections
        for i, j in combinations(range(len(tracks)), 2):
            det1 = tracks[int(i)]
            det2 = tracks[int(j)]

            # If both detections are people proceed to computation
            # TODO: maybe constrain on speed?
            if (det1.class_ids == 0) and (det2.class_ids == 0):
                distance, a1, a2 = self.compute_ned(det1, det2)

                # If distance is below threshold and the ratio between the areas is within the threshold add pair
                if distance <= self.g_distance_threshold and \
                        (self.g_area_threshold <= a1 / a2 <= (1 / self.g_area_threshold)):
                    pairs.append([i, j])

        # Find independent chains in the graph
        crowds = self.get_independent_chains(pairs)

        # For each crowd, compute the bounding box and store it in the results dictionary
        if len(crowds) > 0:
            results = {}
            for k, crowd in enumerate(crowds):
                crowd_box = self.compute_crowd_box(tracks, crowd)
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
            det1 (sv.STrack): first detection.
            det2 (sv.STrack): second detection.
        Returns:
            normalized_distance (float): normalized euclidean distance between the two detections.
            a1 (float): area of the first bounding box.
            a2 (float): area of the second bounding box.
        """
        # Euclidean distance between center of masses
        distance = np.sqrt(np.sum((det1.mean[0:2] - det2.mean[0:2]) ** 2))
        # Mean area of the bounding boxes
        a1 = det1.tlwh[2] * det1.tlwh[3]
        a2 = det2.tlwh[2] * det2.tlwh[3]
        mean_area = (a1 + a2) / 2
        return distance / np.sqrt(mean_area), a1, a2

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
    def compute_crowd_box(tracks, crowd):
        """
        Computes the bounding box of a crowd of people.
        Args:
            tracks (list): list of detections in the frame (sv.STrack objects).
            crowd (list): list of detections that form a crowd.
        Returns:
            crowd_box (list): list containing the coordinates of the bounding box of the crowd.
        """
        # TODO: optimize this
        # Get the coordinates of the bounding boxes of the detections in the crowd
        crowd_boxes = [tracks[i].tlbr for i in crowd]
        # Compute the bounding box of the crowd
        crowd_box = [min([box[0] for box in crowd_boxes]),
                     min([box[1] for box in crowd_boxes]),
                     max([box[2] for box in crowd_boxes]),
                     max([box[3] for box in crowd_boxes])]
        # TODO maybe add more space to the bounding box, not narrow it down to the crowd
        return np.array(crowd_box)

    def recognize_stand_still(self, tracks):
        # TODO: also controlled by sv.STrack.frame_stride
        if not tracks:  # Verificar si la lista de tracks está vacía
            return None
        frame_stride = tracks[0].frame_stride
        ss_results = {}
        for track in tracks:
            if track.tracklet_len > frame_stride and track.class_ids == 0:
                pixel_s, pixel_a, _ = self.get_motion_descriptors(track)
                if pixel_s < self.ss_speed_threshold:  # TODO: condition on acceleration?
                    ss_results[track.track_id] = track.tlbr
        return ss_results if len(ss_results.keys()) > 0 else None

    def get_motion_descriptors(self, track):

        """dif = (track.mean - track.prev_states[-1]) / track.frame_stride
        if len(track.prev_states) > 1:
            dif_prev = (track.prev_states[-1] - track.prev_states[-2]) / track.frame_stride
        else:
            dif_prev = dif

        # Speed
        dif_speed = np.sqrt(np.sum(dif[:4] ** 2))
        dif_prev_speed = np.sqrt(np.sum(dif_prev[:4] ** 2))
        pixel_s = (dif_speed + dif_prev_speed) / 2

        # Acceleration
        pixel_a = abs(dif_speed - dif_prev_speed) / track.frame_stride

        # Direction of movement
        direction = dif[:2] + dif_prev[:2]"""

        ##########################################
        # MEAN SPEED
        ##########################################
        states = track.prev_states + [track.mean]
        X = []
        Y = []
        for state in states:
            X.append(state[0])
            Y.append(state[1])
        X = np.array(X)
        Y = np.array(Y)

        dX = np.diff(X)
        dY = np.diff(Y)
        distance = np.sqrt(dX ** 2 + dY ** 2)

        # Speed
        speeds = distance / track.frame_stride
        avg_speed = np.mean(speeds)
        # Acceleration
        # TODO: we should use instant acceleration, not average, speeds[-3:]?
        acceleration = np.diff(speeds[-2:]) / track.frame_stride
        avg_acceleration = np.mean(acceleration)
        # Direction of movement in Y axis
        direction = np.mean(np.sign(dY))
        # angles = np.arctan2(dY, dX)
        # direction = np.degrees(np.mean(angles))

        return avg_speed, avg_acceleration, direction

    def recognize_fast_approach(self, tracks, frame):
        valid_classes = [0, 1, 2]  # personnel, car, truck
        interest_point = np.array([frame.shape[1] // 2, frame.shape[0]])  # bottom center of the frame

        fa_results = {}
        for track in tracks:
            if track.class_ids in valid_classes and track.frame_id > 1:
                pixel_s, pixel_a, direction = self.get_motion_descriptors(track)
                if pixel_s > self.fa_speed_threshold and direction > 0:
                    # Distance between point of interest and bbox
                    dx = max(abs(track.mean[0] - interest_point[0]) - track.mean[2] / 2, 0)
                    dy = max(abs(track.mean[1] - interest_point[1]) - track.mean[3] / 2, 0)
                    distace_to_interest_point = np.sqrt(dx ** 2 + dy ** 2)
                    # Threshold determined the distance to the bottom of the frame
                    if distace_to_interest_point < (interest_point[1] / self.fa_distance_threshold):
                        # TODO: cars have bigger area, so they are considered to be closer to the interest point
                        # distace_to_interest_point = np.sqrt(np.sum((track.mean[0:2] - interest_point) ** 2))/np.sqrt(track.tlwh[2] * track.tlwh[3])
                        # if distace_to_interest_point < self.fa_distance_threshold:
                        fa_results[track.track_id] = track.tlbr
        return fa_results if len(fa_results.keys()) > 0 else None

    def recognize_suddenly_run(self, tracks):
        sr_results = {}
        for track in tracks:
            if track.class_ids == 0 and track.frame_id > 1:
                # TODO: we should use instant acceleration, not average
                pixel_s, pixel_a, direction = self.get_motion_descriptors(track)
                if pixel_a > self.sr_acceleration_threshold:
                    sr_results[track.track_id] = track.tlbr
        return sr_results if len(sr_results.keys()) > 0 else None

    @staticmethod
    def annotate_gather(frame, crowd_results):
        """
        Annotates the frame with the results of the gathering recognition. Draws a bounding box around each crowd. Text
        is placed on the bottom right corner of the bounding box.
        Args:
            frame (np.array): frame to be annotated.
            crowd_results (dict): dictionary containing the results of the gathering recognition.
        Returns:
            frame (np.array): annotated frame.
        """
        if crowd_results is not None:
            for idx, bbox in crowd_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=background_color,
                    thickness=2,
                )

                text = f"G #{idx + 1}"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                # Text must be top right corner of the bounding box of the crowd but outside the frame
                text_x = x2 - text_padding - text_width
                text_y = y2 + text_padding + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * text_padding - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * text_padding + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame

    @staticmethod
    def annotate_stand_still(frame, ss_results):
        if ss_results is not None:
            for idx, bbox in ss_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                text = f"SS"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                # Text must be top right corner of the bounding box of the crowd but outside the frame
                text_x = x2 - text_padding - text_width
                text_y = y2 + text_padding + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * text_padding - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * text_padding + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame

    @staticmethod
    def annotate_fast_approach(frame, fa_results):
        if fa_results is not None:
            for idx, bbox in fa_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                text = f"FA"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                # Text must be top right corner of the bounding box of the crowd but outside the frame
                text_x = x2 - text_padding - text_width
                text_y = y2 + text_padding + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * text_padding - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * text_padding + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame

    @staticmethod
    def annotate_suddenly_run(frame, sr_results):
        if sr_results is not None:
            for idx, bbox in sr_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                text = f"SR"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                # Text must be top right corner of the bounding box of the crowd but outside the frame
                text_x = x2 - text_padding - text_width
                text_y = y2 + text_padding + text_height

                text_background_x2 = x2
                text_background_x1 = x2 - 2 * text_padding - text_width

                text_background_y1 = y2
                text_background_y2 = y2 + 2 * text_padding + text_height

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame


    """def is_overstep_boundary(self, track):
        bbox = track.tlbr
        x1, y1, x2, y2 = bbox
        bx_center, by_center = (x1 + x2) / 2, (y1 + y2) / 2

        # Check if prev_states has at least one element
        if not track.prev_states or len(track.prev_states) < 1:
            return False

        _,_, movement_direction = self.get_motion_descriptors(track)

        # Calculate the cross product to determine the side of the line on which the bbox is located.
        lx1, ly1, lx2, ly2 = self.osb_line
        line_vec = np.array([lx2 - lx1, ly2 - ly1])
        bbox_vec = np.array([bx_center - lx1, by_center - ly1])
        cross_product = np.cross(line_vec, bbox_vec)

        # Verify if the bbox has crossed the line based on the configured address
        crossed = False
        if self.osb_direction == "up":
            crossed = cross_product < 0
        elif self.osb_direction == "down":
            crossed = cross_product > 0
        else:
            raise ValueError(f"Unknown boundary direction: {self.osb_direction}")

        # Check if the object is moving in the desired direction and is close to the line.
        if crossed:
            if ((self.osb_direction == "down" and movement_direction > 0) or
                    (self.osb_direction == "up" and movement_direction < 0)):
                return True
        return False"""

    def is_overstep_boundary(self, track, frame):
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        bbox = track.tlbr
        bx_center, by_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Centro de la bounding box

        # Crear la región de interés basada en la dirección y la línea
        if self.osb_direction == "down":
            region = [0, self.osb_line[1], frame_width,
                      frame_height - self.osb_line[1]]  # Toda la región debajo de la línea
        elif self.osb_direction == "up":
            region = [0, 0, frame_width, self.osb_line[1]]  # Toda la región por encima de la línea
        else:
            raise ValueError(f"Unknown boundary direction: {self.osb_direction}")

        # Verificar si el centro de la bbox está dentro de la región
        return self.within_bbox((bx_center, by_center), region)

    @staticmethod
    def within_bbox(pt, bbox):
        x, y = pt
        return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]

    def recognize_overstep_boundary(self, tracks, frame):
        osb_results = {}
        for track in tracks:
            # Check if the bbox crosses the boundary line
            if self.is_overstep_boundary(track, frame):
                osb_results[track.track_id] = track.tlbr
        return osb_results if len(osb_results.keys()) > 0 else None

    @staticmethod
    def annotate_overstep_boundary(frame, osb_results):
        if osb_results is not None:
            for idx, bbox in osb_results.items():
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 0, 255),
                    thickness=2,
                )

                text = "OSB"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=font,
                    fontScale=font_scale,
                    thickness=font_thickness,
                )[0]

                text_x = max(x1, min(x2 - text_width - 3 * text_padding, frame.shape[1] - text_width - 3 * text_padding))
                text_y = max(y1 + text_height + 3 * text_padding, text_height + 3 * text_padding)

                text_background_x1 = text_x - text_padding
                text_background_y1 = text_y - text_height - 2 * text_padding
                text_background_x2 = text_x + text_width + text_padding
                text_background_y2 = text_y + text_padding

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=background_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y - text_padding),
                    fontFace=font,
                    fontScale=font_scale,
                    color=text_color,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame
