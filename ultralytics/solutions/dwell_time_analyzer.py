from collections import defaultdict

import cv2
from shapely.geometry import LineString, Polygon

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class DwellTimeAnalyzer(BaseSolution):
    """
    A solution class that extends BaseSolution to measure Dwell Time and Zone Transitions.

    Features:
    - Dwell Time Measurement: How long each tracked object remains in specific zones.
    - Zone Sequence Tracking: Tracks the sequence of zones visited by each object.
    - Class Filtering: Optionally restrict detection to certain classes.

    Optional Advanced Analytics:
    - Funnel Analysis: If enable_funnel=True.
      - If funnel_stages provided and has N stages, it checks how many objects pass through all stages in order.
      - If funnel_stages not provided, it uses all defined zones in order as funnel stages.
    - Average Dwell Times per Zone: If enable_avg_dwell=True.

    Configuration keys of interest:
    - fps: If provided, dwell times are in seconds instead of frames.
    - classes: List of class indices to track if filtering is desired.
    - zones: Dict of zone_name -> list of (x, y) polygon points. If not provided, user selects interactively.
    - enable_funnel: Boolean. If True, compute funnel conversion metrics.
    - funnel_stages: Tuple or list of zone names to form a funnel, e.g., ("Entrance", "Checkout").
    - enable_avg_dwell: Boolean. If True, compute and display average dwell times per zone.
    - detect_mode: String. "all_frames" to detect in all frames, "enter_zones" to detect only when entering zones.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fps = self.CFG.get("fps", None)
        self.analytics_text_color = self.CFG.get("analytics_text_color", (0, 0, 0))
        self.zones = self.CFG.get("zones", {})
        self.classes = self.CFG.get("classes", None)  # If provided, only track these classes

        # Optional features
        self.enable_funnel = self.CFG.get("enable_funnel", False)
        self.funnel_stages = self.CFG.get("funnel_stages", None)  # Could be None initially
        self.enable_avg_dwell = self.CFG.get("enable_avg_dwell", False)
        self.detect_mode = self.CFG.get("detect_mode", "all_frames")  # "all_frames" or "enter_zones"

        # Validate detect_mode
        if self.detect_mode not in ["all_frames", "enter_zones"]:
            raise ValueError("detect_mode must be either 'all_frames' or 'enter_zones'.")

        # If no zones provided, interactively select them
        if not self.zones or len(self.zones) == 0:
            self.zones = self.select_zones()

        self.zone_polygons = {}
        for zone_name, coords in self.zones.items():
            if len(coords) >= 3:
                self.zone_polygons[zone_name] = Polygon(coords)
            else:
                self.zone_polygons[zone_name] = LineString(coords)

        # Dynamically determine funnel stages if not provided and funnel is enabled
        if self.enable_funnel:
            if not self.funnel_stages or len(self.funnel_stages) < 2:
                # Use all zones as funnel stages if possible
                zone_names = list(self.zones.keys())
                if len(zone_names) >= 2:
                    self.funnel_stages = tuple(zone_names)  # All zones form the funnel
                else:
                    # Not enough zones for a funnel
                    self.funnel_stages = None

        # Tracking variables
        self.object_current_zone = {}  # track_id -> zone_name
        self.object_zone_entry_time = {}  # (track_id, zone_name) -> entry_time
        self.object_zone_sequence = defaultdict(list)  # track_id -> [(zone_name, entry_time, exit_time)]
        self.frame_number = 0

    def select_zones(self):
        if not self.CFG.get("source"):
            raise ValueError("No source video provided. Cannot interactively select zones.")

        cap = cv2.VideoCapture(self.CFG["source"])
        if not cap.isOpened():
            raise OSError("Unable to open the video source for zone selection.")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise OSError("Could not read the first frame from the source.")

        num_zones = int(input("How many zones do you want to define? "))

        zones = {}
        zone_points = []
        current_zone_index = 1
        def zone_name(i):
            return f"Zone_{i}"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                zone_points.append((x, y))

        cv2.namedWindow("Select Zones")
        cv2.setMouseCallback("Select Zones", mouse_callback)

        print("Instructions:")
        print("- Left-click to select polygon points.")
        print("- Press 'c' to confirm the polygon for the current zone.")
        print("- Press 'q' to quit without finishing.")

        while True:
            disp_frame = frame.copy()
            for p in zone_points:
                cv2.circle(disp_frame, p, 5, (0, 0, 255), -1)
            if len(zone_points) > 1:
                for i in range(len(zone_points) - 1):
                    cv2.line(disp_frame, zone_points[i], zone_points[i + 1], (0, 255, 0), 2)

            text = f"Define {zone_name(current_zone_index)}: Press 'c' to confirm polygon, 'q' to quit."
            cv2.putText(disp_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Select Zones", disp_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if len(zone_points) >= 3:
                    zones[zone_name(current_zone_index)] = zone_points.copy()
                    zone_points.clear()
                    print(f"Zone {current_zone_index} confirmed.")
                    current_zone_index += 1
                    if current_zone_index > num_zones:
                        # All zones defined
                        break
                else:
                    print("A zone must have at least 3 points. Select more points.")

            elif key == ord("q"):
                print("Quitting zone selection early.")
                break

        cv2.destroyAllWindows()
        return zones

    def get_current_time(self):
        return self.frame_number / self.fps if (self.fps and self.fps > 0) else self.frame_number

    def identify_zone(self, centroid):
        point = self.Point(centroid)
        for zone_name, polygon in self.zone_polygons.items():
            if polygon.contains(point):
                return zone_name
        return None

    def update_dwell_times(self, track_id, old_zone, new_zone):
        current_time = self.get_current_time()
        if old_zone and (track_id, old_zone) in self.object_zone_entry_time:
            entry_time = self.object_zone_entry_time.pop((track_id, old_zone))
            self.object_zone_sequence[track_id].append((old_zone, entry_time, current_time))
        if new_zone:
            self.object_zone_entry_time[(track_id, new_zone)] = current_time
            self.object_current_zone[track_id] = new_zone
        else:
            if track_id in self.object_current_zone:
                del self.object_current_zone[track_id]

    def compute_advanced_analytics(self):
        """
        Compute funnel conversion for multiple funnel stages and average dwell times if enabled.

        For funnel:
        - funnel_stages: [Z1, Z2, ..., Zn]
        - Count how many visited Z1 at all, and how many visited Z1 -> Z2 -> ... -> Zn in order.

        For dwell times:
        - Compute average dwell time per zone if enabled.
        """
        funnel_conversion = None
        visited_first = 0
        visited_all_stages = 0
        avg_dwell_times = {}

        if self.enable_funnel and self.funnel_stages and len(self.funnel_stages) >= 2:
            funnel_stages = self.funnel_stages
        else:
            funnel_stages = None

        zone_dwell_times = defaultdict(float)
        zone_visit_counts = defaultdict(int)

        for track_id, seq in self.object_zone_sequence.items():
            zones_visited = [z[0] for z in seq]

            # Funnel logic for multiple stages
            if funnel_stages:
                # Check if object visited the first stage
                if funnel_stages[0] in zones_visited:
                    visited_first += 1
                    # Check if all stages are visited in order
                    current_index = -1
                    all_stages_found = True
                    for stage in funnel_stages:
                        try:
                            next_index = zones_visited.index(stage, current_index + 1)
                            current_index = next_index
                        except ValueError:
                            all_stages_found = False
                            break
                    if all_stages_found:
                        visited_all_stages += 1

            # Average dwell time if enabled
            if self.enable_avg_dwell:
                for z_name, e_time, x_time in seq:
                    dwell = x_time - e_time
                    zone_dwell_times[z_name] += dwell
                    zone_visit_counts[z_name] += 1

        # Compute funnel conversion rate if funnel enabled
        if funnel_stages and visited_first > 0:
            funnel_conversion = (visited_all_stages / visited_first) * 100.0

        # Compute average dwell times if enabled
        if self.enable_avg_dwell:
            for z_name, total_time in zone_dwell_times.items():
                avg = total_time / zone_visit_counts[z_name] if zone_visit_counts[z_name] > 0 else 0
                avg_dwell_times[z_name] = avg

        return funnel_conversion, visited_first, visited_all_stages, avg_dwell_times

    def display_analytics(self, im0):
        annotator = Annotator(im0, line_width=self.line_width)
        zone_counts = defaultdict(int)
        for track_id, seq in self.object_zone_sequence.items():
            for z, _, _ in seq:
                zone_counts[z] += 1

        offset = 30

        # Display basic zone visit counts
        for z_name, count in zone_counts.items():
            annotator.text(
                (10, offset),
                f"{z_name}: Visited {count}",
                txt_color=self.analytics_text_color,
                box_style=True,
            )
            offset += 30

        # Compute advanced analytics only if needed
        funnel_conversion, visited_first, visited_all_stages, avg_dwell_times = self.compute_advanced_analytics()

        # Display funnel info if funnel is enabled and funnel_stages defined
        if self.enable_funnel and self.funnel_stages and len(self.funnel_stages) >= 2:
            offset += 20
            funnel_stages_str = " -> ".join(self.funnel_stages)
            annotator.text(
                (10, offset),
                f"Funnel: {funnel_stages_str}",
                txt_color=self.analytics_text_color,
                box_style=True,
            )
            offset += 30
            annotator.text(
                (10, offset),
                f"  Visited {self.funnel_stages[0]}: {visited_first}",
                txt_color=self.analytics_text_color,
                box_style=True,
            )
            offset += 30
            annotator.text(
                (10, offset),
                f"  Completed Funnel: {visited_all_stages}",
                txt_color=self.analytics_text_color,
                box_style=True,
            )
            offset += 30
            if funnel_conversion is not None:
                annotator.text(
                    (10, offset),
                    f"  Conversion Rate: {funnel_conversion:.2f}%",
                    txt_color=self.analytics_text_color,
                    box_style=True,
                )
            else:
                annotator.text(
                    (10, offset),
                    "  Conversion Rate: N/A",
                    txt_color=self.analytics_text_color,
                    box_style=True,
                )
            offset += 30

        # Display average dwell times if enabled
        if self.enable_avg_dwell and avg_dwell_times:
            annotator.text(
                (10, offset),
                "Average Dwell Times (per zone):",
                txt_color=self.analytics_text_color,
                box_style=True,
            )
            offset += 30
            for z_name, avg_dt in avg_dwell_times.items():
                unit = "s" if self.fps else "frames"
                annotator.text(
                    (10, offset),
                    f"  {z_name}: {avg_dt:.2f}{unit}",
                    txt_color=self.analytics_text_color,
                    box_style=True,
                )
                offset += 30

        im0[:] = annotator.result()

    def draw_zones(self, im0, current_zone_counts):
        annotator = Annotator(im0, line_width=self.line_width)
        for zone_name, polygon in self.zone_polygons.items():
            if polygon.geom_type == "Polygon":
                coords = list(polygon.exterior.coords)
            else:
                coords = list(polygon.coords)
            coords = [(int(x), int(y)) for (x, y) in coords]
            # Blue color for the zone
            annotator.draw_region(
                reg_pts=coords,
                color=(255, 0, 0),  # BGR for Blue
                thickness=self.line_width,
            )
            # Red color for the zone label
            count_in_zone = current_zone_counts.get(zone_name, 0)
            zone_label = f"{zone_name} | Count: {count_in_zone}"
            if coords:
                annotator.text(
                    (coords[0][0], coords[0][1] - 10),
                    zone_label,
                    txt_color=(0, 0, 255),
                    box_style=True,
                )  # Red label
        im0[:] = annotator.result()

    def count(self, im0):
        self.frame_number += 1
        # Pass classes to model.track if classes are specified
        self.tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"], **self.track_add_args)

        # Extract tracks for OBB or object detection
        self.track_data = self.tracks[0].obb or self.tracks[0].boxes

        if self.track_data and self.track_data.id is not None:
            self.boxes = self.track_data.xyxy.cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
        else:
            self.boxes, self.clss, self.track_ids = [], [], []

        current_time = self.get_current_time()
        annotator = Annotator(im0, line_width=self.line_width)

        # Process each object first to know which zone they are in
        current_zone_counts = defaultdict(int)
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            new_zone = self.identify_zone(current_centroid)
            old_zone = self.object_current_zone.get(track_id, None)
            if new_zone != old_zone:
                self.update_dwell_times(track_id, old_zone, new_zone)

            # Prepare label text: class name + ID and dwell time if in zone
            label = f"{self.names[cls]} | ID: {track_id}"
            if new_zone:
                entry_time = self.object_zone_entry_time.get((track_id, new_zone))
                if entry_time is not None:
                    dwell_time = current_time - entry_time
                    unit = "s" if self.fps else "frames"
                    label += f" | Dw: {dwell_time:.2f}{unit}"
                # Count the object in its current zone
                current_zone_counts[new_zone] += 1

            # Depending on detect_mode, decide whether to display the label
            if self.detect_mode == "all_frames":
                annotator.box_label(box, label=label, color=colors(cls, True))
            elif self.detect_mode == "enter_zones":
                if new_zone:
                    annotator.box_label(box, label=label, color=colors(cls, True))

        im0[:] = annotator.result()

        # Now draw zones with the current object counts
        self.draw_zones(im0, current_zone_counts)

        # Display analytics (if enabled)
        self.display_analytics(im0)

        # Display final output (if show=True)
        self.display_output(im0)
        return im0
