from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors


class DwellTimeAnalyzer(BaseSolution):
    """
    A solution class that extends BaseSolution to measure Dwell Time and Zone Transitions.

    Features:
        - Dwell Time Measurement: Tracks how long each detected object remains within specified zones.
        - Zone Sequence Tracking: Monitors the sequence of zones visited by each object.
        - Class Filtering: Optionally restricts detection to certain object classes.

    Optional Advanced Analytics:
        - Funnel Analysis: If `enable_funnel=True`.
            - If `funnel_stages` are provided and have N stages, it checks how many objects pass through all stages in order.
            - If `funnel_stages` are not provided, it uses all defined zones in order as funnel stages.

    Configuration Keys:
        - fps (Optional[float]): Frames per second. If provided, dwell times are in seconds instead of frames.
        - classes (Optional[List[int]]): List of class indices to track if filtering is desired.
        - zones (Dict[str, List[Tuple[int, int]]]): Mapping of zone names to their polygon points.
        - enable_funnel (bool): If True, compute funnel conversion metrics.
        - funnel_stages (Optional[Union[List[str], Tuple[str, ...]]]): Sequence of zone names forming the funnel.

    Note:
        Since object tracking is already handled in `solutions.py`, we assume `self.track_data` (which includes
        bounding boxes, class IDs, and track IDs) is already populated there.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the DwellTimeAnalyzer with the provided configuration.

        Args:
            **kwargs: Arbitrary keyword arguments for configuration.
        """
        super().__init__(**kwargs)

        self.fps: Optional[float] = self.CFG.get("fps", None)
        self.analytics_text_color: Tuple[int, int, int] = self.CFG.get("analytics_text_color", (0, 0, 0))
        self.zones: Dict[str, List[Tuple[int, int]]] = self.CFG.get("zones", {})
        self.classes: Optional[List[int]] = self.CFG.get("classes", None)  # If provided, only track these classes

        # Optional features
        self.enable_funnel: bool = self.CFG["enable_funnel"]
        self.funnel_stages: Optional[Union[List[str], Tuple[str, ...]]] = self.CFG["funnel_stages"]

        # If no zones are provided, you might want to define them externally (removed video capture approach).

        # Create shapely polygons or lines for each zone
        self.zone_polygons: Dict[str, Union[self.Polygon, self.LineString]] = {}
        for zone_name, coords in self.zones.items():
            if len(coords) >= 3:
                self.zone_polygons[zone_name] = self.Polygon(coords)
            else:
                self.zone_polygons[zone_name] = self.LineString(coords)

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
        self.object_current_zone: Dict[int, str] = {}  # track_id -> zone_name
        self.object_zone_entry_time: Dict[Tuple[int, str], float] = {}  # (track_id, zone_name) -> entry_time
        self.object_zone_sequence: defaultdict = defaultdict(list)  # track_id -> [(zone_name, entry_time, exit_time)]
        self.frame_number: int = 0

    def get_current_time(self) -> float:
        """
        Calculates the current time based on the frame number and FPS.

        Returns:
            float: Current time in seconds if FPS is provided, otherwise in frames.
        """
        return self.frame_number / self.fps if (self.fps and self.fps > 0) else self.frame_number

    def identify_zone(self, centroid: Tuple[float, float]) -> Optional[str]:
        """
        Identifies which zone a given centroid point belongs to.

        Args:
            centroid (Tuple[float, float]): The (x, y) coordinates of the centroid.

        Returns:
            Optional[str]: The name of the zone containing the centroid, or None if not in any zone.
        """
        point = self.Point(centroid)
        for zone_name, polygon in self.zone_polygons.items():
            if polygon.contains(point):
                return zone_name
        return None

    def update_dwell_times(self, track_id: int, old_zone: Optional[str], new_zone: Optional[str]) -> None:
        """
        Updates dwell times and zone sequences when an object transitions between zones.

        Args:
            track_id (int): The unique identifier of the tracked object.
            old_zone (Optional[str]): The previous zone name.
            new_zone (Optional[str]): The new zone name.
        """
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

    def compute_advanced_analytics(self) -> Tuple[Optional[float], int, int]:
        """
        Computes funnel conversion rates (if funnel is enabled).

        Returns:
            Tuple[Optional[float], int, int]:
                - Funnel conversion rate as a percentage.
                - Number of objects that visited the first funnel stage.
                - Number of objects that completed all funnel stages.
        """
        funnel_conversion: Optional[float] = None
        visited_first: int = 0
        visited_all_stages: int = 0

        if self.enable_funnel and self.funnel_stages and len(self.funnel_stages) >= 2:
            funnel_stages = self.funnel_stages
        else:
            funnel_stages = None

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

        # Compute funnel conversion rate if funnel enabled
        if funnel_stages and visited_first > 0:
            funnel_conversion = (visited_all_stages / visited_first) * 100.0

        return funnel_conversion, visited_first, visited_all_stages

    def display_analytics(self, im0: Any) -> None:
        """
        Displays analytics information on the provided image frame.

        Args:
            im0 (Any): The image frame on which to display analytics.
        """
        annotator = Annotator(im0, line_width=self.line_width)
        zone_counts: defaultdict = defaultdict(int)
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

        # Compute advanced analytics
        funnel_conversion, visited_first, visited_all_stages = self.compute_advanced_analytics()

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

        im0[:] = annotator.result()

    def draw_zones(self, im0: Any, current_zone_counts: Dict[str, int]) -> None:
        """
        Draws the defined zones on the image frame along with their labels and current counts.

        Args:
            im0 (Any): The image frame on which to draw the zones.
            current_zone_counts (Dict[str, int]): Current counts of objects in each zone.
        """
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
                )
        im0[:] = annotator.result()

    def analyze(self, im0: Any) -> Any:
        """
        Processes each frame to update dwell times, draw zones, and display analytics.

        We assume that `solutions.py` has already populated `self.track_data`, `self.boxes`, `self.clss`, and
        `self.track_ids`.

        Args:
            im0 (Any): The current image frame to process.

        Returns:
            Any: The annotated image frame.
        """
        # Increment frame count and get current time
        self.frame_number += 1
        current_time = self.get_current_time()
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        # Process each tracked object to determine zones
        current_zone_counts: Dict[str, int] = defaultdict(int)
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            new_zone = self.identify_zone(current_centroid)
            old_zone = self.object_current_zone.get(track_id, None)
            if new_zone != old_zone:
                self.update_dwell_times(track_id, old_zone, new_zone)

            # Create a label with dwell time if the object is in a zone
            label = f"{self.names[cls]} | ID: {track_id}"
            if new_zone:
                entry_time = self.object_zone_entry_time.get((track_id, new_zone))
                if entry_time is not None:
                    dwell_time = current_time - entry_time
                    unit = "s" if self.fps else "frames"
                    label += f" | Dw: {dwell_time:.2f}{unit}"
                # Count the object in its current zone
                current_zone_counts[new_zone] += 1

            # Simply annotate all bounding boxes
            self.annotator.box_label(box, label=label, color=colors(cls, True))

        im0[:] = self.annotator.result()

        # Now draw zones with the current object counts
        self.draw_zones(im0, current_zone_counts)

        # Display analytics (if needed)
        self.display_analytics(im0)

        # Display final output (if show=True)
        self.display_output(im0)
        return im0
