# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class


class Analytics(BaseSolution):
    """
    A class for creating and updating various types of charts for visual analytics.

    This class extends BaseSolution to provide functionality for generating line, bar, pie, and area charts
    based on object detection and tracking data.

    Attributes:
        type (str): The type of analytics chart to generate ('line', 'bar', 'pie', or 'area').
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        bg_color (str): Background color of the chart frame.
        fg_color (str): Foreground color of the chart frame.
        title (str): Title of the chart window.
        max_points (int): Maximum number of data points to display on the chart.
        fontsize (int): Font size for text display.
        color_cycle (cycle): Cyclic iterator for chart colors.
        total_counts (int): Total count of detected objects (used for line charts).
        clswise_count (Dict[str, int]): Dictionary for class-wise object counts.
        fig (Figure): Matplotlib figure object for the chart.
        ax (Axes): Matplotlib axes object for the chart.
        canvas (FigureCanvas): Canvas for rendering the chart.

    Methods:
        process_data: Processes image data and updates the chart.
        update_graph: Updates the chart with new data points.

    Examples:
        >>> analytics = Analytics(analytics_type="line")
        >>> frame = cv2.imread("image.jpg")
        >>> processed_frame = analytics.process_data(frame, frame_number=1)
        >>> cv2.imshow("Analytics", processed_frame)
    """

    def __init__(self, **kwargs):
        """Initialize Analytics class with various chart types for visual data representation."""
        super().__init__(**kwargs)

        self.type = self.CFG["analytics_type"]  # extract type of analytics
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        # Predefined data
        self.bg_color = "#F3F3F3"  # background color of frame
        self.fg_color = "#111E68"  # foreground color of frame
        self.title = "Ultralytics Solutions"  # window name
        self.max_points = 45  # maximum points to be drawn on window
        self.fontsize = 25  # text font size for display
        figsize = (19.2, 10.8)  # Set output image size 1920 * 1080
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # count variable for storing total counts i.e. for line
        self.clswise_count = {}  # dictionary for class-wise counts

        # Ensure line and area chart
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvas(self.fig)  # Set common axis properties
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvas(self.fig)  # Set common axis properties
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":  # Ensure pie chart is circular
                self.ax.axis("equal")

    def process_data(self, im0, frame_number):
        """
        Processes image data and runs object tracking to update analytics charts.

        Args:
            im0 (np.ndarray): Input image for processing.
            frame_number (int): Video frame number for plotting the data.

        Returns:
            (np.ndarray): Processed image with updated analytics chart.

        Raises:
            ModuleNotFoundError: If an unsupported chart type is specified.

        Examples:
            >>> analytics = Analytics(analytics_type="line")
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> processed_frame = analytics.process_data(frame, frame_number=1)
        """
        self.extract_tracks(im0)  # Extract tracks

        if self.type == "line":
            for _ in self.boxes:
                self.total_counts += 1
            im0 = self.update_graph(frame_number=frame_number)
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            self.clswise_count = {}
            for box, cls in zip(self.boxes, self.clss):
                if self.names[int(cls)] in self.clswise_count:
                    self.clswise_count[self.names[int(cls)]] += 1
                else:
                    self.clswise_count[self.names[int(cls)]] = 1
            im0 = self.update_graph(frame_number=frame_number, count_dict=self.clswise_count, plot=self.type)
        else:
            raise ModuleNotFoundError(f"{self.type} chart is not supported ❌")
        return im0

    def update_graph(self, frame_number, count_dict=None, plot="line"):
        """
        Updates the graph with new data for single or multiple classes.

        Args:
            frame_number (int): The current frame number.
            count_dict (Dict[str, int] | None): Dictionary with class names as keys and counts as values for multiple
                classes. If None, updates a single line graph.
            plot (str): Type of the plot. Options are 'line', 'bar', 'pie', or 'area'.

        Returns:
            (np.ndarray): Updated image containing the graph.

        Examples:
            >>> analytics = Analytics()
            >>> frame_number = 10
            >>> count_dict = {"person": 5, "car": 3}
            >>> updated_image = analytics.update_graph(frame_number, count_dict, plot="bar")
        """
        if count_dict is None:
            # Single line update
            x_data = np.append(self.line.get_xdata(), float(frame_number))
            y_data = np.append(self.line.get_ydata(), float(self.total_counts))

            if len(x_data) > self.max_points:
                x_data, y_data = x_data[-self.max_points :], y_data[-self.max_points :]

            self.line.set_data(x_data, y_data)
            self.line.set_label("Counts")
            self.line.set_color("#7b0068")  # Pink color
            self.line.set_marker("*")
            self.line.set_markersize(self.line_width * 5)
        else:
            labels = list(count_dict.keys())
            counts = list(count_dict.values())
            if plot == "area":
                color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])
                # Multiple lines or area update
                x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
                y_data_dict = {key: np.array([]) for key in count_dict.keys()}
                if self.ax.lines:
                    for line, key in zip(self.ax.lines, count_dict.keys()):
                        y_data_dict[key] = line.get_ydata()

                x_data = np.append(x_data, float(frame_number))
                max_length = len(x_data)
                for key in count_dict.keys():
                    y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                    if len(y_data_dict[key]) < max_length:
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])), "constant")
                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict.keys():
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()
                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.7)
                    self.ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        marker="o",
                        markersize=self.line_width * 5,
                        label=f"{key} Data Points",
                    )
            if plot == "bar":
                self.ax.clear()  # clear bar data
                for label in labels:  # Map labels to colors
                    if label not in self.color_mapping:
                        self.color_mapping[label] = next(self.color_cycle)
                colors = [self.color_mapping[label] for label in labels]
                bars = self.ax.bar(labels, counts, color=colors)
                for bar, count in zip(bars, counts):
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha="center",
                        va="bottom",
                        color=self.fg_color,
                    )
                # Create the legend using labels from the bars
                for bar, label in zip(bars, labels):
                    bar.set_label(label)  # Assign label to each bar
                self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)
            if plot == "pie":
                total = sum(counts)
                percentages = [size / total * 100 for size in counts]
                start_angle = 90
                self.ax.clear()

                # Create pie chart and create legend labels with percentages
                wedges, autotexts = self.ax.pie(
                    counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
                )
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

                # Assign the legend using the wedges and manually created labels
                self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                self.fig.subplots_adjust(left=0.1, right=0.75)  # Adjust layout to fit the legend

        # Common plot settings
        self.ax.set_facecolor("#f0f0f0")  # Set to light gray or any other color you like
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

        # Add and format legend
        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.bg_color)
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        # Redraw graph, update view, capture, and display the updated plot
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display_output(im0)

        return im0  # Return the image
