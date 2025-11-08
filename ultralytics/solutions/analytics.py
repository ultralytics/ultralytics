# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from itertools import cycle
from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionResults  # Import a parent class


class Analytics(BaseSolution):
    """A class for creating and updating various types of charts for visual analytics.

    This class extends BaseSolution to provide functionality for generating line, bar, pie, and area charts based on
    object detection and tracking data.

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
        clswise_count (dict[str, int]): Dictionary for class-wise object counts.
        fig (Figure): Matplotlib figure object for the chart.
        ax (Axes): Matplotlib axes object for the chart.
        canvas (FigureCanvasAgg): Canvas for rendering the chart.
        lines (dict): Dictionary to store line objects for area charts.
        color_mapping (dict[str, str]): Dictionary mapping class labels to colors for consistent visualization.

    Methods:
        process: Process image data and update the chart.
        update_graph: Update the chart with new data points.

    Examples:
        >>> analytics = Analytics(analytics_type="line")
        >>> frame = cv2.imread("image.jpg")
        >>> results = analytics.process(frame, frame_number=1)
        >>> cv2.imshow("Analytics", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Analytics class with various chart types for visual data representation."""
        super().__init__(**kwargs)

        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.type = self.CFG["analytics_type"]  # type of analytics i.e "line", "pie", "bar" or "area" charts.
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        # Predefined data
        self.bg_color = "#F3F3F3"  # background color of frame
        self.fg_color = "#111E68"  # foreground color of frame
        self.title = "Ultralytics Solutions"  # window name
        self.max_points = 45  # maximum points to be drawn on window
        self.fontsize = 25  # text font size for display
        figsize = self.CFG["figsize"]  # set output image size i.e (12.8, 7.2) -> w = 1280, h = 720
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # count variable for storing total counts i.e. for line
        self.clswise_count = {}  # dictionary for class-wise counts
        self.update_every = kwargs.get("update_every", 30)  # Only update graph every 30 frames by default
        self.last_plot_im = None  # Cache of the last rendered chart

        # Ensure line and area chart
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvasAgg(self.fig)  # Set common axis properties
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvasAgg(self.fig)  # Set common axis properties
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":  # Ensure pie chart is circular
                self.ax.axis("equal")

    def process(self, im0: np.ndarray, frame_number: int) -> SolutionResults:
        """Process image data and run object tracking to update analytics charts.

        Args:
            im0 (np.ndarray): Input image for processing.
            frame_number (int): Video frame number for plotting the data.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'total_tracks' (int, total number of tracked objects)
                and 'classwise_count' (dict, per-class object count).

        Raises:
            ModuleNotFoundError: If an unsupported chart type is specified.

        Examples:
            >>> analytics = Analytics(analytics_type="line")
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> results = analytics.process(frame, frame_number=1)
        """
        self.extract_tracks(im0)  # Extract tracks
        if self.type == "line":
            for _ in self.boxes:
                self.total_counts += 1
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(frame_number=frame_number)
            plot_im = self.last_plot_im
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            from collections import Counter

            self.clswise_count = Counter(self.names[int(cls)] for cls in self.clss)
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(
                    frame_number=frame_number, count_dict=self.clswise_count, plot=self.type
                )
            plot_im = self.last_plot_im
        else:
            raise ModuleNotFoundError(f"{self.type} chart is not supported ❌")

        # return output dictionary with summary for more usage
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), classwise_count=self.clswise_count)

    def update_graph(
        self, frame_number: int, count_dict: dict[str, int] | None = None, plot: str = "line"
    ) -> np.ndarray:
        """Update the graph with new data for single or multiple classes.

        Args:
            frame_number (int): The current frame number.
            count_dict (dict[str, int], optional): Dictionary with class names as keys and counts as values for multiple
                classes. If None, updates a single line graph.
            plot (str): Type of the plot. Options are 'line', 'bar', 'pie', or 'area'.

        Returns:
            (np.ndarray): Updated image containing the graph.

        Examples:
            >>> analytics = Analytics(analytics_type="bar")
            >>> frame_num = 10
            >>> results_dict = {"person": 5, "car": 3}
            >>> updated_image = analytics.update_graph(frame_num, results_dict, plot="bar")
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
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])))
                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict.keys():
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()
                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.55)
                    self.ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        marker="o",
                        markersize=self.line_width * 5,
                        label=f"{key} Data Points",
                    )
            elif plot == "bar":
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
            elif plot == "pie":
                total = sum(counts)
                percentages = [size / total * 100 for size in counts]
                self.ax.clear()

                start_angle = 90
                # Create pie chart and create legend labels with percentages
                wedges, _ = self.ax.pie(
                    counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
                )
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

                # Assign the legend using the wedges and manually created labels
                self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                self.fig.subplots_adjust(left=0.1, right=0.75)  # Adjust layout to fit the legend

        # Common plot settings
        self.ax.set_facecolor("#f0f0f0")  # Set to light gray or any other color you like
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)  # Display grid for more data insights
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
