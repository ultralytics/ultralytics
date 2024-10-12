# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class


class Analytics(BaseSolution):
    """A class to create and update various types of charts (line, bar, pie, area) for visual analytics."""

    def __init__(self, **kwargs):
        """Initialize the Analytics class with various chart types."""
        super().__init__(**kwargs)

        self.type = self.CFG["analytics_type"]  # extract type of analytics
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        # Predefined data
        self.bg_color = "#00F344"  # background color of frame
        self.fg_color = "#111E68"  # foreground color of frame
        self.title = "Ultralytics Solutions"  # window name
        self.max_points = 45  # maximum points to be drawn on window
        self.fontsize = 25  # text font size for display
        figsize = (19.2, 10.8)  # Set output image size 1920 * 1080
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # count variable for storing total counts i.e for line
        self.clswise_count = {}  # dictionary for classwise counts

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
            self.ax.axis("equal") if type == "pie" else None  # Ensure pie chart is circular

    def process_data(self, im0, frame_number):
        """
        Process the image data, run object tracking.

        Args:
            im0 (ndarray): Input image for processing.
            frame_number (int): Video frame # for plotting the data.
        """
        self.extract_tracks(im0)  # Extract tracks

        if self.type == "line":
            for box in self.boxes:
                self.total_counts += 1
            im0 = self.update_graph(frame_number=frame_number)
            self.total_counts = 0
        elif self.type == "pie" or self.type == "bar" or self.type == "area":
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
        Update the graph (line or area) with new data for single or multiple classes.

        Args:
            frame_number (int): The current frame number.
            count_dict (dict, optional): Dictionary with class names as keys and counts as values for multiple classes.
                                          If None, updates a single line graph.
            plot (str): Type of the plot i.e. line, bar or area.
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
