# Ultralytics YOLO 🚀, AGPL-3.0 license

import warnings
from itertools import cycle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from ultralytics.cfg import get_cfg
from ultralytics.solutions.cfg import extract_cfg_data
from ultralytics.utils.plotting import colors

FILE = Path(__file__).resolve()  # get path of file


class Analytics:
    """A class to create and update various types of charts (line, bar, pie, area) for visual analytics."""

    def __init__(self, **kwargs):
        """Initialize the Analytics class with various chart types."""
        import ast

        self.args = get_cfg(extract_cfg_data(FILE))
        if "writer" not in kwargs:
            raise ValueError("Error: Video writer is required")
        elif "im0_shape" not in kwargs:
            raise ValueError("Error: Im0 is required")
        for key, value in kwargs.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
            else:
                print(f"Warning: Unknown argument Skipping!!! {key}")

        # Set figure size based on image shape
        figsize = (self.args.im0_shape[0] / 100, self.args.im0_shape[1] / 100)

        if self.args.type in {"line", "area"}:
            # Initialize line or area plot
            self.lines = {}
            self.fig = Figure(facecolor=self.args.bg_color, figsize=figsize)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111, facecolor=self.args.bg_color)
            if self.args.type == "line":
                rgb = ast.literal_eval(self.args.line_color)
                hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                (self.line,) = self.ax.plot([], [], color=hex_color, linewidth=self.args.line_width)

        elif self.args.type in {"bar", "pie"}:
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.args.bg_color)
            self.ax.set_facecolor(self.args.bg_color)

            self.color_palette = [(r / 255, g / 255, b / 255, 1) for r, g, b in colors.palette]
            self.color_cycle = cycle(self.color_palette)
            self.color_mapping = {}

            # Ensure pie chart is circular
            self.ax.axis("equal") if type == "pie" else None

        # Set common axis properties
        self.ax.set_title(self.args.title, color=self.args.fg_color, fontsize=self.args.fontsize)
        self.ax.set_xlabel(self.args.x_label, color=self.args.fg_color, fontsize=self.args.fontsize - 3)
        self.ax.set_ylabel(self.args.y_label, color=self.args.fg_color, fontsize=self.args.fontsize - 3)
        self.ax.tick_params(axis="both", colors=self.args.fg_color)

    def update_area(self, frame_number, counts_dict):
        """
        Update the area graph with new data for multiple classes.

        Args:
            frame_number (int): The current frame number.
            counts_dict (dict): Dictionary with class names as keys and counts as values.
        """

        x_data = np.array([])
        y_data_dict = {key: np.array([]) for key in counts_dict.keys()}

        if self.ax.lines:
            x_data = self.ax.lines[0].get_xdata()
            for line, key in zip(self.ax.lines, counts_dict.keys()):
                y_data_dict[key] = line.get_ydata()

        x_data = np.append(x_data, float(frame_number))
        max_length = len(x_data)

        for key in counts_dict.keys():
            y_data_dict[key] = np.append(y_data_dict[key], float(counts_dict[key]))
            if len(y_data_dict[key]) < max_length:
                y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])), "constant")

        # Remove the oldest points if the number of points exceeds max_points
        if len(x_data) > self.args.max_points:
            x_data = x_data[1:]
            for key in counts_dict.keys():
                y_data_dict[key] = y_data_dict[key][1:]

        self.ax.clear()
        color_cycle = cycle(list(colors.hexs))
        for key, y_data in y_data_dict.items():
            color = "#" + next(color_cycle)
            self.ax.fill_between(x_data, y_data, color=color, alpha=0.6)
            self.ax.plot(
                x_data,
                y_data,
                color=color,
                linewidth=self.args.line_width,
                marker="o",
                markersize=self.args.points_width,
                label=f"{key} Data Points",
            )

        self.ax.set_title(self.args.title, color=self.args.fg_color, fontsize=self.args.fontsize)
        self.ax.set_xlabel(self.args.x_label, color=self.args.fg_color, fontsize=self.args.fontsize - 3)
        self.ax.set_ylabel(self.args.y_label, color=self.args.fg_color, fontsize=self.args.fontsize - 3)
        legend = self.ax.legend(
            loc="upper left", fontsize=13, facecolor=self.args.bg_color, edgecolor=self.args.fg_color
        )

        # Set legend text color
        for text in legend.get_texts():
            text.set_color(self.args.fg_color)

        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        self.write_and_display(im0)

    def update_line(self, frame_number, total_counts):
        """
        Update the line graph with new data.

        Args:
            frame_number (int): The current frame number.
            total_counts (int): The total counts to plot.
        """

        # Update line graph data
        x_data = self.line.get_xdata()
        y_data = self.line.get_ydata()
        x_data = np.append(x_data, float(frame_number))
        y_data = np.append(y_data, float(total_counts))
        self.line.set_data(x_data, y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        self.write_and_display(im0)

    def update_multiple_lines(self, counts_dict, labels_list, frame_number):
        """
        Update the line graph with multiple classes.

        Args:
            counts_dict (int): Dictionary include each class counts.
            labels_list (int): list include each classes names.
            frame_number (int): The current frame number.
        """
        warnings.warn("Display is not supported for multiple lines, output will be stored normally!")
        for obj in labels_list:
            if obj not in self.lines:
                (line,) = self.ax.plot([], [], label=obj, marker="o", markersize=self.args.points_width)
                self.lines[obj] = line

            x_data = self.lines[obj].get_xdata()
            y_data = self.lines[obj].get_ydata()

            # Remove the initial point if the number of points exceeds max_points
            if len(x_data) >= self.args.max_points:
                x_data = np.delete(x_data, 0)
                y_data = np.delete(y_data, 0)

            x_data = np.append(x_data, float(frame_number))  # Ensure frame_number is converted to float
            y_data = np.append(y_data, float(counts_dict.get(obj, 0)))  # Ensure total_count is converted to float
            self.lines[obj].set_data(x_data, y_data)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        self.canvas.draw()

        im0 = np.array(self.canvas.renderer.buffer_rgba())
        self.args.view_img = False  # for multiple line view_img not supported yet, coming soon!
        self.write_and_display(im0)

    def write_and_display(self, im0):
        """
        Write and display the line graph
        Args:
            im0 (ndarray): Image for processing
        """
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        cv2.imshow(self.args.title, im0) if self.args.view_img else None
        self.args.writer.write(im0) if self.args.save_img else None

    def update_bar(self, count_dict):
        """
        Update the bar graph with new data.

        Args:
            count_dict (dict): Dictionary containing the count data to plot.
        """

        # Update bar graph data
        self.ax.clear()
        self.ax.set_facecolor(self.args.bg_color)
        labels = list(count_dict.keys())
        counts = list(count_dict.values())

        # Map labels to colors
        for label in labels:
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
                color=self.args.fg_color,
            )

        # Display and save the updated graph
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        im0 = np.asarray(buf)
        self.write_and_display(im0)

    def update_pie(self, classes_dict):
        """
        Update the pie chart with new data.

        Args:
            classes_dict (dict): Dictionary containing the class data to plot.
        """

        # Update pie chart data
        labels = list(classes_dict.keys())
        sizes = list(classes_dict.values())
        total = sum(sizes)
        percentages = [size / total * 100 for size in sizes]
        start_angle = 90
        self.ax.clear()

        # Create pie chart without labels inside the slices
        wedges, autotexts = self.ax.pie(
            sizes, autopct=None, startangle=start_angle, textprops={"color": self.args.fg_color}
        )

        # Construct legend labels with percentages
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Adjust layout to fit the legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.75)

        # Display and save the updated chart
        im0 = self.fig.canvas.draw()
        im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.write_and_display(im0)


if __name__ == "__main__":
    Analytics(type="line", writer=None, im0_shape=None)
