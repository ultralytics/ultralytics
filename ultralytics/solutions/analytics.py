# Ultralytics YOLO 🚀, AGPL-3.0 license

import warnings
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.plotting import colors


class Analytics:
    """A class for creating and updating different types of charts (line, bar, pie, area) for visual analytics."""

    def __init__(self, **kwargs):
        """
        Initializes an instance of the Analytics class, allowing for the creation and customization of various chart
        types.

        Args:
            kwargs (dict): Dictionary of arguments for configuring chart properties such as data sources, titles, axis labels, colors, and other visual parameters.
        """
        import ast

        DEFAULT_CFG_DICT.update(kwargs)
        print(f"Ultralytics Solutions ✅ {DEFAULT_CFG_DICT}")

        # Set figure size based on image shape
        figsize = (DEFAULT_CFG_DICT["im0_shape"][0] / 100, DEFAULT_CFG_DICT["im0_shape"][1] / 100)

        self.facecolor = colors.rgb2hex(DEFAULT_CFG_DICT['bg_color'])
        self.txt_color = colors.rgb2hex(DEFAULT_CFG_DICT['txt_color'])

        if DEFAULT_CFG_DICT["type"] in {"line", "area"}:
            # Initialize line or area plot
            self.lines = {}
            self.fig = Figure(facecolor=self.facecolor, figsize=figsize)
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.add_subplot(111, facecolor=self.facecolor)
            if DEFAULT_CFG_DICT["type"] == "line":
                rgb = ast.literal_eval(DEFAULT_CFG_DICT["line_color"])
                hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                (self.line,) = self.ax.plot([], [], color=hex_color, linewidth=DEFAULT_CFG_DICT["line_width"])

        elif DEFAULT_CFG_DICT["type"] in {"bar", "pie"}:
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.facecolor)
            self.ax.set_facecolor(self.facecolor)

            self.color_cycle = cycle([(r / 255, g / 255, b / 255, 1) for r, g, b in colors.palette])
            self.color_mapping = {}

            # Ensure pie chart is circular
            self.ax.axis("equal") if DEFAULT_CFG_DICT["type"] == "pie" else None

    def set_common_properties(self):
        """Set common axis properties for visual graphs i.e pie chart, line graph, area plot and bar plots."""
        self.ax.set_title(DEFAULT_CFG_DICT["title"], color=self.txt_color, fontsize=DEFAULT_CFG_DICT["fontsize"])
        self.ax.set_xlabel(DEFAULT_CFG_DICT["x_label"], color=self.txt_color, fontsize=DEFAULT_CFG_DICT["fontsize"] - 3)
        self.ax.set_ylabel(DEFAULT_CFG_DICT["y_label"], color=self.txt_color, fontsize=DEFAULT_CFG_DICT["fontsize"] - 3)
        self.ax.tick_params(axis="both", colors=self.txt_color)

    def update_area(self, frame_number, counts_dict):
        """
        Update the area graph with new data for multiple classes.

        Args:
            frame_number (int): The number of the current frame.
            counts_dict (dict): A dictionary where the keys are class names and the values are their respective counts.
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
        if len(x_data) > DEFAULT_CFG_DICT["max_points"]:
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
                linewidth=DEFAULT_CFG_DICT["line_width"],
                marker="o",
                markersize=DEFAULT_CFG_DICT["points_width"],
                label=f"{key} Data Points",
            )

        self.set_common_properties()
        legend = self.ax.legend(
            loc="upper left", fontsize=13, facecolor=self.facecolor, edgecolor=self.txt_color
        )

        # Set legend text color
        for text in legend.get_texts():
            text.set_color(self.facecolor)

        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display(im0)
        return im0

    def update_line(self, frame_number, total_counts):
        """
        Update the line graph with new data.

        Args:
            frame_number (int): The number of the current frame.
            total_counts (int): The total count to be plotted.
        """
        x_data = np.append(self.line.get_xdata(), float(frame_number))
        y_data = np.append(self.line.get_ydata(), float(total_counts))
        self.line.set_data(x_data, y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.set_common_properties()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display(im0)
        return im0

    def update_multiple_lines(self, counts_dict, labels_list, frame_number):
        """
        Update the line graph with data for multiple classes.

        Args:
            counts_dict (int): A dictionary containing the counts for each class.
            labels_list (int): A list of class names.
            frame_number (int): The number of the current frame.
        """
        warnings.warn("Display is not supported for multiple lines, output will be stored normally!")
        for obj in labels_list:
            if obj not in self.lines:
                (line,) = self.ax.plot([], [], label=obj, marker="o", markersize=DEFAULT_CFG_DICT["points_width"])
                self.lines[obj] = line

            x_data = self.lines[obj].get_xdata()
            y_data = self.lines[obj].get_ydata()

            # Remove the initial point if the number of points exceeds max_points
            if len(x_data) >= DEFAULT_CFG_DICT["max_points"]:
                x_data = np.delete(x_data, 0)
                y_data = np.delete(y_data, 0)

            x_data = np.append(x_data, float(frame_number))  # Ensure frame_number is converted to float
            y_data = np.append(y_data, float(counts_dict.get(obj, 0)))  # Ensure total_count is converted to float
            self.lines[obj].set_data(x_data, y_data)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        self.set_common_properties()
        self.canvas.draw()

        im0 = np.array(self.canvas.renderer.buffer_rgba())
        DEFAULT_CFG_DICT["show"] = False  # for multiple line view_img not supported yet, coming soon!
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display(im0)
        return im0

    def update_bar(self, count_dict):
        """
        Refresh the bar graph with updated data.

        Args:
            count_dict (dict): A dictionary containing the data counts to be plotted.
        """
        self.ax.clear()
        self.ax.set_facecolor(self.facecolor)
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
                color=self.facecolor,
            )
        self.set_common_properties()

        # Display and save the updated graph
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        im0 = np.asarray(buf)
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display(im0)
        return im0

    def update_pie(self, classes_dict):
        """
        Refresh the pie chart with updated data.

        Args:
            classes_dict (dict): A dictionary containing the data for each class to be plotted.
        """
        labels = list(classes_dict.keys())
        sizes = list(classes_dict.values())
        total = sum(sizes)
        percentages = [size / total * 100 for size in sizes]
        start_angle = 90
        self.ax.clear()

        # Create pie chart without labels inside the slices
        wedges, autotexts = self.ax.pie(
            sizes, autopct=None, startangle=start_angle, textprops={"color": self.facecolor}
        )

        # Construct legend labels with percentages
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        self.set_common_properties()

        # Adjust layout to fit the legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.75)

        # Display and save the updated chart
        im0 = self.fig.canvas.draw()
        im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display(im0)
        return im0

    def display(self, im0):
        """
        Write and display the line graph
        Generate and display the line graph.
        Args:
            im0 (ndarray): Image for processing
            im0 (ndarray): The image to be processed.
        """
        cv2.imshow(DEFAULT_CFG_DICT["title"], im0) if DEFAULT_CFG_DICT["show"] else None

if __name__ == "__main__":
    Analytics(data={"type": "line", "writer": None, "im0_shape": None})
