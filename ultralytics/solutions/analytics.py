from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class Analytics:
    """A class to create and update various types of charts (line, bar, pie) for visual analytics."""

    def __init__(
        self,
        type,
        writer,
        im0_shape,
        title="ultralytics",
        x_label="x",
        y_label="y",
        bg_color="white",
        fg_color="black",
        line_color="yellow",
        line_width=2,
        fontsize=13,
        view_img=False,
        save_img=True,
    ):
        """
        Initialize the Analytics class with various chart types.

        Args:
            type (str): Type of chart to initialize ('line', 'bar', or 'pie').
            writer: Video writer object to save the frames.
            im0_shape (tuple): Shape of the input image (width, height).
            title (str): Title of the chart.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            bg_color (str): Background color of the chart.
            fg_color (str): Foreground (text) color of the chart.
            line_color (str): Line color for line charts.
            line_width (int): Width of the lines in line charts.
            fontsize (int): Font size for chart text.
            view_img (bool): Whether to display the image.
            save_img (bool): Whether to save the image.
        """

        self.bg_color = bg_color
        self.fg_color = fg_color
        self.view_img = view_img
        self.save_img = save_img
        self.title = title
        self.writer = writer

        # Set figure size based on image shape
        figsize = (im0_shape[0] / 100, im0_shape[1] / 100)

        if type == "line":
            # Initialize line plot
            fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvas(fig)
            self.ax = fig.add_subplot(111, facecolor=self.bg_color)
            (self.line,) = self.ax.plot([], [], color=line_color, linewidth=line_width)

        elif type == "bar" or type == "pie":
            # Initialize bar or pie plot
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.ax.set_facecolor(self.bg_color)
            color_palette = [
                (31, 119, 180),
                (255, 127, 14),
                (44, 160, 44),
                (214, 39, 40),
                (148, 103, 189),
                (140, 86, 75),
                (227, 119, 194),
                (127, 127, 127),
                (188, 189, 34),
                (23, 190, 207),
            ]
            self.color_palette = [(r / 255, g / 255, b / 255, 1) for r, g, b in color_palette]
            self.color_cycle = cycle(self.color_palette)
            self.color_mapping = {}

            # Ensure pie chart is circular
            self.ax.axis("equal") if type == "pie" else None

        # Set common axis properties
        self.ax.set_title(self.title, color=self.fg_color, fontsize=fontsize)
        self.ax.set_xlabel(x_label, color=self.fg_color, fontsize=fontsize - 3)
        self.ax.set_ylabel(y_label, color=self.fg_color, fontsize=fontsize - 3)
        self.ax.tick_params(axis="both", colors=self.fg_color)

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
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)

        # Display and save the updated graph
        cv2.imshow(self.title, im0) if self.view_img else None
        self.writer.write(im0) if self.save_img else None

    def update_bar(self, count_dict):
        """
        Update the bar graph with new data.

        Args:
            count_dict (dict): Dictionary containing the count data to plot.
        """

        # Update bar graph data
        self.ax.clear()
        self.ax.set_facecolor(self.bg_color)
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
                color=self.fg_color,
            )

        # Display and save the updated graph
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        im0 = np.asarray(buf)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGBA2BGR)

        self.writer.write(im0) if self.save_img else None
        cv2.imshow(self.title, im0) if self.view_img else None

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
        wedges, autotexts = self.ax.pie(sizes, autopct=None, startangle=start_angle, textprops={"color": self.fg_color})

        # Construct legend labels with percentages
        legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]
        self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Adjust layout to fit the legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, right=0.75)

        # Display and save the updated chart
        im0 = self.fig.canvas.draw()
        im0 = np.array(self.fig.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.writer.write(im0) if self.save_img else None
        cv2.imshow(self.title, im0) if self.view_img else None


if __name__ == "__main__":
    Analytics("line", writer=None, im0_shape=None)
