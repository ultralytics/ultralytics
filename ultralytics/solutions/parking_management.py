# Ultralytics YOLO 🚀, AGPL-3.0 license

import json

import cv2
import numpy as np

from ultralytics.solutions.solutions import LOGGER, BaseSolution, check_requirements
from ultralytics.utils.plotting import Annotator


class ParkingPtsSelection:
    """Class for selecting and managing parking zone points on images using a Tkinter-based UI."""

    def __init__(self):
        """Class initialization method."""
        check_requirements("tkinter")
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.setup_ui()
        self.initialize_properties()
        self.master.mainloop()

    def setup_ui(self):
        """Sets up the Tkinter UI components."""
        self.master = self.tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        # Canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        # Button frame with buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

    def initialize_properties(self):
        """Initialize the necessary properties."""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1280, 720

    def upload_image(self):
        """Uploads an image, resizes it to fit the canvas, and displays it."""
        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]))
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height), Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event):
        """Handles mouse clicks to add points for bounding boxes."""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box):
        """Draws a bounding box on the canvas."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Removes the last bounding box and redraws the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraws the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """Saves the bounding boxes to a JSON file."""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]
        with open("bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)
        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization."""

    def __init__(self, **kwargs):
        """Initializes the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # Load JSON data
        if self.json_file is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information

        self.arc = (0, 0, 255)  # available region color
        self.occ = (0, 255, 0)  # occupied region color
        self.dc = (255, 0, 189)  # centroid color for each box

    def process_data(self, im0):
        """
        Process the model data for parking lot management.

        Args:
            im0 (ndarray): inference image.
        """
        self.extract_tracks(im0)  # extract tracks from im0
        es, fs = len(self.json), 0  # empty slots, filled slots
        annotator = Annotator(im0, self.line_width)  # init annotator

        for region in self.json:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(self.boxes, self.clss):
                xc, yc = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    # cv2.circle(im0, (xc, yc), radius=self.line_width * 4, color=self.dc, thickness=-1)
                    annotator.display_objects_labels(
                        im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                    )
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # Plotting regions
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)
        self.display_output(im0)  # display output with base class function
        return im0  # return output image for more usage
