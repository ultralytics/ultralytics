# Ultralytics YOLO 🚀, AGPL-3.0 license

import json

import cv2
import numpy as np

from ultralytics.utils.plotting import Annotator
from ultralytics.solutions.solutions import BaseSolution, LOGGER


class ParkingPtsSelection:
    """Class for selecting and managing parking zone points on images using a Tkinter-based UI."""

    def __init__(self):
        """Initializes the UI for selecting parking zone points in a tkinter window."""
        check_requirements("tkinter")

        import tkinter as tk  # scope for multi-environment compatibility

        self.tk = tk
        self.master = tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")

        # Disable window resizing
        self.master.resizable(False, False)

        # Setup canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")

        # Setup buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        self.tk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0)
        self.tk.Button(button_frame, text="Remove Last BBox", command=self.remove_last_bounding_box).grid(
            row=0, column=1
        )
        self.tk.Button(button_frame, text="Save", command=self.save_to_json).grid(row=0, column=2)

        # Initialize properties
        self.image_path = None
        self.image = None
        self.canvas_image = None
        self.rg_data = []  # region coordinates
        self.current_box = []
        self.imgw = 0  # image width
        self.imgh = 0  # image height

        # Constants
        self.canvas_max_width = 1280
        self.canvas_max_height = 720

        self.master.mainloop()

    def upload_image(self):
        """Upload an image and resize it to fit canvas."""
        from tkinter import filedialog

        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package

        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not self.image_path:
            return

        self.image = Image.open(self.image_path)
        self.imgw, self.imgh = self.image.size

        # Calculate the aspect ratio and resize image
        aspect_ratio = self.imgw / self.imgh
        if aspect_ratio > 1:
            # Landscape orientation
            canvas_width = min(self.canvas_max_width, self.imgw)
            canvas_height = int(canvas_width / aspect_ratio)
        else:
            # Portrait orientation
            canvas_height = min(self.canvas_max_height, self.imgh)
            canvas_width = int(canvas_height * aspect_ratio)

        # Check if canvas is already initialized
        if self.canvas:
            self.canvas.destroy()  # Destroy previous canvas

        self.canvas = self.tk.Canvas(self.master, bg="white", width=canvas_width, height=canvas_height)
        resized_image = self.image.resize((canvas_width, canvas_height), Image.LANCZOS)
        self.canvas_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

        self.canvas.pack(side=self.tk.BOTTOM)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Reset bounding boxes and current box
        self.rg_data = []
        self.current_box = []

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")

        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box)
            [
                self.canvas.create_line(self.current_box[i], self.current_box[(i + 1) % 4], fill="blue", width=2)
                for i in range(4)
            ]
            self.current_box = []

    def remove_last_bounding_box(self):
        """Remove the last drawn bounding box from canvas."""
        from tkinter import messagebox  # scope for multi-environment compatibility

        if self.rg_data:
            self.rg_data.pop()  # Remove the last bounding box
            self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # Redraw the image

            # Redraw all bounding boxes
            for box in self.rg_data:
                [self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2) for i in range(4)]
            messagebox.showinfo("Success", "Last bounding box removed.")
        else:
            messagebox.showwarning("Warning", "No bounding boxes to remove.")

    def save_to_json(self):
        """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
        from tkinter import messagebox  # scope for multi-environment compatibility

        rg_data = []  # regions data
        for box in self.rg_data:
            rs_box = [
                (
                    int(x * self.imgw / self.canvas.winfo_width()),  # width scaling
                    int(y * self.imgh / self.canvas.winfo_height()),  # height scaling
                )
                for x, y in box
            ]
            rg_data.append({"points": rs_box})
        with open("bounding_boxes.json", "w") as f:
            json.dump(rg_data, f, indent=4)

        messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement(BaseSolution):
    """Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization."""

    def __init__(self, **kwargs):
        """Initializes the parking management system with a YOLO model and visualization settings."""

        super().__init__(**kwargs)

        self.json = self.CFG["json_file"]   # Load JSON data
        if self.json is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(json_file) as f:
            self.json_data = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information

        self.arc = (0, 0, 255)      # available region color
        self.ocr = (0, 255, 0)      # occupied region color

    def process_data(self, im0):
        """
        Process the model data for parking lot management.

        Args:
            im0 (ndarray): inference image
        """
        es, fs = len(self.json_data), 0  # empty slots, filled slots
        annotator = Annotator(im0)  # init annotator

        for region in self.json_data:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(self.boxes, self.clss):
                xc = int((box[0] + box[2]) / 2)
                yc = int((box[1] + box[3]) / 2)
                annotator.display_objects_labels(
                    im0, self.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
                )
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:
                    rg_occupied = True
                    break
            if rg_occupied:
                fs += 1
                es -= 1

            # Plotting regions
            color = self.occ if rg_occupied else self.arc
            cv2.polylines(im0, [pts_array], isClosed=True, color=color, thickness=2)

        self.pr_info["Occupancy"] = fs
        self.pr_info["Available"] = es

        annotator.display_analytics(im0, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage
