# Ultralytics YOLO 🚀, AGPL-3.0 license

import json

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator


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
            rs_box = []  # rescaled box list
            for x, y in box:
                rs_box.append(
                    (
                        int(x * self.imgw / self.canvas.winfo_width()),  # width scaling
                        int(y * self.imgh / self.canvas.winfo_height()),
                    )
                )  # height scaling
            rg_data.append({"points": rs_box})
        with open("bounding_boxes.json", "w") as f:
            json.dump(rg_data, f, indent=4)

        messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement:
    """Manages parking occupancy and availability using YOLOv8 for real-time monitoring and visualization."""

    def __init__(
        self,
        model,  # Ultralytics YOLO model file path
        json_file,  # Parking management annotation file created from Parking Annotator
        occupied_region_color=(0, 0, 255),  # occupied region color
        available_region_color=(0, 255, 0),  # available region color
    ):
        """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model (str): Path to the YOLOv8 model.
            json_file (str): file that have all parking slot points data
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
        """
        # Model initialization
        from ultralytics import YOLO

        self.model = YOLO(model)

        # Load JSON data
        with open(json_file) as f:
            self.json_data = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information

        self.occ = occupied_region_color
        self.arc = available_region_color

        self.env_check = check_imshow(warn=True)  # check if environment supports imshow

    def process_data(self, im0):
        """
        Process the model data for parking lot management.

        Args:
            im0 (ndarray): inference image
        """
        results = self.model.track(im0, persist=True, show=False)  # object tracking

        es, fs = len(self.json_data), 0  # empty slots, filled slots
        annotator = Annotator(im0)  # init annotator

        # extract tracks data
        if results[0].boxes.id is None:
            self.display_frames(im0)
            return im0

        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for region in self.json_data:
            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            for box, cls in zip(boxes, clss):
                xc = int((box[0] + box[2]) / 2)
                yc = int((box[1] + box[3]) / 2)
                annotator.display_objects_labels(
                    im0, self.model.names[int(cls)], (104, 31, 17), (255, 255, 255), xc, yc, 10
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

        self.display_frames(im0)
        return im0

    def display_frames(self, im0):
        """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
        if self.env_check:
            cv2.imshow("Ultralytics Parking Manager", im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
