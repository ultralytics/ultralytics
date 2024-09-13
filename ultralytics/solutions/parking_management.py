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
        self.bounding_boxes = []
        self.current_box = []
        self.img_width = 0
        self.img_height = 0

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
        self.img_width, self.img_height = self.image.size

        # Calculate the aspect ratio and resize image
        aspect_ratio = self.img_width / self.img_height
        if aspect_ratio > 1:
            # Landscape orientation
            canvas_width = min(self.canvas_max_width, self.img_width)
            canvas_height = int(canvas_width / aspect_ratio)
        else:
            # Portrait orientation
            canvas_height = min(self.canvas_max_height, self.img_height)
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
        self.bounding_boxes = []
        self.current_box = []

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        self.current_box.append((event.x, event.y))
        x0, y0 = event.x - 3, event.y - 3
        x1, y1 = event.x + 3, event.y + 3
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")

        if len(self.current_box) == 4:
            self.bounding_boxes.append(self.current_box)
            self.draw_bounding_box(self.current_box)
            self.current_box = []

    def draw_bounding_box(self, box):
        """
        Draw bounding box on canvas.

        Args:
            box (list): Bounding box data
        """
        for i in range(4):
            [x1, y1, x2, y2] = box[i], box[(i + 1) % 4]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Remove the last drawn bounding box from canvas."""
        from tkinter import messagebox  # scope for multi-environment compatibility

        if self.bounding_boxes:
            self.bounding_boxes.pop()  # Remove the last bounding box
            self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # Redraw the image

            # Redraw all bounding boxes
            for box in self.bounding_boxes:
                self.draw_bounding_box(box)

            messagebox.showinfo("Success", "Last bounding box removed.")
        else:
            messagebox.showwarning("Warning", "No bounding boxes to remove.")

    def save_to_json(self):
        """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
        from tkinter import messagebox  # scope for multi-environment compatibility

        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        width_scaling_factor = self.img_width / canvas_width
        height_scaling_factor = self.img_height / canvas_height
        bounding_boxes_data = []
        for box in self.bounding_boxes:
            rescaled_box = []
            for x, y in box:
                rescaled_x = int(x * width_scaling_factor)
                rescaled_y = int(y * height_scaling_factor)
                rescaled_box.append((rescaled_x, rescaled_y))
            bounding_boxes_data.append({"points": rescaled_box})
        with open("bounding_boxes.json", "w") as f:
            json.dump(bounding_boxes_data, f, indent=4)

        messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement:
    """Manages parking occupancy and availability using YOLOv8 for real-time monitoring and visualization."""

    def __init__(
        self,
        model_path,
        occupied_region_color=(0, 255, 0),
        available_region_color=(0, 0, 255),
    ):
        """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model_path (str): Path to the YOLOv8 model.
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
        """
        # Model initialization
        from ultralytics import YOLO
        self.model = YOLO(model_path)

        # Labels dictionary
        self.labels_dict = {"Occupancy": 0, "Available": 0}

        # Visualization details
        self.occupied_region_color = occupied_region_color
        self.available_region_color = available_region_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

    @staticmethod
    def parking_regions_extraction(json_file):
        """
        Extract parking regions from json file.

        Args:
            json_file (str): file that have all parking slot points
        """
        with open(json_file) as f:
            return json.load(f)

    def process_data(self, json_data, im0, boxes, clss):
        """
        Process the model data for parking lot management.

        Args:
            json_data (str): json data for parking lot management
            im0 (ndarray): inference image
            boxes (list): bounding boxes data
            clss (list): bounding boxes classes list
        """
        annotator = Annotator(im0)
        es, fs = len(json_data), 0   # empty slots, filled slots
        for region in json_data:
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False     # occupied region initialization

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

            color = self.occupied_region_color if rg_occupied else self.available_region_color
            cv2.polylines(im0, [points_array], isClosed=True, color=color, thickness=2)
            if rg_occupied:
                fs += 1
                es -= 1

        self.labels_dict["Occupancy"] = fs
        self.labels_dict["Available"] = es

        annotator.display_analytics(im0, self.labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def display_frames(self, im0):
        """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
        if self.env_check:
            cv2.imshow("Ultralytics YOLOv8 Parking Management System", im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
