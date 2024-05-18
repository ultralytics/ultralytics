import json
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator


class ParkingPtsSelection:
    def __init__(self, master):
        """
        Initializes the UI for selecting parking zone points in a tkinter window.

        Args:
            master (tk.Tk): The main tkinter window object.
        """
        check_requirements("tkinter")
        import tkinter as tk

        self.master = master
        master.title("Ultralytics Parking Zones Points Selector")

        # Disable window resizing
        master.resizable(False, False)

        # Setup canvas for image display
        self.canvas = tk.Canvas(master, bg="white")

        # Setup buttons
        button_frame = tk.Frame(master)
        button_frame.pack(side=tk.TOP)

        tk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=0)
        tk.Button(button_frame, text="Remove Last BBox", command=self.remove_last_bounding_box).grid(row=0, column=1)
        tk.Button(button_frame, text="Save", command=self.save_to_json).grid(row=0, column=2)

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

    def upload_image(self):
        """Upload an image and resize it to fit canvas."""
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

        self.canvas = tk.Canvas(self.master, bg="white", width=canvas_width, height=canvas_height)
        resized_image = self.image.resize((canvas_width, canvas_height), Image.LANCZOS)
        self.canvas_image = ImageTk.PhotoImage(resized_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)

        self.canvas.pack(side=tk.BOTTOM)
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
            x1, y1 = box[i]
            x2, y2 = box[(i + 1) % 4]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Remove the last drawn bounding box from canvas."""
        if self.bounding_boxes:
            self.bounding_boxes.pop()  # Remove the last bounding box
            self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)  # Redraw the image

            # Redraw all bounding boxes
            for box in self.bounding_boxes:
                self.draw_bounding_box(box)

            messagebox.showinfo("Success", "Last bounding box removed.")
        else:
            messagebox.showwarning("Warning", "No bounding boxes to remove.")

    def save_to_json(self):
        """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
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
        with open("bounding_boxes.json", "w") as json_file:
            json.dump(bounding_boxes_data, json_file, indent=4)

        messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


class ParkingManagement:
    def __init__(
        self,
        model_path,
        txt_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        occupied_region_color=(0, 255, 0),
        available_region_color=(0, 0, 255),
        margin=10,
    ):
        """
        Initializes the parking management system with a YOLOv8 model and visualization settings.

        Args:
            model_path (str): Path to the YOLOv8 model.
            txt_color (tuple): RGB color tuple for text.
            bg_color (tuple): RGB color tuple for background.
            occupied_region_color (tuple): RGB color tuple for occupied regions.
            available_region_color (tuple): RGB color tuple for available regions.
            margin (int): Margin for text display.
        """
        # Model path and initialization
        self.model_path = model_path
        self.model = self.load_model()

        # Labels dictionary
        self.labels_dict = {"Occupancy": 0, "Available": 0}

        # Visualization details
        self.margin = margin
        self.bg_color = bg_color
        self.txt_color = txt_color
        self.occupied_region_color = occupied_region_color
        self.available_region_color = available_region_color

        self.window_name = "Ultralytics YOLOv8 Parking Management System"
        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

    def load_model(self):
        """Load the Ultralytics YOLOv8 model for inference and analytics."""
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        return self.model

    @staticmethod
    def parking_regions_extraction(json_file):
        """
        Extract parking regions from json file.

        Args:
            json_file (str): file that have all parking slot points
        """
        with open(json_file, "r") as json_file:
            return json.load(json_file)

    def process_data(self, json_data, im0, boxes, clss):
        """
        Process the model data for parking lot management.

        Args:
            json_data (str): json data for parking lot management
            im0 (ndarray): inference image
            boxes (list): bounding boxes data
            clss (list): bounding boxes classes list
        Returns:
            filled_slots (int): total slots that are filled in parking lot
            empty_slots (int): total slots that are available in parking lot
        """
        annotator = Annotator(im0)
        total_slots, filled_slots = len(json_data), 0
        empty_slots = total_slots

        for region in json_data:
            points = region["points"]
            points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            region_occupied = False

            for box, cls in zip(boxes, clss):
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                text = f"{self.model.names[int(cls)]}"

                annotator.display_objects_labels(
                    im0, text, self.txt_color, self.bg_color, x_center, y_center, self.margin
                )
                dist = cv2.pointPolygonTest(points_array, (x_center, y_center), False)
                if dist >= 0:
                    region_occupied = True
                    break

            color = self.occupied_region_color if region_occupied else self.available_region_color
            cv2.polylines(im0, [points_array], isClosed=True, color=color, thickness=2)
            if region_occupied:
                filled_slots += 1
                empty_slots -= 1

        self.labels_dict["Occupancy"] = filled_slots
        self.labels_dict["Available"] = empty_slots

        annotator.display_analytics(im0, self.labels_dict, self.txt_color, self.bg_color, self.margin)

    def display_frames(self, im0):
        """
        Display frame.

        Args:
            im0 (ndarray): inference image
        """
        if self.env_check:
            cv2.namedWindow(self.window_name)
            cv2.imshow(self.window_name, im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
