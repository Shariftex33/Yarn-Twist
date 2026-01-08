import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import math

class YarnAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Yarn Analysis Tool (TPI, Diameter, Twist Angle)")
        self.root.geometry("1600x950") # Larger window

        # --- Image Data ---
        self.image_path = None
        self.original_image_pil = None # Store PIL image for Tkinter display
        self.original_image_cv = None # Store OpenCV image for processing (BGR)
        self.display_image_tk = None # PhotoImage for the canvas

        # --- Calibration Data ---
        self.pixels_per_mm = None
        self.zoom_magnification = tk.DoubleVar(value=1.0)

        self.calibration_line_coords = None # [(x1,y1), (x2,y2)]
        self.calibration_actual_length_mm = tk.DoubleVar(value=0.0)

        # Flag to prevent recursion when programmatically setting Tkinter variables
        self._updating_vars = False

        # --- Measurement Data ---
        self.diameter_line_coords = None # [(x1,y1), (x2,y2)]
        self.manual_twist_line_coords = None # [(x1,y1), (x2,y2)]
        self.calculated_diameter_mm = None
        self.manual_twist_angle_deg = None
        self.auto_twist_angles_deg = [] # List of angles from Hough

        # --- Application State / Mode ---
        self.current_mode = tk.StringVar(value="none") # "none", "calibration", "diameter", "manual_angle"
        self.line_start_point = None # Used for drawing operations

        # --- Initialize all potentially accessed attributes here ---
        self.twist_direction_var = tk.StringVar(value="backslash") # Default for auto analysis
        self.detected_lines_image = None # Image with green/red Hough lines
        self.processed_edges = None # Stores the Canny edge detection result


        # --- GUI Setup ---
        self.create_menus()
        self.create_control_frame()
        self.create_display_frame()

        self.root.bind("<Configure>", self.on_window_resize) # Bind resize event

    def create_menus(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        mode_menu = tk.Menu(menubar, tearoff=0)
        mode_menu.add_radiobutton(label="Select Mode: None", variable=self.current_mode, value="none", command=self.set_mode)
        mode_menu.add_radiobutton(label="Calibrate (Draw Known Length Line)", variable=self.current_mode, value="calibration", command=self.set_mode)
        mode_menu.add_radiobutton(label="Measure Diameter", variable=self.current_mode, value="diameter", command=self.set_mode)
        mode_menu.add_radiobutton(label="Measure Twist Angle (Manual)", variable=self.current_mode, value="manual_angle", command=self.set_mode)
        menubar.add_cascade(label="Tools", menu=mode_menu)

        auto_analysis_menu = tk.Menu(menubar, tearoff=0)
        auto_analysis_menu.add_radiobutton(label="Automatic Twist Angle (Backslash \\)", variable=self.twist_direction_var, value="backslash", command=self.update_processing)
        auto_analysis_menu.add_radiobutton(label="Automatic Twist Angle (Forwardslash /)", variable=self.twist_direction_var, value="forwardslash", command=self.update_processing)
        menubar.add_cascade(label="Automatic Analysis", menu=auto_analysis_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def create_control_frame(self):
        control_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # --- Mode & Instructions ---
        self.mode_label = tk.Label(control_frame, text="Current Mode: None - Load an Image First", font=("Arial", 12, "bold"), fg="blue")
        self.mode_label.pack(side=tk.TOP, fill=tk.X, pady=5)

        # --- Calibration & Zoom Controls ---
        calib_frame = tk.LabelFrame(control_frame, text="Calibration & Zoom")
        calib_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)

        tk.Label(calib_frame, text="Actual Length (mm):").pack(anchor=tk.W, padx=5, pady=2)
        self.cal_len_entry = tk.Entry(calib_frame, textvariable=self.calibration_actual_length_mm)
        self.cal_len_entry.pack(anchor=tk.W, padx=5, pady=2)
        self.cal_len_entry.bind("<KeyRelease>", self._on_cal_len_entry_change)

        tk.Label(calib_frame, text="Magnification/Zoom:").pack(anchor=tk.W, padx=5, pady=2)
        self.zoom_entry = tk.Entry(calib_frame, textvariable=self.zoom_magnification)
        self.zoom_entry.pack(anchor=tk.W, padx=5, pady=2)
        self.zoom_entry.bind("<KeyRelease>", self._on_zoom_entry_change)


        self.calib_result_label = tk.Label(calib_frame, text="Pixels/mm: N/A")
        self.calib_result_label.pack(anchor=tk.W, padx=5, pady=2)


        # --- Image Processing (Auto Twist) Controls - Now with two columns ---
        img_proc_frame = tk.LabelFrame(control_frame, text="Image Processing (Auto Twist)")
        img_proc_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)

        # Canny Column
        canny_frame = tk.LabelFrame(img_proc_frame, text="Canny Edge Detection")
        canny_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        tk.Label(canny_frame, text="Threshold 1:").pack(side=tk.TOP, padx=5, pady=2)
        self.threshold1_scale = tk.Scale(canny_frame, from_=0, to_=255, orient=tk.HORIZONTAL, length=150, command=self.update_processing)
        self.threshold1_scale.set(100) # Default value
        self.threshold1_scale.pack(side=tk.TOP, padx=5, pady=2)

        tk.Label(canny_frame, text="Threshold 2:").pack(side=tk.TOP, padx=5, pady=2)
        self.threshold2_scale = tk.Scale(canny_frame, from_=0, to_=255, orient=tk.HORIZONTAL, length=150, command=self.update_processing)
        self.threshold2_scale.set(200) # Default value
        self.threshold2_scale.pack(side=tk.TOP, padx=5, pady=2)

        # Hough Column
        hough_frame = tk.LabelFrame(img_proc_frame, text="Hough Line Transform")
        hough_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        tk.Label(hough_frame, text="Hough Threshold:").pack(side=tk.TOP, padx=5, pady=2)
        self.hough_threshold_scale = tk.Scale(hough_frame, from_=10, to_=500, orient=tk.HORIZONTAL, length=150, command=self.update_processing)
        self.hough_threshold_scale.set(50) # Default
        self.hough_threshold_scale.pack(side=tk.TOP, padx=5, pady=2)

        tk.Label(hough_frame, text="Min Line Length:").pack(side=tk.TOP, padx=5, pady=2)
        self.min_line_length_scale = tk.Scale(hough_frame, from_=10, to_=300, orient=tk.HORIZONTAL, length=150, command=self.update_processing)
        self.min_line_length_scale.set(30) # Default
        self.min_line_length_scale.pack(side=tk.TOP, padx=5, pady=2)

        tk.Label(hough_frame, text="Max Line Gap:").pack(side=tk.TOP, padx=5, pady=2)
        self.max_line_gap_scale = tk.Scale(hough_frame, from_=0, to_=100, orient=tk.HORIZONTAL, length=150, command=self.update_processing)
        self.max_line_gap_scale.set(10) # Default
        self.max_line_gap_scale.pack(side=tk.TOP, padx=5, pady=2)

        # Make columns in img_proc_frame expand
        img_proc_frame.grid_columnconfigure(0, weight=1)
        img_proc_frame.grid_columnconfigure(1, weight=1)


        # --- Results Display ---
        results_frame = tk.LabelFrame(control_frame, text="Measurement Results")
        results_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Diameter Results
        tk.Label(results_frame, text="Diameter:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.diameter_px_label = tk.Label(results_frame, text="Pixels: N/A")
        self.diameter_px_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.diameter_mm_label = tk.Label(results_frame, text="Actual (mm): N/A")
        self.diameter_mm_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        # Manual Twist Angle Results
        tk.Label(results_frame, text="Manual Twist Angle:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.manual_angle_label = tk.Label(results_frame, text="Angle: N/A")
        self.manual_angle_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.manual_tpi_label = tk.Label(results_frame, text="Manual TPI: N/A", font=("Arial", 10, "bold"))
        self.manual_tpi_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # Automatic Twist Angle Results
        tk.Label(results_frame, text="Auto Twist Angle:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.auto_angle_label = tk.Label(results_frame, text="Angle: N/A")
        self.auto_angle_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.auto_tpi_label = tk.Label(results_frame, text="Auto TPI: N/A", font=("Arial", 10, "bold"))
        self.auto_tpi_label.grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.auto_lines_count_label = tk.Label(results_frame, text="Lines Used: N/A")
        self.auto_lines_count_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)


        # Configure columns for results_frame to expand proportionally
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_columnconfigure(2, weight=1)


    def create_display_frame(self):
        self.image_canvas = tk.Canvas(self.root, bg="gray", cursor="cross")
        self.image_canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind mouse events for drawing lines
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def on_window_resize(self, event):
        # Redraw image on canvas when window is resized
        if self.original_image_pil:
            self.display_image_on_canvas()

    def set_mode(self):
        mode = self.current_mode.get()
        if not self.original_image_cv and mode != "none":
            messagebox.showwarning("No Image Loaded", "Please load an Image first.")
            self.current_mode.set("none") # Reset mode if no image
            self.mode_label.config(text="Current Mode: None - Load an Image First")
            return

        self.mode_label.config(text=f"Current Mode: {mode.replace('_', ' ').title()}")
        self.display_image_on_canvas() # Redraw to clear any temporary lines or update cursor

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
        )
        if self.image_path:
            try:
                self.original_image_pil = Image.open(self.image_path)
                self.original_image_cv = cv2.imread(self.image_path)
                if self.original_image_cv is None:
                    messagebox.showerror("Error", f"Could not load image from {self.image_path}")
                    self.clear_data()
                    return

                self.current_mode.set("none") # Reset mode on new image
                self.mode_label.config(text="Current Mode: None")
                self.clear_measurements() # Clear previous measurements, including detected_lines_image and processed_edges
                self.display_image_on_canvas() # Initial display of raw image
                self.update_processing(None) # Run auto twist detection and update display with lines
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading image: {e}")
                self.clear_data()

    def display_image_on_canvas(self):
        if self.original_image_pil is None:
            self.image_canvas.delete("all")
            return

        final_display_cv_img = self.original_image_cv.copy()

        if self.detected_lines_image is not None:
            final_display_cv_img = cv2.addWeighted(final_display_cv_img, 1.0, self.detected_lines_image, 0.7, 0)

        display_pil_img = Image.fromarray(cv2.cvtColor(final_display_cv_img, cv2.COLOR_BGR2RGB))

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width = self.root.winfo_width() - 20
            # A more robust way to get control frame height (assuming it's the first child)
            try:
                control_frame_height = self.root.winfo_children()[0].winfo_height()
            except IndexError:
                control_frame_height = 100 # Fallback
            canvas_height = self.root.winfo_height() - control_frame_height - 40

        img_width, img_height = display_pil_img.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        self.display_pil_img_resized = display_pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.display_image_tk = ImageTk.PhotoImage(image=self.display_pil_img_resized)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=self.display_image_tk)
        self.image_canvas.image = self.display_image_tk # Keep a reference!

        self._redraw_persistent_canvas_lines()


    def _redraw_persistent_canvas_lines(self):
        """Redraws the manually drawn lines (calibration, diameter, manual twist) directly on the canvas."""
        if self.original_image_pil is None:
            return

        self.image_canvas.delete("manual_line")

        if self.calibration_line_coords:
            coords_flat = self.calibration_line_coords[0] + self.calibration_line_coords[1]
            canvas_coords = self.image_to_canvas_coords_tuple(coords_flat)
            self.image_canvas.create_line(canvas_coords, fill="blue", width=3, tags="manual_line")

        if self.diameter_line_coords:
            coords_flat = self.diameter_line_coords[0] + self.diameter_line_coords[1]
            canvas_coords = self.image_to_canvas_coords_tuple(coords_flat)
            self.image_canvas.create_line(canvas_coords, fill="red", width=3, tags="manual_line")

        if self.manual_twist_line_coords:
            coords_flat = self.manual_twist_line_coords[0] + self.manual_twist_line_coords[1]
            canvas_coords = self.image_to_canvas_coords_tuple(coords_flat)
            self.image_canvas.create_line(canvas_coords, fill="magenta", width=3, tags="manual_line")


    def on_mouse_press(self, event):
        if self.original_image_cv is None:
            messagebox.showwarning("No Image", "Please load an Image first.")
            return

        mode = self.current_mode.get()
        if mode in ["calibration", "diameter", "manual_angle"]:
            self.line_start_point = self.canvas_to_image_coords(event.x, event.y)
            self.image_canvas.delete("temp_line") # Clear any previous temp line

    def on_mouse_drag(self, event):
        mode = self.current_mode.get()
        if self.line_start_point and mode in ["calibration", "diameter", "manual_angle"]:
            current_point = self.canvas_to_image_coords(event.x, event.y)
            temp_coords_image = (self.line_start_point[0], self.line_start_point[1],
                                 current_point[0], current_point[1])
            temp_coords_canvas = self.image_to_canvas_coords_tuple(temp_coords_image)

            self.image_canvas.delete("temp_line") # Clear previous temporary line
            self.image_canvas.create_line(temp_coords_canvas, fill="yellow", width=2, tags="temp_line")

    def on_mouse_release(self, event):
        mode = self.current_mode.get()
        if self.line_start_point and mode in ["calibration", "diameter", "manual_angle"]:
            end_point = self.canvas_to_image_coords(event.x, event.y)
            line_coords = ((self.line_start_point[0], self.line_start_point[1]), (end_point[0], end_point[1]))

            if mode == "calibration":
                self.calibration_line_coords = line_coords
                self.calculate_calibration()
            elif mode == "diameter":
                self.diameter_line_coords = line_coords
                self.calculate_diameter()
            elif mode == "manual_angle":
                self.manual_twist_line_coords = line_coords
                self.calculate_manual_angle()

            self.line_start_point = None
            self.image_canvas.delete("temp_line") # Clear temp line
            self.display_image_on_canvas() # Redraw to finalize line

    def canvas_to_image_coords(self, canvas_x, canvas_y):
        if self.original_image_pil is None or self.image_canvas.winfo_width() <= 10 or self.image_canvas.winfo_height() <= 10:
            return 0, 0

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        original_width, original_height = self.original_image_pil.size

        ratio = min(canvas_width / original_width, canvas_height / original_height)

        displayed_width = original_width * ratio
        displayed_height = original_height * ratio

        x_offset = (canvas_width - displayed_width) / 2
        y_offset = (canvas_height - displayed_height) / 2

        img_x_displayed = canvas_x - x_offset
        img_y_displayed = canvas_y - y_offset

        original_x = int(img_x_displayed / ratio)
        original_y = int(img_y_displayed / ratio)

        original_x = max(0, min(original_x, original_width - 1))
        original_y = max(0, min(original_y, original_height - 1))

        return original_x, original_y

    def image_to_canvas_coords_tuple(self, image_coords_tuple):
        img_x1, img_y1, img_x2, img_y2 = image_coords_tuple

        if self.original_image_pil is None or self.image_canvas.winfo_width() <= 10 or self.image_canvas.winfo_height() <= 10:
            return (0,0,0,0)

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        original_width, original_height = self.original_image_pil.size

        ratio = min(canvas_width / original_width, canvas_height / original_height)
        
        x_offset = (canvas_width - original_width * ratio) / 2
        y_offset = (canvas_height - original_height * ratio) / 2

        canvas_x1 = int(img_x1 * ratio + x_offset)
        canvas_y1 = int(img_y1 * ratio + y_offset)
        canvas_x2 = int(img_x2 * ratio + x_offset)
        canvas_y2 = int(img_y2 * ratio + y_offset)

        return (canvas_x1, canvas_y1, canvas_x2, canvas_y2)

    def _on_cal_len_entry_change(self, event=None):
        if self._updating_vars:
            return
        self._updating_vars = True
        try:
            val_str = self.cal_len_entry.get()
            if val_str.strip() == "":
                self.pixels_per_mm = None
                self.calib_result_label.config(text="Pixels/mm: N/A")
                self.update_all_measurements()
                return
            
            val = float(val_str)
            self.calibration_actual_length_mm.set(val)

            if val > 0 and self.calibration_line_coords:
                self.calculate_calibration() # Calls update_all_measurements inside
            elif val > 0 and not self.calibration_line_coords:
                self.calib_result_label.config(text="Pixels/mm: Draw line first")
            else:
                self.pixels_per_mm = None
                self.calib_result_label.config(text="Pixels/mm: Invalid input")
                self.update_all_measurements()
        except ValueError:
            self.pixels_per_mm = None
            self.calib_result_label.config(text="Pixels/mm: Invalid input")
            self.update_all_measurements()
        finally:
            self._updating_vars = False

    def _on_zoom_entry_change(self, event=None):
        if self._updating_vars:
            return
        self._updating_vars = True
        try:
            val_str = self.zoom_entry.get()
            if val_str.strip() == "":
                self.zoom_magnification.set(1.0)
                self.update_all_measurements()
                return

            val = float(val_str)
            self.zoom_magnification.set(val)

            if val > 0:
                self.update_all_measurements()
            else:
                messagebox.showwarning("Input Error", "Magnification must be positive.")
                self.zoom_magnification.set(1.0)
                self.update_all_measurements()
        except ValueError:
            pass
        finally:
            self._updating_vars = False

    def calculate_calibration(self):
        if not self.calibration_line_coords:
            self.pixels_per_mm = None
            self.calib_result_label.config(text="Pixels/mm: N/A")
            self.update_all_measurements()
            return
        
        try:
            actual_length = self.calibration_actual_length_mm.get()
            if actual_length <= 0:
                self.pixels_per_mm = None
                self.calib_result_label.config(text="Pixels/mm: Invalid length")
                self.update_all_measurements()
                return

            p1, p2 = self.calibration_line_coords
            pixel_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if pixel_length > 0:
                self.pixels_per_mm = pixel_length / actual_length
                self.calib_result_label.config(text=f"Pixels/mm: {self.pixels_per_mm:.2f}")
            else:
                self.pixels_per_mm = None
                self.calib_result_label.config(text="Pixels/mm: Invalid line length")
            self.update_all_measurements() # Call here to trigger updates for TPI.
        except Exception:
            self.pixels_per_mm = None
            self.calib_result_label.config(text="Pixels/mm: Error")
            self.update_all_measurements()


    def calculate_diameter(self):
        if not self.diameter_line_coords:
            self.diameter_px_label.config(text="Pixels: N/A")
            self.diameter_mm_label.config(text="Actual (mm): N/A")
            self.calculated_diameter_mm = None
            self.update_all_measurements() # Call here to update TPI
            return

        if self.pixels_per_mm is None:
            self.diameter_mm_label.config(text="Actual (mm): Calibrate First")
            self.calculated_diameter_mm = None
            self.update_all_measurements() # Call here to update TPI
            return

        p1, p2 = self.diameter_line_coords
        pixel_diameter = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        self.diameter_px_label.config(text=f"Pixels: {pixel_diameter:.2f}")
        
        magnification_val = self.zoom_magnification.get()
        if magnification_val <= 0:
            self.diameter_mm_label.config(text="Actual (mm): Invalid Mag")
            self.calculated_diameter_mm = None
            self.update_all_measurements() # Call here to update TPI
            return

        actual_diameter_mm = (pixel_diameter / self.pixels_per_mm) / magnification_val
        self.calculated_diameter_mm = actual_diameter_mm
        self.diameter_mm_label.config(text=f"Actual (mm): {actual_diameter_mm:.3f}")
        self.update_all_measurements() # Call here to update TPI

    def calculate_manual_angle(self):
        if not self.manual_twist_line_coords:
            self.manual_angle_label.config(text="Angle: N/A")
            self.manual_tpi_label.config(text="Manual TPI: N/A")
            self.manual_twist_angle_deg = None
            self.update_all_measurements() # Call here to update TPI
            return

        p1, p2 = self.manual_twist_line_coords
        
        delta_x = p2[0] - p1[0]
        delta_y = -(p2[1] - p1[1])

        if delta_x == 0:
            self.manual_twist_angle_deg = 90.0 if delta_y >= 0 else -90.0
        else:
            self.manual_twist_angle_deg = math.degrees(math.atan2(delta_y, delta_x))

        display_angle = abs(self.manual_twist_angle_deg)
        if display_angle > 90:
            display_angle = 180 - display_angle 

        self.manual_angle_label.config(text=f"Angle: {display_angle:.2f} degrees")
        self.update_all_measurements() # Call here to update TPI

    def update_processing(self, val):
        if self.original_image_cv is None:
            self.auto_angle_label.config(text="Angle: N/A")
            self.auto_tpi_label.config(text="Auto TPI: N/A")
            self.auto_lines_count_label.config(text="Lines Used: N/A")
            self.detected_lines_image = None
            self.processed_edges = None
            self.display_image_on_canvas()
            return

        try:
            gray = cv2.cvtColor(self.original_image_cv, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            t1 = self.threshold1_scale.get()
            t2 = self.threshold2_scale.get()
            self.processed_edges = cv2.Canny(blurred, t1, t2)

            self.detected_lines_image = np.zeros_like(self.original_image_cv)

            hough_threshold = self.hough_threshold_scale.get()
            min_line_length = self.min_line_length_scale.get()
            max_line_gap = self.max_line_gap_scale.get()

            lines = cv2.HoughLinesP(self.processed_edges, 1, np.pi / 180,
                                    hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

            self.auto_twist_angles_deg = []
            num_lines = 0
            current_direction = self.twist_direction_var.get()
            display_direction_char = ""

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    angle_rad = math.atan2(-(y2 - y1), (x2 - x1))
                    angle_deg = math.degrees(angle_rad)

                    if angle_deg > 90:
                        angle_deg -= 180
                    elif angle_deg <= -90:
                        angle_deg += 180

                    line_accepted = False

                    if current_direction == "backslash":
                        if -85 <= angle_deg <= -5:
                            line_accepted = True
                        display_direction_char = "\\"
                    elif current_direction == "forwardslash":
                        if 5 <= angle_deg <= 85:
                            line_accepted = True
                        display_direction_char = "/"

                    if line_accepted:
                        cv2.line(self.detected_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.auto_twist_angles_deg.append(abs(angle_deg))
                        num_lines += 1
                    else:
                        cv2.line(self.detected_lines_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

            if self.auto_twist_angles_deg:
                average_angle = np.mean(self.auto_twist_angles_deg)
                self.auto_angle_label.config(text=f"Angle: {average_angle:.2f} degrees")
            else:
                self.auto_angle_label.config(text=f"Angle: No relevant lines detected")
            self.auto_lines_count_label.config(text=f"Lines Used ({display_direction_char}): {num_lines}")

            self.display_image_on_canvas()
            self.update_all_measurements()

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during auto line detection: {e}")
            self.auto_twist_angles_deg = []
            self.detected_lines_image = None
            self.processed_edges = None
            self.display_image_on_canvas()
            self.auto_angle_label.config(text="Angle: N/A")
            self.auto_lines_count_label.config(text="Lines Used: N/A")
            self.update_all_measurements()

    def update_all_measurements(self):
        """
        Recalculates all dependent measurements (Diameter, TPI)
        This is the central function to call when any input affecting calculations changes.
        """
        
        # Diameter calculation (already updates its labels and self.calculated_diameter_mm)
        # We call calculate_diameter here because magnification or calibration changes
        # should re-evaluate the diameter's actual length.
        # This function now calls self.calculate_diameter() itself, which was previously handled
        # by individual mouse release events or input changes.
        # It's important that calculate_diameter and calculate_manual_angle DO NOT call update_all_measurements
        # themselves, as this would lead to recursion.
        
        # Ensure diameter is recalculated if its dependencies change,
        # but avoid re-triggering this entire function from within calculate_diameter.
        # We've already ensured `calculate_diameter` and `calculate_manual_angle`
        # are called directly from mouse release or input changes, and they
        # *now* call `update_all_measurements` at their end.
        # So, no need to call them redundantly here again if the goal is only TPI updates.

        # The core of this function is to calculate TPIs based on current `self.calculated_diameter_mm`
        # and angle measurements.

        # Calculate Manual TPI
        if self.calculated_diameter_mm is not None and self.calculated_diameter_mm > 0 and self.manual_twist_angle_deg is not None:
            angle_rad_manual = math.radians(abs(self.manual_twist_angle_deg))
            if angle_rad_manual != 0 and self.calculated_diameter_mm > 0:
                manual_tpi = (25.4 * math.tan(angle_rad_manual)) / self.calculated_diameter_mm
                self.manual_tpi_label.config(text=f"Manual TPI: {manual_tpi:.2f}")
            else:
                self.manual_tpi_label.config(text="Manual TPI: Angle or Diameter invalid")
        else:
            self.manual_tpi_label.config(text="Manual TPI: N/A")

        # Calculate Automatic TPI
        if self.calculated_diameter_mm is not None and self.calculated_diameter_mm > 0 and self.auto_twist_angles_deg:
            average_auto_angle_deg = np.mean(self.auto_twist_angles_deg)
            angle_rad_auto = math.radians(average_auto_angle_deg)
            if angle_rad_auto != 0 and self.calculated_diameter_mm > 0:
                auto_tpi = (25.4 * math.tan(angle_rad_auto)) / self.calculated_diameter_mm
                self.auto_tpi_label.config(text=f"Auto TPI: {auto_tpi:.2f}")
            else:
                self.auto_tpi_label.config(text="Auto TPI: Angle or Diameter invalid")
        else:
            self.auto_tpi_label.config(text="Auto TPI: N/A")


    def clear_data(self):
        self.image_path = None
        self.original_image_pil = None
        self.original_image_cv = None
        self.detected_lines_image = None
        self.processed_edges = None
        self.image_canvas.delete("all")
        self.current_mode.set("none")
        self.mode_label.config(text="Current Mode: None - Load an Image First")
        self.clear_measurements()

    def clear_measurements(self):
        self._updating_vars = True
        try:
            self.pixels_per_mm = None
            self.zoom_magnification.set(1.0)
            self.calibration_actual_length_mm.set(0.0)

            self.calibration_line_coords = None
            self.diameter_line_coords = None
            self.manual_twist_line_coords = None
            self.manual_twist_angle_deg = None
            
            self.calculated_diameter_mm = None
            self.auto_twist_angles_deg = []
            self.detected_lines_image = None
            self.processed_edges = None

            self.calib_result_label.config(text="Pixels/mm: N/A")
            self.diameter_px_label.config(text="Pixels: N/A")
            self.diameter_mm_label.config(text="Actual (mm): N/A")
            self.manual_angle_label.config(text="Angle: N/A")
            self.manual_tpi_label.config(text="Manual TPI: N/A")
            self.auto_angle_label.config(text="Angle: N/A")
            self.auto_tpi_label.config(text="Auto TPI: N/A")
            self.auto_lines_count_label.config(text="Lines Used: N/A")
            self.display_image_on_canvas()
        finally:
            self._updating_vars = False

    def show_about(self):
        messagebox.showinfo("About Yarn Analyzer",
                            "Yarn Analysis Tool\n\n"
                            "Features:\n"
                            "- Image Calibration (Pixels/mm)\n"
                            "- Yarn Diameter Measurement\n"
                            "- Manual Twist Angle Measurement\n"
                            "- Automatic Twist Angle Detection (using Canny & Hough)\n"
                            "- Automatic & Manual TPI Calculation\n\n"
                            "Developed by Gemini AI")

if __name__ == "__main__":
    root = tk.Tk()
    app = YarnAnalyzerApp(root)
    root.mainloop()