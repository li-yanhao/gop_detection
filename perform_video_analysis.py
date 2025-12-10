import sys
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.widgets import PolygonSelector
# from matplotlib.path import Path

import subprocess
import os
from src.residual_info import get_sorted_residual_info_list
from src.acontrario import AContrarioAnalyser
import glob
from src.util import decode_frames, decode_residuals, convert_to_h264
import argparse


OUTPUT_ROOT = "tmp"


def perform_video_analysis(video_path:str, 
                           d:int, space:str, epsilon:float,
                           roi_mask=None,
                           max_num:int=-1):
    """ Perform video analysis on the given video file.
    :param video_path: Path to the input video file.
    :param d: Parameter d for A Contrario analysis.
    :param space: Color space to use ('Y', 'U', 'V', 'RGB').
    :param epsilon: Epsilon parameter for A Contrario analysis.
    :param roi_mask: Optional mask (NumPy array) defining the region of interest.
    :param max_num: Maximum number of frames to process (-1 for all frames).
    """

    h264_fname = os.path.join(OUTPUT_ROOT, os.path.basename(video_path).split('.')[0] + ".264") 

    # convert the video to h264
    ret = convert_to_h264(video_path, out_fname=h264_fname)

    if not ret:
        print("Conversion to H264 failed!")
        return None, None
    
    # decode frames
    ret, frame_folder = decode_frames(h264_fname, OUTPUT_ROOT)

    if not ret:
        print("Decoding frames failed!")
        return None, None
    
    ret, residual_folder = decode_residuals(h264_fname, OUTPUT_ROOT)

    if not ret:
        print("Decoding residuals failed!")
        return None, None

    # 2. A Contrario analysis
    analyzer = AContrarioAnalyser(epsilon=epsilon, d=d, start_at_0=False,
                              space=space, max_num=max_num)

    residual_info_list = get_sorted_residual_info_list(folder=residual_folder, space=space)

    analyzer.load_frame_info(frame_info_list=residual_info_list, space=space, img_fnames=None, mask_maker=None)

    analyzer.preprocess()

    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"Estimated primary GOP = {GOP_aCont}")
    print(f"NFA = {NFA_aCont}")
    print()

    frame_fname_list = glob.glob(os.path.join(frame_folder, "*.png"))
    frame_fname_list.sort()

    residual_fname_list = [ri.fname for ri in residual_info_list]

    assert len(frame_fname_list) == len(residual_fname_list), "Number of frames ({}) and residuals ({}) do not match!".format(len(frame_fname_list), len(residual_fname_list))


    return frame_fname_list, residual_fname_list


# --------------------------------------------------------

def select_polygon_roi(video_path):
    """
    Opens a Matplotlib window to select a polygon ROI.
    Returns a mask (NumPy array) or None.
    """
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for ROI selection.")
        return None

    ret, initial_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read first frame.")
        return None

    # Convert the frame to RGB for Matplotlib
    initial_frame_rgb = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)

    # Global variables for polygon selection
    points = []

    def onselect(verts):
        nonlocal points
        points = verts
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(initial_frame_rgb)
    ax.set_title("Select Polygon ROI - Close the window when done")

    ax.axis('off')  # Turn off the x and y axis
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Eliminate blank borders
    polygon_selector = PolygonSelector(ax, onselect, useblit=True)

    plt.show()

    if len(points) < 3:
        print("Not enough points for a polygon. No mask created.")
        return None

    # Create the polygon mask
    mask = np.zeros(initial_frame.shape[:2], dtype=np.uint8)
    poly = np.array([[(int(x), int(y)) for x, y in points]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)

    return mask
    

class ResultViewer:
    def __init__(self, master, frame_fname_list, residual_fname_list, roi_used):
        ''' A Tkinter-based GUI to view video analysis results.
        param master: the Tkinter root window
        param frame_fname_list: list of frame image filenames, already sorted in the display order
        param residual_fname_list: list of residual image filenames, already sorted in the display order
        param roi_used: whether ROI is used in the analysis
        '''

        self.master = master
        master.title("Video Analysis Results")
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.frame_fname_list = frame_fname_list
        self.residual_fname_list = residual_fname_list
        self.roi_used = roi_used
        self.current_frame_index = 0
        self.color_space_mode = 'RGB'  # Y, U, V, or RGB (original)

        # 1. Setup Frame for Controls and Statistics
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Statistics Display
        # stats_label_text = "\n".join([f"{k}: {v}" for k, v in self.statistics.items()])
        stats_label_text = f"ROI Used: {'Yes' if self.roi_used else 'No'}"
        # self.stats_label = tk.Label(control_frame, text=stats_label_text, justify=tk.LEFT)
        # self.stats_label.pack(side=tk.LEFT, padx=10)

        # Color Space Radio Buttons
        self.color_mode_var = tk.StringVar(value=self.color_space_mode)
        modes = ['RGB', 'Y', 'U', 'V']
        tk.Label(control_frame, text="Color View:").pack(side=tk.LEFT, padx=(20, 5))
        for mode in modes:
            tk.Radiobutton(control_frame, text=mode, variable=self.color_mode_var, value=mode, 
                           command=self.update_frame).pack(side=tk.LEFT)

        # 2. Setup Frame for Image and Navigation
        view_frame = tk.Frame(master)
        view_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Image Canvas/Label
        self.image_label = tk.Label(view_frame)
        self.image_label.pack(side=tk.LEFT)

        # Frame Navigation Buttons
        nav_frame = tk.Frame(view_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        tk.Button(nav_frame, text="<- Previous Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=10)
        self.master.bind("<Left>", lambda event: self.prev_frame())
        self.frame_index_label = tk.Label(nav_frame, text=f"Frame 1 / {len(self.frame_fname_list)}")
        self.frame_index_label.pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Next Frame ->", command=self.next_frame).pack(side=tk.RIGHT, padx=10)
        self.master.bind("<Right>", lambda event: self.next_frame())

        self.update_frame()

        
    def prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_frame_index < len(self.frame_fname_list) - 1:
            self.current_frame_index += 1
            self.update_frame()

    def update_frame(self):
        self.color_space_mode = self.color_mode_var.get()

        frame_fname = self.frame_fname_list[self.current_frame_index]
        # load the frame image in RGB
        img_rgb = cv2.cvtColor(cv2.imread(frame_fname, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # frame_bgr = self.frames[self.current_frame_index]
        
        # Apply color space conversion
        # processed_frame = self._convert_color_space(frame_bgr, self.color_space_mode)
        
        # Convert OpenCV image (NumPy array) to PIL Image
        # img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # resize the image to the half of the display size
        img_pil = img_pil.resize((img_pil.width // 2, img_pil.height // 2), Image.Resampling.LANCZOS)
        
        # print("img_pil:", np.array(img_pil))
        # Convert PIL Image to PhotoImage for Tkinter
        self.img_tk = ImageTk.PhotoImage(master=self.master, image=img_pil)
        
        # Update Label and Index Text
        self.image_label.config(image=self.img_tk)
        # self.image_label.image = self.img_tk  # Prevent garbage collection
        self.frame_index_label.config(text=f"Frame {self.current_frame_index + 1} / {len(self.frame_fname_list)}")

    def _convert_color_space(self, frame_bgr, mode):
        if mode == 'RGB':
            # Analysis function returns BGR (default for OpenCV), display is updated to RGB in update_frame
            return frame_bgr 
        
        # Convert BGR to YUV
        frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(frame_yuv)
        
        # Create a new 3-channel image where the selected channel is kept
        if mode == 'Y':
            # Y channel (Luminance) on all 3 channels
            return cv2.merge([y, y, y])
        elif mode == 'U':
            # U channel (Chroma)
            # Merge U with 128 (gray) for Y and V to visualize U component intensity
            gray_y = np.full_like(y, 128)
            gray_v = np.full_like(v, 128)
            return cv2.cvtColor(cv2.merge([gray_y, u, gray_v]), cv2.COLOR_YUV2BGR)
        elif mode == 'V':
            # V channel (Chroma)
            # Merge V with 128 (gray) for Y and U to visualize V component intensity
            gray_y = np.full_like(y, 128)
            gray_u = np.full_like(u, 128)
            return cv2.cvtColor(cv2.merge([gray_y, gray_u, v]), cv2.COLOR_YUV2BGR)
        
        return frame_bgr # Default fallback

    def on_closing(self):
        # Optional: Clean up resources
        self.master.destroy()
        self.master.quit()
        print("self.master destroyed.")


def main():
    parser = argparse.ArgumentParser(description="Perform video analysis with optional ROI selection.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--d", type=int, help="number of neighbors to validate a peak residual (default: 3)", default=3)
    parser.add_argument("--space", type=str, help="color space used for detection (default: Y)", default="Y")
    parser.add_argument("--epsilon", type=float, help="threshold for the Number of False Alarms (NFA), (default: 1.0)", default=1.0)
    parser.add_argument("--max_num", type=float, help="maximum number of frames to process (default: -1)", default=-1)


    args = parser.parse_args()


    # --- Step 2: Ask for ROI and select polygon ---
    root = tk.Tk()
    root.withdraw() # Hide the main root window for now

    roi_prompt = tk.Toplevel(root)
    roi_prompt.title("Region of Interest")
    tk.Label(roi_prompt, text="Do you want to define a Region of Interest (ROI)?").pack(padx=20, pady=10)
    
    roi_mask = None
    roi_used = False

    # This prevents the program from exiting before the ROI prompt is handled
    def start_analysis_and_viewer():
        # --- Step 3: Run Core Analysis ---
        frame_fname_list, residual_fname_list = perform_video_analysis(args.video_path, d=args.d, space=args.space, epsilon=args.epsilon, max_num=args.max_num, roi_mask=roi_mask)
        

        # --- Step 4: Launch Visualization GUI ---
        viewer_root = tk.Tk()
        app = ResultViewer(viewer_root, frame_fname_list, residual_fname_list, roi_used)
        viewer_root.mainloop()

        print("end of start_analysis_and_viewer()")


    def on_yes():
        print("User chose to define an ROI.")

        nonlocal roi_mask, roi_used
        roi_prompt.destroy()
        # Launch OpenCV-based polygon selector
        roi_mask = select_polygon_roi(video_path)
        roi_used = roi_mask is not None
        start_analysis_and_viewer()

        root.destroy()

    def on_no():
        print("No ROI selected, analyzing full frame.")

        nonlocal roi_used
        roi_prompt.destroy()
        roi_used = False
        start_analysis_and_viewer()

        root.destroy()


    tk.Button(roi_prompt, text="Yes (Define Polygon)", command=on_yes).pack(side=tk.LEFT, padx=10, pady=10)
    tk.Button(roi_prompt, text="No (Analyze Full Frame)", command=on_no).pack(side=tk.RIGHT, padx=10, pady=10)
    
    # Wait for ROI prompt to be handled
    root.mainloop()


if __name__ == "__main__":
    # Ensure you have the necessary libraries installed:
    # pip install opencv-python numpy pillow
    print("Starting application...")
    # NOTE: To run this, you need to replace `your_script_name.py` with the actual file name
    # and provide a video path as a command line argument (or use the file dialog).
    # E.g.: python video_analysis_app.py my_video.mp4
    main() # Commented out for a clean execution in this environment
