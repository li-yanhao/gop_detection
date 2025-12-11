import os
import glob
import argparse
import threading

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.widgets import PolygonSelector

from src.residual_info import get_sorted_residual_info_list
from src.acontrario import AContrarioAnalyser
from src.util import decode_frames, decode_residuals, convert_to_h264, pad_and_crop, get_rotation
from src.residual_info import read_one_residual


OUTPUT_ROOT = "tmp"
SAVE_VISUALIZED_PATH = None


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

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    h264_fname = os.path.join(OUTPUT_ROOT, os.path.basename(video_path).split('.')[0] + ".264") 

    # convert the video to h264
    ret = convert_to_h264(video_path, out_fname=h264_fname)

    if not ret:
        print("Conversion to H264 failed!")
        return None, None
    
    ret_decode_frames, ret_decode_residuals = False, False
    frame_folder, residual_folder = None, None
    def decode_frames_task():
        nonlocal ret_decode_frames, frame_folder
        print("Decoding frames ...\n")
        ret_decode_frames, frame_folder = decode_frames(h264_fname, OUTPUT_ROOT)
        

    def decode_residuals_task():
        nonlocal ret_decode_residuals, residual_folder
        print("Decoding residuals ...\n")
        ret_decode_residuals, residual_folder = decode_residuals(h264_fname, OUTPUT_ROOT)
        if not ret_decode_residuals:
            print("Decoding residuals failed!\n")

    # Create threads for decoding frames and residuals
    frame_thread = threading.Thread(target=decode_frames_task)
    residual_thread = threading.Thread(target=decode_residuals_task)

    # Start both threads
    frame_thread.start()
    residual_thread.start()

    # Wait for both threads to finish
    frame_thread.join()
    residual_thread.join()

    # Check if either task failed
    if not ret_decode_frames:
        print("Decoding frames failed!\n")
        return None, None

    if not ret_decode_residuals:
        print("Decoding residuals failed!\n")
        return None, None

    # 2. A Contrario analysis
    analyzer = AContrarioAnalyser(epsilon=epsilon, d=d, start_at_0=False,
                              space=space, max_num=max_num)

    residual_info_list = get_sorted_residual_info_list(folder=residual_folder, space=space)

    analyzer.load_frame_info(frame_info_list=residual_info_list, space=space, roi_mask=roi_mask)

    analyzer.preprocess()

    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"Estimated primary GOP = {GOP_aCont}")
    print(f"NFA = {NFA_aCont}")
    print()

    analyzer.visualize(SAVE_VISUALIZED_PATH)

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
    cv2.fillPoly(mask, poly, 1)

    return mask.astype(bool)
    

class ResultViewer:
    def __init__(self, master, frame_fname_list, residual_fname_list, rotation, roi_mask=None):
        ''' A Tkinter-based GUI to view video analysis results.
        param master: the Tkinter root window
        param frame_fname_list: list of frame image filenames, already sorted in the display order
        param residual_fname_list: list of residual image filenames, already sorted in the display order
        param rotation: rotation to correct frame orientation (in degrees, e.g., 0, 90, 180, 270)
        param roi_mask: binary mask (NumPy array) defining the region of interest used in analysis
        '''

        self.master = master
        master.title("Video Analysis Results")
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.frame_fname_list = frame_fname_list
        self.residual_fname_list = residual_fname_list
        self.current_frame_index = 0
        self.color_space_mode = 'Y'  # Y, U, V, or RGB (original)
        self.rotation = rotation

        self.roi_mask = roi_mask

        self.roi_residual_mask = roi_mask
        
        if roi_mask is not None:
            # adjust the size of roi_residual_mask to match the residual image size
            residual_img = read_one_residual(residual_fname_list[0])

            # rotate the residual_img if needed
            residual_img = self._correct_rotation(residual_img)

            residual_shape = residual_img.shape[:2]
            self.roi_residual_mask = pad_and_crop(roi_mask, residual_shape)


            print("ROI mask for frame:", self.roi_residual_mask.shape)
            print("ROI mask for residual:", self.roi_residual_mask.shape)

        # 1. Setup Frame for Controls and Statistics
        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 2. Setup Frame for Image and Navigation
        view_frame = tk.Frame(master)
        view_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Image Canvas/Label
        self.image_label = tk.Label(view_frame)
        self.image_label.pack(side=tk.LEFT)

        # Frame Navigation Buttons
        nav_frame = tk.Frame(view_frame)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Move navigation buttons to the bottom of the window
        nav_frame = tk.Frame(master)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        tk.Button(nav_frame, text="<- Previous Frame", command=self.prev_frame).pack(side=tk.LEFT, padx=10)
        self.master.bind("<Left>", lambda event: self.prev_frame())
        self.frame_index_label = tk.Label(nav_frame, text=f"Frame 1 / {len(self.frame_fname_list)}")
        self.frame_index_label.pack(side=tk.LEFT, padx=10)

        tk.Button(nav_frame, text="Next Frame ->", command=self.next_frame).pack(side=tk.RIGHT, padx=10)
        self.master.bind("<Right>", lambda event: self.next_frame())

        self.update_frame()

    def _correct_rotation(self, frame):
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
        
    def prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_frame_index < len(self.frame_fname_list) - 1:
            self.current_frame_index += 1
            self.update_frame()

    def update_frame(self):
        # self.color_space_mode = self.color_mode_var.get()

        # Load the frame image in RGB
        frame_fname = self.frame_fname_list[self.current_frame_index]
        img_rgb = cv2.cvtColor(cv2.imread(frame_fname, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        # correct rotation if needed
        img_rgb = self._correct_rotation(img_rgb)

        if self.roi_mask is not None:
            # apply roi_mask to img_rgb
            img_rgb = img_rgb * self.roi_mask[:, :, np.newaxis]

        # Load the residual image in grayscale
        residual_fname = self.residual_fname_list[self.current_frame_index]
        img_residual = read_one_residual(residual_fname)

        # correct rotation if needed
        img_residual = self._correct_rotation(img_residual)

        if self.roi_residual_mask is not None:
            img_residual = img_residual * self.roi_residual_mask

        # convert img_residual to grayscale 0-255
        MAX_RESIDUAL_VAL = 10
        img_residual = np.clip(np.abs(img_residual), 0, MAX_RESIDUAL_VAL) * (255.0 / MAX_RESIDUAL_VAL)
        img_residual = img_residual.astype(np.uint8)
        # Convert grayscale residual to RGB for consistent display
        img_residual_rgb = cv2.cvtColor(img_residual, cv2.COLOR_GRAY2RGB)
        
        # Resize both images to half their original size
        img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1] // 2, img_rgb.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        img_residual_rgb = cv2.resize(img_residual_rgb, (img_residual_rgb.shape[1] // 2, img_residual_rgb.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

        # crop both images to the same height and width (minimum of the two)
        height = min(img_rgb.shape[0], img_residual_rgb.shape[0])
        width = min(img_rgb.shape[1], img_residual_rgb.shape[1])
        img_rgb = img_rgb[:height, :width, :]
        img_residual_rgb = img_residual_rgb[:height, :width, :]

        # Concatenate the frame and residual side by side
        combined_img = np.hstack((img_rgb, img_residual_rgb))

        # Convert the combined image to a PIL Image
        img_pil = Image.fromarray(combined_img)

        # Convert PIL Image to PhotoImage for Tkinter
        self.img_tk = ImageTk.PhotoImage(master=self.master, image=img_pil)

        # Update Label and Index Text
        self.image_label.config(image=self.img_tk)
        self.frame_index_label.config(text=f"Frame {self.current_frame_index + 1} / {len(self.frame_fname_list)}")


    def on_closing(self):
        # Optional: Clean up resources
        self.master.destroy()
        self.master.quit()
        print("window destroyed.")


def main():
    parser = argparse.ArgumentParser(description="Perform video analysis with optional ROI selection.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--d", type=int, help="number of neighbors to validate a peak residual (default: 3)", default=3)
    parser.add_argument("--space", type=str, help="color space used for detection (default: Y)", default="Y")
    parser.add_argument("--epsilon", type=float, help="threshold for the Number of False Alarms (NFA), (default: 0.05)", default=0.05)
    # parser.add_argument("--max_num", type=float, help="maximum number of frames to process (default: -1)", default=-1)


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
        frame_fname_list, residual_fname_list = perform_video_analysis(args.video_path, d=args.d, space=args.space, epsilon=args.epsilon, roi_mask=roi_mask)
        
        rotation = get_rotation(args.video_path)
        
        # --- Step 4: Launch Visualization GUI ---
        viewer_root = tk.Tk()
        app = ResultViewer(viewer_root, frame_fname_list, residual_fname_list, rotation, roi_mask=roi_mask)
        viewer_root.mainloop()


    def on_yes():
        print("User chose to define an ROI.")

        nonlocal roi_mask, roi_used
        roi_prompt.destroy()
        # Launch OpenCV-based polygon selector
        roi_mask = select_polygon_roi(args.video_path)
        roi_used = roi_mask is not None
        start_analysis_and_viewer()

        root.destroy()

    def on_no():
        print("No ROI selected, analyzing full frame.")
        print()

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
    print("Starting application...")
    main()


# Usage:
#   python perform_video_analysis.py path/to/video.mp4
# or specify parameters:
#   python perform_video_analysis.py path/to/video.mp4 --d 5 --space U --epsilon 0.1
