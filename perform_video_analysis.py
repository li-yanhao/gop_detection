import os
import glob
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np

from src.residual_info import get_sorted_residual_info_list
from src.acontrario import AContrarioAnalyser
from src.util import decode_frames, decode_residuals, convert_to_h264


OUTPUT_ROOT = "tmp"
SAVE_VISUALIZED_PATH = None



def perform_video_analysis(video_path:str, 
                           d:int, space:str, epsilon:float,
                           roi_mask=None,
                           max_num:int=-1,
                           output_folder:str=None):
    """ Perform video analysis on the given video file.
    :param video_path: Path to the input video file.
    :param d: Parameter d for A Contrario analysis.
    :param space: Color space to use ('Y', 'U', 'V', 'RGB').
    :param epsilon: Epsilon parameter for A Contrario analysis.
    :param roi_mask: Optional mask (NumPy array) defining the region of interest. Suppose the mask is unrotated to be aligned with the video frames.
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

    # GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    detections = analyzer.detect_periodic_signal()

    if len(detections) == 0:
        print("\033[92mNo periodicity detected!\n\033[0m")
    else:
        print("\033[92mDetected candidates (by A Contrario analysis):\033[0m")
        for (p, b, NFA) in detections:
            print(f"\033[92m  Periodicity = {p}, Offset = {b}, NFA = {NFA}\033[0m")
        print()

        print("\033[92mThe most prominent candidate: periodicity = {}, NFA = {}\033[0m".format(analyzer.detected_result[0], analyzer.detected_result[2]))
        print()

    # Save the results to txt if output_folder is given
    if output_folder is not None:
        result_txt_path = os.path.join(output_folder, "detections.txt")
        with open(result_txt_path, 'w') as f:
            for (p, b, NFA) in detections:
                f.write(f"Periodicity = {p}, Offset = {b}, NFA = {NFA}\n")
        print(f"Detections are saved to: {result_txt_path}\n")

    visualize_path = os.path.join(output_folder, "histogram.png") if output_folder is not None else None
    analyzer.visualize(save_fname=visualize_path, open_browser=False)

    frame_fname_list = glob.glob(os.path.join(frame_folder, "*.png"))
    frame_fname_list.sort()

    residual_fname_list = [ri.fname for ri in residual_info_list]

    assert len(frame_fname_list) == len(residual_fname_list), "Number of frames ({}) and residuals ({}) do not match!".format(len(frame_fname_list), len(residual_fname_list))


    return frame_fname_list, residual_fname_list


def main():
    parser = argparse.ArgumentParser(description="Perform video analysis with optional ROI selection.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze.")
    parser.add_argument("--d", type=int, help="number of neighbors to validate a peak residual (default: 3)", default=3)
    parser.add_argument("--space", type=str, help="color space used for detection (default: Y)", default="Y")
    parser.add_argument("--epsilon", type=float, help="threshold for the Number of False Alarms (NFA), (default: 0.05)", default=0.05)
    parser.add_argument("--mask_path", type=str, help="path to the ROI mask image (optional)", default=None)
    parser.add_argument("--out_folder", type=str, help="path to the output folder (default: results/)", default="results")

    args = parser.parse_args()

    # create output folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(args.out_folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # save the full args to a txt file in the output folder
    args_txt_path = os.path.join(output_folder, "args.txt")
    with open(args_txt_path, 'w') as f:
        for arg, val in vars(args).items():
            f.write(f"{arg}: {val}\n")

    # load mask
    if args.mask_path is not None:
        if not os.path.exists(args.mask_path):
            print(f"Mask path '{args.mask_path}' does not exist. Please verify the mask path. Exiting.")
            return
        mask_img = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"Failed to read mask image from '{args.mask_path}'. Please verify the mask path. Exiting.")
            return
        roi_mask = (mask_img > 0).astype(np.uint8)
        print("ROI mask loaded from:", args.mask_path)

    else:
        roi_mask = None

    frame_fname_list, residual_fname_list = perform_video_analysis(args.video_path, d=args.d, space=args.space, epsilon=args.epsilon, roi_mask=roi_mask, output_folder=output_folder)
    
    print(f"Detection finished. All the results are saved in: {output_folder}\n")


if __name__ == "__main__":
    print("Starting application...")
    main()


# Usage:
#   python perform_video_analysis.py path/to/video.mp4
# or specify parameters:
#   python perform_video_analysis.py path/to/video.mp4 --d 5 --space U --epsilon 0.1 --out_folder results/
