import os
import subprocess
import sys

import numpy as np
import ffmpeg
import cv2

OUTPUT_JM = "test_dec.yuv"
BUNDLED_BIN_DIR = "dist_bin"

def get_platform_exe_name(base_name):
    """Add .exe extension on Windows, otherwise return base name as-is."""
    return f"{base_name}.exe" if sys.platform == "win32" else base_name

def get_base_path():
    """Get the base path for resources, handling both PyInstaller bundle and development environments."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running in a normal Python environment
        return os.path.abspath(os.path.dirname(__file__))

def get_executable_path(exe_name, dev_path=None):
    """Get the path to an executable, handling both PyInstaller bundle and development environments.
    
    For PyInstaller bundles, looks in the bundled dist_bin directory first.
    For development, uses dev_path if provided, otherwise assumes the executable is in PATH.
    
    Args:
        exe_name: Name of the executable file (with platform-specific extension)
        dev_path: Optional path to the executable in development environment
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle - look in bundled dist_bin directory
        bundled_exe = os.path.join(sys._MEIPASS, BUNDLED_BIN_DIR, exe_name)
        if os.path.exists(bundled_exe):
            return bundled_exe
    
    # For development environment
    if dev_path and os.path.exists(dev_path):
        return dev_path
    
    # Fall back to assuming it's in PATH
    return exe_name

def convert_to_h264(vid_fname:str, out_fname:str):
    """ Convert the input video to h264 format.
    param vid_fname: input video filename
    param out_fname: output h264 video filename
    return: True if success, False otherwise
    """

    # Get paths to ffprobe and ffmpeg executables
    ffprobe_exe = get_executable_path(get_platform_exe_name("ffprobe"))
    ffmpeg_exe = get_executable_path(get_platform_exe_name("ffmpeg"))

    # 1. Verify the video is encoded by h264
    ffprobe_command = f'"{ffprobe_exe}" -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{vid_fname}"'
    std_msg = subprocess.run(ffprobe_command, shell=True, capture_output=True, text=True)
    found_codec = std_msg.stdout[:-1]

    if found_codec != "h264":
        print(f"Error: The input video '{vid_fname}' needs to be encoded by h264, but codec {found_codec} is found!")
        return False

    # 2. Convert the video file to .h264 file.
    # out_fname = os.path.join(TMP_PATH, H264_VID_FNAME)
    convert_command = f'"{ffmpeg_exe}" -i "{vid_fname}" -an -vcodec copy "{out_fname}" -y'
    std_msg = subprocess.run(convert_command, shell=True, capture_output=True, text=True)
    return True


def decode_residuals(vid_fname:str, output_root:str):
    """ Decode the prediction residuals from a h264 video using JM software.
    :param vid_fname: the filename of a H264 video.
    :param output_root: the root folder to save the output residuals. The video's residuals will be saved in a sub-folder named by the video filename.
    :return: True if success, False otherwise
    """

    assert vid_fname.endswith("264")

    output_folder = os.path.join(output_root, os.path.basename(vid_fname).split('.')[0], "residuals")

    os.makedirs(output_folder, exist_ok=True)

    # Determine the path to ldecod executable
    base_path = get_base_path()
    # In development, ldecod is in jm/bin/ relative to the parent of src/
    dev_ldecod_path = os.path.join(os.path.dirname(base_path), "jm", "bin", get_platform_exe_name("ldecod"))
    
    JM_EXE = get_executable_path(get_platform_exe_name("ldecod"), dev_ldecod_path)
    
    # Verify the executable exists if it's a full path (not just a name in PATH)
    if os.path.dirname(JM_EXE) and not os.path.exists(JM_EXE):
        print(f"Error: ldecod executable not found at {JM_EXE}")
        return False, None

    # 1.2 jm extracts intermediate files
    inspect_command = f'"{JM_EXE}" -i "{vid_fname}" -o "{OUTPUT_JM}" -inspect "{output_folder}"'
    # print(inspect_command)
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    # remove the output yuv file whether it exists or not
    if os.path.exists(OUTPUT_JM):
        os.remove(OUTPUT_JM)

    if std_msg.stderr != '':
        print(std_msg.stderr)
        print(f"Decoding {vid_fname} failed! (from JM software)")
        return False, None

    print("Prediction residuals are saved in: ", output_folder, "   (can be deleted after analysis)")
    print()

    return True, output_folder


def decode_frames(vid_fname:str, output_root:str):
    """ Decode the frames from a h264 video using ffmpeg.
    :param vid_fname: the filename of a H264 video.
    :param output_root: the root folder to save the output frames. The video's frames will be saved in a sub-folder named by the video filename.
    :return: True if success, False otherwise
    """

    assert vid_fname.endswith("264")

    output_folder = os.path.join(output_root, os.path.basename(vid_fname).split('.')[0], "frames")

    os.makedirs(output_folder, exist_ok=True)
    
    # Get path to ffmpeg executable
    ffmpeg_exe = get_executable_path(get_platform_exe_name("ffmpeg"))
    
    # ffmpeg decodes images
    img_out_pattern = os.path.join(output_folder, "img%06d.png")
    ffmpeg_command = f'"{ffmpeg_exe}" -i "{vid_fname}" -start_number 0 "{img_out_pattern}"'
    # print(ffmpeg_command)
    std_msg = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

    print(f"Decoding finished successfully.\n")
    print("Frames are saved in: ", output_folder, "   (can be deleted after analysis)")
    print()

    return True, output_folder


def pad_and_crop(img, target_shape):
    """ Pad or crop the input image to match the target shape.
    :param img: input image
    :param target_shape: target shape (height, width)
    :return: padded or cropped image
    """
    h, w = img.shape[:2]
    target_h, target_w = target_shape

    # Pad in height if needed
    if target_h - h > 0:
        img = np.pad(img, ((0, target_h - h), (0, 0)), mode='constant', constant_values=0)
    elif target_h - h < 0:
        img = img[:target_h, :]
    
    # Pad in width if needed
    if target_w - w > 0:
        img = np.pad(img, ((0, 0), (0, target_w - w)), mode='constant', constant_values=0)
    elif target_w - w < 0:
        img = img[:, :target_w]

    return img

def get_rotation(video_file_path: str):
    try:
        # fetch video metadata
        metadata = ffmpeg.probe(video_file_path)
    except Exception as e:
        print(f'failed to read video: {video_file_path}\n'
              f'{e}\n',
              end='',
              flush=True)
        return None
    # extract rotate info from metadata
    video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
    rotation = int(video_stream.get('tags', {}).get('rotate', 0))
    # extract rotation info from side_data_list, popular for Iphones
    if len(video_stream.get('side_data_list', [])) != 0:
        side_data = next(iter(video_stream.get('side_data_list')))
        side_data_rotation = int(side_data.get('rotation', 0))
        if side_data_rotation != 0:
            rotation -= side_data_rotation
    return rotation


def correct_rotation(frame, rotation):
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame