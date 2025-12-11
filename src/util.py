
import subprocess
import os
import numpy as np



def convert_to_h264(vid_fname:str, out_fname:str):
    """ Convert the input video to h264 format.
    param vid_fname: input video filename
    param out_fname: output h264 video filename
    return: True if success, False otherwise
    """

    # 1. Verify the video is encoded by h264
    ffprobe_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {vid_fname}"
    std_msg = subprocess.run(ffprobe_command, shell=True, capture_output=True, text=True)
    found_codec = std_msg.stdout[:-1]

    if found_codec != "h264":
        print(f"Error: The input video '{vid_fname}' needs to be encoded by h264, but codec {found_codec} is found!")
        return False

    # 2. Convert the video file to .h264 file.
    # out_fname = os.path.join(TMP_PATH, H264_VID_FNAME)
    convert_command = f"ffmpeg -i {vid_fname} -an -vcodec copy {out_fname} -y"
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


    JM_EXE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../jm/bin/ldecod.exe")

    # 1.2 jm extracts intermediate files
    inspect_command = f"{JM_EXE} -i {vid_fname} -inspect {output_folder}"
    # print(inspect_command)
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    if std_msg.stderr != '':
        print(std_msg.stderr)
        print(f"Decoding {vid_fname} failed! (from JM software)")
        return False, None

    print("Prediction residuals are saved in: ", output_folder)
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

    # Check if the output folder exists
    # try:
    os.makedirs(output_folder, exist_ok=True)
    # except Exception as e:
        # print(f"Warning: Failed to create the output folder '{output_folder}'. Error: {e}")
        # return False, None
    
    # ffmpeg decodes images
    img_out_pattern = os.path.join(output_folder, "img%06d.png")
    ffmpeg_command = f"ffmpeg -i {vid_fname} -start_number 0 {img_out_pattern}"
    # print(ffmpeg_command)
    std_msg = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

    print(f"Decoding finished successfully.\n")
    print("Frames are saved in: ", output_folder)
    print()

    return True, output_folder


def decode_one_video(vid_fname:str, output_folder:str):
    assert vid_fname.endswith("264")

    # Check if the output folder exists
    if os.path.exists(output_folder):
        print(f"Warning: The output folder '{output_folder}' already exists!")
        return False
    
    try:
        os.makedirs(output_folder)
    except Exception as e:
        print(f"Warning: Failed to create the output folder '{output_folder}'. Error: {e}")
        return False

    output_residual_folder = os.path.join(output_folder, "residuals")
    output_frame_folder = os.path.join(output_folder, "frames")

    try:
        os.makedirs(output_residual_folder)
        os.makedirs(output_frame_folder)
    except Exception as e:
        print(f"Warning: Failed to create sub-folders in '{output_folder}'. Error: {e}")
        return False

    JM_EXE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../jm/bin/ldecod.exe")

    # 1.2 jm extracts intermediate files
    inspect_command = f"{JM_EXE} -i {vid_fname} -inspect {output_residual_folder}"
    print("inspect_command = ", inspect_command)
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    if std_msg.stderr != '':
        print(std_msg.stderr)
        raise Exception(f"Decoding {vid_fname} failed! (from JM software)")

    # 1.3 ffmpeg decodes images
    img_out_pattern = os.path.join(output_frame_folder, "img%04d.png")
    ffmpeg_command = f"ffmpeg -i {vid_fname} -start_number 0 {img_out_pattern}"
    print(ffmpeg_command)
    std_msg = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

    print(f"Decoding finished successfully.\n")
    print("Prediction residuals are saved in: ", output_residual_folder)
    print("Frames are saved in: ", output_frame_folder)
    print()

    return True

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

import subprocess
import shlex
import json

# def get_rotation(file_path_with_file_name):
#     """
#     Function to get the rotation of the input video file.
#     Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
#     stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

#     Returns a rotation None, 90, 180 or 270
#     """
#     cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
#     args = shlex.split(cmd)
#     args.append(file_path_with_file_name)
#     # run the ffprobe process, decode stdout into utf-8 & convert to JSON
#     ffprobe_output = subprocess.check_output(args).decode('utf-8')
#     if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
#         ffprobe_output = json.loads(ffprobe_output)
#         rotation = ffprobe_output

#     else:
#         rotation = 0

#     return rotation


import ffmpeg

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



# def get_rotation_code(video_file_path: str):
#     rotation = get_rotation(video_file_path)
#     rotation_code = None
#     if rotation == 180:
#         rotation_code = cv2.ROTATE_180
#         print("Rotation code: ROTATE_180")
#     if rotation == 90:
#         rotation_code = cv2.ROTATE_90_CLOCKWISE
#         print("Rotation code: ROTATE_90_CLOCKWISE")
#     if rotation == 270:
#         rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
#         print("Rotation code: ROTATE_90_COUNTERCLOCKWISE")
#     return rotation_code