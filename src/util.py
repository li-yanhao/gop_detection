import os
import subprocess

import numpy as np
import ffmpeg
import cv2

OUTPUT_JM = "test_dec.yuv"

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
    inspect_command = f"{JM_EXE} -i {vid_fname} -o {OUTPUT_JM} -inspect {output_folder}"
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
    
    # ffmpeg decodes images
    img_out_pattern = os.path.join(output_folder, "img%06d.png")
    ffmpeg_command = f"ffmpeg -i {vid_fname} -start_number 0 {img_out_pattern}"
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