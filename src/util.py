
import subprocess
import os



def convert_to_h264(vid_fname:str, out_fname:str):
    assert out_fname.endswith(".264"), "Output filename must end with .264"

    # 1. Verify the video is encoded by h264
    ffprobe_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {vid_fname}"
    std_msg = subprocess.run(ffprobe_command, shell=True, capture_output=True, text=True)
    found_codec = std_msg.stdout[:-1]

    if found_codec != "h264":
        print(f"Error: The input video '{vid_fname}' needs to be encoded by h264, but codec {found_codec} is found!")
        return False

    # 2. Convert the video file to .h264 file.
    convert_command = f"ffmpeg -i {vid_fname} -an -vcodec copy {out_fname} -y"
    std_msg = subprocess.run(convert_command, shell=True, capture_output=True, text=True)
    return True


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

    # Check if the output folder exists
    # try:
    os.makedirs(output_folder, exist_ok=True)
    # except Exception as e:
    #     print(f"Warning: Failed to create the output folder '{output_folder}'. Error: {e}")
    #     return False, None


    JM_EXE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../jm/bin/ldecod.exe")

    # 1.2 jm extracts intermediate files
    inspect_command = f"{JM_EXE} -i {vid_fname} -inspect {output_folder}"
    print("inspect_command = ", inspect_command)
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
    print(ffmpeg_command)
    std_msg = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

    print(f"Decoding finished successfully.")
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

    print(f"Decoding finished successfully.")
    print("Prediction residuals are saved in: ", output_residual_folder)
    print("Frames are saved in: ", output_frame_folder)
    print()

    return True