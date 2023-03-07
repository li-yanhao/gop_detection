import argparse
import glob
import os
import subprocess

from aContrario import StreamAnalyzer


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input', type=str)
parser.add_argument('-d', dest="d", type=int, default=3)
parser.add_argument('--epsilon', dest="epsilon", type=float, default=1.0)

parser.add_argument('--no_show', action="store_false")

args = parser.parse_args()


tmp_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tmp")
jm_exe = os.path.join(os.path.abspath(os.path.dirname(__file__)), "jm/bin/ldecod.exe")
h264_vid_fname = "video.h264"


def convert_to_h264(vid_fname:str):
    print(f"Testing {vid_fname} ...")

    # 1. Verify the video is encoded by h264
    ffprobe_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {vid_fname}"
    std_msg = subprocess.run(ffprobe_command, shell=True, capture_output=True, text=True)
    found_codec = std_msg.stdout[:-1]

    if found_codec != "h264":
        raise Exception(f"Error: The input video '{vid_fname}' needs to be encoded by h264, but codec {found_codec} is found!")

    # 2. Convert the video file to .h264 file.
    out_fname = os.path.join(tmp_path, h264_vid_fname)
    convert_command = f"ffmpeg -i {vid_fname} -an -vcodec copy {out_fname} -y"
    std_msg = subprocess.run(convert_command, shell=True, capture_output=True, text=True)
    return True


def decode_one_video(vid_fname:str):
    assert vid_fname.endswith("264")

    # 1.1 remove existing files
    file_pattern_to_remove = os.path.join(tmp_path, "img*")
    clear_command = f"rm {file_pattern_to_remove}"
    std_msg = subprocess.run(clear_command, shell=True, capture_output=True, text=True)

    # 1.2 jm extracts intermediate files
    inspect_command = f"{jm_exe} -i {vid_fname} -inspect {tmp_path}"
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    if std_msg.stderr != '':
        raise Exception(f"Decoding {vid_fname} failed! ")

    print(f"Decoding finished successfully.")
    print()


def test_one_video(vid_fname: str, reload=True, visualize=False, max_num_frames=-1):
    """
    :param vid_fname: the filename of a H264 video.
    :param reload: whether to decode the video or using existing intermediate data. This is useful when the same video
        needs to be detected again.
    :param visualize: whether to visualize the results.
    :param max_num_frames: the maximum number of frames used for detection. If max_num_frames<=0, all the frames in
        the video will be used.
    :return:
        [0] estimated GOP of the first compression
        [1] NFA
    """



    # 1. JM decodes the video and save it to somewhere
    if reload:
        decode_one_video(vid_fname)

    # 2. A Contrario
    analyzer = StreamAnalyzer(epsilon=1, d=2, start_at_0=False, space="U", max_num=max_num_frames)

    inspect_fnames_Y = glob.glob(os.path.join(tmp_path, "imgY_s*.npy"))
    inspect_fnames_U = glob.glob(os.path.join(tmp_path, "imgU_s*.npy"))
    inspect_fnames_V = glob.glob(os.path.join(tmp_path, "imgV_s*.npy"))

    # load all data
    analyzer.load_from_frames(inspect_fnames_Y, space="Y")
    analyzer.load_from_frames(inspect_fnames_U, space="U")
    analyzer.load_from_frames(inspect_fnames_V, space="V")

    analyzer.preprocess()

    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"Estimated primary GOP = {GOP_aCont}")
    print(f"NFA = {NFA_aCont}", )
    print()

    if visualize:
        analyzer.visualize()

    result = {
        "fname": vid_fname,
        "aContrario": [int(GOP_aCont) if GOP_aCont is not None else -1, NFA_aCont]
    }

    return result


# in_fname = "/Users/yli/phd/deepfake/collection_deepfake/real/table_c2.h264"
# irrelevant_dataset_root = "/Users/yli/phd/sshfs/yanhao/datasets/yuv/compress/"
# in_fname = os.path.join(irrelevant_dataset_root, "real_case/crew_4cif_c1g10.h264")
# in_fname = os.path.join(irrelevant_dataset_root, "real_case/crew_4cif_c1g10_c2g18.h264")
# in_fname = os.path.join(irrelevant_dataset_root, "bframe/cbr1100_c1/gop10_c1/flower_garden_422_cif.h264")
# in_fname = os.path.join(irrelevant_dataset_root, "bframe/cbr1100_c1/gop10_c1/cbr1100_c2/gop16_c2/flower_garden_422_cif.h264")


if __name__ == "__main__":
    os.makedirs(tmp_path, exist_ok=True)

    # 1. convert file to h264
    convert_to_h264(args.input)

    # 2. extract decoding data and detect
    result = test_one_video(os.path.join(tmp_path, h264_vid_fname), reload=True, visualize=args.no_show)
