import argparse
import glob
import os
import subprocess

from aContrario import StreamAnalyzer


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input', type=str)
args = parser.parse_args()


root_inspect = "/Users/yli/phd/video_processing/gop_detection/tmp"
jm_exe = "/Users/yli/phd/video_processing/gop_detection/jm_16.1/bin/ldecod.exe"


def decode_one_video(vid_fname:str):
    # inspect_dir = os.path.dirname(vid_fname.replace("videos_c2", "inspect"))

    # 1.1 remove existing files
    file_pattern_to_remove = os.path.join(root_inspect, "img*")
    clear_command = f"rm {file_pattern_to_remove}"
    std_msg = subprocess.run(clear_command, shell=True, capture_output=True, text=True)

    # 1.2 jm extracts intermediate files
    inspect_command = f"{jm_exe} -i {vid_fname} -inspect {root_inspect}"
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    print(f"Finish decoding {vid_fname}")


def test_one_video(vid_fname: str, reload=True, visualize=False, max_num_frames=-1):
    """
    :param vid_fname: the input video filename
    :return: estimated GOP of the first compression
    """

    print(f"Testing {vid_fname} ...")

    # 1. JM decodes the video and save it to somewhere
    if reload:
        decode_one_video(vid_fname)

    print(f"max_num_frames: {max_num_frames}")

    # 2. A Contrario
    analyzer = StreamAnalyzer(epsilon=1, d=3, start_at_0=False, space="U", max_num=max_num_frames)

    inspect_fnames_Y = glob.glob(os.path.join(root_inspect, "imgY_s*.npy"))
    inspect_fnames_U = glob.glob(os.path.join(root_inspect, "imgU_s*.npy"))
    inspect_fnames_V = glob.glob(os.path.join(root_inspect, "imgV_s*.npy"))

    # load all data
    # analyzer.load_from_frames(inspect_fnames_Y, space="Y")
    analyzer.load_from_frames(inspect_fnames_U, space="U")
    # analyzer.load_from_frames(inspect_fnames_V, space="V")


    analyzer.preprocess()
    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"aContrario estimation: G1 = {GOP_aCont}")

    print()

    if visualize:
        analyzer.visualize()

    result = {
        "fname": vid_fname,
        "aContrario": [int(GOP_aCont) if GOP_aCont is not None else -1, NFA_aCont]
    }

    return result


# in_fname = "/Users/yli/phd/deepfake/collection_deepfake/real/table_c2.h264"

irrelevant_dataset_root = "/Users/yli/phd/sshfs/yanhao/datasets/yuv/compress/"

# in_fname = os.path.join(irrelevant_dataset_root, "real_case/crew_4cif_c1g10.h264")
# in_fname = os.path.join(irrelevant_dataset_root, "real_case/crew_4cif_c1g10_c2g18.h264")

# in_fname = os.path.join(irrelevant_dataset_root, "bframe/cbr1100_c1/gop10_c1/flower_garden_422_cif.h264")
in_fname = os.path.join(irrelevant_dataset_root, "bframe/cbr1100_c1/gop10_c1/cbr1100_c2/gop16_c2/flower_garden_422_cif.h264")


result = test_one_video(in_fname, reload=True, visualize=True)