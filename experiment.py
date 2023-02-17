import numpy as np
import os
import glob

from stream_analyzer import StreamAnalyzer
from PRED import PRED
from Yao import YAO
from Vazquez import Vazquez

""" Process experiments in two steps: decode with jm, and detect with .py script
    1. for each video:
        decode with jm to get residuals and MB types
        2. for each algo
            detect GOP, compare it with GT, store the results
        3. remove the intermediate .npy and .png files
    
    Metrics for comparison:
    1. AUC
    2. False alarms
"""


root_videos = "/mnt/ddisk/yanhao/gop_detection/data/videos_c2"
root_checkpoints = "/mnt/ddisk/yanhao/gop_detection/data/checkpoints"
root_inspect = "/mnt/ddisk/yanhao/gop_detection/data/inspect"
jm_exe = "/mnt/ddisk/yanhao/gop_detection/jm/bin/ldecod.exe"


def fname_from_vid_to_ckpt(vid_fname:str, method):
    # e.g.  for PRED method
    # /mnt/ddisk/yanhao/gop_detection/data/videos_c2/crf04/GOP09/akiyo_cif.h264
    #       to
    # /mnt/ddisk/yanhao/gop_detection/data/checkpoints/crf04/GOP09/akiyo_cif_PRED.ckpt

    ckpt_fname = vid_fname.replace("videos_c2", "checkpoints")[:-5] + f"_{method}.ckpt"
    return ckpt_fname


def test_one_video(vid_fname:str, GOP1):
    """ 

    :param vid_fname: the input video filename
    :return: estimated GOP of the first compression
    """

    
    # 1. JM decodes the video and save it to somewhere
    inspect_dir = os.path.dirname(vid_fname.replace("videos_c2", "inspect"))
    inspect_command = f"{jm_exe} -i {vid_fname}"
    

    # 2. PRED
    analyzer = PRED()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "PRED")
    if not analyzer.load_from_ckpt(ckpt_fname):
        inspect_fnames = glob.glob(inspect_dir, "imgY_s*.npy")
        analyzer.load_from_frames(inspect_fnames)
        analyzer.save_to_ckpt(ckpt_fname)
        GOP = analyzer.detect_periodic_signal()
    analyzer.visualize(None)

    pass
    

def main():
    crf_options = [4, 11, 23, 40]
    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    vid_c2_root = "/mnt/ddisk/yanhao/datasets/yuv/c2/"

    for crf in crf_options:
        for GOP1, GOP2 in zip(GOP_c1_options, GOP_c2_options):
            in_dir = os.path.join(vid_c2_root, f"crf{crf:02d}", f"GOP{GOP2:02d}")
            in_fnames = glob.glob(os.path.join(in_dir, "*.h264"))


