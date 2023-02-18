import numpy as np
import os
import glob

from stream_analyzer import StreamAnalyzer
from PRED import PRED
from Yao import YAO
from Vazquez import Vazquez
import subprocess

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
ffprobe_exe = "/mnt/ddisk/yanhao/ffmpeg/ffmpeg-git-20220910-amd64-static/ffprobe"


def fname_from_vid_to_ckpt(vid_fname:str, method):
    # e.g.  for PRED method
    # /mnt/ddisk/yanhao/gop_detection/data/videos_c2/crf04/GOP09/akiyo_cif.h264
    #       to
    # /mnt/ddisk/yanhao/gop_detection/data/checkpoints/crf04/GOP09/akiyo_cif_PRED.ckpt

    ckpt_fname = vid_fname.replace("videos_c2", "checkpoints")[:-5] + f"_{method}.ckpt"
    return ckpt_fname

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


def test_one_video(vid_fname:str, GOP1):
    """ 

    :param vid_fname: the input video filename
    :return: estimated GOP of the first compression
    """
    print(f"Testing {vid_fname} ...")
    # 1. JM decodes the video and save it to somewhere
    has_decoded = False
    
    print(f"Gt G1: = {GOP1}")
    # 2. PRED
    analyzer = PRED()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "PRED")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames = glob.glob( os.path.join(root_inspect, "imgY_s*.npy") )
        analyzer.load_from_frames(inspect_fnames)
        analyzer.save_to_ckpt(ckpt_fname)
    GOP_est = analyzer.detect_periodic_signal()
    print(f"PRED estimation: G1 = {GOP_est}")

    # 3. Yao
    analyzer = YAO()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "Yao")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames = glob.glob( os.path.join(root_inspect, "imgMB*.png") )

        analyzer.load_SODB(vid_fname, ffprobe_exe)
        analyzer.load_from_frames(inspect_fnames)
        analyzer.preprocess()
        analyzer.save_to_ckpt(ckpt_fname)
    GOP_est = analyzer.detect_periodic_signal()
    print(f"Yao estimation: G1 = {GOP_est}")

    # 4. A Contrario
    analyzer = StreamAnalyzer()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "aContrario")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames = glob.glob( os.path.join(root_inspect, "imgY_s*.npy") )
        analyzer.load_from_frames(inspect_fnames)
        analyzer.save_to_ckpt(ckpt_fname)
    GOP_est = analyzer.detect_periodic_signal()
    print(f"aContrario estimation: G1 = {GOP_est}")

    print()
    pass
    

def main():
    crf_options = [18, 23, 28]
    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]


    for crf_c1 in crf_options:
        for gop1 in GOP_c1_options:
            for crf_c2 in crf_options:
                crf_c2 = 23
                for gop2 in GOP_c2_options:
                    in_dir = os.path.join(root_videos, 
                        f"crf{crf_c1:02d}_c1", f"gop{gop1:02d}_c1",
                        f"crf{crf_c2:02d}_c2", f"gop{gop2:02d}_c2")
                    vid_fnames = glob.glob(os.path.join(in_dir, "*.h264"))
                    vid_fnames.sort()
                    for vid_fname in vid_fnames:
                        try:
                            test_one_video(vid_fname, gop1)
                        except:
                            continue
                    break




def main_cbr():
    valid_prefix = ["akiyo_cif", "coastguard_cif", "deadline_cif",
                    "hall_monitor_cif", "paris_cif", "silent_cif", 
                    "bowing_cif", "container_cif", "foreman_cif", "news_cif", "sign_irene_cif"]
    cbr_c1_options = [300, 700, 1100]
    GOP_c1_options = [10, 15, 30, 40]
    cbr_c2_options = [300, 700, 1100]
    GOP_c2_options = [9, 16, 33, 50]


    for cbr_c1 in cbr_c1_options:
        for cbr_c2 in cbr_c2_options:
        # if cbr_c1 != 300:
            # continue
            for gop1 in GOP_c1_options:
                # if cbr_c2 != 300:
                    # continue
                for gop2 in GOP_c2_options:
                    in_dir = os.path.join(root_videos, 
                        f"cbr{cbr_c1}_c1", f"gop{gop1:02d}_c1",
                        f"cbr{cbr_c2}_c2", f"gop{gop2:02d}_c2")
                    # vid_fnames = glob.glob(os.path.join(in_dir, "*.h264"))
                    # vid_fnames.sort()
                    vid_fnames = [os.path.join(in_dir, s + ".h264") for s in valid_prefix]
                    for vid_fname in vid_fnames:
                        test_one_video(vid_fname, gop1)

if __name__ == "__main__":
    # main()
    main_cbr()