import numpy as np
import os
import glob

from stream_analyzer import StreamAnalyzer
from Chen import Chen
from Yao import YAO
from Vazquez import Vazquez
import subprocess

import json
from datetime import datetime

import argparse



""" Process experiments in two steps: decode with jm, and detect with .py script
    1. for each video:
        decode with jm to get residuals and MB types
        2. for each algo
            detect GOP, compare it with GT, store the results
        3. remove the intermediate .npy and .png files
    
    Metrics for comparison:
    1. AUC
    2. False alarms / Accuracy
"""



root_videos = "/mnt/ddisk/yanhao/gop_detection/data/videos_c2"
root_checkpoints = "/mnt/ddisk/yanhao/gop_detection/data/checkpoints"
jm_exe = "/mnt/ddisk/yanhao/gop_detection/jm/bin/ldecod.exe"
ffprobe_exe = "/mnt/ddisk/yanhao/ffmpeg/ffmpeg-git-20220910-amd64-static/ffprobe"

# root_inspect = "/mnt/ddisk/yanhao/gop_detection/data/inspect"
# root_inspect = "/mnt/ddisk/yanhao/gop_detection/data/inspect_set1"


parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='dataset', type=str, choices=["set1c1", "set1c2", "set2c1", "set2c2"])
args = parser.parse_args()
dataset = args.dataset # ["set1c1", "set1c2", "set2c1", "set2c2"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if "set1" in dataset:
    # list of baseline videos
    valid_prefix = ["akiyo_cif", "coastguard_cif", "deadline_cif",
                    "hall_monitor_cif", "paris_cif", "silent_cif", 
                    "bowing_cif", "container_cif", "foreman_cif", "news_cif", "sign_irene_cif"]
# valid_prefix = ["akiyo_cif"]
elif "set2" in dataset:
    # list of long videos ( >300 frames)
    valid_prefix = ['akiyo_cif', 'bridge_close_cif', 'bridge_far_cif',
                    'city_4cif', 'crew_4cif', 'deadline_cif',
                    'flower_garden_422_cif', 'football_422_cif', 'foreman_cif',
                    'galleon_422_cif', 'harbour_4cif', 'ice_4cif', 
                    'mad900_cif', 'paris_cif', 'students_cif',
                    'paris_cif', 'soccer_4cif', 'sign_irene_cif']

root_inspect = f"/mnt/ddisk/yanhao/gop_detection/data/inspect_{dataset}"
json_fname = f"results_{dataset}_{timestamp}.json"


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


def test_one_video(vid_fname:str, GOP1, params={}):
    """ 

    :param vid_fname: the input video filename
    :return: estimated GOP of the first compression
    """

    print(f"Testing {vid_fname} ...")
    # 1. JM decodes the video and save it to somewhere
    has_decoded = False
    
    print(f"Gt G1: = {GOP1}")
    # 2. PRED
    analyzer = Chen()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "PRED")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames = glob.glob( os.path.join(root_inspect, "imgY_s*.npy") )
        analyzer.load_from_frames(inspect_fnames)
        analyzer.save_to_ckpt(ckpt_fname)
    GOP_Chen, score_Chen = analyzer.detect_periodic_signal()
    print(f"Chen estimation: G1 = {GOP_Chen}")

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
    GOP_Yao, score_Yao = analyzer.detect_periodic_signal()
    print(f"Yao estimation: G1 = {GOP_Yao}")

    # 4. Vazquez
    analyzer = Vazquez()
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "Vazquez")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames = glob.glob( os.path.join(root_inspect, "imgMB*.png") )
        analyzer.load_from_frames(inspect_fnames)
        analyzer.preprocess()
        analyzer.save_to_ckpt(ckpt_fname)
    GOP_Vazquez, score_Vazquez = analyzer.detect_periodic_signal()
    print(f"Vazquez estimation: G1 = {GOP_Vazquez}")


    # 5. A Contrario
    analyzer = StreamAnalyzer(epsilon=100000, d=3, start_at_0=True, space="U")
    ckpt_fname = fname_from_vid_to_ckpt(vid_fname, "aContrario")
    if not analyzer.load_from_ckpt(ckpt_fname):
        if not has_decoded:
            decode_one_video(vid_fname)
            has_decoded = True
        inspect_fnames_Y = glob.glob( os.path.join(root_inspect, "imgY_s*.npy") )
        inspect_fnames_U = glob.glob( os.path.join(root_inspect, "imgU_s*.npy") )
        inspect_fnames_V = glob.glob( os.path.join(root_inspect, "imgV_s*.npy") )

        # load all data
        analyzer.load_from_frames(inspect_fnames_Y, space="Y")
        analyzer.load_from_frames(inspect_fnames_U, space="U")
        analyzer.load_from_frames(inspect_fnames_V, space="V")

        analyzer.save_to_ckpt(ckpt_fname)

    analyzer.preprocess()
    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"aContrario estimation: G1 = {GOP_aCont}")

    print()


    result = {
        "fname": vid_fname,
        "B1": params["B1"],
        "B2": params["B2"],
        "G1": params["G1"],
        "G2": params["G2"],
        "Chen": [int(GOP_Chen), score_Chen],
        "Yao": [int(GOP_Yao), score_Yao],
        "Vazquez": [int(GOP_Vazquez), score_Vazquez],
        "aContrario": [int(GOP_aCont) if GOP_aCont is not None else -1, NFA_aCont]
    }

    return result
    
    

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

def main_videos_c1():
    os.makedirs(root_inspect, exist_ok=True)

    cbr_c1_options = [300, 700, 1100]
    GOP_c1_options = [10, 15, 30, 40]

    # json_file = open("results_set1.json", "w")
    assert "c1" in json_fname
    json_file = open(json_fname, "w")
    json_file.write("[ \n")

    for cbr_c1 in cbr_c1_options:
        for gop1 in GOP_c1_options:
            params = {
                "B1": cbr_c1,
                "B2": -1,
                "G1": gop1,
                "G2": -1
            }

            in_dir = os.path.join(root_videos, 
                f"cbr{cbr_c1}_c1", f"gop{gop1:02d}_c1")
            # vid_fnames = glob.glob(os.path.join(in_dir, "*.h264"))
            vid_fnames = [os.path.join(in_dir, s + ".h264") for s in valid_prefix]
            for i in range(len(vid_fnames)):
                vid_fname = vid_fnames[i]
                result = test_one_video(vid_fname, gop1, params)
                print(result)
                json.dump(result, json_file, indent=4)
                json_file.write(", \n")
                json_file.flush()
    json_file.close()

    with open(json_fname, 'rb+') as json_file:
        json_file.seek(-3, os.SEEK_END)
        json_file.truncate()
    
    with open(json_fname, 'a') as json_file:
        json_file.write("\n")
        json_file.write("] \n")




def main_videos_c2():
    os.makedirs(root_inspect, exist_ok=True)

    cbr_c1_options = [300, 700, 1100]
    GOP_c1_options = [10, 15, 30, 40]
    cbr_c2_options = [300, 700, 1100]
    GOP_c2_options = [9, 16, 33, 50]

    # json_file = open("results_set1.json", "w")
    assert "c2" in json_fname
    json_file = open(json_fname, "w")
    json_file.write("[ \n")

    for cbr_c1 in cbr_c1_options:
        for gop1 in GOP_c1_options:
            for cbr_c2 in cbr_c2_options:
                for gop2 in GOP_c2_options:
                    # if cbr_c1 != 700 or cbr_c2 != 300 or gop1 != 30 or gop2 !=16:
                    # if gop1 != 40 or gop2 != 16:
                        # continue
                    params = {
                        "B1": cbr_c1,
                        "B2": cbr_c2,
                        "G1": gop1,
                        "G2": gop2
                    }

                    in_dir = os.path.join(root_videos, 
                        f"cbr{cbr_c1}_c1", f"gop{gop1:02d}_c1",
                        f"cbr{cbr_c2}_c2", f"gop{gop2:02d}_c2")
                    # vid_fnames = glob.glob(os.path.join(in_dir, "*.h264"))
                    vid_fnames = [os.path.join(in_dir, s + ".h264") for s in valid_prefix]

                    for i in range(len(vid_fnames)):
                        vid_fname = vid_fnames[i]
                        result = test_one_video(vid_fname, gop1, params)
                        print(result)
                        json.dump(result, json_file, indent=4)
                        json_file.write(", \n")
                        json_file.flush()

    json_file.close()

    with open(json_fname, 'rb+') as json_file:
        json_file.seek(-3, os.SEEK_END)
        json_file.truncate()
    
    with open(json_fname, 'a') as json_file:
        json_file.write("\n")
        json_file.write("] \n")


if __name__ == "__main__":
    if "c1" in dataset:
        main_videos_c1()
    else:
        main_videos_c2()