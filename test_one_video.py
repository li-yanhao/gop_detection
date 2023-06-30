import argparse
import glob
import os
import subprocess

from aContrario import StreamAnalyzer


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input', type=str)
parser.add_argument('-d', dest="d", type=int, default=3)
parser.add_argument('--epsilon', dest="epsilon", type=float, default=1.0)
parser.add_argument('--space', dest="space", type=str, default="Y")
parser.add_argument('--no_show', action="store_false")
parser.add_argument('--plot', dest='plot', type=str, default=None)
parser.add_argument('--detect_zone', dest='detect_zone', type=str, default="all", choices=['all', 'face', 'background'])

args = parser.parse_args()


tmp_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tmp")
jm_exe = os.path.join(os.path.abspath(os.path.dirname(__file__)), "jm/bin/ldecod.exe")
h264_vid_fname = "video.h264"


def get_file_prefix(fname):
    return os.path.basename(fname).split('.')[0]

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
    subprocess.run(clear_command, shell=True, capture_output=True, text=True)

    # 1.2 jm extracts intermediate files
    inspect_command = f"{jm_exe} -i {vid_fname} -inspect {tmp_path}"
    std_msg = subprocess.run(inspect_command, shell=True, capture_output=True, text=True)

    if std_msg.stderr != '':
        print(std_msg.stderr)
        raise Exception(f"Decoding {vid_fname} failed! (from JM)")

    # 1.3 ffmpeg decodes images
    img_out_pattern = os.path.join(tmp_path, "img%04d.png")
    ffmpeg_command = f"ffmpeg -i {vid_fname} -start_number 0 {img_out_pattern}"
    print(ffmpeg_command)
    std_msg = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)

    # if std_msg.stderr != '':
    #     raise Exception(f"Decoding {vid_fname} failed! (from ffmpeg)")

    print(f"Decoding finished successfully.")
    print()


def test_one_video(vid_fname:str, reload=True, visualize=False, max_num_frames=-1,
                   params_in={}):
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

    params = {"d": 3,
              "space": "U",
              "epsilon": 1,
              "detect_zone": "all"
              }
    params.update(params_in)

    # 2. A Contrario
    analyzer = StreamAnalyzer(epsilon=params["epsilon"], d=params["d"], start_at_0=False,
                              space=params["space"], max_num=max_num_frames)

    res_U_fnames = glob.glob(os.path.join(tmp_path, "img" + params["space"] + "_d*.npy"))

    if params["detect_zone"] == 'all':
        mask_maker = None
    else:
        from face_segmenter import FaceSegmenter
        if params["detect_zone"] == 'face':
            mask_maker = FaceSegmenter(invert=False)
        elif params["detect_zone"] == 'background':
            mask_maker = FaceSegmenter(invert=True)

    bgr_fnames = glob.glob(os.path.join(tmp_path, f"img[0-9]*.png"))
    bgr_fnames.sort()

    analyzer.load_from_frames(res_fnames=res_U_fnames, space=params["space"], img_fnames=bgr_fnames, mask_maker=mask_maker)

    analyzer.preprocess()

    GOP_aCont, NFA_aCont = analyzer.detect_periodic_signal()
    print(f"Estimated primary GOP = {GOP_aCont}")
    print(f"NFA = {NFA_aCont}", )
    print()


    if visualize:
        analyzer.visualize(params["plot_fname"])

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
    # tmp_path = get_file_prefix(args.input)

    os.makedirs(tmp_path, exist_ok=True)

    # 1. convert file to h264
    convert_to_h264(args.input)

    # 2. extract decoding data and detect
    # plot_fname = None

    args.detect_zone = 'all'
    plot_fname = os.path.basename(args.input).split(".")[0] + "_" + args.space + "_" + args.detect_zone + ".html"
    params = {"d": args.d, "space": args.space, "epsilon": args.epsilon, "detect_zone": args.detect_zone,
              "plot_fname": plot_fname}
    result = test_one_video(os.path.join(tmp_path, h264_vid_fname), reload=False, visualize=args.no_show,
                            params_in=params)
    print("Zone `all` is done")
    print()
    #
    # args.detect_zone = 'face'
    # plot_fname = os.path.basename(args.input).split(".")[0] + "_" + args.space + "_" + args.detect_zone + ".html"
    # params = {"d": args.d, "space": args.space, "epsilon": args.epsilon, "detect_zone": args.detect_zone,
    #           "plot_fname": plot_fname}
    # result = test_one_video(os.path.join(tmp_path, h264_vid_fname), reload=True, visualize=args.no_show,
    #                         params_in=params)
    # print(f"Zone {args.detect_zone} is done")
    # print()
    # #
    # args.detect_zone = 'background'
    # plot_fname = os.path.basename(args.input).split(".")[0] + "_" + args.space + "_" + args.detect_zone + ".html"
    # params = {"d": args.d, "space": args.space, "epsilon": args.epsilon, "detect_zone": args.detect_zone,
    #           "plot_fname": plot_fname}
    # result = test_one_video(os.path.join(tmp_path, h264_vid_fname), reload=False, visualize=args.no_show,
    #                         params_in=params)
    # print(f"Zone {args.detect_zone} is done")
    # print()
