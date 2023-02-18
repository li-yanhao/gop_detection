import subprocess
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import mplcursors
import cv2
import torch



class YAO:
    def __init__(self):
        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None
        self.detected_result = {}

        self.SODBs = None

    def load_SODB(self, vid_fname, ffprobe_exe="ffprobe"):
        """
        :param vid_fname: input video filename
        :return: a list of packet sizes in bytes
        """
        command = f"{ffprobe_exe} -show_frames {vid_fname} | grep pkt_size"
        # print(command)

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = result.stdout
        pkt_sizes = [int(s.split("=")[-1]) for s in result.split("\n") if s != ""]
        # print("pkt_sizes:", pkt_sizes)
        # print("len pkt_sizes:", len(pkt_sizes))

        self.SODBs = np.array(pkt_sizes)

    def load_from_frames(self, fnames, max_num=10000):
        fnames = np.array(fnames)

        self.frame_types = np.array([fname.split(".")[-2][-1] for fname in fnames])
        self.stream_nums = np.array([int(fname.split("_")[-3][1:]) for fname in fnames])
        self.display_nums = np.array([int(fname.split("_")[-2][1:]) for fname in fnames])

        SMB_features = []
        for fname in fnames:
            # SMB_feature = 1 / (cv2.imread(fname).mean() + 1e-5)
            SMB_feature = 1 / ((cv2.imread(fname) == 3).mean() + 1e-5) # 3 is the value for SMB
            SMB_features.append(SMB_feature)
        self.SMB_features = np.array(SMB_features)

        sorted_ind = np.argsort(self.display_nums)

        self.frame_types = self.frame_types[sorted_ind][:max_num]
        self.stream_nums = self.stream_nums[sorted_ind][:max_num]
        self.display_nums = self.display_nums[sorted_ind][:max_num]
        self.SODBs = self.SODBs[sorted_ind][:max_num]
        self.SMB_features = self.SMB_features[sorted_ind][:max_num]


    def visualize(self, save_fname=None):
        fig, ax = plt.subplots(figsize=(20, 5))
        #
        colors = []
        for type in self.frame_types:
            if type == 'I':
                colors.append('red')
            elif type == 'P':
                colors.append('blue')
            elif type == 'B':
                colors.append('green')
            else:
                raise Exception(f"Invalid type {type}")
        # print("colors", colors)

        if len(self.detected_result) > 0:
            for frame_num in self.detected_result["frame_nums"]:
                colors[frame_num] = 'cyan'

        # plt.text(x=2, y=self.residuals.max(), s=f"periodicity={p} \noffset={b} \nNFA={NFA}",
        #          horizontalalignment='center',
        #          verticalalignment='center',
        #          ha='left', va='top')

        display_nums = np.arange(len(self.SODBs))
        bars = ax.bar(display_nums, self.Et_features, color=colors)


        plt.xlabel('frame number')
        plt.ylabel('String of data bytes')
        # plt.title('Sr')

        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
        @cursor.connect("add")
        def on_add(sel):
            x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            sel.annotation.set(text=f"num={round(x)}, SODB={height}",
                               position=(0, 20), anncoords="offset points")
            sel.annotation.xy = (x, y + height)

        if save_fname is not None:
            plt.savefig(save_fname, bbox_inches='tight')

        plt.show()

    def preprocess(self):
        I_indices = np.where(self.frame_types == 'I')[0]
        for i in I_indices:
            if i == 0:
                self.SODBs[i] = self.SODBs[i + 1]
            elif i == len(self.SODBs) - 1:
                self.SODBs[i] = self.SODBs[i - 1]
            else:
                self.SODBs[i] = (self.SODBs[i - 1] + self.SODBs[i + 1]) / 2
        # print(I_indices)

        self.Et_features = self.SODBs * self.SMB_features
        self.Et_features[I_indices] = self.SODBs[I_indices] / self.SMB_features[I_indices]

    def detect_periodic_signal(self):
        T = len(self.Et_features)
        G_max = min(150, T // 10)
        G_candidates = np.arange(2, G_max+1)

        def Lambda(G):
            return self.Et_features[::G].mean()

        Lambda_candidates = [Lambda(m) for m in G_candidates]

        # for i in range(len(G_candidates)):
        #     print(f"G={G_candidates[i]}, Lambda={Lambda_candidates[i]}")

        idx_sorted = np.argsort(Lambda_candidates)
        idx_max1, idx_max2 = idx_sorted[-1], idx_sorted[-2]

        self.T_Lambda = -0.01 # ?
        if Lambda_candidates[idx_max1] - Lambda_candidates[idx_max2] > self.T_Lambda:
            G1 = G_candidates[idx_max1]
            self.detected_result["frame_nums"] = np.arange(0, len(self.Et_features), G1)
            return G1, Lambda_candidates[idx_max1]
        else:
            return None, None


    def load_from_ckpt(self, ckpt_fname):
        try:
            checkpoint = torch.load(ckpt_fname)
            self.Et_features = checkpoint["Et_features"]
            return True
        except:
            return False
    
    def save_to_ckpt(self, ckpt_fname):
        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "Et_features": self.Et_features
        }, ckpt_fname)
        return True
        


if __name__ == '__main__':
    # vid_fname = "/Users/yli/phd/deepfake/collection_deepfake/tom_cruise/tom_2.h264"
    vid_fname = "/Users/yli/phd/deepfake/collection_deepfake/real/office_c2.h264"

    root = "/Users/yli/phd/video_processing/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgSB_*.png"))

    analyzer = YAO()
    analyzer.load_SODB(vid_fname)
    analyzer.load_from_frames(fnames)
    analyzer.preprocess()


    analyzer.visualize()

    G1 = analyzer.detect_periodic_signal()
    print("Detected G1 = ", G1)

    analyzer.visualize()
