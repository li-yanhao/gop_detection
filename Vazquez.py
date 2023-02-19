import numpy as np
from skimage.util.shape import view_as_blocks
import os
import glob
import cv2
import matplotlib.pyplot as plt
import mplcursors
import torch
# epsilon = 1e-5


class Vazquez:
    def __init__(self):
        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None

        self.detected_result = {}

        self.S_arr = None
        self.I_arr = None
        self.P = None

    def load_from_frames(self, fnames, max_num=10000):
        fnames = np.array(fnames)

        frame_types = np.array([fname.split(".")[-2][-1] for fname in fnames])
        stream_nums = np.array([int(fname.split("_")[-3][1:]) for fname in fnames])
        display_nums = np.array([int(fname.split("_")[-2][1:]) for fname in fnames])

        S_arr = []
        I_arr = []
        for fname in fnames:
            img_type = cv2.imread(fname).reshape(-1)
            num_SMB = (img_type == 3).sum() / 256
            num_IMB = (img_type == 1).sum() / 256
            S_arr.append(num_SMB)
            I_arr.append(num_IMB)
        self.S_arr = np.array(S_arr)
        self.I_arr = np.array(I_arr)

        sorted_indices = np.argsort(display_nums)

        self.display_nums = display_nums[sorted_indices]
        self.frame_types = frame_types[sorted_indices]
        self.stream_nums = stream_nums[sorted_indices]
        self.I_arr = self.I_arr[sorted_indices]
        self.S_arr = self.S_arr[sorted_indices]

    def preprocess(self):
        for i in range(len(self.frame_types)):
            if self.frame_types[i] == 'I':
                if i == 0:
                    self.I_arr[i] = self.I_arr[i + 1]
                elif i == len(self.frame_types) - 1:
                    self.I_arr[i] = self.I_arr[i - 1]
                else:
                    self.I_arr[i] = (self.I_arr[i + 1] + self.I_arr[i - 1]) / 2

        P = []
        for n in range(1, len(self.I_arr) - 1):
            if self.I_arr[n - 1] < self.I_arr[n] and \
                    self.I_arr[n + 1] < self.I_arr[n] and \
                    self.S_arr[n - 1] > self.S_arr[n] and \
                    self.S_arr[n + 1] > self.S_arr[n]:
                P.append(n)
        self.P = np.array(P)

        E_arr = np.zeros(len(self.I_arr))
        E_arr[1:-1] = np.abs((self.I_arr[1:-1] - self.I_arr[0:-2]) * (self.S_arr[1:-1] - self.S_arr[0:-2])) + \
            np.abs((self.I_arr[2:] - self.I_arr[1:-1]) * (self.S_arr[2:] - self.S_arr[1:-1]))

        self.V_arr = np.zeros(len(self.I_arr))
        # print("self.P", self.P)
        if len(self.P) > 0:
            self.V_arr[self.P] = E_arr[self.P]

    def load_from_ckpt(self, ckpt_fname):
        try:
            checkpoint = torch.load(ckpt_fname)
            self.V_arr = checkpoint["V_arr"]
            self.P = checkpoint["P"]
            return True
        except:
            return False
    
    def save_to_ckpt(self, ckpt_fname):
        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "V_arr": self.V_arr,
            "P": self.P
        }, ckpt_fname)
        return True
        

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

        if len(self.detected_result) > 0:
            for frame_num in self.detected_result["frame_nums"]:
                colors[frame_num] = 'cyan'

        # plt.text(x=2, y=self.residuals.max(), s=f"periodicity={p} \noffset={b} \nNFA={NFA}",
        #          horizontalalignment='center',
        #          verticalalignment='center',
        #          ha='left', va='top')

        display_nums = np.arange(len(self.V_arr))
        bars = ax.bar(display_nums, self.V_arr, color=colors)


        plt.xlabel('frame number i')
        plt.ylabel('v(i)')

        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
        @cursor.connect("add")
        def on_add(sel):
            x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            sel.annotation.set(text=f"num={round(x)}, v(i)={height}",
                               position=(0, 20), anncoords="offset points")
            sel.annotation.xy = (x, y + height)

        if save_fname is not None:
            plt.savefig(save_fname, bbox_inches='tight')

        plt.show()

    def detect_periodic_signal(self):
        # P = np.where(self.S_PRED > 0)[0]
        # T = len(self.S_PRED)

        C = set()
        # print(self.P)
        for i in range(len(self.P)):
            n1 = self.P[i]
            for j in range(i + 1, len(self.P)):
                n2 = self.P[j]
                c = np.gcd(n1, n2)
                if c >= 2 and c < len(self.V_arr):
                    C.add(c)
        if len(C) == 0:
            if len(self.P) == 1:
                C.add(self.P[0])
            else:
                C.add(1)

        C = np.sort(np.array(list(C)))
        
        phi_arr = []
        for i in range(len(C)):
            c = C[i]
            phi = self.compute_phi(c)
            phi_arr.append(phi)
        idx_best = np.argmax(phi_arr)

        phi_max = phi_arr[idx_best]
        T_phi = -np.inf # TODO: how to set?

        if phi_max > T_phi:
            G1 = C[idx_best]
            self.detected_result["G1"] = G1
        else:
            G1 = None
            phi_max = None

        return G1, phi_max

    def compute_phi(self, c):
        indices = np.arange(0, len(self.V_arr), c)

        phi1 = self.V_arr[np.intersect1d(indices, self.P)].sum()

        beta = 0.1 * np.max(self.V_arr)
        phi2 = len(np.setdiff1d(indices, self.P)) * beta

        energy_arr = []
        for z in range(1, c):
            energy = self.V_arr[::z].sum()
            energy_arr.append(energy)
        if len(energy_arr) == 0:
            phi3 = 0
        else:
            phi3 = np.max(np.array(energy_arr))

        return phi1 - phi2 - phi3


        # C = np.array(list(C))
        # phi_arr = np.zeros_like(C).astype(float)
        #
        # for i in range(len(C)):
        #     c = C[i]
        #     res = compute_phi(self.S_PRED, c, P)
        #     phi_arr[i] = res
        #
        # GOP = C[np.argmax(phi_arr)]
        # print("The estimated GOP is", GOP)


# def compute_phi()


def test():
    root = "/Users/yli/phd/video_processing/gop_detection/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgMB*.png"))

    analyzer = Vazquez()
    analyzer.load_from_frames(fnames, max_num=1000)
    analyzer.preprocess()
    analyzer.visualize()
    GOP = analyzer.detect_periodic_signal()
    print("GOP:", GOP)


if __name__ == "__main__":
    test()
