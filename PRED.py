import numpy as np
from skimage.util.shape import view_as_blocks
import os
import glob
import matplotlib.pyplot as plt
import mplcursors
import torch


epsilon = 1e-5

class PRED:
    def __init__(self):
        self.residuals = None
        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None
        self.detected_result = {}

        self.S_PRED = None

    def load_from_frames(self, fnames, max_num=10000):
        fnames = np.array(fnames)

        self.frame_types = np.array([fname.split(".")[-2][-1] for fname in fnames])
        self.stream_nums = np.array([int(fname.split("_")[-3][1:]) for fname in fnames])
        self.display_nums = np.array([int(fname.split("_")[-2][1:]) for fname in fnames])


        PRED_arr = []
        for fname in fnames:
            img_res = np.load(fname).squeeze()
            PRED = compute_PRED(img_res)
            PRED_arr.append(PRED)


        PRED_arr = np.array(PRED_arr)

        # PRED_arr[::31, :] = PRED_arr[1::31, :].copy()

        S_JSD_arr = []

        # first element
        S_JSD_arr.append(JSD(PRED_arr[1], PRED_arr[0]) + JSD(PRED_arr[0], PRED_arr[1]))

        for t in range(len(PRED_arr) - 2):
            S_JDS = JSD(PRED_arr[t+1], PRED_arr[t]) + JSD(PRED_arr[t+1], PRED_arr[t+2])
            S_JSD_arr.append(S_JDS)

        # last element
        S_JSD_arr.append(JSD(PRED_arr[-1], PRED_arr[-2]) + JSD(PRED_arr[-2], PRED_arr[-1]))

        S_JSD_arr = np.array(S_JSD_arr)
        S_MF_arr = median_filter(S_JSD_arr)
        self.S_PRED = np.maximum(S_JSD_arr - S_MF_arr, 0)

        

        sorted_indices = np.argsort(self.display_nums)
        self.S_PRED = self.S_PRED[sorted_indices]
        self.display_nums = self.display_nums[sorted_indices]
        self.stream_nums = self.stream_nums[sorted_indices]
        self.frame_types = self.frame_types[sorted_indices]

        # suppress I frames
        for t in np.where(self.frame_types == 'I')[0]:
            if t == 0:
                self.S_PRED[t] = self.S_PRED[t+1]
            elif t == len(self.S_PRED) - 1:
                self.S_PRED[t] = self.S_PRED[t-1]
            else:
                self.S_PRED[t] = (self.S_PRED[t-1] + self.S_PRED[t+1]) / 2

    def load_from_ckpt(self, ckpt_fname):
        try:
            checkpoint = torch.load(ckpt_fname)
            self.S_PRED = checkpoint["S_PRED"]
            return True
        except:
            return False
    
    def save_to_ckpt(self, ckpt_fname):
        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "S_PRED": self.S_PRED
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

        display_nums = np.arange(len(self.S_PRED))
        bars = ax.bar(display_nums, self.S_PRED, color=colors)


        plt.xlabel('frame number')
        plt.ylabel('String of data bytes')
        # plt.title('Sr')

        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
        @cursor.connect("add")
        def on_add(sel):
            x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            sel.annotation.set(text=f"num={round(x)}, PRED={height}",
                               position=(0, 20), anncoords="offset points")
            sel.annotation.xy = (x, y + height)

        if save_fname is not None:
            plt.savefig(save_fname, bbox_inches='tight')

        plt.show()

    def detect_periodic_signal(self):
        P = np.where(self.S_PRED > 0)[0]
        T = len(self.S_PRED)

        # print("self.S_PRED: ", self.S_PRED)
        # print("P:", P)
        C = set()
        for i in range(len(P)):
            n1 = P[i]
            for j in range(i + 1, len(P)):
                n2 = P[j]
                c = np.gcd(n1, n2)
                if c >= 2 and c < T / 5:
                    C.add(c)

        C = np.sort(np.array(list(C)))
        phi_arr = np.zeros_like(C).astype(float)

        for i in range(len(C)):
            c = C[i]
            small_candidates = C[:i]
            res = compute_phi(self.S_PRED, c, P)
            phi_arr[i] = res

            # print(f"c={c}, phi={res}")

        G1 = C[np.argmax(phi_arr)]
        # print("The estimated G1 is", G1)

        threshold = -100000 # TODO: what's the value?

        if phi_arr.max() > threshold:
            self.detected_result["frame_nums"] = np.arange(0, len(self.S_PRED), G1)
            self.detected_result["G1"] = np.arange(0, len(self.S_PRED), G1)
            return G1, phi_arr.max()
        else:
            return None, None



def compute_PRED(img_res):
    """
    :param img_res: a residual image, of size (H, W)
    :return: a PRED feature of size (11,) counting the quantized values in [0, 10]
    """
    blocks = view_as_blocks(img_res, block_shape=(4,4))
    N = blocks.shape[0] * blocks.shape[1]
    R_arr = np.mean(np.abs(blocks), axis=(-1,-2))
    R_arr = np.floor(np.round(R_arr) + 0.5)
    R_arr = np.clip(R_arr, 0, 10).reshape(-1)
    PRED = np.histogram(R_arr, bins=11, range=(-0.5, 10.5))[0]
    PRED = PRED / N

    return PRED


def KLD(p, q):
    """
    :param p: a vector of size (n,)
    :param q: the other vector of size (n,)
    :return:
    """
    return np.sum(p * (np.log(p+epsilon) - np.log(q+epsilon)))


def JSD(p, q):
    """
    :param p: a PRED of size (n,)
    :param q: the other PRED of size (n,)
    :return:
    """
    m = (p + q) / 2
    return (KLD(p, m) + KLD(q, m)) / 2


def median_filter(s):
    """
    :param s: a sequence of S_JSD(t), of size (n,)
    :return:
    """
    res = s.copy()
    res[1:-1] = np.median([s[0:-2], s[1:-1], s[2:]], axis=0)
    return res

def compute_phi(S, c, P) -> float:
    """
    :param S: the S_PRED, of size (T,)
    :param c: candidate c
    :param P: indices of peaks in S_PRED

    :return:
    """
    T = len(S)
    indices = np.arange(0, T, c)
    phi_1 = np.sum(S[np.intersect1d(indices, P)])

    beta = np.max(S) * 0.3
    phi_2 = beta * (len(indices) - len(np.intersect1d(indices, P)))

    phi_3_arr = []
    for z in range(1, c):
        indices = np.arange(0, T, z)
        phi_3 = np.sum(S[indices])
        phi_3_arr.append(phi_3)
    
    if len(phi_3_arr) == 0:
        phi_3 = 0
    else:
        phi_3 = np.max(np.array(phi_3_arr))

    # print(f"c={c}, phi_1={phi_1}, phi_2={phi_2}, phi_3={phi_3}")

    return phi_1 - phi_2 - phi_3


def test():
    root = "/Users/yli/phd/video_processing/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgU_s*.npy"))

    analyzer = PRED()
    analyzer.load_from_frames(fnames, max_num=1000)
    analyzer.visualize()
    GOP = analyzer.detect_periodic_signal()
    analyzer.visualize(None)



if __name__ == "__main__":
    test()
