import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import mplcursors
from scipy import stats
# from skimage.util.shape import view_as_blocks
import matplotlib.patches as mpatches
import torch

class StreamAnalyzer:
    def __init__(self, epsilon=1, d=3, start_at_0=False, space="Y", max_num=-1):
        self.residuals_Y = None
        self.residuals_U = None
        self.residuals_V = None

        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None
        self.detected_result = None
        self.valid_sequence_mask = None

        self.epsilon = epsilon
        self.d = d
        self.start_at_0 = start_at_0
        self.space = space

        self.max_num = max_num if max_num > 0 else 100000

    def load_from_frames(self, fnames, space="Y"):
        fnames = np.array(fnames)

        self.frame_types = np.array([fname.split(".")[-2][-1] for fname in fnames])
        self.stream_nums = np.array([int(fname.split("_")[-3][1:]) for fname in fnames])
        self.display_nums = np.array([int(fname.split("_")[-2][1:]) for fname in fnames])
        residuals = []
        for fname in fnames:
            img_res = np.load(fname)
            img_res = np.squeeze(img_res)
            residual = compute_residual(img_res)
            residuals.append(residual)
        residuals = np.array(residuals)

        sorted_ind = np.argsort(self.display_nums)

        self.frame_types = self.frame_types[sorted_ind]
        self.stream_nums = self.stream_nums[sorted_ind]
        self.display_nums = self.display_nums[sorted_ind]
        if space == "Y":
            self.residuals_Y = residuals[sorted_ind]
        if space == "U":
            self.residuals_U = residuals[sorted_ind]
        if space == "V":
            self.residuals_V = residuals[sorted_ind]

    def preprocess(self):
        if self.space == "Y":
            self.residuals = self.residuals_Y.copy()
        elif self.space == "U":
            self.residuals = self.residuals_U.copy()
        elif self.space == "V":
            self.residuals = self.residuals_V.copy()
        elif self.space == "YUV":
            self.residuals = self.residuals_Y + self.residuals_U + self.residuals_V
            

        self.valid_peak_mask = np.zeros(len(self.residuals), dtype=bool)

        # a map indicating that the current i-th signal is related to the map[i]-th signal which is a peak
        self.map_to_peak_pos = np.zeros(len(self.residuals), dtype=int) - 1

        P_indices = np.where(self.frame_types == 'P')[0]
        for pivot in P_indices:
            # BETA: a P-frame with prediction residual based on I frame usually has higher residual than other P frames
            # if self.frame_types[pivot - 1] == 'I':
            #     continue

            delta = -1
            count = 0
            while count < self.d and pivot + delta >= 0:
                if self.frame_types[pivot + delta] == 'P':
                    if self.residuals[pivot] < self.residuals[pivot + delta]:
                        break
                    else:
                        count += 1
                delta -= 1

            if count < self.d:
                continue

            # compare the pivot with residuals on the right side
            delta = 1
            count = 0
            while count < self.d and pivot + delta < len(self.residuals):
                if self.frame_types[pivot + delta] == 'P':
                    if self.residuals[pivot] < self.residuals[pivot + delta]:
                        break
                    else:
                        count += 1
                delta += 1

            if count == self.d:
                self.valid_peak_mask[pivot] = True
                self.map_to_peak_pos[pivot] = pivot

                pos = pivot - 1
                while self.frame_types[pos] == 'B':
                    self.valid_peak_mask[pos] = True
                    self.map_to_peak_pos[pos] = pivot
                    pos -= 1
        # print(np.where(self.valid_peak_mask)[0])
        # print(self.map_to_peak_pos)

        # I frames cover the I-P residual peaks. An I index and its previous B indice until 
        # its previous P index are ignored.
        I_indices = np.where(self.frame_types == 'I')[0]
        self.valid_sequence_mask = np.ones(len(self.residuals), dtype=bool)
        for pos in I_indices:
            while pos >= 0 and self.frame_types[pos] != 'P':
                self.valid_sequence_mask[pos] = False
                pos -= 1

    def load_from_ckpt(self, ckpt_fname):
        try:
            checkpoint = torch.load(ckpt_fname)
            self.residuals_Y = checkpoint["residuals_Y"]
            self.residuals_U = checkpoint["residuals_U"]
            self.residuals_V = checkpoint["residuals_V"]
            self.frame_types = checkpoint["frame_types"]
            # self.valid_peak_mask = checkpoint["valid_peak_mask"]
            # self.valid_sequence_mask = checkpoint["valid_sequence_mask"]
            # self.map_to_peak_pos = checkpoint["map_to_peak_pos"]

            return True
        except:
            return False

    def save_to_ckpt(self, ckpt_fname):
        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "residuals_Y": self.residuals_Y,
            "residuals_U": self.residuals_U,
            "residuals_V": self.residuals_V,
            "frame_types": self.frame_types,
            # "valid_peak_mask": self.valid_peak_mask,
            # "map_to_peak_pos": self.map_to_peak_pos,
            # "valid_sequence_mask": self.valid_sequence_mask
        }, ckpt_fname)
        return True


    def visualize(self, save_fname=None):
        fig, ax = plt.subplots(figsize=(20, 5))

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

        # use cyan color for detected periodic residuals
        if self.detected_result is not None:
            for i in self.detected_result[3]:
                colors[i] = 'cyan'
                plt.text(x=i, y=self.residuals[i]+0.1, s=str(i),
                         horizontalalignment='center',
                         verticalalignment='center')

            p, b, NFA, _ = self.detected_result
            plt.text(x=2, y=self.residuals.max(), s=f"periodicity={p} \noffset={b} \nNFA={NFA}",
                     horizontalalignment='center',
                     verticalalignment='center',
                     ha='left', va='top')

        bars = ax.bar(self.display_nums, self.residuals, color=colors)

        color_patches = []
        color_patches.append(mpatches.Patch(color='red', label='I frame'))
        color_patches.append(mpatches.Patch(color='blue', label='P frame'))
        color_patches.append(mpatches.Patch(color='green', label='B frame'))
        if self.detected_result is not None:
            color_patches.append(mpatches.Patch(color='cyan', label='detected change frame'))
        ax.legend(handles=color_patches)

        plt.xlabel('frame number')
        plt.ylabel('residual')
        plt.title('Frame residual')

        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
        @cursor.connect("add")
        def on_add(sel):
            x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            sel.annotation.set(text=f"num={round(x)}, res={height:.2f} \ntype={self.frame_types[sel.index]}",
                               position=(0, 20), anncoords="offset points")
            sel.annotation.xy = (x, y + height)

        # plt.xticks(frame_numbers)
        # plt.yticks(np.arange(0, residuals.max(), 10))

        if save_fname is not None:
            plt.savefig(save_fname, bbox_inches='tight')

        plt.show()

    def compute_NFA(self, pi, bij, d, N_test):
        """ Compute the NFA of a candidate (pi, bij)

        :param pi: the tested period
        :param bij: the tested offset
        :param d: the range of neighborhood to valid a peak residual
        :return:
        """
        n = len(self.residuals)

        # the candidate (pi, bi) has periodic indices within the range [d, n-d)
        positions = np.arange((bij - d) % pi + d, n - d, pi)

        k = (self.valid_peak_mask[positions]).sum()

        length_sequence = self.valid_sequence_mask[positions].sum()
        prob = 1 / (2 * d + 1)

        if length_sequence > 0:
            NFA = stats.binom.sf(k - 0.1, length_sequence, prob) * N_test
        else:
            NFA = np.inf

        # if pi == 120:
        #     print("positions", positions)
        #     print("self.frame_types[positions]", self.frame_types[positions])
        #     print(f"pi={pi}, bij={bij}, k={k}, len={length_sequence}, NFA={NFA}")

        positions_valid = self.map_to_peak_pos[positions]
        positions_valid = positions_valid[positions_valid >= 0]
        return NFA, positions_valid

    def detect_periodic_signal(self):
        """ Compute the NFA of a periodic sequence starting at qi with spacing of pi.

        :param d: the range of neighborhood to valid a peak residual.
        :return: the detected periodicity
        """

        assert self.d >= 1, "the range of neighborhood must be larger than 1"

        self.residuals = self.residuals[:self.max_num]
        self.frame_types = self.frame_types[:self.max_num]
        
        # print(self.residuals_U)
        # print(self.frame_types)

        detected_results = []
        for p in range(2 * self.d, len(self.residuals) // 2):
            if self.start_at_0:
                b_candidates = [0]
                # N_test = len(self.residuals - 1) // 2 - 2 * self.d
                N_test = p * (len(self.residuals - 1) // 2 - 2 * self.d)
            else:
                b_candidates = np.arange(0, p)
                N_test = p * (len(self.residuals - 1) // 2 - 2 * self.d)

            for b in b_candidates:
                NFA, tested_indices = self.compute_NFA(p, b, self.d, N_test)
                if NFA < self.epsilon:
                    print(f"periodicity={p} offset={b} NFA={NFA}")
                    detected_results.append((p, b, NFA, tested_indices))

        if len(detected_results) == 0:
            print("No periodic residual sequence is detected.")
            return -1, np.inf

        best_NFA = self.epsilon
        best_i = 0
        for i in range(len(detected_results)):
            if best_NFA > detected_results[i][2]:
                best_NFA = detected_results[i][2]
                best_i = i

        self.detected_result = detected_results[best_i]

        # return the periodicity
        return self.detected_result[0], best_NFA


def compute_residual(img_res):
    """
    :param img_res: of size (H, W)
    :return: mean residual
    """
    img_res = np.abs(img_res)
    # block_reduce(img_res, block_size=(8, 8), func=np.mean)
    return np.mean(img_res)


def main():
    root = "/Users/yli/phd/video_processing/gop_detection/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgY_s*.npy"))

    d = 3
    analyzer = StreamAnalyzer(epsilon=10, start_at_0=False)
    analyzer.load_from_frames(fnames, max_num=10000)

    vis_fname = None

    # vis_fname = "residuals_Y_c2.eps"
    analyzer.visualize(vis_fname)

    import time
    start = time.time()
    analyzer.preprocess(d)
    gop, NFA = analyzer.detect_periodic_signal(d)
    end = time.time()

    # vis_fname = "detection_Y_c2.eps"
    analyzer.visualize(vis_fname)

    print("Estimated GOP:", gop)
    print("Elapsed time:", end - start)

if __name__ == '__main__':
    main()
