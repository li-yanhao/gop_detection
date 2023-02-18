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
    def __init__(self):
        self.residuals = None
        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None
        self.detected_result = None

        self.epsilon = 1

    def load_from_frames(self, fnames, max_num=10000):
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
        self.residuals = np.array(residuals)

        sorted_ind = np.argsort(self.display_nums)

        self.frame_types = self.frame_types[sorted_ind][:max_num]
        self.stream_nums = self.stream_nums[sorted_ind][:max_num]
        self.display_nums = self.display_nums[sorted_ind][:max_num]
        self.residuals = self.residuals[sorted_ind][:max_num]

    def preprocess(self, d):
        self.valid_peak_mask = np.zeros(len(self.residuals), dtype=bool)

        # a map indicating that the current i-th signal is related to the map[i]-th signal which is a peak
        self.map_to_peak_pos = np.zeros(len(self.residuals), dtype=int)

        P_indices = np.where(self.frame_types == 'P')[0]
        for pivot in P_indices:
            delta = -1
            count = 0
            while count < d and pivot + delta >= 0:
                if self.frame_types[pivot + delta] == 'P':
                    if self.residuals[pivot] < self.residuals[pivot + delta]:
                        break
                    else:
                        count += 1
                delta -= 1

            if count < d:
                continue

            # compare the pivot with residuals on the right side
            delta = 1
            count = 0
            while count < d and pivot + delta < len(self.residuals):
                if self.frame_types[pivot + delta] == 'P':
                    if self.residuals[pivot] < self.residuals[pivot + delta]:
                        break
                    else:
                        count += 1
                delta += 1

            if count == d:
                self.valid_peak_mask[pivot] = True
                self.map_to_peak_pos[pivot] = pivot

                pos = pivot - 1
                while self.frame_types[pos] != 'P':
                    self.valid_peak_mask[pos] = True
                    pos -= 1

        print(np.where(self.valid_peak_mask)[0])

    def load_from_ckpt(self, ckpt_fname):
        try:
            checkpoint = torch.load(ckpt_fname)
            self.residuals = checkpoint["residuals"]
            self.frame_types = checkpoint["frame_types"]
            return True
        except:
            return False

    def save_to_ckpt(self, ckpt_fname):
        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "residuals": self.residuals,
            "frame_types": self.frame_types
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

    def compute_NFA(self, pi, bij, d):
        """ Compute the NFA of a candidate (pi, bij)

        :param pi: the tested periodicity
        :param bij: the tested offset
        :param d: the range of neighborhood to valid a peak residual
        :return:
        """
        n = len(self.residuals)

        # the candidate (pi, bi) has periodic indices within the range [d, n-d)
        positions = np.arange((bij - d) % pi + d, n - d, pi)

        k = (self.valid_peak_mask[positions]).sum()
        prob = 1 / (2 * d + 1)
        NFA = stats.binom.sf(k - 0.5, len(positions), prob) * pi * (( n - 1) // 2 - 2 * d)

        positions_valid = self.map_to_peak_pos[positions]
        return NFA, positions_valid

    def detect_periodic_signal(self, d=2):
        """ Compute the NFA of a periodic sequence starting at qi with spacing of pi.

        :param d: the range of neighborhood to valid a peak residual.
        :return: the detected periodicity
        """

        assert d >= 1, "the range of neighborhood must be larger than 1"

        detected_results = []
        for p in range(2 * d, len(self.residuals) // 2):
            for b in range(0, p):
                NFA, tested_indices = self.compute_NFA(p, b, d)
                if NFA < self.epsilon:
                    print(f"periodicity={p} offset={b} NFA={NFA}")
                    detected_results.append((p, b, NFA, tested_indices))

        if len(detected_results) == 0:
            print("No periodic residual sequence is detected.")
            return None

        best_NFA = self.epsilon
        best_i = 0
        for i in range(len(detected_results)):
            if best_NFA > detected_results[i][2]:
                best_NFA = detected_results[i][2]
                best_i = i

        self.detected_result = detected_results[best_i]

        # return the periodicity
        return self.detected_result[0]


def compute_residual(img_res):
    """
    :param img_res: of size (H, W)
    :return: mean residual
    """
    img_res = np.abs(img_res)
    # block_reduce(img_res, block_size=(8, 8), func=np.mean)
    return np.mean(img_res[0:, :])


def main():
    root = "/Users/yli/phd/video_processing/gop_detection/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgU_s*.npy"))

    d = 2
    analyzer = StreamAnalyzer()
    analyzer.load_from_frames(fnames, max_num=10000)

    vis_fname = None

    # vis_fname = "residuals_Y_c2.eps"
    # analyzer.visualize(vis_fname)

    import time
    start = time.time()
    analyzer.preprocess(d)
    gop = analyzer.detect_periodic_signal(d)
    end = time.time()

    # vis_fname = "detection_Y_c2.eps"
    analyzer.visualize(vis_fname)

    print("Detected result:", gop)
    print("Elapsed time:", end - start)


if __name__ == '__main__':
    main()
