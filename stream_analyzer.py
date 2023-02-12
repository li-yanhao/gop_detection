import numpy as np
from skimage.measure import block_reduce
import os
import glob
import matplotlib.pyplot as plt
import mplcursors
import scipy
from skimage.util.shape import view_as_blocks
import matplotlib.patches as mpatches


class StreamAnalyzer:
    def __init__(self):
        self.residuals = None
        self.frame_types = None
        self.stream_nums = None
        self.display_nums = None
        self.detected_result = None

        self.epsilon = 1

    def load_residuals(self, fnames):
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

        self.frame_types = self.frame_types[sorted_ind]
        self.stream_nums = self.stream_nums[sorted_ind]
        self.display_nums = self.display_nums[sorted_ind]
        self.residuals = self.residuals[sorted_ind]

    def visualize(self):
        # fig = plt.figure(figsize=(20, 5))
        fig, ax = plt.subplots(figsize=(8, 6))

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

        bars = ax.bar(self.display_nums, self.residuals, color=colors)

        color_patches = []
        color_patches.append(mpatches.Patch(color='red', label='I frame'))
        color_patches.append(mpatches.Patch(color='blue', label='P frame'))
        color_patches.append(mpatches.Patch(color='green', label='B frame'))
        if self.detected_result is not None:
            color_patches.append(mpatches.Patch(color='cyan', label='detected frame'))
        ax.legend(handles=color_patches)


        plt.xlabel('frame number')
        plt.ylabel('residual')

        plt.title('Frame residual')

        # mplcursors.cursor(bars)

        cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)

        @cursor.connect("add")
        def on_add(sel):
            x, y, width, height = sel.artist[sel.index].get_bbox().bounds
            sel.annotation.set(text=f"num={round(x)}, res={height:.2f} \ntype={self.frame_types[sel.index]}",
                               position=(0, 20), anncoords="offset points")
            sel.annotation.xy = (x, y + height)

        # plt.xticks(frame_numbers)
        # plt.yticks(np.arange(0, residuals.max(), 10))

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
        positions_corrected = []
        for pos in positions:
            while self.frame_types[pos] != 'P':
                pos += 1
            positions_corrected.append(pos)
        positions_valid = []

        k = 0
        for pos in positions_corrected:
            delta = 1
            count = 0
            valid = True
            while count < d and pos - delta >= 0:
                if self.frame_types[pos - delta] == 'P':
                    count += 1
                    if self.residuals[pos] < self.residuals[pos - delta]:
                        valid = False
                        break
                delta += 1

            if not valid:
                continue

            delta = 1
            count = 0
            while count < d and pos + delta < n:
                if self.frame_types[pos + delta] == 'P':
                    count += 1
                    if self.residuals[pos] < self.residuals[pos + delta]:
                        valid = False
                        break
                delta += 1

            if valid:
                k += 1
                positions_valid.append(pos)

        prob = 1 / (2 * d + 1)
        NFA = scipy.stats.binom.sf(k - 0.5, len(positions), prob, loc=0) * pi * (( n - 1) // 2 - 2 * d)

        return NFA, positions_valid

    def detect_periodic_signal(self, d=2):
        """ Compute the NFA of a periodic sequence starting at qi with spacing of pi.

        :param d: the range of neighborhood to valid a peak residual.
        """
        detected_results = []
        for p in range(2 * d, len(self.residuals) // 2):
            for b in range(0, p):
                NFA, tested_indices = self.compute_NFA(p, b, d)
                if NFA < self.epsilon:
                    print(f"periodicity={p} offset={b} NFA={NFA}")
                    detected_results.append((p, b, NFA, tested_indices))

        if len(detected_results) == 0:
            return

        best_NFA = self.epsilon
        best_i = 0
        for i in range(len(detected_results)):
            if best_NFA > detected_results[i][2]:
                best_NFA = detected_results[i][2]
                best_i = i

        self.detected_result = detected_results[best_i]


def compute_residual(img_res):
    """
    :param img_res: of size (H, W)
    :return: mean residual
    """
    img_res = np.abs(img_res)
    # block_reduce(img_res, block_size=(8, 8), func=np.mean)
    return np.mean(img_res)


def main():
    root = "/Users/yli/phd/video_processing/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgU_s*.npy"))

    analyzer = StreamAnalyzer()
    analyzer.load_residuals(fnames)

    analyzer.visualize()
    analyzer.detect_periodic_signal(d=6)
    analyzer.visualize()


if __name__ == '__main__':
    main()
