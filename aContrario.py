import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import mplcursors
from scipy import stats
import matplotlib.patches as mpatches
import pandas as pd

import plotly.graph_objects as go
import pickle


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
        self.stream_nums = np.array([int(fname.split("_")[-2][1:]) for fname in fnames])
        self.display_nums = np.array([int(fname.split("_")[-3][1:]) for fname in fnames])
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
            checkpoint = pickle.load( open( ckpt_fname, "rb" ) )
            self.residuals_Y = checkpoint["residuals_Y"]
            self.residuals_U = checkpoint["residuals_U"]
            self.residuals_V = checkpoint["residuals_V"]
            self.frame_types = checkpoint["frame_types"]

            return True
        except:
            return False

    def save_to_ckpt(self, ckpt_fname):

        dir = os.path.dirname(ckpt_fname)
        os.makedirs(dir, exist_ok=True)
        saved_contents = {
            "residuals_Y": self.residuals_Y,
            "residuals_U": self.residuals_U,
            "residuals_V": self.residuals_V,
            "frame_types": self.frame_types,
        }

        pickle.dump(saved_contents, open( ckpt_fname, "wb" ))

        return True


    def visualize_plt(self, save_fname=None):
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
            plt.text(x=2, y=self.residuals.max(), s=f"period={p} \noffset={b} \nNFA={NFA}",
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

        if save_fname is not None:
            plt.savefig(save_fname, bbox_inches='tight')

        plt.show()

    def visualize(self, save_fname:str=None):
        color_map = {
            "I": "red",
            "P": "blue",
            "B": "green"
        }

        df = pd.DataFrame({
            "frame_type": self.frame_types,
            "frame_number": self.display_nums,
            "residuals": self.residuals,
            "color": [color_map[type] for type in self.frame_types]
        })

        if self.detected_result is not None:
            for i in self.detected_result[3]:
                df.at[i, "color"] = "cyan"

        fig = go.Figure()

        hover_template = "frame number: %{x:d} <br>" \
                         "residual: %{y:.3f} <br>" \
                         "frame type: %{customdata}"

        I_df = df.query("color == 'red'")
        fig.add_trace(go.Bar(
            x=I_df["frame_number"],
            y=I_df["residuals"],
            name='I frame',
            marker_color="red",
            customdata=I_df["frame_type"],
            hovertemplate=hover_template
        ))

        P_df = df.query("color == 'blue'")
        fig.add_trace(go.Bar(
            x=P_df["frame_number"],
            y=P_df["residuals"],
            name='P frame',
            marker_color="blue",
            customdata=P_df["frame_type"],
            hovertemplate=hover_template
        ))

        B_df = df.query("color == 'green'")
        fig.add_trace(go.Bar(
            x=B_df["frame_number"],
            y=B_df["residuals"],
            name='B frame',
            marker_color="green",
            customdata=B_df["frame_type"],
            hovertemplate=hover_template
        ))

        abnormal_df = df.query("color == 'cyan'")
        fig.add_trace(go.Bar(
            x=abnormal_df["frame_number"],
            y=abnormal_df["residuals"],
            name='P frame',
            marker_color="blue",
            showlegend=False,
            customdata=abnormal_df["frame_type"],
            hovertemplate=hover_template
        ))

        buttons = list([
            dict(
                args=[{"marker.color": ["red", "blue", "green", "blue"],
                       "name": ["I frame", "P frame", "B frame", "P frame"],
                       "showlegend": [True, True, True, False]},
                      [0, 1, 2, 3]],
                label="raw",
                method="restyle"
            ),
            dict(
                # args=[{"marker.color": [df["color"]]}, [0]],
                args=[{"marker.color": ["red", "blue", "green", "cyan"],
                       "name": ["I frame", "P frame", "B frame", "peak"],
                       "showlegend": [True, True, True, True]},
                      [0, 1, 2, 3]],
                label="detection",
                method="restyle"
            )
        ])

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=buttons,
                    direction="left",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.2,
                    yanchor="top"
                ),
            ]
        )

        # fig.update_layout(
            # autosize=False,
            # height=500,
            # margin=dict(
            #     l=50,
            #     r=50,
            #     b=100,
            #     t=100,
            #     pad=4
            # ),
            # paper_bgcolor="LightSteelBlue",
        # )

        fig.update_layout(
            autosize=False,
            width=1300,
            height=500,
            # font_family="Courier New",
            # font_color="blue",
            # title_font_family="Times New Roman",
            # title_font_color="red",
            # legend_title_font_color="green"
            font_size=14
        )

        fig.update_layout(
            xaxis={'title': 'frame number'},
            yaxis={'title': 'prediction residual'}
        )

        # fig.show()
        if save_fname.endswith(".html"):
            fig.write_html(save_fname, include_plotlyjs="cdn")

    def compute_NFA(self, pi, bij, d, N_test):
        """ Compute the NFA of a candidate (pi, bij)

        :param pi: the tested period
        :param bij: the tested offset
        :param d: the range of neighborhood to valid a peak residual
        :param N_test: the number of tests
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

        positions_valid = self.map_to_peak_pos[positions]
        positions_valid = positions_valid[positions_valid >= 0]
        return NFA, positions_valid

    def detect_periodic_signal(self):
        """ Compute the NFA of a periodic sequence starting at qi with spacing of pi.

        :return:
            [0] the detected periodicity
            [1] the NFA of the detection
        """

        assert self.d >= 1, "the range of neighborhood must be larger than 1"

        self.residuals = self.residuals[:self.max_num]
        self.frame_types = self.frame_types[:self.max_num]
        
        detected_results = []
        for p in range(2 * self.d, len(self.residuals) // 2):
            if self.start_at_0:
                b_candidates = [0]
                N_test = p * (len(self.residuals - 1) // 2 - 2 * self.d)
            else:
                b_candidates = np.arange(0, p)
                N_test = p * (len(self.residuals - 1) // 2 - 2 * self.d)

            for b in b_candidates:
                NFA, tested_indices = self.compute_NFA(p, b, self.d, N_test)
                if NFA < self.epsilon:
                    # print(f"periodicity={p} offset={b} NFA={NFA}")
                    detected_results.append((p, b, NFA, tested_indices))

        if len(detected_results) == 0:
            # print("No periodic residual sequence is detected.")
            return -1, np.inf

        print("Detected candidates are:")
        best_NFA = self.epsilon
        best_i = 0
        for i in range(len(detected_results)):
            p, b, NFA, _ = detected_results[i]
            print(f"period={p} offset={b} NFA={NFA}")
            if best_NFA > NFA:
                best_NFA = NFA
                best_i = i
        print()
        self.detected_result = detected_results[best_i]

        # return the periodicity
        return self.detected_result[0], best_NFA


def compute_residual(img_res):
    """
    :param img_res: of size (H, W)
    :return: mean residual
    """
    img_res = np.abs(img_res)
    return np.mean(img_res)


def main():
    root = "/Users/yli/phd/video_processing/gop_detection/jm_16.1/bin"
    fnames = glob.glob(os.path.join(root, "imgY_s*.npy"))

    d = 3
    analyzer = StreamAnalyzer(epsilon=10, start_at_0=False)
    analyzer.load_from_frames(fnames, space='Y')

    vis_fname = None

    analyzer.visualize(vis_fname)

    import time
    start = time.time()
    analyzer.preprocess()
    gop, NFA = analyzer.detect_periodic_signal()
    end = time.time()

    # vis_fname = "detection_Y_c2.eps"
    analyzer.visualize(vis_fname)

    print("Estimated GOP", gop)
    print("NFA:", NFA)

    print("Elapsed time:", end - start)

if __name__ == '__main__':
    main()
