import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc 


def load_gop_df_from_json(json_fname:str):
    json_file = open(json_fname, "r")
    data = json.load(json_file)

    raw_data = [
        [
            item["B1"], item["B2"], item["G1"], item["G2"],
            item["Chen"][0], item["Yao"][0], item["Vazquez"][0], item["aContrario"][0]
        ] 
    for item in data]

    df = pd.DataFrame(raw_data, 
        columns=["B1", "B2", "G1", "G2", "Chen", "Yao", "Vazquez", "aContrario"])
    
    return df


def load_score_df_from_json(json_fname:str):
    json_file = open(json_fname, "r")
    data = json.load(json_file)

    raw_data = [
        [
            item["B1"], item["B2"], item["G1"], item["G2"],
            item["Chen"][1], item["Yao"][1], item["Vazquez"][1], -item["aContrario"][1]
        ] 
    for item in data]

    df = pd.DataFrame(raw_data, 
        columns=["B1", "B2", "G1", "G2", "Chen", "Yao", "Vazquez", "aContrario"])
    
    return df


def load_gop_and_NFA_df_from_json(json_fname:str):
    json_file = open(json_fname, "r")
    data = json.load(json_file)

    raw_data = [
        [
            item["B1"], item["B2"], item["G1"], item["G2"],
            item["aContrario"][0], item["aContrario"][1]
        ] 
    for item in data]

    df = pd.DataFrame(raw_data, 
        columns=["B1", "B2", "G1", "G2", "aContrario", "NFA"])
    
    return df


# fname_c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c1_20230220_011016.json"
# fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c2_20230220_011026.json"

# fname_c1 = '/mnt/ddisk/yanhao/gop_detection/results_cbr_set1c1b_20230221_004420.json'

fname_c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c1_20230220_185913.json"
fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c2_20230220_185921.json"

# set3
# fname_c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230221_012003.json"
# fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230221_012008.json"

fname_set2c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c1_20230221_130650.json"
fname_set2c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c2_20230221_130658.json"


fname_set2c1b = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c1b_20230222_180210.json"
fname_set2c2b = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set2c2b_20230222_180217.json"

def GOP_accuracy():

    df = load_gop_df_from_json(fname_c2)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100

    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    # for G1 in GOP_c1_options:
    #     sub_df = df.query(f'G1 == {G1}')
    #     print(f"G1={G1}")
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            sub_df = df.query(f'B1 == {B1} and B2 == {B2}')
            for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
                if method != "aContrario":
                    continue
                acc = len(sub_df.query(f'{method} == G1')) / len(sub_df)
                # print(f"{method}: acc={acc:.2f}")
                print(f"{acc:.3f}")
            # print()



def GOP_accuracy_bframe():

    # df = load_gop_df_from_json(fname_c2)
    df = load_gop_and_NFA_df_from_json(fname_set2c2b)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100

    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    # for G1 in GOP_c1_options:
    #     sub_df = df.query(f'G1 == {G1}')
    #     print(f"G1={G1}")
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            sub_df = df.query(f'B1 == {B1} and B2 == {B2}')
            for method in ["aContrario"]:
                acc = len(sub_df.query(f'{method} == G1')) / len(sub_df)
                # print(f"{method}: acc={acc:.2f}")
                print(f"{acc:.3f}")
            # print()

def compute_auc(df, method) -> float:
    y_true = (df["B2"] > 0)
    y_score = df[method]
    auc = roc_auc_score(y_true, y_score)
    return auc 


def detection_AUC_old():
    df_c1 = load_score_df_from_json(fname_set3c1) # fname_c1, fname_set2c1
    df_c2 = load_score_df_from_json(fname_set3c2) # fname_c2, fname_set2c2

    df_c2 = df_c2.sample(n=len(df_c1) * 3, random_state=1)
    df = pd.concat([df_c1, df_c2])

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100
    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]
    methods = ["Chen", "Yao", "Vazquez", "aContrario"]

    # for B1 in B1_arr:
    #     results = []
    #     print(f"B1={B1}")
    # for B2 in B1_arr:
    #         results_B2 = []
    #         sub_df = df.query(f'B1 == {B1} and (B2 == {B2} or B2 == -1)')
            # print(f"B1={B1}, B2={B2}")
    # results = []
    # for G1 in GOP_c1_options:
        # g1_results = []
        # sub_df = df.query(f'G1 == {G1}')
        # print(f"G1={G1}")
    # for B1 in B1_arr:
        # sub_df1 = df_c1.query(f'B1 == {B1}')

        
    for B2 in B2_arr:
        
        # print(f"B1={B1}, B2={B2}")

        sub_df1 = df_c1.query(f'B2 == {B2} or B2 == -1')
        sub_df2 = df_c2.query(f'B2 == {B2} or B2 == -1')
        
        
        for method in methods:
            # auc_sum = 0
            # for n in range(len(sub_df2) // len(sub_df1)):
            #     start, end = n * len(sub_df1), (n+1) * len(sub_df1)
            #     auc_sum += compute_auc(pd.concat([sub_df1, sub_df2[start:end]]), method)
            # auc = auc_sum / (len(sub_df2) // len(sub_df1))

            auc = compute_auc(pd.concat([sub_df1, sub_df2]), method)
            print(f"{method}: {auc}")
        print()


        # y_true = (sub_df["B2"] > 0)
        # for method in methods:

        #     y_score = sub_df[method]
        #     auc = roc_auc_score(y_true, y_score)
        #     print(f"{method}: {auc}")
        #     # print(f"{auc:.3f}")
        # print()





def detection_AUC():
    df_c1 = pd.concat([load_score_df_from_json(fname_set2c1), load_score_df_from_json(fname_set3c1)]) # fname_set2c1
    df_c2 = pd.concat([load_score_df_from_json(fname_set2c2), load_score_df_from_json(fname_set3c2)]) # fname_set2c2
    # df_c2 = df_c2.sample(n=len(df_c1), random_state=1)


    # df = pd.concat([df_c1, df_c2])
    # print(df)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100
    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]
    methods = ["Chen", "Yao", "Vazquez", "aContrario"]


    # for B1 in B1_arr:
    #     results = []
    #     print(f"B1={B1}")
    #     for B2 in B1_arr:
    #         results_B2 = []
    #         sub_df = df.query(f'B1 == {B1} and (B2 == {B2} or B2 == -1)')
            # print(f"B1={B1}, B2={B2}")
    results = []
    # for G1 in GOP_c1_options:
        # g1_results = []
        # sub_df = df.query(f'G1 == {G1}')
        # print(f"G1={G1}")
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            
            sub_df1 = df_c1.query(f'B1 == {B1} and B2 == -1')
            sub_df2 = df_c2.query(f'B1 == {B1} and B2 == {B2}')
            for method in methods:
                auc_sum = 0
                for n in range(len(sub_df2) // len(sub_df1)):
                    start, end = n * len(sub_df1), (n+1) * len(sub_df1)
                    auc_sum += compute_auc(pd.concat([sub_df1, sub_df2[start:end]]), method)
                auc = auc_sum / (len(sub_df2) // len(sub_df1))

                # print(f"{method}: {auc}")
                print(f"{auc}")
            print()

        # optional: mix    
        # print(f"B1=mix, B2={B2}")
        # sub_df1 = df_c1.query(f'B2 == {B2} or B2 == -1')
        # sub_df2 = df_c2.query(f'B2 == {B2} or B2 == -1')
        # for method in methods:
        #     auc_sum = 0
        #     for n in range(len(sub_df2) // len(sub_df1)):
        #         start, end = n * len(sub_df1), (n+1) * len(sub_df1)
        #         auc_sum += compute_auc(pd.concat([sub_df1, sub_df2[start:end]]), method)
        #     auc = auc_sum / (len(sub_df2) // len(sub_df1))
        #     # print(f"{method}: {auc}")
        #     print(f"{auc}")

        print()
        
    for method in methods:
        auc_sum = 0
        for n in range(len(df_c2) // len(df_c1)):
            start, end = n * len(df_c1), (n+1) * len(df_c1)
            auc_sum += compute_auc(pd.concat([df_c1, df_c2[start:end]]), method)
        auc = auc_sum / (len(df_c2) // len(df_c1))

        # print(f"{method}: {auc}")
        print(f"{auc}")


def aCont_accuracy():
    df = load_gop_and_NFA_df_from_json(fname_c2)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100

    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    for G1 in GOP_c1_options:
        # for G2 in GOP_c2_options:
        sub_df = df.query(f'G1 == {G1}')
        print(f"G1={G1}")
    # for B1 in B1_arr:
        # for B2 in B2_arr:
            # print(f"B1={B1}, B2={B2}")
        # sub_df = df.query(f'B1 == {B1} and B2 == {B2} and NFA < 1')
        method = "aContrario"
        # print(sub_df.query(f'{method} != G1'))
        acc = len(sub_df.query(f'{method} == G1')) / len(sub_df) # TP / (FP + TP) = 1 - FP / (FP + TP)
            # print(f"{method}: acc={acc:.2f}")
        print(f"{acc:.3f}")
        print()


def plot_precision_recall(plot_fname_prefix=None):
    df_c1 = load_score_df_from_json(fname_set2c1)
    df_c2 = load_score_df_from_json(fname_set2c2)
    df_c2 = df_c2.sample(n=len(df_c1) * 1, random_state=2)
    df = pd.concat([df_c1, df_c2])
    
    print(df)
    
    fig, ax = plt.subplots()
    for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
        y_true = (df['B2'] > 0)
        probas_pred = df[method]
        precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)

        # print("precision:", precision)
        # print("recall:", recall)
        # print("thresholds:", thresholds)

        #create precision recall curve
        ax.plot(recall, precision, label=method + " (I and P)")
    
    



    # optional: show b-frame
    df_c1b = load_gop_and_NFA_df_from_json(fname_set2c1b)
    df_c2b = load_gop_and_NFA_df_from_json(fname_set2c2b)
    df_c2b = df_c2b.sample(n=len(df_c1b), random_state=5)
    dfb = pd.concat([df_c1b, df_c2b])

    y_true = (dfb['B2'] > 0)
    probas_pred = -dfb['NFA']
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    ax.plot(recall, precision, '--', label="aContrario (I, P and B)")
    
    legend = ax.legend(loc="lower left")
    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    if plot_fname_prefix is not None:
        plt.savefig(f"{plot_fname_prefix}.png")
        plt.savefig(f"{plot_fname_prefix}.eps")

        #display plot
        # plt.show()

def plot_roc(plot_fname_prefix):

    df_c1 = load_score_df_from_json(fname_c1)
    df_c2 = load_score_df_from_json(fname_c2)
    df_c2 = df_c2.sample(n=len(df_c1) * 2, random_state=1)
    df = pd.concat([df_c1, df_c2])

    fig, ax = plt.subplots()
    for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
        y_true = (df['B2'] > 0)
        probas_pred = df[method]

        fpr, tpr, _ = roc_curve(y_true,  probas_pred)

    #create ROC curve
        ax.plot(fpr, tpr, label=method)

    legend = ax.legend()
    ax.set_title('ROC Curve')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    if plot_fname_prefix is not None:
        plt.savefig(f"{plot_fname_prefix}.png")
        plt.savefig(f"{plot_fname_prefix}.eps")


def decide_threshold():
    """ choose threshold with precision > 90%
    """
    df_c1 = load_score_df_from_json(fname_set2c1)
    df_c2 = load_score_df_from_json(fname_set2c2)
    df_c2 = df_c2.sample(n=len(df_c2), random_state=1)
    # df = pd.concat([df_c1, df_c2])

    thresholds_result = {}
    for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
        
        threshold_subset = []
        aucpr_sample = 0
        for n in range(len(df_c2) // len(df_c1)):
            start, end = n * len(df_c1), (n + 1) * len(df_c1)
            sub_df = pd.concat([df_c1, df_c2[start:end]])

            y_true = (sub_df['B2'] > 0)
            probas_pred = sub_df[method]
            precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
            
            aucpr_sample += auc(recall, precision)
            

            # take the threshold corresponding to Precision = 0.90
            tolerance = 0.95
            i_best = np.where(precision >= tolerance)[0].min()

            # take th threshold of the best F1 score
            # i_best = np.argmax(precision * recall / (precision + recall))

            # precision_best = precision[i_best]
            recall_best = recall[i_best]
            threshold = thresholds[i_best]
            threshold_subset.append(threshold)
        
        aucpr = aucpr_sample / (len(df_c2) // len(df_c1))
        print(f"AUC-PR = {aucpr}")

        threshold = np.median(np.array(threshold_subset))

        print(f"At precision>={tolerance}, {method} has threshold={threshold} with recall={recall_best}")
        # print(f"The best F1 score of {method} is associated to precision={precision_best} and recall={recall_best}")

        thresholds_result[method] = threshold

    return thresholds_result


# fname_set3c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230221_001327.json"
# fname_set3c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230221_001830.json"

# TODO: Reprocess the videos in set3

# 900 frames
# fname_set3c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230221_190804.json"
# fname_set3c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230221_193157.json"

# 600 frames
# fname_set3c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230221_193828.json"
# fname_set3c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230221_193909.json"

# 300 frames
# fname_set3c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230221_194208.json"
# fname_set3c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230221_194221.json"

# 400 frames, updated NFA
fname_set3c1 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c1_20230222_173843.json"
fname_set3c2 = "/mnt/ddisk/yanhao/gop_detection/results_cbr_set3c2_20230222_173851.json"


def compute_precision_recall(thresholds):
    df_c1 = load_score_df_from_json(fname_set3c1)
    df_c2 = load_score_df_from_json(fname_set3c2)
    # df_c2 = df_c2.sample(n=len(df_c1), random_state=1)
    # df = pd.concat([df_c1, df_c2])
    methods = ["Chen", "Yao", "Vazquez", "aContrario"]

    # print(df)
    file = open("dump.csv", 'w')

    
    
    for method in methods:
        threshold = thresholds[method]
        precision_sample, recall_sample = 0, 0
        for n in range(len(df_c2) // len(df_c1)):
            # probas_pred = df[method]
            start, end = n * len(df_c1), (n + 1) * len(df_c1)
            sub_df = pd.concat([df_c1, df_c2[start:end]])

            y_true = (sub_df['B2'] > 0)
            pred_true = sub_df[method] > threshold

            if pred_true.sum() > 0:
                precision_sample += (pred_true & y_true).sum() / pred_true.sum()
            else:
                precision_sample += 0.5
            recall_sample += (pred_true & y_true).sum() / y_true.sum()
        
        precision = precision_sample / (len(df_c2) // len(df_c1))
        recall = recall_sample / (len(df_c2) // len(df_c1))

        F1 = 2 * precision * recall / (precision + recall)
        print(f"{method}: precision= {precision}")
        print(f"{method}: recall= {recall}")
        print(f"{method}: F1= {F1}")
        print()

        file.write(f"{precision},{recall},{F1} \n")

    e_arr = [-1, -0.1]
    # add case with epsilon=0.1
    
    method = "aContrario"
    for threshold in e_arr:
        precision_sample, recall_sample = 0, 0
        for n in range(len(df_c2) // len(df_c1)):
            start, end = n * len(df_c1), (n + 1) * len(df_c1)
            sub_df = pd.concat([df_c1, df_c2[start:end]])

            y_true = (sub_df['B2'] > 0)
            pred_true = sub_df[method] > threshold

            precision_sample += (pred_true & y_true).sum() / pred_true.sum()
            recall_sample += (pred_true & y_true).sum() / y_true.sum()
            
        precision = precision_sample / (len(df_c2) // len(df_c1))       
        recall = recall_sample / (len(df_c2) // len(df_c1))       
        F1 = 2 * precision * recall / (precision + recall)

        print(f"{method}: precision= {precision}")
        print(f"{method}: recall= {recall}")
        print(f"{method}: F1= {F1}")
        print()

        file.write(f"{precision},{recall},{F1}\n")

    file.close()


if __name__ == "__main__":
    # GOP_accuracy()
    # detection_AUC_old()
    # detection_AUC()

    # aCont_accuracy()

    # plot_PR_prefix = "plot_PR"
    # plot_precision_recall(plot_PR_prefix)

    # plot_ROC_fname = "plot_ROC.eps"
    # plot_roc(plot_ROC_fname)

    thresholds = decide_threshold()
    # print(thresholds)

    # compute_precision_recall(thresholds)

    # GOP_accuracy_bframe()