import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


# fname_c1 = "/mnt/ddisk/yanhao/gop_detection/results_set1c1_20230219_175525.json"
# fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_set1c2_20230219_175536.json"

fname_c1 = "/mnt/ddisk/yanhao/gop_detection/results_set2c1_20230219_184317.json"
fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_set2c2_20230219_184327.json"

def GOP_accuracy():
    # fname_c2 = "/mnt/ddisk/yanhao/gop_detection/results_set1c2_20230219_175207.json"

    df = load_gop_df_from_json(fname_c2)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100

    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    # for G1 in GOP_c1_options:
        # for G2 in GOP_c2_options:
        # sub_df = df.query(f'G1 == {G1}')
        # print(f"G1={G1}")
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            sub_df = df.query(f'B1 == {B1} and B2 == {B2}')
            for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
                acc = len(sub_df.query(f'{method} == G1')) / len(sub_df)
                # print(f"{method}: acc={acc:.2f}")
                print(f"{acc:.3f}")
            print()


def detection_AUC():
    df_c1 = load_score_df_from_json(fname_c1)

    df_c2 = load_score_df_from_json(fname_c2)
    df_c2 = df_c2.sample(n=len(df_c1), random_state=1)


    df = pd.concat([df_c1, df_c2])
    print(df)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100
    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]
    # for B1 in B1_arr:
        # for B2 in B1_arr:
            # for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
            # sub_df = df.query(f'B1 == {B1} and (B2 == {B2} or B2 == -1)')
        # print(f"B1={B1}, B2={B2}")
    for G1 in GOP_c1_options:
        # sub_df = df.query(f'B1 == {B1} and G1 == {G1}')
        sub_df = df.query(f'G1 == {G1}')
        # print(f"B1={B1}, G1={G1}")
        print(f"G1={G1}")

        y_true = (sub_df["B2"] > 0)
        # print(y_true)
        for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
            y_score = sub_df[method]
            # print(y_score)
            auc = roc_auc_score(y_true, y_score)
            # print(f"{method}: AUC = {auc}")
            print(f"{auc:.3f}")

        print()
    pass


def aCont_accuracy():
    df = load_gop_and_NFA_df_from_json(fname_c2)

    B1_arr = [300, 700, 1100] # 1100
    B2_arr = [300, 700, 1100] # 1100

    GOP_c1_options = [10, 15, 30, 40]
    GOP_c2_options = [9, 16, 33, 50]

    # for G1 in GOP_c1_options:
        # for G2 in GOP_c2_options:
        # sub_df = df.query(f'G1 == {G1}')
        # print(f"G1={G1}")
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            sub_df = df.query(f'B1 == {B1} and B2 == {B2} and NFA < 1')
            method = "aContrario"
            print(sub_df.query(f'{method} != G1'))
            acc = len(sub_df.query(f'{method} == G1')) / len(sub_df)
                # print(f"{method}: acc={acc:.2f}")
            print(f"{acc:.3f}")
            print()


if __name__ == "__main__":
    # GOP_accuracy()
    # detection_AUC()
    aCont_accuracy()