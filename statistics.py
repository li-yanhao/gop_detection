import json
import numpy as np
import pandas as pd


def GOP_accuracy():
    fname = "results_cp.json"
    json_file = open(fname, "r")
    data = json.load(json_file)
    # print(data)

    raw_data = [
        [
            item["B1"], item["B2"], item["G1"], item["G2"],
            item["Chen"][0], item["Yao"][0], item["Vazquez"][0], item["aContrario"][0]
        ] 
    for item in data]

    df = pd.DataFrame(raw_data, 
        columns=["B1", "B2", "G1", "G2", "Chen", "Yao", "Vazquez", "aContrario"])
    print(df)

    B1_arr = [300, 700, 1100]
    B2_arr = [300, 700, 1100]
    
    for B1 in B1_arr:
        for B2 in B2_arr:
            print(f"B1={B1}, B2={B2}")
            sub_df = df.query(f'B1 == {B1} and B2 == {B2}')
            for method in ["Chen", "Yao", "Vazquez", "aContrario"]:
                acc = len(sub_df.query(f'{method} == G1')) / len(sub_df)
                print(f"{method}: acc={acc:.2f}")
            print()


def detection_AUC():
    pass


if __name__ == "__main__":
    GOP_accuracy()
