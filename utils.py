import numpy as np
import pandas as pd
import os
import torch
import random
import json


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #gpu randomseed fixed
    torch.backends.cudnn.deterministic = True

def preprocess_df(df):
    individual_mapping = {k:i for i, k in enumerate(df["individual_id"].unique())}
    with open("output/individual_mapping.json", "w") as f:
        json.dump(individual_mapping, f)
    df["individual_id_integer"] = df["individual_id"].map(individual_mapping)


def get_margins(df, min,max):
    with open("output/individual_mapping.json", "r") as f:
        individual_mapping = json.load(f)
        tmp = np.sqrt(1 / np.sqrt(df['individual_id'].value_counts().loc[list(individual_mapping)].values))
        margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (max -min) + min
        return margins
