import pandas as pd
import numpy as np
import torch
import os
import math
import copy
import torchmetrics
import torchvision
import pytorch_lightning as pl
import wandb
import json
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms, models
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.neighbors import NearestNeighbors
import timm
from Model import WandDIDNet
from Dataset import TrainDataset, TestDataset
import utils
import train

WandBLOGIN = "7f337ecfba84fc0060f00206c9d85053744a76ab"
SEED = 69
BATCH_SIZE = 8
NUM_WORKERS = 2
CHECKPOINT_PATH = 'output/W_AND_D-final.ckpt'
CENTERS_PER_CLASS = 3
MIXED_PRECISION = True
KNN = 100
Q_NEW = 0.112 # Proportion of new individuals expected in the dataset

class Solution:
    def __init__(self, database, q_prior, individual_mapping):
        self.individual_mapping = individual_mapping
        self.database_embeddings = np.array(database["embeddings"]["embedding"].values.tolist())
        self.database_individuals = database["embeddings"]["individual_id"].values
        self.q_prior = q_prior
        self.embed_neigh = NearestNeighbors(n_neighbors=KNN,metric='cosine')
        self.embed_neigh.fit(self.database_embeddings)
        self.class_neigh = NearestNeighbors(n_neighbors=KNN,metric='cosine')
        self.class_neigh.fit(database["class_centers"])
        self.default = ['938b7e931166', '5bf17305f073', '7593d2aee842', '7362d7a01d00','956562ff2888']        
    
    def predict(self, queries):
        embed_distances, embed_idxs = self.embed_neigh.kneighbors(queries, KNN, return_distance=True)
        class_distances, class_idxs = self.class_neigh.kneighbors(queries, KNN, return_distance=True)
        
        print("embed_distances\n\n",embed_distances,"\n\n")
        print("embed_idxs\n\n",embed_idxs,"\n\n")
        print("class_distances\n\n",class_distances,"\n\n")
        print("class_idxs\n\n",class_idxs,"\n\n")

        class_individuals = np.repeat(list(self.individual_mapping), CENTERS_PER_CLASS)[class_idxs]
        embed_individuals = self.database_individuals[embed_idxs]
        
        print(" class_individuals\n\n", class_individuals,"\n\n")
        print(" embed_individuals\n\n",embed_individuals,"\n\n")

        n = embed_distances.size
        embeddings_df = pd.DataFrame(data={
            'distance': embed_distances.ravel(),
            'individual': embed_individuals.ravel(),
            'query_id': np.repeat(np.arange(len(queries)), KNN)
        }, index=np.arange(n))
        
        class_df = pd.DataFrame(data={
            'distance': class_distances.ravel(),
            'individual': class_individuals.ravel(),
            'query_id': np.repeat(np.arange(len(queries)), KNN)
        }, index=np.arange(n))
        print(embeddings_df.head(),'\n\n')
        print(class_df.head(),'\n\n')

        embeddings_topk = embeddings_df.groupby(["query_id", "individual"]).agg("min")['distance'].groupby('query_id', group_keys=False).nsmallest(5)
        class_topk = class_df.groupby(["query_id", "individual"]).agg("min")['distance'].groupby('query_id', group_keys=False).nsmallest(5)
        embeddings_topk = embeddings_topk.reset_index().groupby("query_id").agg(list)
        class_topk = class_topk.reset_index().groupby("query_id").agg(list)
        class_t_new = np.quantile(class_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)
        embeddings_t_new = np.quantile(embeddings_topk["distance"].apply(lambda x: x[0]), 1 - self.q_prior)
        
        print(" embeddings_topk\n\n",embeddings_topk,"\n\n")
        print(" class_topk\n\n",class_topk,"\n\n")
        print(" embeddings_topk\n\n",embeddings_topk,"\n\n")
        print(" class_topk\n\n",class_topk,"\n\n")
        print(" class_t_new\n\n",class_t_new,"\n\n")
        print(" embeddings_t_new\n\n",embeddings_t_new,"\n\n")

        def insert_new_individuals(x, t_new):
            m = np.array(x["distance"]) > t_new
            preds = x["individual"]
            if m.any():
                preds.insert(np.argmax(m), "new_individual")
            preds = preds + [y for y in self.default if y not in preds]
            return preds[:5]
        
        preds1 = class_topk.apply(insert_new_individuals,t_new=class_t_new,  axis=1)
        preds2 = embeddings_topk.apply(insert_new_individuals,t_new=embeddings_t_new, axis=1)
        return preds1.values.tolist(),preds2.values.tolist()

def predict():
    wandb.login(key=WandBLOGIN)
    df = pd.read_csv(os.path.join("input/", "train.csv"))
    utils.preprocess_df(df)
    utils.fix_seed(SEED)
    

    train_path = "input/train/"

    test_dataset = TestDataset(df,train_path)

    # Prediction on the training data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    model = WandDIDNet.load_from_checkpoint(CHECKPOINT_PATH, model_path = train.BASEMODEL_PATH,model_name=train.FEATURE_EXTRACTOR, margins = utils.get_margins(df,0.2,0.4))
    trainer = Trainer(
        accelerator="cpu",# Use the one GPU we have
        precision=16 if MIXED_PRECISION else 32,# Mixed precision
        )


    preds = trainer.predict(model, dataloaders=test_loader)
    preds = torch.cat(preds, dim=0)

    train_data = df.copy()
    train_data["embedding"] = preds.tolist()
    train_data.to_csv("output/train.csv")

    # Prediction on test data
    test_path = "input/test/"
    test_data = pd.read_csv("input/test.csv")

    pred_dataset = TestDataset(test_data, test_path)    
    pred_loader = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )
    preds = trainer.predict(model, dataloaders=pred_loader)
    preds = torch.cat(preds, dim=0)

    print("PREDS\n\n",preds,"\n\n")

    test_data["embedding"] = preds.tolist()
    test_data.to_csv("output/test.csv")

    individual_mapping = None
    with open("output/individual_mapping.json", "r") as f:
        individual_mapping = json.load(f)

    train_data["individual_id_integer"] = train_data["individual_id"].map(individual_mapping).fillna(-1)
    train_embeddings = np.array(train_data["embedding"].values.tolist())
    test_embeddings = np.array(test_data["embedding"].values.tolist())
    class_centers = model.metric_classify.weight.detach().numpy()

    print("CENTERS\n\n",class_centers,"\n\n")


    solution = Solution({
        "embeddings": train_data,
        "class_centers": class_centers
    }, Q_NEW, individual_mapping)
    predictions1,predictions2 = solution.predict(test_embeddings)
    predictions1 = pd.Series(predictions1, test_data.index, name="predictions").map(lambda x: " ".join(x))
    predictions1.to_csv("output/submission_class.csv")

    predictions2 = pd.Series(predictions2, test_data.index, name="predictions").map(lambda x: " ".join(x))
    predictions2.to_csv("output/submission_embendings.csv")


    artifact = wandb.log_artifact("output/submission_class.csv", name='submission', type="submission") 
    artifact = wandb.log_artifact("output/submission_embendings.csv", name='submission', type="submission") 

    wandb_logger.finalize("success")
    wandb.finish()

if __name__ == "__main__":
    predict()