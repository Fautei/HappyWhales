from Model import WandDIDNet
from Dataset import TrainDataset
import utils
import wandb
import torch
import pandas as pd
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import numpy as np
import json

WandBLOGIN = "7f337ecfba84fc0060f00206c9d85053744a76ab"
SEED = 69
BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 1
NUM_WORKERS = 2
EPOCHS = 2
MIXED_PRECISION = True
BASEMODEL_PATH = 'output/efficientnet_b0-final.ckpt'
FEATURE_EXTRACTOR = "efficientnet_b0"
MARGIN_MIN = 0.2
MARGIN_MAX = 0.4

def train():
    utils.fix_seed(SEED)
    wandb.login(key=WandBLOGIN)

    df = pd.read_csv(os.path.join("input/", "train.csv"))
    utils.preprocess_df(df)

    margins = utils.get_margins(df,MARGIN_MIN, MARGIN_MAX)

    base_path = "input/train/"

    train_dataset = TrainDataset(df, base_path, augment = False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True
    )

    #train_dataset.plot_sample(3)

    model = WandDIDNet(BASEMODEL_PATH,FEATURE_EXTRACTOR, margins)
    print(model)
    wandb_logger = WandbLogger(project="W&D - identification")

    checkpoint_callback = ModelCheckpoint(
        dirpath='output/checkpoints/',
        filename='W_AND_D-{epoch}-{train_loss:.2f}-{train_acc:.2f}')

    trainer = Trainer(
        callbacks = [checkpoint_callback],
        profiler="simple", # Profiling
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,# Accumulate gradient over multiple batches
        accelerator="cpu",# Use the one GPU we have
        precision=16 if MIXED_PRECISION else 32,# Mixed precision
        max_epochs=EPOCHS,
        #logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True
        )

    # Let's go ⚡
    trainer.fit(model, train_loader)

    trainer.save_checkpoint("output/W_AND_D-final.ckpt")
    artifact = wandb.log_artifact("output/W_AND_D-final.ckpt", name='w_and_d-id-normal', type='model') 
    model.save_class_weights()
    artifact = wandb.log_artifact('class_weights.pt', name='w_and_d-id-normal-weights', type="class_weights")

    wandb_logger.finalize("success")
    wandb.finish()

if __name__ == "__main__":
    train()
