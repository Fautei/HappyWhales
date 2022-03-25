from Model import BaseModel
from Dataset import TrainDataset
import utils
import wandb
import torch
import pandas as pd
import os
import json
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn import model_selection

WandBLOGIN = "7f337ecfba84fc0060f00206c9d85053744a76ab"
SEED = 69
BATCH_SIZE = 256
ACCUMULATE_GRAD_BATCHES = 8
IMAGE_SIZE = 256
NUM_WORKERS = 2
EPOCHS = 30
MIXED_PRECISION = True
FEATURE_EXTRACTOR = "efficientnet_b0"

def pretrain(df_path, images):
    utils.fix_seed(SEED)
    wandb.login(key=WandBLOGIN)

    df = pd.read_csv(df_path)
    df =  df[:1000]
    utils.preprocess_df(df)
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=SEED, stratify=df.species.values)


    train_dataset = TrainDataset(df_train, images, augment = True, image_size = IMAGE_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=True
    )

    val_dataset = TrainDataset(df_valid, images, augment = False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False
    )

    #train_dataset.plot_sample(3)

    model = BaseModel(feature_extractor = FEATURE_EXTRACTOR, classify = True)
    print(model)
    wandb_logger = WandbLogger(project="W&D - identification")

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='output/checkpoints/',
        filename=FEATURE_EXTRACTOR+'-{epoch}-{train_loss:.2f}'
)
    trainer = Trainer(
        auto_lr_find = True,
        callbacks = [checkpoint_callback],
        profiler="simple", # Profiling
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,# Accumulate gradient over multiple batches
        devices = -1,
        accelerator="cpu",# Use the one GPU we have
        precision=16 if MIXED_PRECISION else 32,# Mixed precision
        max_epochs=EPOCHS,
        logger=wandb_logger,
        log_every_n_steps=10,
        deterministic=True
        )

    lr_finder = trainer.tuner.lr_find(model,train_loader,val_loader)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.learning_rate = new_lr
    trainer.fit(model, train_loader, val_loader)

    model_name = "output/"+FEATURE_EXTRACTOR + "-final.ckpt"
    trainer.save_checkpoint(model_name)
    artifact = wandb.log_artifact(model_name, name='w_and_d-id-normal', type='model') 
    
    wandb_logger.finalize("success")
    wandb.finish()






if __name__ == "__main__":
    pretrain("train2.csv","cropped_train_images/cropped_train_images")
